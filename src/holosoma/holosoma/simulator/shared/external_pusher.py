"""MuJoCo-compatible wrench applier for inference-time force disturbances.

Applies timed external forces to allow-listed rigid bodies via xfrc_applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from holosoma.config_types.simulator import ExternalPusherCfg
from holosoma.utils.safe_torch_import import torch


@dataclass
class _ActiveWindow:
    """Runtime state for the currently active push window."""

    body_id: int
    body_name: str
    force_vec: np.ndarray   # shape (3,) — world-frame force
    torque_vec: np.ndarray   # shape (3,) — world-frame torque
    start_time: float
    end_time: float  # start_time + duration_s


class MuJoCoEEWrenchApplier:
    """MuJoCo-compatible wrench applier for inference-time force disturbances.

    Fills the sim2sim gap left by EEWrenchRandomizerState (IsaacSim-only).
    Applies timed external forces to allow-listed rigid bodies via xfrc_applied.
    """

    def __init__(
        self,
        sim: Any,
        cfg: ExternalPusherCfg,
    ) -> None:
        """Initialise the wrench applier.

        Parameters
        ----------
        sim : Any
            Simulator instance exposing ``applied_forces`` and
            ``find_rigid_body_indice``.
        cfg : ExternalPusherCfg
            Pusher configuration dataclass.

        Raises
        ------
        ValueError
            If ``interval_range_s`` is invalid or a schedule event has a
            ``force_vec`` with length != 3.
        TypeError
            If a schedule event has a non-string ``body_name``.
        """
        self.sim = sim
        self.cfg = cfg

        # Backend detection — same pattern as VirtualGantry
        self._is_gpu: bool = isinstance(sim.applied_forces, torch.Tensor)

        # --- Resolve body names to IDs ---
        self._body_id_map: dict[str, int] = {}
        for name in cfg.allowed_bodies:
            try:
                body_id = sim.find_rigid_body_indice(name)
                self._body_id_map[name] = body_id
            except RuntimeError:
                logger.warning(
                    f"ExternalPusher: body '{name}' not found in model, "
                    "excluding from runtime allow-list"
                )

        if not self._body_id_map:
            logger.warning(
                "ExternalPusher: no bodies resolved — applier is effectively disabled"
            )

        # --- Validate schedule (hardcoded mode) ---
        for i, event in enumerate(cfg.schedule):
            if not isinstance(event.body_name, str):
                raise TypeError(
                    f"Schedule event {i}: body_name must be str, "
                    f"got {type(event.body_name).__name__}"
                )
            if len(event.force_vec) != 3:
                raise ValueError(
                    f"Schedule event {i}: force_vec must have length 3, "
                    f"got {len(event.force_vec)}"
                )

        # --- Validate interval_range_s ---
        if cfg.interval_range_s[0] > cfg.interval_range_s[1]:
            raise ValueError(
                f"interval_range_s[0] ({cfg.interval_range_s[0]}) must be "
                f"<= interval_range_s[1] ({cfg.interval_range_s[1]})"
            )

        # --- Runtime state ---
        self._active_windows: list[_ActiveWindow] = []
        self._schedule_idx: int = 0  # next event index for hardcoded mode

        # --- Randomized mode RNG ---
        self._rng: np.random.Generator = np.random.default_rng(cfg.rng_seed)
        self._next_push_time: float | None = None
        # Sample first interval so it's ready when wrench is activated
        self._first_interval: float = float(
            self._rng.uniform(cfg.interval_range_s[0], cfg.interval_range_s[1])
        )
        self._wrench_activated_once: bool = False

        # --- Eval logger (optional) ---
        self._eval_logger: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, current_time: float, apply_wrench: bool) -> None:
        """Called each physics step. Manages push window lifecycle and force writes.

        Parameters
        ----------
        current_time : float
            Current simulation time in seconds (``MuJoCo.time()``).
        apply_wrench : bool
            Whether EE wrench application is active (toggled by 'P' key).
        """
        # Task 2.2 — disabled / enabled gate (pure no-op)
        if not self.cfg.enabled or self.cfg.mode == "disabled":
            return

        # --- Auto-reset on deactivation ---
        if self._wrench_activated_once and not apply_wrench and not self._active_windows:
            self.reset()

        # --- Expire finished windows ---
        self._active_windows = [
            w for w in self._active_windows if current_time < w.end_time
        ]

        # --- Attempt to open new windows ---
        # Hardcoded mode can open multiple windows on distinct bodies even
        # while others are already active. Randomized mode keeps its
        # original single-salvo behaviour (one window per interval).
        if self.cfg.mode == "hardcoded":
            self._try_open_hardcoded(current_time, apply_wrench)
        elif self.cfg.mode == "randomized":
            if not self._active_windows:
                self._try_open_randomized(current_time, apply_wrench)

        # --- Apply wrenches (backend-aware, assignment not accumulation) ---
        if self._active_windows:
            for w in self._active_windows:
                self._apply_wrench(w.body_id, w.force_vec, w.torque_vec)
            # DEBUG: print applied forces
            if int(current_time * 100) % 50 == 0:  # every 0.5s
                for w in self._active_windows:
                    remaining = w.end_time - current_time
                    logger.info(
                        f"[PUSHER DEBUG] t={current_time:.2f} body={w.body_name}({w.body_id}) "
                        f"force={w.force_vec} torque={w.torque_vec} remaining={remaining:.2f}s"
                    )
        else:
            self._clear_allow_list_forces()

    def attach_logger(self, eval_logger: Any) -> None:
        """Store eval logger reference for push event emission.

        Parameters
        ----------
        eval_logger : Any
            The eval logger instance (``EvalLogger``).
        """
        self._eval_logger = eval_logger
    def reset(self) -> None:
        """Reset all runtime state for a fresh push sequence.

        Call this when the policy stops / simulation resets so the next
        policy activation gets a clean schedule. Re-seeds the RNG to
        produce the same sequence as a fresh construction.
        Also clears applied_forces immediately so no stale force persists.
        """
        self._active_windows = []
        self._schedule_idx = 0
        self._wrench_activated_once = False
        self._next_push_time = None

        # Clear any residual forces immediately
        self._clear_allow_list_forces()

        # Re-seed RNG so the next run is identical to a fresh start
        self._rng = np.random.default_rng(self.cfg.rng_seed)
        self._first_interval = float(
            self._rng.uniform(self.cfg.interval_range_s[0], self.cfg.interval_range_s[1])
        )
        logger.info("ExternalPusher: reset — schedule and RNG re-initialised")


    # ------------------------------------------------------------------
    # Hardcoded mode (task 2.3)
    # ------------------------------------------------------------------

    def _try_open_hardcoded(
        self, current_time: float, apply_wrench: bool
    ) -> None:
        """Consume all hardcoded schedule events whose time has arrived.

        Multiple simultaneous events are allowed as long as they target
        **distinct bodies** (e.g. one event per EE at the same ``sim_time_s``).
        If an event targets a body that already has an active window, we
        stop scanning and leave it for the next step so the previous window
        can expire first (preserves the single-window-per-body invariant).
        Events are still gated on ``apply_wrench``; if false, the next event
        is deferred until it becomes true.
        """
        schedule = self.cfg.schedule
        while self._schedule_idx < len(schedule):
            event = schedule[self._schedule_idx]

            # Not yet time for this event — stop scanning, leave pointer here
            if current_time < event.sim_time_s:
                return

            # Time has arrived — but gate on apply_wrench
            if not apply_wrench:
                # Defer: don't advance index, wait for wrench to be enabled
                return

            # If this body already has an active window, defer this event
            # until the existing window expires (preserves old semantics for
            # back-to-back events on the same body).
            if any(w.body_name == event.body_name for w in self._active_windows):
                return

            # Advance past this event regardless of body resolution
            self._schedule_idx += 1

            # Skip if body not in runtime allow-list
            if event.body_name not in self._body_id_map:
                logger.warning(
                    f"ExternalPusher: hardcoded event body '{event.body_name}' "
                    "not in runtime allow-list, skipping"
                )
                continue

            # Open push window
            body_id = self._body_id_map[event.body_name]
            force_vec = np.array(event.force_vec, dtype=np.float64)
            window = _ActiveWindow(
                body_id=body_id,
                body_name=event.body_name,
                force_vec=force_vec,
                torque_vec=np.zeros(3, dtype=np.float64),
                start_time=current_time,
                end_time=current_time + event.duration_s,
            )
            self._active_windows.append(window)
            self._emit_push_event(window)
            # Continue scanning — more events may share this sim time (on other bodies)

    # ------------------------------------------------------------------
    # Randomized mode (task 2.4)
    # ------------------------------------------------------------------

    def _try_open_randomized(
        self, current_time: float, apply_wrench: bool
    ) -> None:
        """Sample independent wrenches for each body in the allow-list."""
        if not apply_wrench:
            return

        if not self._body_id_map:
            return

        # Anchor schedule on first activation
        if not self._wrench_activated_once:
            self._wrench_activated_once = True
            self._next_push_time = current_time + self._first_interval

        if self._next_push_time is None:
            return

        if current_time < self._next_push_time:
            return

        # --- Sample independent wrench for each body ---
        for body_name, body_id in self._body_id_map.items():
            direction = self._sample_direction()
            magnitude = float(
                self._rng.uniform(self.cfg.force_range[0], self.cfg.force_range[1])
            )
            force_vec = direction * magnitude

            torque_lo, torque_hi = self.cfg.torque_range_nm
            if torque_hi > 1e-9:
                torque_dir = self._sample_direction()
                torque_mag = float(self._rng.uniform(torque_lo, torque_hi))
                torque_vec = torque_dir * torque_mag
            else:
                torque_vec = np.zeros(3, dtype=np.float64)

            window = _ActiveWindow(
                body_id=body_id,
                body_name=body_name,
                force_vec=force_vec,
                torque_vec=torque_vec,
                start_time=current_time,
                end_time=current_time + self.cfg.duration_s,
            )
            self._active_windows.append(window)
            self._emit_push_event(window)

        # Sample next interval
        next_interval = float(
            self._rng.uniform(
                self.cfg.interval_range_s[0], self.cfg.interval_range_s[1]
            )
        )
        self._next_push_time = current_time + next_interval

    def _sample_direction(self) -> np.ndarray:
        """Sample a unit direction vector according to ``direction_mode``."""
        if self.cfg.direction_mode == "axes":
            # Uniform over 6 signed unit axes: +x, -x, +y, -y, +z, -z
            axes = np.array([
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ], dtype=np.float64)
            idx = self._rng.integers(0, 6)
            return axes[idx]
        else:
            # "sphere": uniform on unit sphere via normalised Gaussian
            raw = self._rng.standard_normal(3)
            norm = np.linalg.norm(raw)
            while norm < 1e-10:
                raw = self._rng.standard_normal(3)
                norm = np.linalg.norm(raw)
            return raw / norm

    # ------------------------------------------------------------------
    # Wrench application — assignment, not accumulation
    # ------------------------------------------------------------------

    def _apply_wrench(self, body_id: int, force_vec: np.ndarray, torque_vec: np.ndarray) -> None:
        """Write wrench into ``applied_forces`` using direct assignment (``=``).

        The pusher owns the EE rows exclusively. Using ``=`` avoids
        accidental accumulation across sub-steps.

        Parameters
        ----------
        body_id : int
            MuJoCo body index.
        force_vec : np.ndarray
            3-element force vector in world frame (N).
        torque_vec : np.ndarray
            3-element torque vector in world frame (Nm).
        """
        if self._is_gpu:
            # WarpBackend: [num_envs, num_bodies, 6]
            force_t = torch.from_numpy(force_vec).float().to(self.sim.device)
            torque_t = torch.from_numpy(torque_vec).float().to(self.sim.device)
            self.sim.applied_forces[0, body_id, :3] = force_t
            self.sim.applied_forces[0, body_id, 3:6] = torque_t
        else:
            # ClassicBackend: [num_bodies, 6]
            self.sim.applied_forces[body_id, :3] = force_vec
            self.sim.applied_forces[body_id, 3:6] = torque_vec

    def _clear_allow_list_forces(self) -> None:
        """Zero allow-list body rows.

        Called when no push window is active. Required for both backends
        since we now use assignment (``=``) — stale values from the
        previous window must be explicitly cleared.
        """
        for body_id in self._body_id_map.values():
            if self._is_gpu:
                self.sim.applied_forces[0, body_id, :] = 0.0
            else:
                self.sim.applied_forces[body_id, :] = 0.0

    # ------------------------------------------------------------------
    # Eval logger integration (task 2.6)
    # ------------------------------------------------------------------

    def _emit_push_event(self, window: _ActiveWindow | None = None) -> None:
        """Emit one push event to the eval logger for a window."""
        if self._eval_logger is None:
            return

        w = window
        if w is None:
            return
        push_event = {
            "body_name": w.body_name,
            "body_id": w.body_id,
            "force_vec": w.force_vec.tolist(),
            "torque_vec": w.torque_vec.tolist(),
            "sim_time_s": w.start_time,
            "duration_s": w.end_time - w.start_time,
            "mode": self.cfg.mode,
        }

        try:
            self._eval_logger.log_event("push_event", push_event)
        except Exception:
            logger.debug("ExternalPusher: failed to emit push event to logger")
