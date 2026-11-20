"""Locomotion command sampling and gait state terms."""

from __future__ import annotations

from typing import Any, Sequence, cast

from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.rotations import quat_rotate_inverse
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float


class LocomotionCommand(CommandTermBase):
    """Stateful command term that owns command buffers and resampling logic."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        ranges = params.get("command_ranges")
        if ranges is None:
            raise ValueError("LocomotionCommand requires 'command_ranges' in params.")
        self.command_ranges: dict[str, Sequence[float]] = {key: tuple(value) for key, value in ranges.items()}
        self.stand_prob: float = float(params.get("stand_prob", 0.0))
        self.turn_in_place_prob: float = float(params.get("turn_in_place_prob", 0.0))
        self.command_dim: int = params.get("command_dim", 3)
        self.commands: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        env = self.env
        commands = torch.zeros(env.num_envs, self.command_dim, dtype=torch.float32, device=env.device)
        self.commands = commands
        if hasattr(env, "simulator"):
            env.simulator.commands = commands

    def reset(self, env_ids: torch.Tensor | None) -> None:
        commands = self.commands
        if commands is None:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if self.env.is_evaluating:
            commands[idx] = 0.0
            return

        self._resample(idx)

    def step(self) -> None:
        commands = self.commands
        if commands is None or self.env.is_evaluating:
            return

        command_cfg = getattr(self.manager, "command_cfg", None) if hasattr(self, "manager") else None
        resample_time = getattr(command_cfg, "locomotion_command_resampling_time", None) if command_cfg else None
        if resample_time is None or resample_time <= 0:
            return

        interval = int(resample_time / self.env.dt)
        if interval <= 0 or not hasattr(self.env, "episode_length_buf"):
            return

        env_ids = (self.env.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        self._resample(env_ids.to(device=self.env.device, dtype=torch.long))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resample(self, env_ids: torch.Tensor) -> None:
        commands = self.commands
        if commands is None or env_ids.numel() == 0:
            return

        device = self.env.device
        ranges = self.command_ranges

        commands[env_ids, 0] = torch_rand_float(
            ranges["lin_vel_x"][0],
            ranges["lin_vel_x"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)
        commands[env_ids, 1] = torch_rand_float(
            ranges["lin_vel_y"][0],
            ranges["lin_vel_y"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)
        commands[env_ids, 2] = torch_rand_float(
            ranges["ang_vel_yaw"][0],
            ranges["ang_vel_yaw"][1],
            (env_ids.shape[0], 1),
            device=device,
        ).squeeze(1)

        manager = getattr(self, "manager", None)
        if manager is not None:
            gait_state = manager.get_state("locomotion_gait")
        else:
            gait_state = None

        if gait_state is not None:
            cast("LocomotionGait", gait_state).resample_frequency(env_ids)

        if self.stand_prob > 0.0:
            stand_mask = torch.rand(env_ids.shape[0], device=device) <= self.stand_prob
            if stand_mask.any():
                commands[env_ids[stand_mask], :3] = 0.0

        if self.turn_in_place_prob > 0.0:
            # Among envs NOT already standing, zero linear vel but keep yaw
            remaining = ~stand_mask if self.stand_prob > 0.0 else torch.ones(env_ids.shape[0], dtype=torch.bool, device=device)
            turn_mask = remaining & (torch.rand(env_ids.shape[0], device=device) <= self.turn_in_place_prob)
            if turn_mask.any():
                commands[env_ids[turn_mask], :2] = 0.0

    def _ensure_index_tensor(self, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)


class LocomotionGait(CommandTermBase):
    """Stateful term that owns gait phase buffers and updates them each step."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.gait_period: float = float(params.get("gait_period", 1.0))
        self.gait_period_randomization_width: float = float(params.get("gait_period_randomization_width", 0.0))
        # Optional explicit period range [min, max] sampled UNIFORMLY in the
        # period domain, then converted to frequency. When provided it takes
        # precedence over gait_period_randomization_width (which is symmetric in
        # the frequency domain and cannot express an asymmetric period range).
        # Defaults to None to preserve legacy frequency-width behavior.
        gait_period_range = params.get("gait_period_range", None)
        self.gait_period_range: tuple[float, float] | None = (
            (float(gait_period_range[0]), float(gait_period_range[1]))
            if gait_period_range is not None
            else None
        )
        self.randomize_phase: bool = bool(params.get("randomize_phase", True))
        self.stand_phase_value: float = float(params.get("stand_phase_value", torch.pi))

        self.phase_offset: torch.Tensor | None = None
        self.phase: torch.Tensor | None = None
        self.gait_freq: torch.Tensor | None = None
        self.phase_dt: torch.Tensor | None = None
        self._gait_steps: torch.Tensor | None = None
        self.mean_gait_freq: float = 1.0 / self.gait_period

    def setup(self) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs

        self.phase_offset = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.phase = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.gait_freq = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        self.phase_dt = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        # Walking-only gait clock: advances only on non-standing steps so the
        # phase freezes during standing and resumes without a random teleport
        # from a free-running absolute clock.
        self._gait_steps = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)

        self._initialize_indices(None, evaluating=env.is_evaluating)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        self._initialize_indices(env_ids, evaluating=self.env.is_evaluating)

    def step(self) -> None:
        if self.phase is None or self.phase_offset is None or self.phase_dt is None:
            return
        if self._gait_steps is None:
            return

        env = self.env

        # Determine standing envs from the locomotion command (near-zero
        # velocity command). If no command tensor is available, treat all envs
        # as walking (advance every step), matching the original behavior.
        command_tensor = getattr(self.manager, "commands", None) if hasattr(self, "manager") else None
        if command_tensor is None:
            stand_mask = torch.zeros(self.phase.shape[0], dtype=torch.bool, device=env.device)
        else:
            stand_mask = torch.logical_and(
                torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
                torch.abs(command_tensor[:, 2]) < 0.01,
            )

        # Walking-only gait clock: advance the step counter only for walking
        # (non-standing) envs, so the phase resumes from where it stopped
        # instead of teleporting to a free-running absolute clock.
        walking_mask = ~stand_mask
        self._gait_steps[walking_mask] += 1.0

        phase_tp1 = self._gait_steps * self.phase_dt + self.phase_offset
        self.phase.copy_(torch.fmod(phase_tp1 + torch.pi, 2 * torch.pi) - torch.pi)

        # Freeze phase for standing envs (clean double-support display)
        if stand_mask.any():
            self.phase[stand_mask] = torch.full(
                (int(stand_mask.sum().item()), 2), self.stand_phase_value, device=env.device
            )

    def set_eval_mode(self, evaluating: bool) -> None:
        self._initialize_indices(None, evaluating=evaluating)

    def resample_frequency(self, env_ids: torch.Tensor) -> None:
        if self.gait_freq is None or self.phase_dt is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if self.env.is_evaluating:
            self.gait_freq[idx] = self.mean_gait_freq
        elif self.gait_period_range is not None:
            # Uniform sampling in the PERIOD domain, then convert to frequency.
            lo, hi = self.gait_period_range
            periods = torch_rand_float(lo, hi, (idx.shape[0], 1), device=self.env.device)
            self.gait_freq[idx] = 1.0 / periods
        elif self.gait_period_randomization_width <= 0.0:
            self.gait_freq[idx] = self.mean_gait_freq
        else:
            low = self.mean_gait_freq - self.gait_period_randomization_width
            high = self.mean_gait_freq + self.gait_period_randomization_width
            self.gait_freq[idx] = torch_rand_float(low, high, (idx.shape[0], 1), device=self.env.device)

        self.phase_dt[idx] = 2 * torch.pi * self.env.dt * self.gait_freq[idx]

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    def _initialize_indices(self, env_ids: torch.Tensor | None, *, evaluating: bool) -> None:
        if self.phase_offset is None or self.phase is None or self.gait_freq is None or self.phase_dt is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if evaluating:
            self.phase_offset[idx, 0] = 0.0
            self.phase_offset[idx, 1] = -torch.pi
        elif self.randomize_phase:
            self.phase_offset[idx, 0] = torch_rand_float(
                -torch.pi, torch.pi, (idx.shape[0], 1), device=self.env.device
            ).squeeze(1)
            self.phase_offset[idx, 1] = torch.fmod(self.phase_offset[idx, 0] + 2 * torch.pi, 2 * torch.pi) - torch.pi
        else:
            self.phase_offset[idx] = 0.0

        self.phase[idx] = self.phase_offset[idx]
        # Reset the walking-only gait clock so a fresh episode starts at
        # phase == phase_offset (same clean reset as the absolute formulation).
        if self._gait_steps is not None:
            self._gait_steps[idx] = 0.0
        self.resample_frequency(idx)

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)


class GaitAveragedVelocity(CommandTermBase):
    """Tracks gait-cycle-averaged body-frame yaw rate and lateral velocity.

    A bipedal gait is an oscillator: even for a perfectly straight command, the
    pelvis yaws back and forth and drifts laterally each step, oscillating near
    the gait frequency with (ideally) zero mean over a stride. Penalizing the
    *instantaneous* yaw rate / lateral velocity therefore conflates legitimate
    intra-stride oscillation with the net directional drift we actually want to
    suppress. Cranking the instantaneous penalty up forces a stiff, stilted
    gait.

    This term maintains a per-env *running mean over the current gait cycle* of:
        - ``wz``      : body-frame yaw rate (rad/s)
        - ``v_perp``  : signed lateral (perpendicular-to-command) body velocity (m/s)

    The mean-so-far is updated every step (dense signal) and the accumulator is
    reset at each gait-phase wrap (so the average locks to each env's own,
    possibly randomized, stride). The just-completed full-cycle mean is retained
    as ``cycle_mean_*`` so reward terms can fall back to it during the noisy
    early-stride window (warmup guard).

    Exposed buffers (all shape ``[num_envs]`` unless noted):
        wz_running_mean        : mean-so-far of wz within the current cycle
        v_perp_running_mean    : mean-so-far of v_perp within the current cycle
        wz_cycle_mean          : last completed full-cycle mean of wz
        v_perp_cycle_mean      : last completed full-cycle mean of v_perp
        sample_count           : samples accumulated so far this cycle (float)
        wz_instant             : instantaneous wz this step
        v_perp_instant         : instantaneous v_perp this step

    Config params (via cfg.params):
        gait_term_name: str — name of the gait state term providing ``phase``
            and (optionally) ``is_walking`` (default "locomotion_gait").
        warmup_samples: int — min samples before the running mean is trusted;
            below this, ``*_running_mean`` mirrors the last cycle mean
            (default 5).
        min_cmd_speed: float — min commanded planar speed to define the command
            direction for the lateral decomposition (default 0.03).
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.gait_term_name: str = str(params.get("gait_term_name", "locomotion_gait"))
        self.warmup_samples: int = int(params.get("warmup_samples", 5))
        self.min_cmd_speed: float = float(params.get("min_cmd_speed", 0.03))

        # Accumulators (current cycle)
        self._sum_wz: torch.Tensor | None = None
        self._sum_v_perp: torch.Tensor | None = None
        self._count: torch.Tensor | None = None
        self._prev_phase_sign: torch.Tensor | None = None

        # Exposed running / cycle means
        self.wz_running_mean: torch.Tensor | None = None
        self.v_perp_running_mean: torch.Tensor | None = None
        self.wz_cycle_mean: torch.Tensor | None = None
        self.v_perp_cycle_mean: torch.Tensor | None = None
        self.sample_count: torch.Tensor | None = None
        self.wz_instant: torch.Tensor | None = None
        self.v_perp_instant: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        env = self.env
        device = env.device
        n = env.num_envs
        z = lambda: torch.zeros(n, dtype=torch.float32, device=device)  # noqa: E731
        self._sum_wz = z()
        self._sum_v_perp = z()
        self._count = z()
        self._prev_phase_sign = torch.ones(n, dtype=torch.float32, device=device)
        self.wz_running_mean = z()
        self.v_perp_running_mean = z()
        self.wz_cycle_mean = z()
        self.v_perp_cycle_mean = z()
        self.sample_count = z()
        self.wz_instant = z()
        self.v_perp_instant = z()

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self._sum_wz is None:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return
        for buf in (
            self._sum_wz, self._sum_v_perp, self._count,
            self.wz_running_mean, self.v_perp_running_mean,
            self.wz_cycle_mean, self.v_perp_cycle_mean,
            self.sample_count, self.wz_instant, self.v_perp_instant,
        ):
            buf[idx] = 0.0
        if self._prev_phase_sign is not None:
            self._prev_phase_sign[idx] = 1.0

    def step(self) -> None:
        if self._sum_wz is None or self.manager is None:
            return

        # --- instantaneous body-frame quantities ---
        ang_vel_world = self.env.simulator.robot_root_states[:, 10:13]
        ang_vel_body = quat_rotate_inverse(
            self.env.simulator.base_quat, ang_vel_world, w_last=True
        )
        wz = ang_vel_body[:, 2]

        lin_vel_world = self.env.simulator.robot_root_states[:, 7:10]
        lin_vel_body = quat_rotate_inverse(
            self.env.simulator.base_quat, lin_vel_world, w_last=True
        )
        v_xy = lin_vel_body[:, :2]

        # Lateral (perpendicular-to-command) component, signed via 2D cross.
        try:
            commands = self.manager.commands
            c_xy = commands[:, :2]
        except (AttributeError, TypeError):
            c_xy = torch.zeros_like(v_xy)
        cmd_speed = torch.linalg.norm(c_xy, dim=-1, keepdim=True)
        moving = (cmd_speed.squeeze(-1) > self.min_cmd_speed)
        cmd_dir = c_xy / torch.clamp(cmd_speed, min=1e-6)
        # signed perpendicular speed = cross(cmd_dir, v) z-component
        v_perp = cmd_dir[:, 0] * v_xy[:, 1] - cmd_dir[:, 1] * v_xy[:, 0]
        v_perp = torch.where(moving, v_perp, torch.zeros_like(v_perp))

        self.wz_instant.copy_(wz)
        self.v_perp_instant.copy_(v_perp)

        # --- gait cycle detection (left-leg phase wrap, neg -> pos) ---
        gait_state = self.manager.get_state(self.gait_term_name)
        phase = getattr(gait_state, "phase", None) if gait_state is not None else None
        is_walking = getattr(gait_state, "is_walking", None) if gait_state is not None else None
        if phase is None:
            # No gait state: degrade to instantaneous (means mirror instant).
            self.wz_running_mean.copy_(wz)
            self.v_perp_running_mean.copy_(v_perp)
            return

        walking_mask = (
            is_walking if is_walking is not None
            else torch.ones_like(wz, dtype=torch.bool)
        )

        # Accumulate for walking envs only (standing has frozen phase).
        self._sum_wz[walking_mask] += wz[walking_mask]
        self._sum_v_perp[walking_mask] += v_perp[walking_mask]
        self._count[walking_mask] += 1.0

        cur_sign = (phase[:, 0] >= 0).float()
        cycle_done = (
            (self._prev_phase_sign < 0.5)
            & (cur_sign >= 0.5)
            & (self._count > self.warmup_samples)
            & walking_mask
        )
        if cycle_done.any():
            ids = cycle_done.nonzero(as_tuple=False).flatten()
            denom = self._count[ids].clamp(min=1.0)
            self.wz_cycle_mean[ids] = self._sum_wz[ids] / denom
            self.v_perp_cycle_mean[ids] = self._sum_v_perp[ids] / denom
            self._sum_wz[ids] = 0.0
            self._sum_v_perp[ids] = 0.0
            self._count[ids] = 0.0
        self._prev_phase_sign.copy_(cur_sign)

        # --- running mean with warmup guard ---
        live_denom = self._count.clamp(min=1.0)
        live_wz = self._sum_wz / live_denom
        live_v_perp = self._sum_v_perp / live_denom
        warm = self._count >= self.warmup_samples
        self.wz_running_mean.copy_(
            torch.where(warm, live_wz, self.wz_cycle_mean)
        )
        self.v_perp_running_mean.copy_(
            torch.where(warm, live_v_perp, self.v_perp_cycle_mean)
        )
        self.sample_count.copy_(self._count)

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
