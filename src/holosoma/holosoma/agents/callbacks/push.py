"""Eval callback that applies random force pulses to test policy robustness.

Periodically pushes the robot at random body locations with configurable
force magnitude, duration, and interval. Uses IsaacLab's
set_external_force_and_torque API to apply forces in the physics simulation.

Push events are optionally recorded for visualization in viser by sharing
buffer/metadata dicts with EvalRecordingCallback.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import PushConfig
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.agents.callbacks.recording import EvalRecordingCallback


class EvalPushCallback(RLEvalCallback):
    """Applies random force pulses during evaluation and optionally records them.

    Forces are applied at the physics substep level by patching the env's
    _apply_force_in_physics_step method, ensuring the force is active for the
    full duration across all substeps (control decimation).

    Recorded arrays (per env step):
        push_active    [T]    1.0 when pushing, 0.0 otherwise
        push_force_w   [T, 3] force vector in world frame (zeros when inactive)
        push_body_pos_w[T, 3] world position of pushed body (zeros when inactive)
        push_body_idx  [T]    index into body_names (-1 when inactive)
    """

    def __init__(
        self,
        config: PushConfig,
        training_loop: Any = None,
    ) -> None:
        super().__init__(config, training_loop)
        self.env_id = config.env_id
        self.candidate_body_names = [s.strip() for s in config.body_names.split(",")]
        self._record_buffers: dict[str, list[npt.NDArray[Any]]] | None = None
        self._record_metadata: dict[str, Any] | None = None

        # Resolved at on_pre_evaluate_policy
        self._resolved_bodies: list[tuple[str, int, int]] = []
        self._dt: float = 0.0
        self._sim: Any = None
        self._original_apply_force: Any = None

        # Push state machine
        self._push_active: bool = False
        self._push_steps_remaining: int = 0
        self._idle_steps_remaining: int = 0
        self._current_force_w: torch.Tensor | None = None
        self._current_body_names_idx: int = -1
        self._current_isaac_body_id: int = -1

        self._step_count: int = 0

    def _get_env(self) -> Any:
        return self.training_loop._unwrap_env()

    def _find_recording_callback(self) -> EvalRecordingCallback | None:
        from holosoma.agents.callbacks.recording import EvalRecordingCallback

        for cb in self.training_loop.eval_callbacks:
            if isinstance(cb, EvalRecordingCallback):
                return cb
        return None

    def on_pre_evaluate_policy(self) -> None:
        # Discover recording callback for shared buffer/metadata access
        recording_cb = self._find_recording_callback()
        if recording_cb is not None:
            self._record_buffers = recording_cb._buffers
            self._record_metadata = recording_cb._metadata

        env = self._get_env()
        self._sim = env.simulator
        if self._sim.simulator_config.name != "isaacsim":
            raise RuntimeError(f"EvalPushCallback requires IsaacSim, got '{self._sim.simulator_config.name}'")
        self._dt = float(env.dt)

        # Resolve body names to simulator indices
        self._resolved_bodies = []
        available = list(getattr(self._sim, "body_names", []))
        for name in self.candidate_body_names:
            body_names_idx = self._sim.find_rigid_body_indice(name)
            if isinstance(body_names_idx, int) and body_names_idx >= 0:
                isaac_body_id = self._sim.body_ids[body_names_idx]
                self._resolved_bodies.append((name, body_names_idx, isaac_body_id))
            else:
                raise ValueError(f"EvalPushCallback: body '{name}' not found. Available: {available}")

        resolved_names = [name for name, _, _ in self._resolved_bodies]
        logger.info(
            f"EvalPushCallback: {len(self._resolved_bodies)} push bodies: {resolved_names}, "
            f"force={self.config.force_range}N, duration={self.config.duration_s}s, interval={self.config.interval_s}s"
        )

        # Store push config in recording metadata
        if self._record_metadata is not None:
            self._record_metadata["push_config"] = {
                "body_names": resolved_names,
                "force_range": list(self.config.force_range),
                "duration_s": list(self.config.duration_s),
                "interval_s": list(self.config.interval_s),
            }

        # Initialize recording buffers
        buffers = self._get_buffers()
        if buffers is not None:
            for key in ("push_active", "push_force_w", "push_body_pos_w", "push_body_idx"):
                buffers[key] = []

        # Start with a random idle period
        self._idle_steps_remaining = self._sample_interval_steps()
        self._push_active = False

        # Patch _apply_force_in_physics_step to inject push forces at every substep
        self._original_apply_force = env._apply_force_in_physics_step

        def _patched_apply_force() -> None:
            self._original_apply_force()
            if self._push_active:
                self._apply_push_force()

        env._apply_force_in_physics_step = _patched_apply_force

    def on_post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        # Manage push schedule
        if self._push_active:
            self._push_steps_remaining -= 1
            if self._push_steps_remaining <= 0:
                self._deactivate_push()
                self._idle_steps_remaining = self._sample_interval_steps()
        else:
            self._idle_steps_remaining -= 1
            if self._idle_steps_remaining <= 0:
                self._activate_push()

        buffers = self._get_buffers()
        eid = self.env_id

        if self._push_active:
            assert self._current_force_w is not None
            force_w = self._current_force_w.detach().cpu().numpy().copy()
            body_pos = self._sim._robot.data.body_pos_w[eid, self._current_isaac_body_id].detach().cpu().numpy().copy()
            body_idx = self._current_body_names_idx
            active = 1.0
        else:
            force_w = np.zeros(3, dtype=np.float32)
            body_pos = np.zeros(3, dtype=np.float32)
            body_idx = -1
            active = 0.0

        if buffers is not None:
            buffers["push_active"].append(np.array(active, dtype=np.float32))
            buffers["push_force_w"].append(force_w.astype(np.float32))
            buffers["push_body_pos_w"].append(body_pos.astype(np.float32))
            buffers["push_body_idx"].append(np.array(body_idx, dtype=np.int32))

        self._step_count += 1
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        if self._push_active:
            self._deactivate_push()
        self._get_env()._apply_force_in_physics_step = self._original_apply_force
        logger.info(f"EvalPushCallback: completed {self._step_count} steps")

    # ------------------------------------------------------------------
    # Push lifecycle
    # ------------------------------------------------------------------

    def _activate_push(self) -> None:
        name, body_names_idx, isaac_body_id = random.choice(self._resolved_bodies)
        self._current_body_names_idx = body_names_idx
        self._current_isaac_body_id = isaac_body_id

        # Random force magnitude
        mag = random.uniform(*self.config.force_range)

        # Random horizontal direction (XY plane)
        angle = random.uniform(0, 2 * math.pi)
        direction = torch.tensor(
            [math.cos(angle), math.sin(angle), 0.0],
            dtype=torch.float32,
            device=self._sim.sim_device,
        )
        self._current_force_w = direction * mag

        # Duration in env steps
        dur_s = random.uniform(*self.config.duration_s)
        self._push_steps_remaining = max(1, round(dur_s / self._dt))

        self._push_active = True
        logger.info(
            f"PUSH START body='{name}' force={mag:.0f}N "
            f"dir=[{direction[0].item():.2f},{direction[1].item():.2f}] "
            f"duration={self._push_steps_remaining} steps ({dur_s:.2f}s)"
        )

    def _deactivate_push(self) -> None:
        self._clear_push_force()
        self._push_active = False
        self._current_force_w = None
        self._current_body_names_idx = -1
        self._current_isaac_body_id = -1
        logger.info("PUSH END")

    # ------------------------------------------------------------------
    # Force application (called each physics substep via monkey-patch)
    # ------------------------------------------------------------------

    def _apply_push_force(self) -> None:
        from isaaclab.utils.math import quat_apply_inverse

        eid = self.env_id
        isaac_body_id = self._current_isaac_body_id

        # Transform world-frame force to body-local frame
        # (IsaacLab 2.1 hardcodes is_global=False)
        body_quat_w = self._sim._robot.data.body_quat_w[eid, isaac_body_id]
        force_body = quat_apply_inverse(body_quat_w, self._current_force_w)

        forces = force_body.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        torques = torch.zeros_like(forces)  # [1, 1, 3]

        self._sim._robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            env_ids=torch.tensor([eid], device=self._sim.sim_device),
            body_ids=torch.tensor([isaac_body_id], device=self._sim.sim_device),
        )

    def _clear_push_force(self) -> None:
        if self._current_isaac_body_id < 0:
            return
        zero = torch.zeros(1, 1, 3, device=self._sim.sim_device)
        self._sim._robot.set_external_force_and_torque(
            forces=zero,
            torques=zero,
            env_ids=torch.tensor([self.env_id], device=self._sim.sim_device),
            body_ids=torch.tensor([self._current_isaac_body_id], device=self._sim.sim_device),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_interval_steps(self) -> int:
        s = random.uniform(*self.config.interval_s)
        return max(1, round(s / self._dt))

    def _get_buffers(self) -> dict[str, list[npt.NDArray[Any]]] | None:
        return self._record_buffers
