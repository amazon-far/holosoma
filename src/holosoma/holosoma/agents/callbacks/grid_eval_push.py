"""Grid evaluation push callback for deterministic push perturbation sweeps.

Sweeps push perturbations across body x direction x gait_phase (x force_magnitude),
cross-producted with velocity conditions from GridEvalVelocityCallback. Each condition
env receives exactly one push. Forces are injected every physics substep by
monkey-patching ``env._apply_force_in_physics_step``.

Push lifecycle:
    1. **Assign** (_deferred_setup): map each env to its push params (body, direction,
       phase, force). No-gait-phase path computes a random fire step per env.
    2. **Activate** (_try_activate_pushes): when the target gait phase is detected (or
       the random fire step is reached), compute world-frame forces and mark env as active.
    3. **Apply force** (_apply_push_forces): called every physics substep via monkey-patch.
       Groups active envs by body and applies forces via IsaacSim API.
    4. **Expire** (on_post_eval_env_step): decrement timers, clear forces when done.
    5. **Record** (on_post_eval_env_step): append push_active and push_force_w each step.

Push timing:
    With gait_phases: pushes fire when the target phase is detected by GaitAnalyser
        after a warmup + gait analysis window.
    Without gait_phases: pushes fire at a random step in
        [warmup + gait_analysis_s, + 1s] to ensure stable walking.

Push directions are in the robot's local frame, rotated by yaw at push time.
Direction vectors are defined in GridEvalPushConfig.DIRECTION_VECTORS.

Usage example (4 vel x 2 bodies x 4 dirs x 4 phases = 128 conditions):

    --grid-velocity.config.enabled
    --grid-velocity.config.lin-vel-x 0.5
    --grid-push.config.enabled
    --grid-push.config.body-names torso_link pelvis
    --grid-push.config.body-labels torso pelvis
    --grid-push.config.directions forward backward left right
    --grid-push.config.gait-phases swing_to_stance stance_to_swing mid_stance mid_swing
    --grid-push.config.force-n 150.0
    --grid-push.config.duration-s 0.2
    --training.num-envs 128
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.callbacks.gait_analysis import GaitAnalyser
from holosoma.agents.callbacks.sim_utils import apply_body_force_world, clear_body_force, resolve_body
from holosoma.config_types.eval_callback import GridEvalPushConfig
from holosoma.utils.safe_torch_import import torch


class GridEvalPushCallback(RLEvalCallback):
    """Sweeps push perturbations across body x direction x gait_phase.

    Registers axes with the condition manager (via recording callback).
    When gait_phases are configured, validates that the env is a
    locomotion manager (the robot must be walking).
    """

    def __init__(self, config: GridEvalPushConfig, training_loop: Any = None):
        super().__init__(config, training_loop)

        # External refs
        self._recording_cb: Any = None
        self._sim: Any = None
        self._original_apply_force: Any = None

        self._resolved_bodies: list[tuple[str, int]] = []

        # Gait analysis timing
        self._gait_analyser: GaitAnalyser | None = None
        self._gait_analysis_min_steps: int = 0
        self._gait_analysis_max_steps: int = 0

        # Per-env push assignment (set in _deferred_setup)
        self._push_body_idx_per_env: list[int] = []
        self._push_local_dir_per_env: torch.Tensor | None = None
        self._push_gait_phase_per_env: list[str] = []
        self._push_force_n_per_env: torch.Tensor | None = None

        # Push execution state
        self._push_active: torch.Tensor | None = None
        self._push_force_w: torch.Tensor | None = None
        self._push_steps_remaining: torch.Tensor | None = None
        self._push_fired: torch.Tensor | None = None
        self._push_fire_step: torch.Tensor | None = None

        self._step_count = 0
        self._setup_done = False

    def on_pre_evaluate_policy(self) -> None:
        # --- Validate prerequisites ---
        self._recording_cb = self._require_recording_cb()

        env = self._get_env()
        self._sim = env.simulator

        if self.config.gait_phases:
            from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager

            if not isinstance(env, LeggedRobotLocomotionManager):
                raise RuntimeError(
                    "GridEvalPushCallback with gait_phases requires a locomotion policy "
                    f"(LeggedRobotLocomotionManager), got {type(env).__name__}"
                )

        if self._sim.simulator_config.name != "isaacsim":
            raise RuntimeError(f"GridEvalPushCallback requires IsaacSim, got '{self._sim.simulator_config.name}'")

        # --- Resolve body names to IsaacSim body IDs ---
        body_names = self.config.body_names
        body_labels = self.config.body_labels

        self._resolved_bodies = [resolve_body(self._sim, n) for n in body_names]

        left_foot_isaac_id = resolve_body(self._sim, self.config.left_foot_body)[1]
        right_foot_isaac_id = resolve_body(self._sim, self.config.right_foot_body)[1]

        # --- Determine force magnitudes ---
        if self.config.force_magnitudes:
            self._force_magnitudes = list(self.config.force_magnitudes)
        else:
            self._force_magnitudes = [self.config.force_n]

        logger.info(
            f"GridEvalPushCallback: {len(self._resolved_bodies)} bodies, "
            f"directions={list(self.config.directions)}, "
            f"gait_phases={list(self.config.gait_phases)}, "
            f"forces={self._force_magnitudes}N, duration={self.config.duration_s}s, "
            f"gait_analysis={self.config.gait_analysis_s}s"
        )

        # --- Register sweep axes with condition manager ---
        cm = self._recording_cb.condition_manager
        cm.register_axis("push_body_label", list(body_labels), group="push")
        cm.register_axis("push_direction", list(self.config.directions), group="push")
        if self.config.gait_phases:
            cm.register_axis("push_gait_phase", list(self.config.gait_phases), group="push")
        cm.register_axis("push_force_n", self._force_magnitudes, group="push")

        # --- Register recording buffers and metadata ---
        self._recording_cb.register_buffer_key("push_active")
        self._recording_cb.register_buffer_key("push_force_w")

        self._recording_cb.set_metadata("push_body_labels", list(body_labels))
        self._recording_cb.set_metadata("push_body_names", [n for n, _ in self._resolved_bodies])
        self._recording_cb.set_metadata("push_directions", list(self.config.directions))
        self._recording_cb.set_metadata("push_gait_phases", list(self.config.gait_phases))
        self._recording_cb.set_metadata("push_force_magnitudes", self._force_magnitudes)
        self._recording_cb.set_metadata("push_duration_s", self.config.duration_s)
        self._recording_cb.set_metadata("push_gait_analysis_s", self.config.gait_analysis_s)
        self._recording_cb.set_metadata("push_left_foot_body", self.config.left_foot_body)
        self._recording_cb.set_metadata("push_right_foot_body", self.config.right_foot_body)

        # --- Create GaitAnalyser (only for gait-phase path) ---
        if self.config.gait_phases:
            self._gait_analyser = GaitAnalyser(
                sim=self._sim,
                left_foot_isaac_id=left_foot_isaac_id,
                right_foot_isaac_id=right_foot_isaac_id,
                num_conditions=0,  # set in _deferred_setup after finalization
                foot_contact_threshold=self.config.foot_contact_threshold,
                min_gait_cycles=self.config.min_gait_cycles,
            )

    def _deferred_setup(self) -> None:
        self._recording_cb.ensure_finalized()

        env = self._get_env()
        cm = self._recording_cb.condition_manager
        conditions = cm.conditions
        num_c = cm.num_conditions
        device = env.device
        dt = float(env.dt)

        # --- Gait analysis timing ---
        warmup_steps = cm.warmup_steps
        self._gait_analysis_min_steps = warmup_steps + int(self.config.gait_analysis_s / dt)
        self._gait_analysis_max_steps = warmup_steps + int(self.config.max_gait_analysis_s / dt)

        if self._gait_analyser is not None:
            self._gait_analyser._num_conditions = num_c
            self._gait_analyser._prev_left_on_ground = np.zeros(num_c, dtype=bool)

        # --- Map each env to its push params from conditions ---
        body_labels = self.config.body_labels
        label_to_body_idx = {label: i for i, label in enumerate(body_labels)}

        push_duration_steps = max(1, round(self.config.duration_s / dt))

        self._push_body_idx_per_env = []
        self._push_gait_phase_per_env = []
        local_dirs = []
        force_ns = []
        has_gait_phases = bool(self.config.gait_phases)

        for cond in conditions:
            body_label = cond.get("push_body_label", "")
            direction = cond.get("push_direction", "forward")
            force = float(cond.get("push_force_n", self.config.force_n))

            self._push_body_idx_per_env.append(label_to_body_idx.get(body_label, 0))
            local_dirs.append(self.config.DIRECTION_VECTORS[direction])
            if has_gait_phases:
                self._push_gait_phase_per_env.append(cond.get("push_gait_phase", "swing_to_stance"))
            force_ns.append(force)

        self._push_local_dir_per_env = torch.tensor(local_dirs, dtype=torch.float32, device=device)
        self._push_force_n_per_env = torch.tensor(force_ns, dtype=torch.float32, device=device)

        # --- Initialize push execution state ---
        self._push_active = torch.zeros(num_c, dtype=torch.bool, device=device)
        self._push_force_w = torch.zeros(num_c, 3, dtype=torch.float32, device=device)
        self._push_steps_remaining = torch.zeros(num_c, dtype=torch.long, device=device)
        self._push_fired = torch.zeros(num_c, dtype=torch.bool, device=device)
        self._push_duration_steps = push_duration_steps

        if not self.config.gait_phases:
            delay_steps = int(1.0 / dt)
            earliest = self._gait_analysis_min_steps
            self._push_fire_step = torch.randint(earliest, earliest + delay_steps + 1, (num_c,), device=device)

        # --- Per-condition metadata ---
        self._recording_cb.set_metadata(
            "push_body_label_per_condition", [c.get("push_body_label", "") for c in conditions]
        )
        self._recording_cb.set_metadata(
            "push_direction_per_condition", [c.get("push_direction", "") for c in conditions]
        )
        self._recording_cb.set_metadata(
            "push_gait_phase_per_condition", [c.get("push_gait_phase", "") for c in conditions]
        )
        self._recording_cb.set_metadata("push_force_n_per_condition", force_ns)
        self._recording_cb.set_metadata("push_duration_steps", push_duration_steps)

        # --- Monkey-patch physics step to inject push forces every substep ---
        self._original_apply_force = env._apply_force_in_physics_step

        def _patched_apply_force():
            self._original_apply_force()
            self._apply_push_forces()

        env._apply_force_in_physics_step = _patched_apply_force

        self._setup_done = True
        logger.info(
            f"GridEvalPushCallback: deferred setup, {num_c} conditions, "
            f"push_duration={push_duration_steps} steps, "
            f"gait_analysis min step {self._gait_analysis_min_steps}, "
            f"max step {self._gait_analysis_max_steps}, "
            f"min_cycles={self.config.min_gait_cycles}"
        )

    def _compute_world_forces_batch(self, env_indices: np.ndarray) -> torch.Tensor:
        """Rotate local-frame push directions by each robot's yaw, scale by force magnitude."""
        assert self._push_local_dir_per_env is not None
        assert self._push_force_n_per_env is not None
        device = self._sim.sim_device
        idx = torch.tensor(env_indices, dtype=torch.long, device=device)

        local_dirs = self._push_local_dir_per_env[idx]

        # Extract yaw from root quaternion (xyzw)
        root_quats = self._sim.robot_root_states[idx, 3:7]
        x, y, z, w = root_quats[:, 0], root_quats[:, 1], root_quats[:, 2], root_quats[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        # 2D rotation: local frame -> world frame
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        world_x = local_dirs[:, 0] * cos_yaw - local_dirs[:, 1] * sin_yaw
        world_y = local_dirs[:, 0] * sin_yaw + local_dirs[:, 1] * cos_yaw

        force_dirs = torch.stack([world_x, world_y, local_dirs[:, 2]], dim=1)
        return force_dirs * self._push_force_n_per_env[idx].unsqueeze(1)

    def _try_activate_pushes(self) -> None:
        assert self._push_fired is not None
        assert self._push_active is not None
        assert self._push_steps_remaining is not None
        assert self._push_force_w is not None
        num_c = self._recording_cb.num_conditions

        # Find candidate envs ready to fire
        if self.config.gait_phases:
            assert self._gait_analyser is not None
            phases = self._gait_analyser.detect_phases()
            candidates = [
                i
                for i in range(num_c)
                if not self._push_fired[i]
                and not self._push_active[i]
                and phases[i] != ""
                and phases[i] == self._push_gait_phase_per_env[i]
            ]
        else:
            assert self._push_fire_step is not None
            candidates = [
                i
                for i in range(num_c)
                if not self._push_fired[i] and not self._push_active[i] and self._step_count >= self._push_fire_step[i]
            ]

        if not candidates:
            return

        # Compute world-frame forces and activate
        env_indices = np.array(candidates)
        forces = self._compute_world_forces_batch(env_indices)

        idx_tensor = torch.tensor(env_indices, device=self.device, dtype=torch.long)
        self._push_fired[idx_tensor] = True
        self._push_active[idx_tensor] = True
        self._push_steps_remaining[idx_tensor] = self._push_duration_steps
        self._push_force_w[idx_tensor] = forces

        for i in candidates:
            body_name = self._resolved_bodies[self._push_body_idx_per_env[i]][0]
            logger.debug(f"GridEvalPushCallback: env {i} push at step {self._step_count} (body={body_name})")

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        if not self._setup_done:
            self._deferred_setup()

        cm = self._recording_cb.condition_manager
        num_c = cm.num_conditions

        # Gait analysis: record foot heights, try to finalize
        if self._gait_analyser is not None and not self._gait_analyser.done:
            if self._step_count >= cm.warmup_steps:
                self._gait_analyser.record_foot_heights()
                if self._step_count >= self._gait_analysis_min_steps:
                    force = self._step_count >= self._gait_analysis_max_steps
                    if self._gait_analyser.try_finalize(force=force):
                        for key, value in self._gait_analyser.get_metadata().items():
                            self._recording_cb.set_metadata(key, value)

        # Activate pushes (gait-phase path waits for analyser.done, no-gait-phase uses fire_step)
        can_activate = self._gait_analyser is None or self._gait_analyser.done
        if can_activate:
            self._try_activate_pushes()

        # Decrement push timers, expire finished pushes
        assert self._push_active is not None
        assert self._push_steps_remaining is not None
        assert self._push_force_w is not None
        assert self._push_fired is not None
        active_mask = self._push_active[:num_c]
        self._push_steps_remaining[:num_c][active_mask] -= 1

        expired = active_mask & (self._push_steps_remaining[:num_c] <= 0)
        if expired.any():
            self._push_active[:num_c][expired] = False
            self._push_force_w[:num_c][expired] = 0.0
            self._clear_push_forces(torch.where(expired)[0])

        # Record push state
        self._recording_cb.append_buffer(
            "push_active",
            self._push_active[:num_c].float().cpu().numpy().copy(),
        )
        self._recording_cb.append_buffer(
            "push_force_w",
            self._push_force_w[:num_c].cpu().numpy().copy(),
        )

        self._step_count += 1
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        # Restore original physics step
        if self._original_apply_force is not None:
            self._get_env()._apply_force_in_physics_step = self._original_apply_force

        # Record which conditions actually fired
        assert self._push_fired is not None
        num_c = self._recording_cb.num_conditions
        self._recording_cb.set_metadata(
            "push_fired_per_condition",
            self._push_fired[:num_c].cpu().tolist(),
        )

        num_fired = int(self._push_fired[:num_c].sum().item())
        logger.info(f"GridEvalPushCallback: completed, {num_fired}/{num_c} pushes fired")

    def _apply_push_forces(self) -> None:
        """Substep hot path: apply forces for all active pushes."""
        if self._push_active is None:
            return
        assert self._push_force_w is not None

        num_c = self._recording_cb.num_conditions
        active_mask = self._push_active[:num_c]
        if not active_mask.any():
            return

        # Group active envs by body for batched IsaacSim API calls
        device = self._sim.sim_device
        active_indices = torch.where(active_mask)[0]

        body_groups: dict[int, list[int]] = {}
        for env_idx in active_indices.tolist():
            body_groups.setdefault(self._push_body_idx_per_env[env_idx], []).append(env_idx)

        for body_idx, env_idxs in body_groups.items():
            _name, isaac_body_id = self._resolved_bodies[body_idx]
            env_ids = torch.tensor(env_idxs, dtype=torch.long, device=device)
            apply_body_force_world(self._sim, env_ids, isaac_body_id, self._push_force_w[env_ids])

    def _clear_push_forces(self, env_indices: torch.Tensor) -> None:
        """Zero out forces on expired envs, grouped by body."""
        device = self._sim.sim_device
        body_groups: dict[int, list[int]] = {}
        for env_idx in env_indices.tolist():
            body_groups.setdefault(self._push_body_idx_per_env[env_idx], []).append(env_idx)

        for body_idx, env_idxs in body_groups.items():
            _name, isaac_body_id = self._resolved_bodies[body_idx]
            env_ids = torch.tensor(env_idxs, dtype=torch.long, device=device)
            clear_body_force(self._sim, env_ids, isaac_body_id)
