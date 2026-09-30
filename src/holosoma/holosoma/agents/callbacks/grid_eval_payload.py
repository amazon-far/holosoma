"""Grid evaluation payload callback for multi-condition payload sweeps.

Applies constant downward forces (simulating carried payload) during evaluation. Sweeps across
body-group placement (where the payload sits) and mass. Forces are injected every physics substep
by monkey-patching ``env._apply_force_in_physics_step``. Records payload body positions to the NPZ.

Usage example (4 vel x 2 body_groups x 3 masses = 24 conditions):

    --grid-velocity.config.enabled
    --grid-velocity.config.lin-vel-x 0.25 0.5 0.75 1.0
    --grid-payload.config.enabled
    --grid-payload.config.body-groups pelvis left_wrist_yaw_link,right_wrist_yaw_link
    --grid-payload.config.body-labels torso wrists
    --grid-payload.config.mass-kg 0.0 1.0 2.0
    --training.num-envs 24
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.callbacks.sim_utils import apply_body_force_world, resolve_body
from holosoma.config_types.eval_callback import GridEvalPayloadConfig
from holosoma.utils.safe_torch_import import torch

GRAVITY = 9.81


class GridEvalPayloadCallback(RLEvalCallback):
    """Sweeps payload across body-group x mass dimensions in grid evaluation.

    Registers axes with the condition manager (via recording callback),
    then applies per-env forces at each physics substep.
    """

    def __init__(self, config: GridEvalPayloadConfig, training_loop: Any = None):
        super().__init__(config, training_loop)

        self._recording_cb: Any = None
        self._sim: Any = None
        self._original_apply_force: Any = None

        self._resolved_groups: list[tuple[str, list[tuple[str, int]]]] = []

        # Per-env assignment (set in _deferred_setup):
        #   _payload_forces_per_env[i] = force per body (N) for env i
        #   _payload_group_idx_per_env[i] = which body-group env i uses
        self._payload_forces_per_env: torch.Tensor | None = None
        self._payload_group_idx_per_env: list[int] = []

        # Same data reorganized by group for the substep hot path:
        #   _group_env_indices[g] = env indices with non-zero payload in group g
        #   _group_force_world[g] = [N, 3] world-frame force vectors for those envs
        self._group_env_indices: list[torch.Tensor] = []
        self._group_force_world: list[torch.Tensor] = []

    def on_pre_evaluate_policy(self) -> None:
        self._recording_cb = self._require_recording_cb()

        env = self._get_env()
        self._sim = env.simulator

        if self._sim.simulator_config.name != "isaacsim":
            raise RuntimeError(f"GridEvalPayloadCallback requires IsaacSim, got '{self._sim.simulator_config.name}'")

        body_groups = self.config.body_groups
        body_labels = self.config.body_labels

        # --- Resolve comma-separated body name strings to (name, isaac_body_id) pairs ---
        self._resolved_groups = [
            (
                body_labels[i],
                [resolve_body(self._sim, n.strip()) for n in bg_str.split(",") if n.strip()],
            )
            for i, bg_str in enumerate(body_groups)
        ]

        logger.info(
            f"GridEvalPayloadCallback: {len(self._resolved_groups)} body groups, masses={list(self.config.mass_kg)}kg"
        )
        for label, bodies in self._resolved_groups:
            logger.info(f"  group '{label}': bodies={[n for n, _ in bodies]}")

        # --- Register sweep axes: body_group x mass_kg ---
        cm = self._recording_cb.condition_manager
        cm.register_axis("payload_body_label", list(body_labels), group="payload")
        cm.register_axis("payload_mass_kg", list(self.config.mass_kg), group="payload")

        self._recording_cb.register_buffer_key("payload_body_pos_w")

        self._recording_cb.set_metadata(
            "payload_body_groups", {label: [n for n, _ in bodies] for label, bodies in self._resolved_groups}
        )
        self._recording_cb.set_metadata("payload_mass_kg_values", list(self.config.mass_kg))
        self._recording_cb.set_metadata("payload_body_labels", list(body_labels))

    def _deferred_setup(self) -> None:
        """Set up per-env forces after condition manager is finalized."""
        self._recording_cb.ensure_finalized()

        env = self._get_env()
        cm = self._recording_cb.condition_manager
        conditions = cm.conditions
        num_c = cm.num_conditions
        device = env.device

        # --- Map each env to its body-group and compute per-body force ---
        label_to_group_idx = {label: i for i, (label, _) in enumerate(self._resolved_groups)}
        self._payload_forces_per_env = torch.zeros(num_c, dtype=torch.float32, device=device)
        self._payload_group_idx_per_env = []

        for i, cond in enumerate(conditions):
            mass_kg = cond.get("payload_mass_kg", 0.0)
            body_label = cond.get("payload_body_label", "")
            group_idx = label_to_group_idx.get(body_label, 0)
            self._payload_group_idx_per_env.append(group_idx)

            if mass_kg > 0:
                _label, bodies = self._resolved_groups[group_idx]
                self._payload_forces_per_env[i] = (mass_kg * GRAVITY) / len(bodies)

        # --- Reorganize by group for substep hot path ---
        group_idx_tensor = torch.tensor(self._payload_group_idx_per_env, device=device)
        self._group_env_indices = []
        self._group_force_world = []

        for group_idx, (_label, _bodies) in enumerate(self._resolved_groups):
            mask = (group_idx_tensor == group_idx) & (self._payload_forces_per_env > 0)
            if mask.any():
                indices = torch.where(mask)[0]
                force_world = torch.zeros(len(indices), 3, dtype=torch.float32, device=device)
                force_world[:, 2] = -self._payload_forces_per_env[indices]
            else:
                indices = torch.empty(0, dtype=torch.long, device=device)
                force_world = torch.empty(0, 3, dtype=torch.float32, device=device)
            self._group_env_indices.append(indices)
            self._group_force_world.append(force_world)

        self._recording_cb.set_metadata(
            "payload_mass_kg_per_condition", [c.get("payload_mass_kg", 0.0) for c in conditions]
        )
        self._recording_cb.set_metadata(
            "payload_body_label_per_condition", [c.get("payload_body_label", "") for c in conditions]
        )

        # --- Monkey-patch physics step to inject payload forces every substep ---
        self._original_apply_force = env._apply_force_in_physics_step

        def _patched_apply_force():
            self._original_apply_force()
            self._apply_payload_forces()

        env._apply_force_in_physics_step = _patched_apply_force

        non_zero = (self._payload_forces_per_env > 0).sum().item()
        logger.info(f"GridEvalPayloadCallback: deferred setup, {num_c} conditions, non-zero payload on {non_zero} envs")

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        if self._payload_forces_per_env is None:
            self._deferred_setup()

        num_c = self._recording_cb.num_conditions

        # --- Record world positions of each group's bodies for each env ---
        max_bodies = max(len(bodies) for _, bodies in self._resolved_groups)
        body_positions: np.ndarray = np.zeros((num_c, max_bodies, 3), dtype=np.float32)

        for group_idx, (_label, bodies) in enumerate(self._resolved_groups):
            env_mask = np.array(self._payload_group_idx_per_env) == group_idx
            env_indices = np.where(env_mask)[0]
            if len(env_indices) == 0:
                continue
            for b_idx, (_name, isaac_body_id) in enumerate(bodies):
                pos = self._sim._robot.data.body_pos_w[env_indices, isaac_body_id]
                body_positions[env_indices, b_idx] = pos.detach().cpu().numpy()

        self._recording_cb.append_buffer("payload_body_pos_w", body_positions)
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        if self._original_apply_force is not None:
            self._get_env()._apply_force_in_physics_step = self._original_apply_force
        logger.info("GridEvalPayloadCallback: completed")

    def _apply_payload_forces(self) -> None:
        for group_idx, (_label, bodies) in enumerate(self._resolved_groups):
            env_indices = self._group_env_indices[group_idx]
            if len(env_indices) == 0:
                continue
            force_world = self._group_force_world[group_idx]
            for _name, isaac_body_id in bodies:
                apply_body_force_world(
                    self._sim,
                    env_indices,
                    isaac_body_id,
                    force_world,
                )
