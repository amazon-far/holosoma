"""Eval callback that applies constant downward forces on wrists to simulate a held payload.

Applies F = mass * g / num_bodies on each resolved wrist/elbow link
as a sustained downward (-Z) force throughout the evaluation episode. Uses the same
IsaacLab set_external_force_and_torque API as the push callback, with substep-level
force application via monkey-patching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import PayloadConfig
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.agents.callbacks.recording import EvalRecordingCallback

GRAVITY = 9.81


class EvalPayloadCallback(RLEvalCallback):
    """Applies constant downward forces on wrist links to simulate holding a payload.

    The total payload weight (mass * g) is split evenly across all resolved
    wrist bodies. Forces are applied at every physics substep via monkey-patch.

    Recorded arrays (per env step):
        payload_force_per_body [T]     force magnitude per body in N
        payload_body_pos_w     [T, N, 3] world positions of payload bodies
    """

    def __init__(
        self,
        config: PayloadConfig,
        training_loop: Any = None,
    ) -> None:
        super().__init__(config, training_loop)
        self.env_id = config.env_id
        self.candidate_body_names = [s.strip() for s in config.body_names.split(",")]
        self._record_buffers: dict[str, list[npt.NDArray[Any]]] | None = None
        self._record_metadata: dict[str, Any] | None = None

        # Resolved at on_pre_evaluate_policy
        self._resolved_bodies: list[tuple[str, int, int]] = []
        self._force_per_body: float = 0.0
        self._sim: Any = None
        self._original_apply_force: Any = None
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
            raise RuntimeError(f"EvalPayloadCallback requires IsaacSim, got '{self._sim.simulator_config.name}'")

        # Resolve body names
        self._resolved_bodies = []
        available = list(getattr(self._sim, "body_names", []))
        for name in self.candidate_body_names:
            body_names_idx = self._sim.find_rigid_body_indice(name)
            if isinstance(body_names_idx, int) and body_names_idx >= 0:
                isaac_body_id = self._sim.body_ids[body_names_idx]
                self._resolved_bodies.append((name, body_names_idx, isaac_body_id))
            else:
                raise ValueError(f"EvalPayloadCallback: body '{name}' not found. Available: {available}")

        total_force = self.config.mass_kg * GRAVITY
        self._force_per_body = total_force / len(self._resolved_bodies)

        resolved_names = [name for name, _, _ in self._resolved_bodies]
        logger.info(
            f"EvalPayloadCallback: {len(self._resolved_bodies)} payload bodies: {resolved_names}, "
            f"mass={self.config.mass_kg}kg, total_force={total_force:.1f}N, "
            f"force_per_body={self._force_per_body:.1f}N"
        )

        # Store config in recording metadata
        if self._record_metadata is not None:
            self._record_metadata["payload_config"] = {
                "body_names": resolved_names,
                "mass_kg": self.config.mass_kg,
                "total_force_n": total_force,
                "force_per_body_n": self._force_per_body,
            }

        # Initialize recording buffers
        buffers = self._get_buffers()
        if buffers is not None:
            buffers["payload_force_per_body"] = []
            buffers["payload_body_pos_w"] = []

        # Monkey-patch substep force application
        self._original_apply_force = env._apply_force_in_physics_step

        def _patched_apply_force() -> None:
            self._original_apply_force()
            self._apply_payload_forces()

        env._apply_force_in_physics_step = _patched_apply_force

    def on_post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        buffers = self._get_buffers()
        if buffers is not None:
            eid = self.env_id
            body_positions = []
            for _name, _body_names_idx, isaac_body_id in self._resolved_bodies:
                pos = self._sim._robot.data.body_pos_w[eid, isaac_body_id].detach().cpu().numpy().copy()
                body_positions.append(pos)

            buffers["payload_force_per_body"].append(np.array(self._force_per_body, dtype=np.float32))
            buffers["payload_body_pos_w"].append(np.stack(body_positions, axis=0).astype(np.float32))

        self._step_count += 1
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        self._get_env()._apply_force_in_physics_step = self._original_apply_force
        logger.info(
            f"EvalPayloadCallback: completed {self._step_count} steps, "
            f"payload={self.config.mass_kg}kg on {len(self._resolved_bodies)} bodies"
        )

    def _apply_payload_forces(self) -> None:
        """Apply downward force on all resolved wrist bodies (called each substep)."""
        from isaaclab.utils.math import quat_apply_inverse

        eid = self.env_id
        device = self._sim.sim_device

        # Force in world frame: straight down (-Z)
        force_world = torch.tensor(
            [0.0, 0.0, -self._force_per_body],
            dtype=torch.float32,
            device=device,
        )

        for _name, _body_names_idx, isaac_body_id in self._resolved_bodies:
            # Transform world-frame force to body-local frame
            body_quat_w = self._sim._robot.data.body_quat_w[eid, isaac_body_id]
            force_body = quat_apply_inverse(body_quat_w, force_world)

            forces = force_body.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
            torques = torch.zeros_like(forces)

            self._sim._robot.set_external_force_and_torque(
                forces=forces,
                torques=torques,
                env_ids=torch.tensor([eid], device=device),
                body_ids=torch.tensor([isaac_body_id], device=device),
            )

    def _get_buffers(self) -> dict[str, list[npt.NDArray[Any]]] | None:
        return self._record_buffers
