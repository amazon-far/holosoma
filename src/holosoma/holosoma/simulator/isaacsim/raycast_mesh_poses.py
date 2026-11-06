"""Batched robot poses for scene-discovered raycast meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from holosoma.utils.rotations import quat_apply, quat_mul

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


@dataclass(frozen=True)
class RobotMeshPoses:
    """Initialization-time bindings to Holosoma body order and mesh-local frames.

    Several mesh slots may reference the same body. Offset tensors have shape
    ``[num_envs, num_bound_meshes, k]`` and orientations use xyzw.
    """

    target_paths: frozenset[str]
    body_ids: torch.Tensor
    mesh_slots: torch.Tensor
    position_offsets: torch.Tensor
    orientation_offsets: torch.Tensor

    def update(
        self,
        sim: BaseSimulator,
        mesh_positions: torch.Tensor,
        mesh_orientations_wxyz: torch.Tensor,
    ) -> None:
        """Read refreshed body tensors and write persistent mesh buffers in place."""
        body_positions = sim.rigid_body_pos_w[:, self.body_ids]
        body_orientations = sim.rigid_body_quat_w[:, self.body_ids].contiguous()
        mesh_positions[:, self.mesh_slots] = body_positions + quat_apply(
            body_orientations, self.position_offsets, w_last=True
        )
        orientations = quat_mul(body_orientations, self.orientation_offsets, w_last=True)
        mesh_orientations_wxyz[:, self.mesh_slots] = orientations[..., [3, 0, 1, 2]]
