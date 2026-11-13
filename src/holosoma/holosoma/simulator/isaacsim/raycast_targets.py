"""Isaac-SDK-free ray-cast target helpers for Isaac Sim sensors.

Default target-list assembly and batched robot mesh poses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from holosoma.utils.rotations import quat_apply, quat_mul

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

TERRAIN_PRIM_PATH = "/World/ground"


def _is_path_at_or_below(path: str, root: str) -> bool:
    """Return whether ``path`` is ``root`` or one of its descendants."""
    return path == root or path.startswith(f"{root.rstrip('/')}/")


def assemble_default_mesh_prim_paths(
    *,
    excluded_target: tuple[str, str] | None,
    robot_mesh_prim_paths: Callable[[str | None], list[str]],
    actor_names: Iterable[str],
    discover_mesh_prim_paths: Callable[[str, str], Iterable[str]],
) -> list[str]:
    """Combine visual robot, ground, and scene-object targets after applying the body filter.

    The ground is always the cached terrain mesh at :data:`TERRAIN_PRIM_PATH`.
    """
    exclude_robot = excluded_target is not None and excluded_target[0] == "robot"
    excluded_robot_link = (
        excluded_target[1] if excluded_target is not None and excluded_target[0] == "robot_link" else None
    )
    paths = [] if exclude_robot else list(robot_mesh_prim_paths(excluded_robot_link))
    paths.append(TERRAIN_PRIM_PATH)
    for actor_name in actor_names:
        if excluded_target != ("actor", actor_name):
            paths.extend(
                discover_mesh_prim_paths(f"/World/envs/env_0/{actor_name}", f"/World/envs/env_.*/{actor_name}")
            )
    return paths


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
