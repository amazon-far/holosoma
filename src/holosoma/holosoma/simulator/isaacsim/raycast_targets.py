"""Isaac-SDK-free ray-cast target helpers for Isaac Sim sensors.

Target-list assembly and include/exclude selection, batched robot mesh poses, and sharing the
terrain manager's mesh with IsaacLab's ray-caster cache.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from loguru import logger

from holosoma.config_types.sensor import IsaacSimRaycastMeshConfig
from holosoma.utils.rotations import quat_apply, quat_mul

if TYPE_CHECKING:
    import warp as wp

    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

TERRAIN_PRIM_PATH = "/World/ground"
TERRAIN_MESH_PRIM_PATH = f"{TERRAIN_PRIM_PATH}/mesh"


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
    """Combine visual robot, ground, and scene-object targets after applying the body filter."""
    exclude_robot = excluded_target is not None and excluded_target[0] == "robot"
    excluded_robot_link = (
        excluded_target[1] if excluded_target is not None and excluded_target[0] == "robot_link" else None
    )
    paths = [] if exclude_robot else robot_mesh_prim_paths(excluded_robot_link)
    candidates = [(TERRAIN_PRIM_PATH, TERRAIN_PRIM_PATH)]
    candidates.extend(
        (
            f"/World/envs/env_0/{actor_name}",
            f"/World/envs/env_.*/{actor_name}",
        )
        for actor_name in actor_names
        if excluded_target != ("actor", actor_name)
    )
    for source, expression in candidates:
        paths.extend(discover_mesh_prim_paths(source, expression))
    if not paths:
        if exclude_robot:
            raise RuntimeError(
                "Isaac Sim ray-cast auto-discovery found no supported visual scene meshes "
                "(body filter 'robot' excluded the robot; ground and scene actors contributed none)."
            )
        raise RuntimeError("Isaac Sim ray-cast auto-discovery found no supported visual scene meshes.")
    return paths


def select_mesh_prim_paths(
    mesh_paths: Iterable[str],
    config: IsaacSimRaycastMeshConfig,
    *,
    resolve_matching_paths: Callable[[str], Iterable[str]],
) -> list[str]:
    """Filter discovered meshes below roots resolved with IsaacLab's native path matcher.

    Discovery and configuration use LiDAR's cloned ``env_.*`` namespace. Resolve rules against
    env_0 once, retaining the discovered expressions for native per-environment mesh tracking.
    """
    paths = sorted(set(mesh_paths))

    def source_path(path: str) -> str:
        return path.replace("/World/envs/env_.*", "/World/envs/env_0", 1)

    def matches(expression: str) -> set[str]:
        roots = tuple(resolve_matching_paths(source_path(expression)))
        return {path for path in paths if any(_is_path_at_or_below(source_path(path), root) for root in roots)}

    included = set(paths)
    if config.include_paths is not None:
        included = set()
        for expression in config.include_paths:
            matched = matches(expression)
            if not matched:
                raise ValueError(f"Raycast mesh include {expression!r} matched no supported visual scene geometry.")
            included.update(matched)
    for expression in config.exclude_paths:
        matched = matches(expression)
        if not matched:
            # Allowed, so one config can exclude objects some scenes lack; a typo surfaces here.
            logger.warning(f"Raycast mesh exclude {expression!r} matched no supported visual scene geometry.")
        included.difference_update(matched)
    selected = sorted(included)
    if not selected:
        raise ValueError("Raycast mesh inclusion/exclusion selected no supported visual scene geometry.")
    return selected


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


def register_managed_terrain_mesh(cache: MutableMapping[str, wp.Mesh], mesh: wp.Mesh, device: wp.Device) -> None:
    """Borrow the existing world-frame mesh authored at identity pose by Holosoma.

    The height scanner looks up the ground root; depth and LiDAR discover the exact mesh leaf.
    Both keys must retain the same mesh, whose original owner is the terrain manager.
    """
    if mesh.device != device:
        raise ValueError(f"Managed terrain Warp mesh is on {mesh.device}, but the IsaacLab scene is on {device}.")

    paths = (TERRAIN_PRIM_PATH, TERRAIN_MESH_PRIM_PATH)
    for path in paths:
        if path in cache and cache[path] is not mesh:
            raise RuntimeError(f"IsaacLab raycaster cache already contains a different Warp mesh at {path!r}.")
    for path in paths:
        cache[path] = mesh
