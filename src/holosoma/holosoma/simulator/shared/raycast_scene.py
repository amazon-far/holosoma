"""Geometry interface between scene initialization and the fused depth renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import warp as wp


@dataclass(frozen=True)
class RaycastSceneMeshes:
    """Persistent per-environment mesh IDs and world poses, owned by the scene provider.

    IDs and poses have shape ``[num_envs, num_meshes]``; positions are Warp vec3 and orientations
    are Warp quaternions (xyzw). The provider retains the meshes and updates poses in place.
    """

    mesh_ids: wp.array[Any]
    positions: wp.array[Any]
    orientations: wp.array[Any]
    num_meshes: int


class RaycastScene(Protocol):
    """Scene geometry initialized after assets load; pose updates do not perform ray queries."""

    def prepare_meshes(self) -> RaycastSceneMeshes: ...

    def update_mesh_poses(self) -> None: ...
