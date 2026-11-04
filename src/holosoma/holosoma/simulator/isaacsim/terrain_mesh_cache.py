"""Publish managed terrain geometry to IsaacLab's shared raycaster cache."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import warp as wp

TERRAIN_PRIM_PATH = "/World/ground"
TERRAIN_MESH_PRIM_PATH = f"{TERRAIN_PRIM_PATH}/mesh"


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
