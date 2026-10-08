"""Warp utility methods.

From: https://github.com/escontra/gauss_gym/blob/main/gauss_gym/utils/warp_utils.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import warp as wp

from holosoma.utils.warp_interop import from_torch_vec3

wp.init()


@wp.kernel
def raycast_kernel(
    mesh: wp.uint64,
    ray_starts_world: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    ray_directions_world: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    ray_hits_world: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
) -> None:
    tid = wp.tid()

    # NOTE: float()/int() wrappers are load-bearing in Warp kernels: a bare literal is
    # inferred as const, which cannot bind to mesh_query_ray's mutable float&/int& out-params.
    # UP018 (which strips these) is disabled for this file in pyproject.toml.
    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index
    max_dist = float(1e6)  # max raycast disance
    # ray cast against the mesh
    if wp.mesh_query_ray(  # type: ignore[call-arg, truthy-bool]
        mesh,
        ray_starts_world[tid],
        ray_directions_world[tid],
        max_dist,  # type: ignore[arg-type]
        t,
        u,
        v,
        sign,
        n,
        f,
    ):
        ray_hits_world[tid] = ray_starts_world[tid] + t * ray_directions_world[tid]


def ray_cast(ray_starts_world: torch.Tensor, ray_directions_world: torch.Tensor, wp_mesh: wp.Mesh) -> torch.Tensor:
    """Performs ray casting on the terrain mesh.

    Args:
        ray_starts_world (Torch.tensor): The starting position of the ray.
        ray_directions_world (Torch.tensor): The ray direction.

    Returns:
        [Torch.tensor]: The ray hit position. Returns float('inf') for missed hits.
    """
    shape = ray_starts_world.shape
    ray_starts_world = ray_starts_world.view(-1, 3)
    ray_directions_world = ray_directions_world.view(-1, 3)
    num_rays = len(ray_starts_world)
    ray_starts_world_wp = from_torch_vec3(ray_starts_world, wp_mesh.device)
    ray_directions_world_wp = from_torch_vec3(ray_directions_world, wp_mesh.device)
    ray_hits_world = torch.full_like(ray_starts_world, float("inf"))
    # full_like preserves the checked layout and device, so this output only needs wrapping.
    ray_hits_world_wp = wp.from_torch(ray_hits_world, dtype=wp.vec3)
    wp.launch(
        kernel=raycast_kernel,
        dim=num_rays,
        inputs=[
            wp_mesh.id,
            ray_starts_world_wp,
            ray_directions_world_wp,
            ray_hits_world_wp,
        ],
        device=wp_mesh.device,
    )
    wp.synchronize()
    return ray_hits_world.view(shape)


@wp.kernel
def nearest_point_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    mesh_points: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
) -> None:
    tid = wp.tid()

    max_dist = float(1e6)  # max raycast disance
    query = wp.mesh_query_point(mesh, points[tid], max_dist=max_dist)  # type: ignore[arg-type]
    if not query.result:  # type: ignore[attr-defined]
        return

    # Evaluate the position of the nearest location found.
    mesh_points[tid] = wp.mesh_eval_position(
        mesh,
        query.face,  # type: ignore[attr-defined]
        query.u,  # type: ignore[attr-defined]
        query.v,  # type: ignore[attr-defined]
    )


def nearest_point(points: torch.Tensor, wp_mesh: wp.Mesh) -> torch.Tensor:
    """Find the nearest terrain-mesh position for each query point.

    Args:
        points (Torch.tensor): The query points.

    Returns:
        [Torch.tensor]: The nearest mesh positions. Returns float('inf') when no point is found.
    """
    shape = points.shape
    points = points.view(-1, 3)
    num_points = len(points)
    points_wp = from_torch_vec3(points, wp_mesh.device)
    mesh_points = torch.full_like(points, float("inf"))
    # full_like preserves the checked layout and device, so this output only needs wrapping.
    mesh_points_wp = wp.from_torch(mesh_points, dtype=wp.vec3)
    wp.launch(
        kernel=nearest_point_kernel,
        dim=num_points,
        inputs=[wp_mesh.id, points_wp, mesh_points_wp],
        device=wp_mesh.device,
    )
    wp.synchronize()
    return mesh_points.view(shape)


def convert_to_wp_mesh(
    vertices: npt.NDArray[np.floating[Any]], triangles: npt.NDArray[np.integer[Any]], device: str
) -> wp.Mesh:
    return wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(triangles.astype(np.int32).flatten(), dtype=int, device=device),
    )
