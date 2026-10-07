from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy.typing as npt
from isaacgym import gymapi, gymutil

if TYPE_CHECKING:
    from holosoma.simulator.isaacgym.isaacgym import IsaacGym


def clear_lines(simulator: IsaacGym) -> None:
    """Clear all debug lines from the viewer."""
    if simulator.viewer:
        simulator.gym.clear_lines(simulator.viewer)


def draw_sphere(
    simulator: IsaacGym,
    pos: list[float] | tuple[float, float, float] | npt.NDArray[Any],
    radius: float,
    color: list[float] | tuple[float, float, float],
    env_id: int,
    pos_id: int | None = None,
    num_lats: int = 8,
    num_lons: int = 8,
) -> None:
    """Draw a wireframe sphere at the specified position."""
    sphere_geom = gymutil.WireframeSphereGeometry(radius, num_lats, num_lons, None, color=color)
    sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
    gymutil.draw_lines(sphere_geom, simulator.gym, simulator.viewer, simulator.envs[env_id], sphere_pose)


def draw_line(
    simulator: IsaacGym,
    start_point: list[float] | tuple[float, float, float] | npt.NDArray[Any],
    end_point: list[float] | tuple[float, float, float] | npt.NDArray[Any],
    color: list[float] | tuple[float, float, float],
    env_id: int,
) -> None:
    """Draw a line from start_point to end_point."""
    p1 = gymapi.Vec3(start_point[0], start_point[1], start_point[2])
    p2 = gymapi.Vec3(end_point[0], end_point[1], end_point[2])
    color = gymapi.Vec3(color[0], color[1], color[2])
    gymutil.draw_line(p1, p2, color, simulator.gym, simulator.viewer, simulator.envs[env_id])


def draw_points(
    simulator: IsaacGym,
    points: list[list[float]] | list[tuple[float, float, float]] | npt.NDArray[Any],
    colors: list[list[float]] | list[tuple[float, float, float]] | npt.NDArray[Any],
    sizes: list[float] | npt.NDArray[Any],
    env_id: int,
) -> None:
    """Draw points at the specified positions."""
    if simulator.viewer:
        for _i, (point, color, size) in enumerate(zip(points, colors, sizes)):
            sphere_geom = gymutil.WireframeSphereGeometry(size / 100.0, 4, 4, None, color=color)
            sphere_pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
            gymutil.draw_lines(sphere_geom, simulator.gym, simulator.viewer, simulator.envs[env_id], sphere_pose)


def draw_height_points(
    simulator: IsaacGym,
    base_pos: list[float] | tuple[float, float, float] | npt.NDArray[Any],
    height_points: npt.NDArray[Any],
    heights: npt.NDArray[Any],
    env_id: int,
    point_color: list[float] | tuple[float, float, float] = (1, 1, 0),
    point_size: float = 0.02,
) -> None:
    """Draw height measurement points."""
    if simulator.viewer:
        sphere_geom = gymutil.WireframeSphereGeometry(point_size, 4, 4, None, color=point_color)
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(
                sphere_geom,
                simulator.gym,
                simulator.viewer,
                simulator.envs[env_id],
                sphere_pose,
            )


def draw_foot_height_points(
    simulator: IsaacGym,
    foot_pos: list[float] | tuple[float, float, float] | npt.NDArray[Any],
    terrain_height: float,
    env_id: int,
    foot_color: list[float] | tuple[float, float, float] = (1, 0, 0),
    line_color: list[float] | tuple[float, float, float] = (1, 0, 0),
) -> None:
    """Draw foot position and height points."""
    if simulator.viewer:
        # Draw the actual foot position
        foot_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=foot_color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(foot_pos[0], foot_pos[1], foot_pos[2]), r=None)
        gymutil.draw_lines(
            foot_sphere_geom,
            simulator.gym,
            simulator.viewer,
            simulator.envs[env_id],
            sphere_pose,
        )

        # Draw the sampled height point on the terrain
        sphere_pose = gymapi.Transform(gymapi.Vec3(foot_pos[0], foot_pos[1], terrain_height), r=None)
        gymutil.draw_lines(
            foot_sphere_geom,
            simulator.gym,
            simulator.viewer,
            simulator.envs[env_id],
            sphere_pose,
        )

        # Draw a line connecting foot to terrain point
        gymutil.draw_line(
            gymapi.Vec3(foot_pos[0], foot_pos[1], foot_pos[2]),
            gymapi.Vec3(foot_pos[0], foot_pos[1], terrain_height),
            gymapi.Vec3(*line_color),
            simulator.gym,
            simulator.viewer,
            simulator.envs[env_id],
        )
