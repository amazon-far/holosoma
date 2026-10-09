from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import numpy.typing as npt

from holosoma.utils.adapters.draw_utils import convert_to_numpy
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.mujoco import MuJoCo


def clear_lines(simulator: MuJoCo) -> None:
    """Clear all debug visualization by resetting user_scn geoms.

    Parameters
    ----------
    simulator : MuJoCo
        MuJoCo simulator instance with viewer.
    """
    if simulator.viewer is None:
        return

    # Reset the number of user geoms to 0
    # This effectively clears all debug objects
    simulator.viewer.user_scn.ngeom = 0


def draw_sphere(
    simulator: MuJoCo,
    pos: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    radius: float,
    color: list[float] | tuple[float, float, float] | torch.Tensor,
    env_id: int,
    num_lats: int = 4,  # Ignored for MuJoCo
    num_lons: int = 4,  # Ignored for MuJoCo
    pos_id: int | None = None,
) -> None:
    """Draw a sphere using MuJoCo's user_scn API.

    Parameters
    ----------
    simulator : MuJoCo
        MuJoCo simulator instance with viewer.
    pos : list[float] | tuple[float, float, float] | np.ndarray | torch.Tensor
        Position [x, y, z] where to draw the sphere.
    radius : float
        Radius of the sphere.
    color : list[float] | tuple[float, float, float]
        RGB color tuple (0-1 range).
    env_id : int
        Environment ID (ignored for MuJoCo single env).
    num_lats : int, default=4
        Number of latitude lines (ignored for MuJoCo).
    num_lons : int, default=4
        Number of longitude lines (ignored for MuJoCo).
    pos_id : int | None, default=None
        Position ID (optional, ignored for MuJoCo).
    """
    if simulator.viewer is None:
        return

    # Get current number of user geoms
    current_geoms = simulator.viewer.user_scn.ngeom

    # Check if we have space for another geom
    if current_geoms >= simulator.viewer.user_scn.maxgeom:
        return  # No space for more debug objects

    # Convert position to numpy array if needed
    pos_array = convert_to_numpy(pos)

    # Convert color to numpy array with alpha
    color_array: npt.NDArray[np.float32] = np.array([color[0], color[1], color[2], 0.7], dtype=np.float32)

    # Initialize the geometry using MuJoCo's official API
    mujoco.mjv_initGeom(
        simulator.viewer.user_scn.geoms[current_geoms],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=pos_array,
        mat=np.eye(3).flatten(),  # Identity rotation matrix
        rgba=color_array,
    )

    # Increment the geom count
    simulator.viewer.user_scn.ngeom += 1


def draw_line(
    simulator: MuJoCo,
    start_point: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    end_point: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    color: list[float] | tuple[float, float, float] | torch.Tensor,
    env_id: int,
) -> None:
    """Draw a line using MuJoCo's capsule geometry with very small radius.

    Parameters
    ----------
    simulator : MuJoCo
        MuJoCo simulator instance with viewer.
    start_point : list[float] | tuple[float, float, float] | np.ndarray | torch.Tensor
        Starting point [x, y, z] of the line.
    end_point : list[float] | tuple[float, float, float] | np.ndarray | torch.Tensor
        Ending point [x, y, z] of the line.
    color : list[float] | tuple[float, float, float]
        RGB color tuple (0-1 range).
    env_id : int
        Environment ID (ignored for MuJoCo single env).
    """
    if simulator.viewer is None:
        return

    # Get current number of user geoms
    current_geoms = simulator.viewer.user_scn.ngeom

    # Check if we have space for another geom
    if current_geoms >= simulator.viewer.user_scn.maxgeom:
        return  # No space for more debug objects

    # Convert points to numpy arrays if needed
    start_array = convert_to_numpy(start_point)
    end_array = convert_to_numpy(end_point)

    # Calculate line properties
    midpoint = (start_array + end_array) / 2.0
    direction = end_array - start_array
    length = np.linalg.norm(direction)

    if length < 1e-6:  # Avoid degenerate lines
        return

    # Normalize direction vector
    direction = direction / length

    # Create rotation matrix to align capsule with line direction
    # MuJoCo capsules are aligned along Z-axis by default
    z_axis = np.array([0, 0, 1])

    # Handle the case where direction is parallel to z-axis
    if np.abs(np.dot(direction, z_axis)) > 0.999:
        # Use identity or 180-degree rotation
        if direction[2] > 0:
            rotation_matrix = np.eye(3)
        else:
            rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        # Create rotation matrix using cross product
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

        # Rodrigues' rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        cross_matrix = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation_matrix = cos_angle * np.eye(3) + sin_angle * cross_matrix + (1 - cos_angle) * np.outer(axis, axis)

    # Convert color to numpy array with alpha
    color_array: npt.NDArray[np.float32] = np.array([color[0], color[1], color[2], 0.8], dtype=np.float32)

    # Initialize the capsule geometry
    mujoco.mjv_initGeom(
        simulator.viewer.user_scn.geoms[current_geoms],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.002, length / 2, 0],  # Small radius, half-length, unused
        pos=midpoint,
        mat=rotation_matrix.flatten(),
        rgba=color_array,
    )

    # Increment the geom count
    simulator.viewer.user_scn.ngeom += 1


def draw_height_points(
    simulator: MuJoCo,
    base_pos: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    height_points: npt.NDArray[Any] | torch.Tensor,
    heights: npt.NDArray[Any] | torch.Tensor,
    env_id: int,
    point_color: list[float] | tuple[float, float, float] = (1, 1, 0),
    point_size: float = 0.02,
) -> None:
    """Draw height measurement points as spheres at terrain height locations.

    Parameters
    ----------
    simulator : MuJoCo
        MuJoCo simulator instance with viewer.
    base_pos : list[float] | tuple[float, float, float] | np.ndarray | torch.Tensor
        Base position [x, y, z] for relative height point calculations.
    height_points : np.ndarray | torch.Tensor
        Array of height measurement points relative to base position.
    heights : np.ndarray | torch.Tensor
        Array of terrain heights corresponding to each height point.
    env_id : int
        Environment ID (ignored for MuJoCo single env).
    point_color : list[float] | tuple[float, float, float], default=(1, 1, 0)
        RGB color tuple (0-1 range) for the height point spheres.
    point_size : float, default=0.02
        Radius of the height point spheres.
    """
    if simulator.viewer is None:
        return

    # Convert numpy arrays if needed
    base_pos_array = convert_to_numpy(base_pos)
    height_points_array = convert_to_numpy(height_points)
    heights_array = convert_to_numpy(heights)

    # Draw a sphere at each height measurement point
    num_points = height_points_array.shape[0]
    for i in range(num_points):
        # Calculate world position: height_points are relative to base, heights are terrain heights
        world_x = height_points_array[i, 0] + base_pos_array[0]
        world_y = height_points_array[i, 1] + base_pos_array[1]
        world_z = heights_array[i]  # Use terrain height directly

        world_pos = [world_x, world_y, world_z]

        # Draw sphere using existing draw_sphere function
        draw_sphere(simulator, world_pos, point_size, point_color, env_id)


def draw_foot_height_points(
    simulator: MuJoCo,
    foot_pos: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    terrain_height: float,
    env_id: int,
    foot_color: list[float] | tuple[float, float, float] = (1, 0, 0),
    line_color: list[float] | tuple[float, float, float] = (1, 0, 0),
) -> None:
    """Draw foot position and terrain height points with connecting line.

    Parameters
    ----------
    simulator : MuJoCo
        MuJoCo simulator instance with viewer.
    foot_pos : list[float] | tuple[float, float, float] | np.ndarray | torch.Tensor
        Current foot position [x, y, z].
    terrain_height : float
        Height of terrain directly below the foot.
    env_id : int
        Environment ID (ignored for MuJoCo single env).
    foot_color : list[float] | tuple[float, float, float], default=(1, 0, 0)
        RGB color tuple (0-1 range) for the foot and terrain point spheres.
    line_color : list[float] | tuple[float, float, float], default=(1, 0, 0)
        RGB color tuple (0-1 range) for the connecting line.
    """
    if simulator.viewer is None:
        return

    # Convert foot_pos to numpy array if needed
    foot_pos_array = convert_to_numpy(foot_pos)

    # Draw sphere at actual foot position
    draw_sphere(simulator, foot_pos_array, 0.02, foot_color, env_id)

    # Draw sphere at terrain height below foot (same X,Y but terrain Z)
    terrain_pos = [foot_pos_array[0], foot_pos_array[1], terrain_height]
    draw_sphere(simulator, terrain_pos, 0.02, foot_color, env_id)

    # Draw line connecting foot to terrain point
    draw_line(simulator, foot_pos_array, terrain_pos, line_color, env_id)


def draw_points(
    simulator: MuJoCo,
    points: list[list[float]] | list[tuple[float, float, float]] | npt.NDArray[Any] | torch.Tensor,
    colors: list[list[float]] | list[tuple[float, float, float]] | npt.NDArray[Any] | torch.Tensor,
    sizes: list[float] | npt.NDArray[Any] | torch.Tensor,
    env_id: int,
) -> None:
    """Stub: Log draw_points calls to understand usage patterns.

    This is a placeholder implementation that logs the function call parameters
    to help understand how this function is used in the codebase.

    Parameters
    ----------
    simulator : MuJoCo
        MuJoCo simulator instance with viewer.
    points : list[list[float]] | list[tuple[float, float, float]] | np.ndarray | torch.Tensor
        Array of 3D points to draw.
    colors : list[list[float]] | list[tuple[float, float, float]] | np.ndarray | torch.Tensor
        Array of RGB colors corresponding to each point.
    sizes : list[float] | np.ndarray | torch.Tensor
        Array of sizes for each point.
    env_id : int
        Environment ID (ignored for MuJoCo single env).
    """
