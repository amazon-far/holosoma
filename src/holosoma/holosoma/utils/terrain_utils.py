# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab v2.0.0 (https://github.com/isaac-sim/IsaacLab)
# Contributors: https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md
#
# This file contains terrain generation utilities adapted from Isaac Lab's terrain generation code.

"""Terrain generation utilities adapted from Isaac Lab.

This module provides terrain generation functions for creating heightfield-based
terrains with various patterns including slopes, stairs, and random variations.
The code is adapted from Isaac Lab's terrain generation utilities with modifications.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import interpolate


def random_uniform_terrain(
    terrain: SubTerrain,
    min_height: float,
    max_height: float,
    step: float = 1,
    downsampled_scale: float | None = None,
) -> SubTerrain:
    """Generate uniform noise terrain with random heights.

    Creates a terrain with heights sampled uniformly from a specified range.
    Heights are sampled at a coarse resolution and then interpolated to the
    full terrain resolution using bivariate spline interpolation.

    Adapted from Isaac Lab's random_uniform_terrain function.

    Parameters
    ----------
    terrain : SubTerrain
        The terrain object to modify in-place.
    min_height : float
        Minimum height of the terrain in meters.
    max_height : float
        Maximum height of the terrain in meters.
    step : float, optional
        Minimum height change between sampled points in meters. Default is 1.
    downsampled_scale : float, optional
        Distance between randomly sampled points in meters. Must be larger than
        or equal to terrain.horizontal_scale. If None, uses terrain.horizontal_scale.

    Returns
    -------
    SubTerrain
        The modified terrain object (same as input).

    Raises
    ------
    ValueError
        If downsampled_scale is smaller than horizontal_scale.
    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale
    elif downsampled_scale < terrain.horizontal_scale:
        raise ValueError(
            f"Downsampled scale must be larger than or equal to horizontal scale: "
            f"{downsampled_scale} < {terrain.horizontal_scale}"
        )

    # Convert parameters to discrete units
    min_height_px = int(min_height / terrain.vertical_scale)
    max_height_px = int(max_height / terrain.vertical_scale)
    step_px = max(1, int(step / terrain.vertical_scale))  # Ensure at least 1 to avoid division by zero

    # Calculate downsampled dimensions
    width_downsampled = int(terrain.width * terrain.horizontal_scale / downsampled_scale)
    length_downsampled = int(terrain.length * terrain.horizontal_scale / downsampled_scale)

    # Create height range and randomly sample heights
    height_range = np.arange(min_height_px, max_height_px + step_px, step_px)
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))

    # Create coordinate arrays for interpolation
    x = np.linspace(0, terrain.width * terrain.horizontal_scale, width_downsampled)
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, length_downsampled)
    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)

    # Use RectBivariateSpline for linear interpolation
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled, kx=1, ky=1)
    z_upsampled = func(x_upsampled, y_upsampled, grid=False)

    # Add interpolated heights to terrain
    terrain.height_field_raw += np.rint(z_upsampled).astype(np.int16)
    return terrain


def pyramid_sloped_terrain(terrain: SubTerrain, slope: float = 1, platform_size: float = 1.0) -> SubTerrain:
    """Generate pyramid-shaped sloped terrain.

    Creates a terrain with a truncated pyramid structure where the slope increases
    from the edges toward the center, creating a pyramid shape that trims into a
    flat platform at the center. The slope can be positive (pyramid rising to center)
    or negative (inverted pyramid/pit).

    Adapted from Isaac Lab's pyramid_sloped_terrain function.

    Parameters
    ----------
    terrain : SubTerrain
        The terrain object to modify in-place.
    slope : float, optional
        Slope steepness as the ratio of height change to horizontal distance.
        Can be negative for inverted pyramids. Default is 1.
    platform_size : float, optional
        Size of the flat platform at the center in meters. Default is 1.0.

    Returns
    -------
    SubTerrain
        The modified terrain object (same as input).
    """
    # Get terrain dimensions
    width_pixels = terrain.width
    length_pixels = terrain.length
    center_x = width_pixels // 2
    center_y = length_pixels // 2

    # Calculate maximum height at pyramid peak
    height_max = int(slope * terrain.horizontal_scale / terrain.vertical_scale * width_pixels / 2)

    # Create coordinate meshgrid
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)

    # Calculate pyramid shape: distance from edges normalized
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    # Apply pyramid height field
    terrain.height_field_raw += (height_max * xx * yy).astype(np.int16)

    # Create flat platform at center by clipping heights
    platform_pixels = int(platform_size / terrain.horizontal_scale / 2)
    x1 = max(0, width_pixels // 2 - platform_pixels)
    y1 = max(0, length_pixels // 2 - platform_pixels)

    # Get height at platform corner for clipping reference
    z_pf = terrain.height_field_raw[x1, y1] if x1 < width_pixels and y1 < length_pixels else 0
    # Clip terrain heights to create flat platform
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min(0, z_pf), max(0, z_pf))

    return terrain


def pyramid_stairs_terrain(
    terrain: SubTerrain, step_width: float, step_height: float, platform_size: float = 1.0
) -> SubTerrain:
    """Generate pyramid-shaped stairs terrain.

    Creates a terrain with a pyramid stair pattern which trims to a flat platform
    at the center. Stairs descend/ascend from all edges toward the center in a
    symmetric pyramid pattern.

    Adapted from Isaac Lab's pyramid_stairs_terrain function.

    Parameters
    ----------
    terrain : SubTerrain
        The terrain object to modify in-place.
    step_width : float
        Width of each step in meters.
    step_height : float
        Height of each step in meters. Can be negative for descending stairs.
    platform_size : float, optional
        Size of the flat platform at the center in meters. Default is 1.0.

    Returns
    -------
    SubTerrain
        The modified terrain object (same as input).
    """
    # Convert parameters to discrete units
    width_pixels = terrain.width
    length_pixels = terrain.length
    step_width_px = int(step_width / terrain.horizontal_scale)
    step_height_px = int(step_height / terrain.vertical_scale)
    platform_pixels = int(platform_size / terrain.horizontal_scale)

    # Build pyramid stairs from edges toward center
    current_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels

    while (stop_x - start_x) > platform_pixels and (stop_y - start_y) > platform_pixels:
        # Move inward by step width
        start_x += step_width_px
        stop_x -= step_width_px
        start_y += step_width_px
        stop_y -= step_width_px
        # Increase height
        current_height += step_height_px
        # Apply height to current step region
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = current_height

    return terrain


def convert_heightfield_to_trimesh(
    height_field_raw: npt.NDArray[np.int_],
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    """Convert a heightfield array to a triangle mesh.

    Converts a 2D heightfield array to a triangle mesh represented by vertices
    and triangles. Optionally corrects vertical surfaces above a slope threshold
    to avoid unrealistically steep terrain features.

    The correction works by moving vertices horizontally when the slope between
    adjacent points exceeds the threshold, effectively creating vertical walls
    instead of extremely steep slopes.

    Adapted from Isaac Lab's convert_height_field_to_mesh function.

    Parameters
    ----------
    height_field_raw : np.ndarray
        Input heightfield as a 2D array of integers representing discretized heights.
    horizontal_scale : float
        Horizontal discretization scale in meters per pixel.
    vertical_scale : float
        Vertical discretization scale in meters per height unit.
    slope_threshold : float, optional
        The slope threshold (rise/run) above which surfaces are made vertical.
        If None, no correction is applied. Default is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - vertices : np.ndarray of shape (num_vertices, 3)
            Each row is a vertex position [x, y, z] in meters.
        - triangles : np.ndarray of shape (num_triangles, 3)
            Each row contains indices of the 3 vertices forming a triangle.
    """
    num_rows, num_cols = height_field_raw.shape

    # Create coordinate grids for the heightfield
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    # Copy heightfield to avoid modifying original
    hf = height_field_raw.copy()

    # Apply slope threshold correction if specified
    if slope_threshold is not None:
        # Scale threshold based on discretization
        slope_threshold *= horizontal_scale / vertical_scale

        # Arrays to track vertex movement
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))

        # Detect steep slopes in x direction and mark for correction
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold

        # Detect steep slopes in y direction and mark for correction
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold

        # Detect steep slopes at corners (diagonal)
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )

        # Apply corrections: move vertices to create vertical surfaces
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # Create vertex array
    vertices: npt.NDArray[np.float32] = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale

    # Create triangle indices
    # Each grid cell becomes 2 triangles
    triangles: npt.NDArray[np.uint32] = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        # Calculate vertex indices for current row
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1

        # Calculate triangle range for this row
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)

        # First triangle of each quad
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1

        # Second triangle of each quad
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles


def sloped_terrain(terrain: SubTerrain, slope: float = 1) -> SubTerrain:
    """Generate a simple sloped terrain.

    Creates a terrain with a linear slope in one direction. The terrain slopes
    uniformly from one edge to the other along the x-axis.

    Parameters
    ----------
    terrain : SubTerrain
        The terrain object to modify in-place.
    slope : float, optional
        Slope steepness (positive or negative). Default is 1.

    Returns
    -------
    SubTerrain
        The modified terrain object (same as input).
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(
        terrain.height_field_raw.dtype
    )
    return terrain


def discrete_obstacles_terrain(
    terrain: SubTerrain,
    max_height: float,
    min_size: float,
    max_size: float,
    num_rects: int,
    platform_size: float = 1.0,
) -> SubTerrain:
    """Generate terrain with randomly placed discrete box obstacles.

    Creates a terrain with randomly placed rectangular obstacles of varying heights.
    A flat platform is maintained at the center for spawning.

    Adapted from Isaac Lab's discrete_obstacles_terrain function.

    Parameters
    ----------
    terrain : SubTerrain
        The terrain object to modify in-place.
    max_height : float
        Maximum height of obstacles in meters (range: [-max, -max/2, max/2, max]).
    min_size : float
        Minimum size of rectangular obstacles in meters.
    max_size : float
        Maximum size of rectangular obstacles in meters.
    num_rects : int
        Number of randomly generated obstacles.
    platform_size : float, optional
        Size of the flat platform at the center in meters. Default is 1.0.

    Returns
    -------
    SubTerrain
        The modified terrain object (same as input).
    """
    # Convert to discrete units
    max_height_px = int(max_height / terrain.vertical_scale)
    min_size_px = int(min_size / terrain.horizontal_scale)
    max_size_px = int(max_size / terrain.horizontal_scale)
    platform_pixels = int(platform_size / terrain.horizontal_scale)

    # Create discrete ranges for obstacles
    height_range = [-max_height_px, -max_height_px // 2, max_height_px // 2, max_height_px]
    width_range = range(min_size_px, max_size_px, 4)
    length_range = range(min_size_px, max_size_px, 4)

    # Generate random obstacles
    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, terrain.width - width, 4))
        start_j = np.random.choice(range(0, terrain.length - length, 4))
        terrain.height_field_raw[start_i : start_i + width, start_j : start_j + length] = np.random.choice(height_range)

    # Create flat platform at center
    x1 = (terrain.width - platform_pixels) // 2
    x2 = (terrain.width + platform_pixels) // 2
    y1 = (terrain.length - platform_pixels) // 2
    y2 = (terrain.length + platform_pixels) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def wave_terrain(terrain: SubTerrain, num_waves: int = 1, amplitude: float = 1.0) -> SubTerrain:
    """Generate terrain with sinusoidal wave patterns.

    Creates a wavy terrain using sine and cosine functions. The waves are applied
    in both x and y directions to create a natural undulating surface.

    Adapted from Isaac Lab's wave_terrain function.

    Parameters
    ----------
    terrain : SubTerrain
        The terrain object to modify in-place.
    num_waves : int, optional
        Number of sine waves across the terrain length. Default is 1.
    amplitude : float, optional
        Amplitude of the waves in meters. Default is 1.0.

    Returns
    -------
    SubTerrain
        The modified terrain object (same as input).
    """
    amplitude_px = int(0.5 * amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw += (amplitude_px * np.cos(yy / div) + amplitude_px * np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype
        )
    return terrain


def stairs_terrain(terrain: SubTerrain, step_width: float, step_height: float) -> SubTerrain:
    """Generate linear stairs terrain.

    Creates a simple staircase that ascends linearly in one direction across
    the terrain width.

    Parameters
    ----------
    terrain : SubTerrain
        The terrain object to modify in-place.
    step_width : float
        Width of each step in meters.
    step_height : float
        Height of each step in meters.

    Returns
    -------
    SubTerrain
        The modified terrain object (same as input).
    """
    # Convert to discrete units
    step_width_px = int(step_width / terrain.horizontal_scale)
    step_height_px = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width_px
    height = step_height_px
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width_px : (i + 1) * step_width_px, :] += height
        height += step_height_px
    return terrain


def stepping_stones_terrain(
    terrain: SubTerrain,
    stone_size: float,
    stone_distance: float,
    max_height: float,
    platform_size: float = 1.0,
    depth: float = -10,
) -> SubTerrain:
    """Generate terrain with stepping stones pattern.

    Creates a terrain with discrete stepping stones separated by gaps. Each stone
    has a random height, and a flat platform is maintained at the center.

    Adapted from Isaac Lab's stepping_stones_terrain function.

    Parameters
    ----------
    terrain : SubTerrain
        The terrain object to modify in-place.
    stone_size : float
        Horizontal size of each stepping stone in meters.
    stone_distance : float
        Distance between stones (size of gaps) in meters.
    max_height : float
        Maximum height variation of stones in meters (positive and negative).
    platform_size : float, optional
        Size of the flat platform at the center in meters. Default is 1.0.
    depth : float, optional
        Depth of the gaps/holes in meters. Default is -10.

    Returns
    -------
    SubTerrain
        The modified terrain object (same as input).
    """
    # Convert to discrete units
    stone_size_px = int(stone_size / terrain.horizontal_scale)
    stone_distance_px = int(stone_distance / terrain.horizontal_scale)
    max_height_px = int(max_height / terrain.vertical_scale)
    platform_pixels = int(platform_size / terrain.horizontal_scale)
    depth_px = int(depth / terrain.vertical_scale)

    # Create height range for stones
    height_range = np.arange(-max_height_px - 1, max_height_px, step=1)

    # Fill entire terrain with depth (gaps)
    terrain.height_field_raw[:, :] = depth_px

    # Place stepping stones
    start_x, start_y = 0, 0
    # Choose pattern based on terrain shape
    if terrain.length >= terrain.width:
        # Fill column by column
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size_px)
            start_x = np.random.randint(0, stone_size_px)
            # Fill first hole
            stop_x = max(0, start_x - stone_distance_px)
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(height_range)
            # Fill row with stones
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size_px)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_x += stone_size_px + stone_distance_px
            start_y += stone_size_px + stone_distance_px
    else:
        # Fill row by row
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size_px)
            start_y = np.random.randint(0, stone_size_px)
            # Fill first hole
            stop_y = max(0, start_y - stone_distance_px)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(height_range)
            # Fill column with stones
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size_px)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_y += stone_size_px + stone_distance_px
            start_x += stone_size_px + stone_distance_px

    # Create flat platform at center
    x1 = (terrain.width - platform_pixels) // 2
    x2 = (terrain.width + platform_pixels) // 2
    y1 = (terrain.length - platform_pixels) // 2
    y2 = (terrain.length + platform_pixels) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


class SubTerrain:
    """Container for terrain heightfield data.

    This class holds the heightfield array and metadata for a sub-terrain patch.
    It provides a simple data structure for terrain generation functions to work with.

    Parameters
    ----------
    terrain_name : str
        Name identifier for this terrain patch.
    width : int
        Width of the terrain in pixels (number of points along x-axis).
    length : int
        Length of the terrain in pixels (number of points along y-axis).
    vertical_scale : float
        Vertical discretization scale in meters per height unit.
    horizontal_scale : float
        Horizontal discretization scale in meters per pixel.
    """

    def __init__(
        self,
        terrain_name: str = "terrain",
        width: int = 256,
        length: int = 256,
        vertical_scale: float = 1.0,
        horizontal_scale: float = 1.0,
    ):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw: npt.NDArray[np.int16] = np.zeros((self.width, self.length), dtype=np.int16)
