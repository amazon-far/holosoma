"""ROS-free encoding of sensor-frame XYZ points for ``sensor_msgs/PointCloud2``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class EncodedPointCloud:
    """Point bytes and layout metadata needed to construct a PointCloud2 message."""

    data: bytes
    height: int
    width: int
    point_step: int
    row_step: int
    is_dense: bool


@lru_cache(maxsize=64)
def _transform_components(
    translation: tuple[float, ...],
    rotation: tuple[float, ...],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    w, x, y, z = rotation
    matrix = np.asarray(Rotation.from_quat([x, y, z, w]).as_matrix(), dtype=np.float32)
    offset = np.asarray(translation, dtype=np.float32)
    matrix.setflags(write=False)
    offset.setflags(write=False)
    return matrix, offset


def transform_xyz_points(
    points: npt.NDArray[Any],
    *,
    translation: list[float],
    rotation: list[float],
) -> npt.NDArray[np.float32]:
    """Apply a fixed transform to XYZ points as ``p_out = R p_in + t``."""
    if translation == [0.0, 0.0, 0.0] and rotation == [1.0, 0.0, 0.0, 0.0]:
        return points

    matrix, offset = _transform_components(tuple(translation), tuple(rotation))
    transformed = np.asarray(points, dtype=np.float32) @ matrix.T
    transformed += offset
    return transformed


def encode_xyz_points(
    points: npt.NDArray[Any],
    *,
    height: int,
    width: int,
) -> EncodedPointCloud:
    """Encode ``[R,3]`` XYZ points as contiguous little-endian float32 records."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"PointCloud2 XYZ encoding needs an [R,3] array, got shape {points.shape}.")
    if height <= 0 or width <= 0 or height * width != points.shape[0]:
        raise ValueError(f"PointCloud2 organized shape {height}x{width} does not contain {points.shape[0]} points.")
    xyz = np.ascontiguousarray(points, dtype="<f4")
    point_step = 3 * np.dtype("<f4").itemsize
    return EncodedPointCloud(
        data=xyz.tobytes(),
        height=height,
        width=width,
        point_step=point_step,
        row_step=point_step * width,
        is_dense=bool(np.isfinite(xyz).all()),
    )
