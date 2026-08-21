"""Shared scalar-first quaternion, rotation-matrix, and rigid-transform conversions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

__all__ = [
    "normalize_quaternions_wxyz",
    "position_quaternion_from_transform",
    "rotation_as_wxyz",
    "rotation_matrices_as_wxyz",
    "rotation_matrices_from_wxyz",
    "rotations_from_wxyz",
    "transform_from_position_quaternion",
]


def normalize_quaternions_wxyz(
    quaternions_wxyz: np.ndarray,
    *,
    canonical: bool = True,
) -> np.ndarray:
    """Normalize scalar-first quaternion(s), optionally choosing a nonnegative scalar part."""

    values = np.asarray(quaternions_wxyz, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError(f"Expected scalar-first quaternion array ending in 4, got {values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("Quaternions must contain finite, nonzero values")
    normalized = values / norms
    if not canonical:
        return normalized
    signs = np.where(normalized[..., :1] < 0.0, -1.0, 1.0)
    return normalized * signs


def rotations_from_wxyz(quaternions_wxyz: np.ndarray) -> Rotation:
    """Build one or more SciPy rotations from scalar-first quaternion(s)."""

    values = normalize_quaternions_wxyz(quaternions_wxyz)
    if values.ndim > 2:
        raise ValueError(f"SciPy Rotation requires shape [4] or [N, 4], got {values.shape}")
    return Rotation.from_quat(values[..., [1, 2, 3, 0]])


def rotation_as_wxyz(rotation: Rotation, *, canonical: bool = True) -> np.ndarray:
    """Return scalar-first quaternion(s) from one or more SciPy rotations."""

    xyzw = np.asarray(rotation.as_quat(), dtype=float)
    return normalize_quaternions_wxyz(xyzw[..., [3, 0, 1, 2]], canonical=canonical)


def rotation_matrices_from_wxyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion arrays with arbitrary leading dimensions to matrices."""

    values = normalize_quaternions_wxyz(quaternions_wxyz)
    leading_shape = values.shape[:-1]
    xyzw = values[..., [1, 2, 3, 0]].reshape(-1, 4)
    return Rotation.from_quat(xyzw).as_matrix().reshape(leading_shape + (3, 3))


def rotation_matrices_as_wxyz(
    rotation_matrices: np.ndarray,
    *,
    canonical: bool = True,
) -> np.ndarray:
    """Convert rotation-matrix arrays with arbitrary leading dimensions to scalar-first quaternions."""

    matrices = np.asarray(rotation_matrices, dtype=float)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError(f"Expected finite rotation matrices ending in [3, 3], got {matrices.shape}")
    leading_shape = matrices.shape[:-2]
    rotations = Rotation.from_matrix(matrices.reshape(-1, 3, 3))
    return rotation_as_wxyz(rotations, canonical=canonical).reshape(leading_shape + (4,))


def transform_from_position_quaternion(
    position_m: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Construct a homogeneous transform from a position and scalar-first quaternion."""

    position = np.asarray(position_m, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"Expected a finite position with shape [3], got {position.shape}")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotations_from_wxyz(quaternion_wxyz).as_matrix()
    transform[:3, 3] = position
    return transform


def position_quaternion_from_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract a position and canonical scalar-first quaternion from a transform."""

    value = np.asarray(transform, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"Expected a finite homogeneous transform with shape [4, 4], got {value.shape}")
    return value[:3, 3].copy(), rotation_as_wxyz(Rotation.from_matrix(value[:3, :3]))
