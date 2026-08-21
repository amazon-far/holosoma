"""Tests for shared scalar-first rotation and transform conversions."""

from __future__ import annotations

import numpy as np
import pytest
from holosoma_retargeting.transformation_utils import (
    normalize_quaternions_wxyz,
    position_quaternion_from_transform,
    rotation_as_wxyz,
    rotation_matrices_as_wxyz,
    rotation_matrices_from_wxyz,
    rotations_from_wxyz,
    transform_from_position_quaternion,
)
from scipy.spatial.transform import Rotation


def test_scalar_first_rotation_conversions_support_single_and_batched_values() -> None:
    rotations = Rotation.from_euler("xyz", [[10.0, 20.0, 30.0], [-25.0, 5.0, 170.0]], degrees=True)
    quaternions = rotation_as_wxyz(rotations)
    assert quaternions.shape == (2, 4)
    assert np.all(quaternions[:, 0] >= 0.0)
    np.testing.assert_allclose(
        (rotations_from_wxyz(quaternions).inv() * rotations).magnitude(),
        0.0,
        atol=1e-12,
    )
    single = rotation_as_wxyz(rotations[0])
    assert single.shape == (4,)
    np.testing.assert_allclose(rotations_from_wxyz(single).as_matrix(), rotations[0].as_matrix())


def test_rotation_matrix_conversions_preserve_arbitrary_leading_dimensions() -> None:
    quaternions = np.zeros((2, 3, 4), dtype=float)
    quaternions[..., 0] = -1.0
    matrices = rotation_matrices_from_wxyz(quaternions)
    assert matrices.shape == (2, 3, 3, 3)
    np.testing.assert_allclose(matrices, np.broadcast_to(np.eye(3), matrices.shape), atol=1e-12)
    restored = rotation_matrices_as_wxyz(matrices)
    np.testing.assert_allclose(restored[..., 0], 1.0)
    np.testing.assert_allclose(restored[..., 1:], 0.0, atol=1e-12)


def test_quaternion_normalization_rejects_invalid_values() -> None:
    np.testing.assert_allclose(normalize_quaternions_wxyz([-2.0, 0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        normalize_quaternions_wxyz([-2.0, 0.0, 0.0, 0.0], canonical=False),
        [-1.0, 0.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="nonzero"):
        normalize_quaternions_wxyz(np.zeros(4))
    with pytest.raises(ValueError, match="ending in 4"):
        normalize_quaternions_wxyz(np.zeros((2, 3)))


def test_position_quaternion_transform_round_trip() -> None:
    position = np.array([0.4, -0.2, 1.3])
    quaternion = rotation_as_wxyz(Rotation.from_euler("zyx", [45.0, -10.0, 25.0], degrees=True))
    transform = transform_from_position_quaternion(position, quaternion)
    restored_position, restored_quaternion = position_quaternion_from_transform(transform)
    np.testing.assert_allclose(restored_position, position)
    np.testing.assert_allclose(restored_quaternion, quaternion)
