"""Tests for the shared kinematic model."""

from __future__ import annotations

import numpy as np
from holosoma_retargeting.kinematics import (
    KinematicTree,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
    compute_joint_positions,
    rotate_vector,
    rotate_vectors,
    validate_kinematic_tree,
)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Former scalar production formula retained as an independent oracle."""

    normalized = quaternion / np.linalg.norm(quaternion)
    conjugate = np.array([normalized[0], -normalized[1], -normalized[2], -normalized[3]])
    pure = np.array([0.0, vector[0], vector[1], vector[2]])
    return _quaternion_multiply(
        _quaternion_multiply(normalized, pure),
        conjugate,
    )[1:]


def test_rotation_apis_match_previous_scalar_implementation() -> None:
    rng = np.random.default_rng(8146)
    quaternions = rng.normal(size=(17, 1, 4))
    quaternions *= rng.uniform(0.1, 4.0, size=(17, 1, 1))
    vectors = rng.normal(size=(1, 23, 3))

    batch = rotate_vectors(quaternions, vectors)
    scalar = np.stack(
        [
            np.stack(
                [
                    _rotate_vector(quaternions[frame, 0], vectors[0, point])
                    for point in range(23)
                ]
            )
            for frame in range(17)
        ]
    )

    np.testing.assert_allclose(batch, scalar, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        rotate_vector(quaternions[3, 0], vectors[0, 7]),
        scalar[3, 7],
        rtol=2e-14,
        atol=2e-14,
    )


def test_generic_tree_validates_and_computes_joint_positions() -> None:
    model = KinematicTree(
        name="TwoBody",
        root_body="Root",
        bodies=(
            RigidBodyDefinition("Root", Transform.identity()),
            RigidBodyDefinition("Child", Transform(np.array([0.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0]))),
        ),
        joints=(
            SphericalJointDefinition(
                "Joint",
                "Root",
                "Child",
                Transform(np.array([0.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0])),
                Transform.identity(),
            ),
        ),
    )

    report = validate_kinematic_tree(model)
    assert report.is_valid
    np.testing.assert_allclose(
        compute_joint_positions(model, {body.name: body.reference_pose for body in model.bodies})["Joint"],
        [0.0, 0.0, 0.5],
    )


def test_generic_tree_rejects_disconnected_and_inconsistent_anchors() -> None:
    model = KinematicTree(
        name="Invalid",
        root_body="Root",
        bodies=(
            RigidBodyDefinition("Root", Transform.identity()),
            RigidBodyDefinition("Child", Transform.identity()),
            RigidBodyDefinition("Orphan", Transform.identity()),
        ),
        joints=(
            SphericalJointDefinition(
                "Joint",
                "Root",
                "Child",
                Transform(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
                Transform.identity(),
            ),
        ),
    )

    report = validate_kinematic_tree(model)
    assert not report.is_valid
    assert any("reference anchors differ" in error for error in report.errors)
    assert any("Orphan" in error for error in report.errors)
