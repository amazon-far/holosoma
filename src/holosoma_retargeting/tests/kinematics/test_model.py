"""Tests for the shared kinematic model."""

from __future__ import annotations

import numpy as np
from holosoma_retargeting.kinematics import (
    KinematicTree,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
    compute_joint_positions,
    validate_kinematic_tree,
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
