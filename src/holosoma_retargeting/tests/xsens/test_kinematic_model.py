"""Tests for the Xsens kinematic model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    JointRotationMetadata,
    SegmentPoseSet,
    XsensHdf5Calibration,
)
from holosoma_retargeting.kinematics import compute_reference_joint_positions, validate_kinematic_tree
from holosoma_retargeting.xsens.kinematic_model import (
    TENNIS_RACKET_BODY,
    TENNIS_RACKET_JOINT,
    XSENS_JOINT_SPECS,
    XSENS_RACKET_SOURCE_JOINT,
    XSENS_RACKET_SOURCE_SEGMENT,
    build_xsens_kinematic_tree,
)


def synthetic_calibration() -> XsensHdf5Calibration:
    segment_names = ("Pelvis",) + tuple(spec.child_segment for spec in XSENS_JOINT_SPECS)
    joint_names = tuple(spec.source_joint for spec in XSENS_JOINT_SPECS)
    positions = {"Pelvis": np.zeros(3)}
    landmarks: dict[str, dict[str, np.ndarray]] = {name: {} for name in segment_names}
    for index, spec in enumerate(XSENS_JOINT_SPECS):
        if spec.source_joint == XSENS_RACKET_SOURCE_JOINT:
            positions[spec.child_segment] = positions[spec.parent_segment] + np.array([0.02, -0.1, 0.0])
            continue
        offset = np.array([0.002 * (index + 1), 0.001 * ((index % 3) - 1), 0.08 + 0.001 * index])
        landmarks[spec.parent_segment][spec.landmark] = offset
        landmarks[spec.child_segment][spec.landmark] = np.zeros(3)
        positions[spec.child_segment] = positions[spec.parent_segment] + offset
    pose = SegmentPoseSet(
        positions_m=np.asarray([positions[name] for name in segment_names]),
        quaternions_wijk=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(segment_names), 1)),
        variant="Tpose",
    )
    rotation_metadata = {
        name: JointRotationMetadata(name, ("x", "y", "z"), ("body_joint_angles_eulerZXY_xyz_rad",))
        for name in joint_names
    }
    return XsensHdf5Calibration(
        source_path=Path("recording.hdf5"),
        source_stream_name="body_position_xyz_m",
        segment_names=segment_names,
        joint_names=joint_names,
        tpose=pose,
        tpose_isb=None,
        identity_pose=None,
        landmarks_m=landmarks,
        joint_rotation_metadata=rotation_metadata,
        joint_stream_names=("body_joint_angles_eulerZXY_xyz_rad",),
        mvn_version="test",
        mvnx_version="4",
    )


def test_xsens_conversion_renames_racket_and_preserves_source_provenance() -> None:
    calibration = synthetic_calibration()
    model = build_xsens_kinematic_tree(calibration)

    assert len(model.bodies) == 24
    assert len(model.joints) == 23
    assert model.bodies[-1].name == TENNIS_RACKET_BODY
    assert model.joints[-1].name == TENNIS_RACKET_JOINT
    assert model.bodies[-1].metadata["xsens:sourceSegmentName"] == XSENS_RACKET_SOURCE_SEGMENT
    assert model.joints[-1].metadata["xsens:sourceJointName"] == XSENS_RACKET_SOURCE_JOINT
    assert all("Sword" not in body.name for body in model.bodies)
    assert all("Sword" not in joint.name for joint in model.joints)
    assert validate_kinematic_tree(model).is_valid


def test_every_xsens_joint_position_equals_its_child_segment_origin() -> None:
    calibration = synthetic_calibration()
    model = build_xsens_kinematic_tree(calibration)
    body_map = model.body_map()
    joint_positions = compute_reference_joint_positions(model)

    for joint in model.joints:
        np.testing.assert_allclose(joint_positions[joint.name], body_map[joint.child_body].reference_pose.translation_m)


def test_human_only_model_omits_racket_body_and_joint() -> None:
    model = build_xsens_kinematic_tree(synthetic_calibration(), include_tennis_racket=False)

    assert len(model.bodies) == 23
    assert len(model.joints) == 22
    assert TENNIS_RACKET_BODY not in model.body_map()
    assert TENNIS_RACKET_JOINT not in model.joint_map()
    assert validate_kinematic_tree(model).is_valid
