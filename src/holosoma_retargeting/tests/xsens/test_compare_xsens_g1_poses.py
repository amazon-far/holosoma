"""Tests for the canonical Xsens and G1 reference-pose comparison."""

from __future__ import annotations

import mujoco  # type: ignore[import-not-found]
import numpy as np
import pytest
from holosoma_retargeting.examples.xsens_tennis.compare_xsens_g1_poses import (
    body_poses_from_xsens_pose,
    build_canonical_xsens_npose,
    canonical_xsens_npose_orientations,
    g1_npose_qpos_from_tpose,
    orientation_correspondence_body_names,
    side_by_side_offsets,
    tree_vertical_bounds,
)
from holosoma_retargeting.kinematics import KinematicPose
from holosoma_retargeting.kinematics.model import rotate_vector
from holosoma_retargeting.xsens.g1_kinematic_reduction import (
    build_g1_proportioned_xsens_tree,
    extract_g1_anthropometry,
)
from holosoma_retargeting.xsens.kinematic_model import normalize_xsens_name


def test_side_by_side_offsets_preserve_requested_column_order() -> None:
    assert side_by_side_offsets(2.0) == (-2.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="positive"):
        side_by_side_offsets(0.0)


def test_tree_vertical_bounds_include_procedural_meshes() -> None:
    model = build_g1_proportioned_xsens_tree(extract_g1_anthropometry())
    minimum_z, maximum_z = tree_vertical_bounds(model)

    body_z = np.asarray([body.reference_pose.translation_m[2] for body in model.bodies])
    assert minimum_z < float(body_z.min())
    assert maximum_z > float(body_z.max())
    # The G1 is roughly 1.3 m tall.  This catches accidental deletion of the
    # pelvis-to-hip anchor or the collapsed hip-cluster extent.
    assert maximum_z - minimum_z > 1.25


def test_g1_xsens_render_pose_preserves_supplied_global_orientations() -> None:
    model = build_g1_proportioned_xsens_tree(extract_g1_anthropometry())
    source_names = tuple(str(body.metadata["xsens:sourceSegmentName"]) for body in model.bodies)
    positions = np.asarray([body.reference_pose.translation_m for body in model.bodies], dtype=float)
    quaternions = np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (len(model.bodies), 1))

    body_poses = body_poses_from_xsens_pose(
        model,
        KinematicPose(source_names, positions, quaternions),
    )

    assert set(body_poses) == {body.name for body in model.bodies}
    for pose in body_poses.values():
        np.testing.assert_allclose(pose.rotation_wxyz, [0.5, 0.5, 0.5, 0.5])


def test_orientation_correspondence_selection_uses_xsens_semantics() -> None:
    model = build_g1_proportioned_xsens_tree(extract_g1_anthropometry())

    selected = orientation_correspondence_body_names(model, ["L5", "Left Hand", "Right Hand"])

    assert selected == {"L5", "LeftHand", "RightHand"}


def test_canonical_npose_rotates_complete_arm_chains_and_closes_joint_anchors() -> None:
    model = build_g1_proportioned_xsens_tree(extract_g1_anthropometry())
    source_names = tuple(str(body.metadata["xsens:sourceSegmentName"]) for body in model.bodies)
    positions = np.asarray([body.reference_pose.translation_m for body in model.bodies], dtype=float)
    orientations = np.zeros((len(model.bodies), 4), dtype=float)
    orientations[:, 0] = 1.0
    tpose = KinematicPose(source_names, positions, orientations)

    npose_orientations = canonical_xsens_npose_orientations(tpose)
    npose = build_canonical_xsens_npose(model, tpose)
    source_indices = {normalize_xsens_name(name): index for index, name in enumerate(source_names)}
    left_upper_arm = npose_orientations[source_indices[normalize_xsens_name("Left Upper Arm")]]
    right_upper_arm = npose_orientations[source_indices[normalize_xsens_name("Right Upper Arm")]]
    assert abs(float(np.dot(left_upper_arm, [np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0]))) == pytest.approx(1.0)
    assert abs(float(np.dot(right_upper_arm, [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]))) == pytest.approx(1.0)
    for side in ("Left", "Right"):
        for segment in ("Upper Arm", "Forearm", "Hand"):
            np.testing.assert_allclose(
                npose_orientations[source_indices[normalize_xsens_name(f"{side} {segment}")]],
                npose_orientations[source_indices[normalize_xsens_name(f"{side} Upper Arm")]],
                atol=1e-12,
            )

    body_to_source = {
        body.name: source_indices[normalize_xsens_name(str(body.metadata["xsens:sourceSegmentName"]))]
        for body in model.bodies
    }
    for joint in model.joints:
        parent_index = body_to_source[joint.parent_body]
        child_index = body_to_source[joint.child_body]
        parent_anchor = npose.positions_m[parent_index] + rotate_vector(
            npose.orientations_wxyz[parent_index],
            joint.parent_frame.translation_m,
        )
        child_anchor = npose.positions_m[child_index] + rotate_vector(
            npose.orientations_wxyz[child_index],
            joint.child_frame.translation_m,
        )
        np.testing.assert_allclose(parent_anchor, child_anchor, atol=1e-12)


def test_g1_npose_neutralizes_only_shoulder_roll_joints() -> None:
    model_path = extract_g1_anthropometry().model_path
    model = mujoco.MjModel.from_xml_path(str(model_path))
    tpose_qpos = np.linspace(-0.5, 0.5, model.nq)

    npose_qpos = g1_npose_qpos_from_tpose(model_path, tpose_qpos)

    changed_indices: set[int] = set()
    for joint_name in ("left_shoulder_roll_joint", "right_shoulder_roll_joint"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_index = int(model.jnt_qposadr[joint_id])
        changed_indices.add(qpos_index)
        assert npose_qpos[qpos_index] == 0.0
    unchanged_indices = sorted(set(range(model.nq)) - changed_indices)
    np.testing.assert_array_equal(npose_qpos[unchanged_indices], tpose_qpos[unchanged_indices])
