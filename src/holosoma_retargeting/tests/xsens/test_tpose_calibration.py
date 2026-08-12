"""Tests for Xsens T-pose calibration."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import XSENS_BODY_SEGMENT_NAMES, XsensHdf5Tpose
from holosoma_retargeting.examples.robot_retarget import create_task_constants
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.src.mujoco_utils import evaluate_mujoco_frame_poses
from holosoma_retargeting.xsens.orientation_tracking import (
    XSENS_AXIS_SPECS,
    build_xsens_axis_calibration_metadata,
    build_xsens_orientation_targets_from_calibration,
    load_xsens_orientation_targets,
    matrix_to_quat_wijk,
    quat_wijk_to_matrix,
)
from holosoma_retargeting.xsens.tpose_calibration import (
    CALIBRATION_POSITION_MAPPING,
    XsensTposeCalibrationConfig,
    align_and_scale_tpose_targets,
    axis_alignment_error_deg,
    compute_canonical_axes,
    elbow_bend_angle_deg,
    evaluate_head_candidate,
    limb_axis_error_deg,
    solve_xsens_tpose_calibration_from_data,
    symmetry_residual,
)
from scipy.spatial.transform import Rotation

import holosoma_retargeting

MODEL_DIR = Path(holosoma_retargeting.__file__).parent / "models" / "g1"


def _symmetric_tpose_positions() -> np.ndarray:
    coords = {name: np.zeros(3, dtype=float) for name in XSENS_BODY_SEGMENT_NAMES}
    coords.update(
        {
            "Pelvis": np.array([0.0, 0.0, 0.86]),
            "L5": np.array([0.0, 0.0, 0.97]),
            "L3": np.array([0.0, 0.0, 1.04]),
            "T12": np.array([0.0, 0.0, 1.12]),
            "T8": np.array([0.0, 0.0, 1.22]),
            "Neck": np.array([0.0, 0.0, 1.36]),
            "Head": np.array([0.0, 0.0, 1.48]),
            "Left Shoulder": np.array([0.0, 0.16, 1.28]),
            "Right Shoulder": np.array([0.0, -0.16, 1.28]),
            "Left Upper Arm": np.array([0.0, 0.25, 1.28]),
            "Right Upper Arm": np.array([0.0, -0.25, 1.28]),
            "Left Forearm": np.array([0.0, 0.52, 1.28]),
            "Right Forearm": np.array([0.0, -0.52, 1.28]),
            "Left Hand": np.array([0.0, 0.74, 1.28]),
            "Right Hand": np.array([0.0, -0.74, 1.28]),
            "Left Upper Leg": np.array([0.0, 0.09, 0.84]),
            "Right Upper Leg": np.array([0.0, -0.09, 0.84]),
            "Left Lower Leg": np.array([-0.02, 0.09, 0.46]),
            "Right Lower Leg": np.array([-0.02, -0.09, 0.46]),
            "Left Foot": np.array([-0.05, 0.09, 0.08]),
            "Right Foot": np.array([-0.05, -0.09, 0.08]),
            "Left Toe": np.array([0.12, 0.09, 0.0]),
            "Right Toe": np.array([0.12, -0.09, 0.0]),
        }
    )
    return np.asarray([coords[name] for name in XSENS_BODY_SEGMENT_NAMES], dtype=float)


def test_align_and_scale_tpose_targets_ground_aligns_and_centers() -> None:
    positions = _symmetric_tpose_positions()
    positions[:, :2] += np.array([2.0, -3.0])

    aligned, axes = align_and_scale_tpose_targets(positions, scale_factor=2.0)

    np.testing.assert_allclose(aligned[XSENS_BODY_SEGMENT_NAMES.index("Pelvis"), :2], [0.0, 0.0])
    foot_indices = [
        XSENS_BODY_SEGMENT_NAMES.index(name) for name in ("Left Foot", "Right Foot", "Left Toe", "Right Toe")
    ]
    assert np.min(aligned[foot_indices, 2]) == 0.0
    np.testing.assert_allclose(np.cross(axes[0], axes[1]), axes[2], atol=1e-8)
    assert axes[2, 2] > 0.0


def test_compute_canonical_axes_is_right_handed() -> None:
    axes = compute_canonical_axes(_symmetric_tpose_positions())

    np.testing.assert_allclose(axes[0], [1.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(axes[1], [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(axes[2], [0.0, 0.0, 1.0], atol=1e-8)


def test_symmetry_and_limb_axis_residual_helpers() -> None:
    np.testing.assert_allclose(symmetry_residual([1.0, 2.0, 3.0], [1.0, -2.0, 3.0]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(symmetry_residual([1.1, 2.0, 3.0], [1.0, -2.0, 3.0]), [0.1, 0.0, 0.0])
    assert limb_axis_error_deg([0, 0, 0], [1, 0, 0], [2, 2, 2], [3, 2, 2]) < 1e-8


def test_head_candidate_diagnostics_accept_and_reject() -> None:
    cfg = XsensTposeCalibrationConfig(verbose=0)

    accepted, status = evaluate_head_candidate(
        head_axis_error_deg=5.0,
        head_orientation_offset_wijk=np.array([1.0, 0.0, 0.0, 0.0]),
        config=cfg,
    )
    assert accepted
    assert status.startswith("accepted")

    rejected, status = evaluate_head_candidate(
        head_axis_error_deg=20.0,
        head_orientation_offset_wijk=np.array([1.0, 0.0, 0.0, 0.0]),
        config=cfg,
    )
    assert not rejected
    assert status.startswith("rejected")


def test_axis_calibration_metadata_matches_tracking_matrix() -> None:
    positions = _symmetric_tpose_positions()
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(XSENS_BODY_SEGMENT_NAMES), 1))

    metadata = build_xsens_axis_calibration_metadata(
        tpose_positions_m=positions,
        tpose_quaternions_wijk=quaternions,
    )

    assert metadata["axis_names"].tolist() == [spec.name for spec in XSENS_AXIS_SPECS]
    assert metadata["axis_xsens_segment_names"].tolist() == [spec.xsens_segment for spec in XSENS_AXIS_SPECS]
    assert metadata["axis_robot_start_link_names"].tolist() == [spec.robot_axis_start for spec in XSENS_AXIS_SPECS]
    assert metadata["axis_robot_end_link_names"].tolist() == [spec.robot_axis_end or "" for spec in XSENS_AXIS_SPECS]
    np.testing.assert_allclose(
        metadata["axis_robot_local_vectors"],
        np.asarray([spec.robot_local_axis or (0.0, 0.0, 0.0) for spec in XSENS_AXIS_SPECS]),
    )
    np.testing.assert_allclose(np.linalg.norm(metadata["axis_local_tpose_xyz"], axis=1), 1.0)


def test_orientation_targets_reconstruct_from_dynamic_orientation_and_saved_offsets(tmp_path) -> None:
    angle = np.pi / 2.0
    dynamic_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    offset_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    calibration_path = tmp_path / "calibration.npz"
    positions = _symmetric_tpose_positions()
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(XSENS_BODY_SEGMENT_NAMES), 1))
    axis_metadata = build_xsens_axis_calibration_metadata(
        tpose_positions_m=positions,
        tpose_quaternions_wijk=quaternions,
    )
    np.savez(
        calibration_path,
        active_orientation_mapping_names=np.asarray(["L5"], dtype=str),
        robot_link_names=np.asarray(["torso_link"], dtype=str),
        orientation_offsets_wijk=matrix_to_quat_wijk(offset_rotation.reshape(1, 3, 3)),
        **axis_metadata,
    )

    motion_quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, len(XSENS_BODY_SEGMENT_NAMES), 1))
    motion_quaternions[:, XSENS_BODY_SEGMENT_NAMES.index("L5")] = matrix_to_quat_wijk(
        np.tile(dynamic_rotation, (2, 1, 1))
    )

    targets = load_xsens_orientation_targets(
        calibration_path=calibration_path,
        motion_quaternions_wijk=motion_quaternions,
        segment_names=XSENS_BODY_SEGMENT_NAMES,
    )

    assert targets.orientation_names == ["L5"]
    assert targets.orientation_robot_link_names == ["torso_link"]
    np.testing.assert_allclose(
        targets.orientation_target_rotations[0, 0],
        dynamic_rotation @ offset_rotation,
        atol=1e-12,
    )
    assert targets.axis_target_vectors.shape == (2, len(XSENS_AXIS_SPECS), 3)


def test_orientation_tracking_jacobians_match_finite_difference() -> None:
    robot_urdf = str(MODEL_DIR / "g1_29dof.urdf")
    constants = create_task_constants(
        robot_config=RobotConfig(robot_type="g1", robot_urdf_file=robot_urdf),
        motion_data_config=MotionDataConfig(data_format="xsens", robot_type="g1"),
        task_config=TaskConfig(),
        task_type="robot_only",
    )
    retargeter = InteractionMeshRetargeter(
        task_constants=constants,
        object_urdf_path=None,
        activate_foot_sticking=False,
        activate_obj_non_penetration=False,
    )
    q = np.zeros(retargeter.nq, dtype=float)
    q[3] = 1.0
    retargeter.robot_data.qpos[:] = q
    mujoco.mj_forward(retargeter.robot_model, retargeter.robot_data)

    joint_id = mujoco.mj_name2id(retargeter.robot_model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")
    q_idx = int(retargeter.robot_model.jnt_qposadr[joint_id])
    eps = 1e-6

    J_rot, current_rotation = retargeter._frame_rotational_jacobian("left_rubber_hand_link")
    q_eps = q.copy()
    q_eps[q_idx] += eps
    retargeter.robot_data.qpos[:] = q_eps
    mujoco.mj_forward(retargeter.robot_model, retargeter.robot_data)
    _, rotation_eps, _ = retargeter._frame_pose("left_rubber_hand_link")
    rotvec_fd = Rotation.from_matrix(rotation_eps @ current_rotation.T).as_rotvec() / eps
    np.testing.assert_allclose(J_rot[:, q_idx], rotvec_fd, atol=1e-4)

    retargeter.robot_data.qpos[:] = q
    mujoco.mj_forward(retargeter.robot_model, retargeter.robot_data)
    axis, J_axis = retargeter._axis_jacobian("left_elbow_link", "left_rubber_hand_link")
    q_eps = q.copy()
    q_eps[q_idx] += eps
    retargeter.robot_data.qpos[:] = q_eps
    mujoco.mj_forward(retargeter.robot_model, retargeter.robot_data)
    axis_eps, _ = retargeter._axis_jacobian("left_elbow_link", "left_rubber_hand_link")
    axis_fd = (axis_eps - axis) / eps
    np.testing.assert_allclose(J_axis[:, q_idx], axis_fd, atol=1e-4)

    ankle_pitch_id = mujoco.mj_name2id(
        retargeter.robot_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "left_ankle_pitch_joint",
    )
    ankle_pitch_q_idx = int(retargeter.robot_model.jnt_qposadr[ankle_pitch_id])
    retargeter.robot_data.qpos[:] = q
    mujoco.mj_forward(retargeter.robot_model, retargeter.robot_data)
    body_axis, J_body_axis = retargeter._body_fixed_axis_jacobian(
        "left_ankle_roll_link",
        np.array([1.0, 0.0, 0.0]),
    )
    q_eps = q.copy()
    q_eps[ankle_pitch_q_idx] += eps
    retargeter.robot_data.qpos[:] = q_eps
    mujoco.mj_forward(retargeter.robot_model, retargeter.robot_data)
    body_axis_eps, _ = retargeter._body_fixed_axis_jacobian(
        "left_ankle_roll_link",
        np.array([1.0, 0.0, 0.0]),
    )
    body_axis_fd = (body_axis_eps - body_axis) / eps
    np.testing.assert_allclose(J_body_axis[:, ankle_pitch_q_idx], body_axis_fd, atol=1e-4)


def test_g1_tpose_calibration_smoke_on_synthetic_symmetric_tpose() -> None:
    positions = _symmetric_tpose_positions()
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(XSENS_BODY_SEGMENT_NAMES), 1))
    tpose = XsensHdf5Tpose(
        positions_m=positions,
        quaternions_wijk=quaternions,
        variant="Tpose",
        segment_names=XSENS_BODY_SEGMENT_NAMES,
        source_indices=list(range(len(XSENS_BODY_SEGMENT_NAMES))),
    )

    calibration_config = XsensTposeCalibrationConfig(max_nfev=8, verbose=0)
    result = solve_xsens_tpose_calibration_from_data(
        tpose,
        config=calibration_config,
    )

    assert result.qpos.shape == (1, 36)
    assert np.isfinite(result.qpos).all()
    assert result.head_candidate_status == "not evaluated"
    assert result.axis_names == [spec.name for spec in XSENS_AXIS_SPECS]
    assert result.axis_local_tpose_xyz.shape == (len(XSENS_AXIS_SPECS), 3)
    assert result.position_offsets_robot_minus_xsens_m.shape == (len(CALIBRATION_POSITION_MAPPING), 3)

    link_poses = evaluate_mujoco_frame_poses(
        MODEL_DIR / "g1_29dof.xml",
        result.qpos[0],
        result.robot_link_names,
    )
    targets = build_xsens_orientation_targets_from_calibration(
        result,
        motion_quaternions_wijk=tpose.quaternions_wijk[None, ...],
        segment_names=tpose.segment_names,
    )
    np.testing.assert_allclose(
        quat_wijk_to_matrix(link_poses.quaternions_wxyz),
        targets.orientation_target_rotations[0],
        atol=1e-10,
    )

    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "g1_29dof.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = result.qpos[0]
    mujoco.mj_forward(model, data)
    left_hand = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_rubber_hand_link")]
    right_hand = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_rubber_hand_link")]
    np.testing.assert_allclose(left_hand[[0, 2]], right_hand[[0, 2]], atol=0.2)
    np.testing.assert_allclose(left_hand[1], -right_hand[1], atol=0.2)
    left_knee = result.qpos[0, 10]
    right_knee = result.qpos[0, 16]
    assert abs(left_knee) < 0.5
    assert abs(right_knee) < 0.5
    forward = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    neutral_data = mujoco.MjData(model)
    neutral_data.qpos[3] = 1.0
    mujoco.mj_forward(model, neutral_data)
    neutral_hip_positions = [
        neutral_data.xanchor[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_pitch_joint")]
        for side in ("left", "right")
    ]
    nominal_half_hip_width = 0.5 * np.linalg.norm(neutral_hip_positions[0] - neutral_hip_positions[1])
    for side in ("left", "right"):
        joint_positions = []
        for joint_name in ("shoulder_yaw_joint", "elbow_joint", "wrist_roll_joint"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint_name}")
            joint_positions.append(data.xanchor[joint_id])
        assert elbow_bend_angle_deg(*joint_positions) < 2.0
        expected_arm_axis = np.array([0.0, 1.0 if side == "left" else -1.0, 0.0])
        assert axis_alignment_error_deg(joint_positions[-1] - joint_positions[0], expected_arm_axis) < 2.0
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_rubber_hand_link")
        hand_rotation = data.xmat[hand_id].reshape(3, 3)
        thumb_axis = hand_rotation[:, 2]
        palm_normal = hand_rotation[:, 1]
        expected_palm_normal = up if side == "left" else -up
        assert axis_alignment_error_deg(thumb_axis, forward) < 8.0
        assert axis_alignment_error_deg(palm_normal, expected_palm_normal) < 8.0

        contact_positions = []
        contact_radii = []
        for index in range(1, 6):
            geom_name = f"{side}_ankle_roll_sphere_{index}_link"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            contact_positions.append(data.geom_xpos[geom_id])
            contact_radii.append(model.geom_size[geom_id, 0])
        contact_positions = np.asarray(contact_positions)
        contact_bottom_heights = contact_positions[:, 2] - np.asarray(contact_radii)
        heel_center = np.mean(contact_positions[:2], axis=0)
        toe_center = np.mean(contact_positions[2:], axis=0)
        positive_lateral = np.mean(contact_positions[[0, 2]], axis=0)
        negative_lateral = np.mean(contact_positions[[1, 3]], axis=0)
        sole_forward = toe_center - heel_center
        sole_lateral = positive_lateral - negative_lateral
        sole_normal = np.cross(sole_forward, sole_lateral)
        assert np.min(contact_bottom_heights) >= -1e-8
        assert np.max(contact_bottom_heights) < 0.005
        assert axis_alignment_error_deg(sole_forward, forward) < 2.0
        assert axis_alignment_error_deg(sole_normal, up) < 2.0

        hip_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_pitch_joint")
        knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_knee_joint")
        ankle_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_pitch_joint")
        ankle_roll_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_roll_joint")
        hip_position = data.xanchor[hip_joint_id]
        knee_position = data.xanchor[knee_joint_id]
        ankle_position = data.xanchor[ankle_joint_id]
        assert axis_alignment_error_deg(knee_position - hip_position, ankle_position - knee_position) < 12.0
        assert axis_alignment_error_deg(ankle_position - hip_position, -up) < 3.0
        ankle_roll_qpos = data.qpos[model.jnt_qposadr[ankle_roll_joint_id]]
        assert abs(np.degrees(ankle_roll_qpos)) < 4.0

        foot_center = np.mean(contact_positions, axis=0)
        target_lateral = data.qpos[1] + (1.0 if side == "left" else -1.0) * nominal_half_hip_width
        # The G1's offset hip linkage places a neutral knee about 54 mm outside
        # its hip line. Calibration reduces the offset while allowing a modestly
        # wider stance that preserves straightness and ankle-roll workspace.
        assert abs(knee_position[1] - target_lateral) < 0.04
        assert abs(foot_center[1] - target_lateral) < 0.025

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    torso_rotation = data.xmat[torso_id].reshape(3, 3)
    assert axis_alignment_error_deg(torso_rotation[:, 0], forward) < 1.0
    torso_up = torso_rotation[:, 2]
    assert axis_alignment_error_deg(torso_up, up) < 1.0


def test_g1_proportioned_tpose_calibration_uses_explicit_unit_scale() -> None:
    positions = _symmetric_tpose_positions()
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(XSENS_BODY_SEGMENT_NAMES), 1))
    tpose = XsensHdf5Tpose(
        positions_m=positions,
        quaternions_wijk=quaternions,
        variant="G1ProportionedTpose",
        segment_names=XSENS_BODY_SEGMENT_NAMES,
        source_indices=list(range(len(XSENS_BODY_SEGMENT_NAMES))),
    )

    result = solve_xsens_tpose_calibration_from_data(
        tpose,
        config=XsensTposeCalibrationConfig(max_nfev=1, verbose=0),
        position_scale_factor=1.0,
    )
    expected_positions, _ = align_and_scale_tpose_targets(positions, scale_factor=1.0)

    assert result.scale_factor == 1.0
    assert result.variant == "G1ProportionedTpose"
    np.testing.assert_allclose(result.xsens_tpose_positions_m, expected_positions)


@pytest.mark.parametrize("scale_factor", [0.0, -1.0, np.inf, np.nan])
def test_tpose_calibration_rejects_invalid_explicit_position_scale(scale_factor: float) -> None:
    positions = _symmetric_tpose_positions()
    tpose = XsensHdf5Tpose(
        positions_m=positions,
        quaternions_wijk=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]),
            (len(XSENS_BODY_SEGMENT_NAMES), 1),
        ),
        variant="Tpose",
        segment_names=XSENS_BODY_SEGMENT_NAMES,
        source_indices=list(range(len(XSENS_BODY_SEGMENT_NAMES))),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        solve_xsens_tpose_calibration_from_data(
            tpose,
            config=XsensTposeCalibrationConfig(max_nfev=1, verbose=0),
            position_scale_factor=scale_factor,
        )
