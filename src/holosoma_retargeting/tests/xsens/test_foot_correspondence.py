"""Tests for the single production Xsens-to-G1 foot correspondence."""

from holosoma_retargeting.config_types.data_type import JOINTS_MAPPINGS
from holosoma_retargeting.xsens.orientation_tracking import XSENS_AXIS_SPECS
from holosoma_retargeting.xsens.tpose_calibration import (
    CALIBRATION_POSITION_MAPPING,
    CANDIDATE_ORIENTATION_MAPPING,
    LIMB_AXIS_TARGETS,
    SYMMETRY_LINK_PAIRS,
)


def test_runtime_and_calibration_use_anatomical_foot_frames() -> None:
    runtime_mapping = JOINTS_MAPPINGS[("xsens", "g1")]
    expected_positions = {
        "Left Foot": "left_ankle_roll_link",
        "Right Foot": "right_ankle_roll_link",
        "Left Toe": "left_ankle_roll_metatarsal_site",
        "Right Toe": "right_ankle_roll_metatarsal_site",
    }

    assert {name: runtime_mapping[name] for name in expected_positions} == expected_positions
    assert {name: CALIBRATION_POSITION_MAPPING[name] for name in expected_positions} == expected_positions
    assert CANDIDATE_ORIENTATION_MAPPING["Left Foot"] == "left_ankle_roll_link"
    assert CANDIDATE_ORIENTATION_MAPPING["Right Foot"] == "right_ankle_roll_link"


def test_runtime_and_calibration_use_matching_foot_axes() -> None:
    runtime_axes = {
        spec.name: (spec.robot_axis_start, spec.robot_axis_end, spec.robot_local_axis)
        for spec in XSENS_AXIS_SPECS
        if spec.name in {"left_shank", "right_shank", "left_foot_forward", "right_foot_forward"}
    }
    expected_axes = {
        "left_shank": ("left_knee_link", "left_ankle_pitch_link", None),
        "right_shank": ("right_knee_link", "right_ankle_pitch_link", None),
        "left_foot_forward": ("left_ankle_roll_link", None, (1.0, 0.0, 0.0)),
        "right_foot_forward": ("right_ankle_roll_link", None, (1.0, 0.0, 0.0)),
    }
    calibration_axes = {
        target.name: (target.robot_start, target.robot_end, target.robot_local_axis)
        for target in LIMB_AXIS_TARGETS
    }
    expected_calibration_axes = {
        "left_shank": expected_axes["left_shank"],
        "right_shank": expected_axes["right_shank"],
        "left_foot": expected_axes["left_foot_forward"],
        "right_foot": expected_axes["right_foot_forward"],
    }

    assert runtime_axes == expected_axes
    assert {name: calibration_axes[name] for name in expected_calibration_axes} == expected_calibration_axes
    assert ("left_ankle_pitch_link", "right_ankle_pitch_link", "ankle") in SYMMETRY_LINK_PAIRS
    assert (
        "left_ankle_roll_metatarsal_site",
        "right_ankle_roll_metatarsal_site",
        "toe",
    ) in SYMMETRY_LINK_PAIRS


def test_runtime_and_calibration_use_distal_compound_joint_origins() -> None:
    runtime_mapping = JOINTS_MAPPINGS[("xsens", "g1")]
    expected_positions = {
        "Left Upper Leg": "left_hip_yaw_link",
        "Right Upper Leg": "right_hip_yaw_link",
        "Left Upper Arm": "left_shoulder_yaw_link",
        "Right Upper Arm": "right_shoulder_yaw_link",
        "Left Hand": "left_wrist_yaw_link",
        "Right Hand": "right_wrist_yaw_link",
    }
    expected_axes = {
        "left_thigh": ("left_hip_pitch_link", "left_knee_link"),
        "right_thigh": ("right_hip_pitch_link", "right_knee_link"),
        "left_upper_arm": ("left_shoulder_yaw_link", "left_elbow_link"),
        "right_upper_arm": ("right_shoulder_yaw_link", "right_elbow_link"),
        "left_forearm": ("left_elbow_link", "left_wrist_yaw_link"),
        "right_forearm": ("right_elbow_link", "right_wrist_yaw_link"),
    }
    runtime_axes = {
        spec.name: (spec.robot_axis_start, spec.robot_axis_end)
        for spec in XSENS_AXIS_SPECS
        if spec.name in expected_axes
    }
    calibration_axes = {
        target.name: (target.robot_start, target.robot_end)
        for target in LIMB_AXIS_TARGETS
    }

    assert {name: runtime_mapping[name] for name in expected_positions} == expected_positions
    assert {name: CALIBRATION_POSITION_MAPPING[name] for name in expected_positions} == expected_positions
    assert runtime_axes == expected_axes
    assert {name: calibration_axes[name] for name in expected_axes} == expected_axes
    assert ("left_hip_yaw_link", "right_hip_yaw_link", "hip") in SYMMETRY_LINK_PAIRS
    assert ("left_shoulder_yaw_link", "right_shoulder_yaw_link", "shoulder") in SYMMETRY_LINK_PAIRS
    assert ("left_wrist_yaw_link", "right_wrist_yaw_link", "hand") in SYMMETRY_LINK_PAIRS


def test_head_motion_is_not_applied_to_the_fixed_g1_head() -> None:
    assert "Head" not in CANDIDATE_ORIENTATION_MAPPING
    assert "head_neck_up" not in {spec.name for spec in XSENS_AXIS_SPECS}
    assert "head" not in {target.name for target in LIMB_AXIS_TARGETS}
