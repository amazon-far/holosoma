"""Tests for the Xsens retargeting morphology-selection seam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tyro
from holosoma_retargeting.config_types.retargeter import OrientationTrackingConfig, RetargeterConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig, XsensMorphologyConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XSENS_BODY_SEGMENT_NAMES,
    XsensHdf5Motion,
    XsensHdf5Tpose,
)
from holosoma_retargeting.examples import robot_retarget
from holosoma_retargeting.kinematics import KinematicMotion
from holosoma_retargeting.xsens.morphology_adaptation import XsensRootMotionConfig
from holosoma_retargeting.xsens.orientation_tracking import XsensOrientationTargets


def _motion() -> XsensHdf5Motion:
    body_count = len(XSENS_BODY_SEGMENT_NAMES)
    quaternions = np.zeros((1, body_count, 4), dtype=float)
    quaternions[..., 0] = 1.0
    return XsensHdf5Motion(
        positions_m=np.arange(body_count * 3, dtype=float).reshape(1, body_count, 3),
        times_s=np.array([0.0]),
        stream_name="body_position_xyz_m",
        segment_names=list(XSENS_BODY_SEGMENT_NAMES),
        source_indices=list(range(body_count)),
        quaternions_wijk=quaternions,
        orientation_stream_name="body_orientation_quaternion_wijk",
    )


def test_xsens_morphology_defaults_to_g1_proportioned_dynamic_grounding() -> None:
    config = XsensMorphologyConfig()

    assert config.mode == "g1_proportioned"
    assert config.grounding == "match_lowest_soles"
    assert config.root_motion.mode == "scale_by_leg_length"
    assert config.root_motion.ground_height_m is None
    assert config.preserve_joint_offsets is False
    assert config.g1_model_path is None
    assert config.track_orientations is True


def test_g1_proportioned_xsens_enables_orientation_tracking_by_default() -> None:
    resolved = robot_retarget.resolve_orientation_tracking_config(
        retargeter_config=RetargeterConfig(),
        morphology_config=XsensMorphologyConfig(),
        data_format="xsens",
        task_type="robot_only",
        robot="g1",
    )
    disabled = robot_retarget.resolve_orientation_tracking_config(
        retargeter_config=RetargeterConfig(),
        morphology_config=XsensMorphologyConfig(track_orientations=False),
        data_format="xsens",
        task_type="robot_only",
        robot="g1",
    )

    assert resolved.orientation.enable is True
    assert disabled.orientation.enable is False


def _retargeter_summary_stub():
    return SimpleNamespace(
        laplacian_match_links={"Left Hand": "left_rubber_hand_link"},
        laplacian_weights=10.0,
        smooth_weight=0.2,
        Q_diag=np.array([0.0, 1.0]),
        w_nominal_tracking_init=5.0,
        track_nominal_indices=[0],
        activate_foot_sticking=True,
        q_a_init_idx=-7,
        activate_joint_limits=True,
        activate_obj_non_penetration=True,
        foot_lock=SimpleNamespace(enable=False),
        _self_collision_config=SimpleNamespace(enable=False),
        step_size=0.2,
        orientation_config=RetargeterConfig().orientation,
    )


def test_retargeting_summary_lists_active_objectives_and_rotation_offsets() -> None:
    targets = XsensOrientationTargets(
        orientation_names=["Left Hand"],
        orientation_robot_link_names=["left_rubber_hand_link"],
        orientation_offsets_wijk=np.array([[0.5, 0.5, 0.5, 0.5]]),
        orientation_target_rotations=np.eye(3).reshape(1, 1, 3, 3),
        axis_names=["left_forearm"],
        axis_xsens_segment_names=["Left Forearm"],
        axis_robot_start_link_names=["left_elbow_link"],
        axis_robot_end_link_names=["left_rubber_hand_link"],
        axis_robot_local_vectors=np.zeros((1, 3), dtype=float),
        axis_target_vectors=np.array([[[1.0, 0.0, 0.0]]]),
        axis_weights=np.ones(1),
    )

    summary = "\n".join(
        robot_retarget.describe_retargeting_setup(
            retargeter=_retargeter_summary_stub(),
            orientation_targets=targets,
            q_nominal_list=None,
        )
    )

    assert "[active] interaction-mesh positional/relational tracking" in summary
    assert "[active] full segment-orientation tracking" in summary
    assert "[active] segment-axis direction tracking" in summary
    assert "R_G1_target_world(t) = R_Xsens_segment_world(t) @ R_offset" in summary
    assert "Left Hand -> left_rubber_hand_link" in summary
    assert "offset_wxyz=(+0.500000, +0.500000, +0.500000, +0.500000)" in summary
    assert "offset_angle=120.00 deg" in summary


def test_retargeting_summary_explicitly_reports_inactive_orientation_tracking() -> None:
    summary = "\n".join(
        robot_retarget.describe_retargeting_setup(
            retargeter=_retargeter_summary_stub(),
            orientation_targets=None,
            q_nominal_list=None,
        )
    )

    assert "[inactive] full segment-orientation tracking" in summary
    assert "[inactive] segment-axis direction tracking" in summary
    assert "robot link orientations are unconstrained" in summary


def test_direct_mode_preserves_raw_positions_and_human_scale() -> None:
    motion = _motion()

    positions, scale = robot_retarget.prepare_xsens_motion_for_retargeting(
        motion,
        hdf5_path=Path("recording.hdf5"),
        direct_scale=0.75,
        robot_config=RobotConfig(robot_type="g1"),
        morphology_config=XsensMorphologyConfig(mode="direct"),
    )

    np.testing.assert_array_equal(positions, motion.positions_m)
    assert not np.shares_memory(positions, motion.positions_m)
    assert scale == 0.75


def test_g1_mode_uses_robot_xml_and_disables_uniform_rescaling(monkeypatch) -> None:
    motion = _motion()
    adapted_positions = motion.positions_m + 10.0
    captured: dict[str, object] = {}

    def fake_adapt(input_motion, **kwargs):
        captured.update(kwargs)
        return KinematicMotion(
            tuple(input_motion.segment_names),
            adapted_positions,
            input_motion.quaternions_wijk,
            input_motion.times_s,
        )

    monkeypatch.setattr(robot_retarget, "adapt_xsens_motion_to_g1", fake_adapt)
    positions, scale = robot_retarget.prepare_xsens_motion_for_retargeting(
        motion,
        hdf5_path=Path("recording.hdf5"),
        direct_scale=0.75,
        robot_config=RobotConfig(robot_type="g1", robot_urdf_file="models/g1/custom.urdf"),
        morphology_config=XsensMorphologyConfig(),
    )

    np.testing.assert_array_equal(positions, adapted_positions)
    assert scale == 1.0
    assert captured["g1_model_path"] == Path("models/g1/custom.xml")
    assert captured["grounding"] == "match_lowest_soles"
    assert captured["root_motion"] == XsensMorphologyConfig().root_motion


def test_g1_mode_calibrates_orientations_from_g1_proportioned_tpose(monkeypatch) -> None:
    motion = _motion()
    body_count = len(XSENS_BODY_SEGMENT_NAMES)
    adapted_tpose = XsensHdf5Tpose(
        positions_m=np.arange(body_count * 3, dtype=float).reshape(body_count, 3),
        quaternions_wijk=motion.quaternions_wijk[0].copy(),
        variant="G1ProportionedTpose",
        segment_names=list(XSENS_BODY_SEGMENT_NAMES),
        source_indices=list(range(body_count)),
    )
    sentinel_calibration = object()
    sentinel_targets = object()
    captured: dict[str, object] = {}

    def fake_adapt_tpose(**kwargs):
        captured["adapt"] = kwargs
        return adapted_tpose

    def fake_solve(tpose, config, *, position_scale_factor):
        captured["solve_tpose"] = tpose
        captured["solve_config"] = config
        captured["position_scale_factor"] = position_scale_factor
        return sentinel_calibration

    def fake_build(calibration, **kwargs):
        captured["build_calibration"] = calibration
        captured["build"] = kwargs
        return sentinel_targets

    monkeypatch.setattr(robot_retarget, "adapt_xsens_tpose_to_g1", fake_adapt_tpose)
    monkeypatch.setattr(robot_retarget, "solve_xsens_tpose_calibration_from_data", fake_solve)
    monkeypatch.setattr(robot_retarget, "build_xsens_orientation_targets_from_calibration", fake_build)

    result = robot_retarget.load_orientation_targets_for_retargeting(
        orientation_config=OrientationTrackingConfig(enable=True),
        robot_config=RobotConfig(robot_type="g1", robot_urdf_file="models/g1/custom.urdf"),
        robot="g1",
        data_format="xsens",
        task_type="robot_only",
        xsens_motion=motion,
        hdf5_path=Path("recording.hdf5"),
        morphology_config=XsensMorphologyConfig(),
    )

    assert result is sentinel_targets
    assert captured["adapt"] == {
        "hdf5_path": Path("recording.hdf5"),
        "g1_model_path": Path("models/g1/custom.xml"),
        "grounding": "match_lowest_soles",
        "preserve_joint_offsets": False,
    }
    assert captured["solve_tpose"] is adapted_tpose
    assert captured["position_scale_factor"] == 1.0
    assert captured["build_calibration"] is sentinel_calibration


def test_direct_mode_retains_human_scaled_orientation_calibration(monkeypatch) -> None:
    motion = _motion()
    sentinel_calibration = object()
    sentinel_targets = object()
    captured: dict[str, object] = {}

    def fake_solve(hdf5_path, *, config):
        captured["hdf5_path"] = hdf5_path
        captured["config"] = config
        return sentinel_calibration

    def fail_adapt(**_kwargs):
        raise AssertionError("Direct mode must not morphology-adapt the calibration T-pose")

    monkeypatch.setattr(robot_retarget, "solve_xsens_tpose_calibration", fake_solve)
    monkeypatch.setattr(robot_retarget, "adapt_xsens_tpose_to_g1", fail_adapt)
    monkeypatch.setattr(
        robot_retarget,
        "build_xsens_orientation_targets_from_calibration",
        lambda _calibration, **_kwargs: sentinel_targets,
    )

    result = robot_retarget.load_orientation_targets_for_retargeting(
        orientation_config=OrientationTrackingConfig(enable=True),
        robot_config=RobotConfig(robot_type="g1"),
        robot="g1",
        data_format="xsens",
        task_type="robot_only",
        xsens_motion=motion,
        hdf5_path=Path("recording.hdf5"),
        morphology_config=XsensMorphologyConfig(mode="direct"),
    )

    assert result is sentinel_targets
    assert captured["hdf5_path"] == Path("recording.hdf5")


def test_g1_mode_rejects_unsupported_task_or_robot() -> None:
    with pytest.raises(ValueError, match="robot_only"):
        robot_retarget.validate_xsens_morphology_selection(
            task_type="object_interaction",
            data_format="xsens",
            robot="g1",
            config=XsensMorphologyConfig(),
        )
    with pytest.raises(ValueError, match="robot='g1'"):
        robot_retarget.validate_xsens_morphology_selection(
            task_type="robot_only",
            data_format="xsens",
            robot="t1",
            config=XsensMorphologyConfig(),
        )

    robot_retarget.validate_xsens_morphology_selection(
        task_type="robot_only",
        data_format="xsens",
        robot="t1",
        config=XsensMorphologyConfig(
            mode="direct",
            root_motion=XsensRootMotionConfig(mode="preserve_world"),
        ),
    )


def test_direct_mode_rejects_scaled_root_motion() -> None:
    with pytest.raises(ValueError, match="g1_proportioned"):
        robot_retarget.validate_xsens_morphology_selection(
            task_type="robot_only",
            data_format="xsens",
            robot="g1",
            config=XsensMorphologyConfig(
                mode="direct",
                root_motion=XsensRootMotionConfig(mode="scale_by_leg_length"),
            ),
        )


def test_tyro_parses_nested_root_motion_configuration() -> None:
    config = tyro.cli(
        RetargetingConfig,
        args=[
            "--xsens-morphology.root-motion.mode",
            "scale_by_leg_length_contact_aware",
            "--xsens-morphology.root-motion.ground-height-m",
            "0.12",
            "--xsens-morphology.root-motion.contact-height-tolerance-m",
            "0.04",
        ],
    )

    assert config.xsens_morphology.root_motion.mode == "scale_by_leg_length_contact_aware"
    assert config.xsens_morphology.root_motion.ground_height_m == pytest.approx(0.12)
    assert config.xsens_morphology.root_motion.contact_height_tolerance_m == pytest.approx(0.04)
