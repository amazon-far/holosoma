"""Tests for the Xsens hand/racket relative-motion analysis."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from holosoma_retargeting.examples.xsens_tennis.analyze_hand_racket_motion import (
    Config,
    angular_speed,
    compute_orientation_residual,
    compute_relative_pose,
    infer_analysis_windows,
    rotations_from_wxyz,
    run_analysis,
)
from scipy.spatial.transform import Rotation


def _identity_rotations(count: int) -> Rotation:
    return Rotation.from_quat(np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1)))


def test_constant_rigid_transform_has_zero_relative_pose_residual() -> None:
    frame_count = 8
    hand_positions = np.column_stack(
        [np.linspace(0.0, 1.0, frame_count), np.zeros(frame_count), np.zeros(frame_count)]
    )
    hand_rotations = Rotation.from_euler("z", np.linspace(0.0, 90.0, frame_count)[:, None], degrees=True)
    relative_translation = np.array([0.02, -0.09, -0.01])
    relative_rotation = Rotation.from_euler("x", -90.0, degrees=True)
    racket_positions = hand_positions + hand_rotations.apply(np.tile(relative_translation, (frame_count, 1)))
    racket_rotations = hand_rotations * relative_rotation

    relative_pose = compute_relative_pose(
        hand_positions,
        racket_positions,
        hand_rotations,
        racket_rotations,
    )
    residual = compute_orientation_residual(relative_pose.rotations, relative_rotation)

    np.testing.assert_allclose(
        relative_pose.translations_m,
        np.tile(relative_translation, (frame_count, 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(residual.geodesic_deg, 0.0, atol=1e-12)
    np.testing.assert_allclose(residual.longitudinal_axis_misalignment_deg, 0.0, atol=1e-12)
    np.testing.assert_allclose(residual.twist_deg, 0.0, atol=1e-12)


def test_twist_and_longitudinal_axis_misalignment_have_expected_geometry() -> None:
    identity = Rotation.identity()
    twist = compute_orientation_residual(Rotation.from_euler("x", [[30.0]], degrees=True), identity)
    tilt = compute_orientation_residual(Rotation.from_euler("y", [[20.0]], degrees=True), identity)

    np.testing.assert_allclose(twist.geodesic_deg, [30.0], atol=1e-10)
    np.testing.assert_allclose(twist.longitudinal_axis_misalignment_deg, [0.0], atol=1e-10)
    np.testing.assert_allclose(twist.twist_deg, [30.0], atol=1e-10)
    np.testing.assert_allclose(tilt.geodesic_deg, [20.0], atol=1e-10)
    np.testing.assert_allclose(tilt.longitudinal_axis_misalignment_deg, [20.0], atol=1e-10)
    np.testing.assert_allclose(tilt.twist_deg, [0.0], atol=1e-10)


def test_quaternion_sign_flips_do_not_change_orientation_residual() -> None:
    angle_rad = np.deg2rad(42.0)
    quaternion_wxyz = np.array([np.cos(angle_rad / 2.0), 0.0, 0.0, np.sin(angle_rad / 2.0)])
    rotations = rotations_from_wxyz(np.stack([quaternion_wxyz, -quaternion_wxyz]))
    baseline = rotations_from_wxyz(quaternion_wxyz)
    residual = compute_orientation_residual(rotations, baseline)

    np.testing.assert_allclose(residual.geodesic_deg, 0.0, atol=1e-12)
    np.testing.assert_allclose(residual.rotvec_deg, 0.0, atol=1e-12)


def test_angular_speed_respects_irregular_timestamps() -> None:
    times_s = np.array([0.0, 0.1, 0.4])
    rotations = Rotation.from_euler("z", [[0.0], [0.1], [0.4]])

    np.testing.assert_allclose(angular_speed(rotations, times_s), 1.0, atol=1e-12)


def test_calibration_window_inference_uses_first_good_tpose_and_next_calibration() -> None:
    stream_start_s = 1000.0
    relative_times_s = [5.0, 8.0, 10.0, 20.0, 100.0, 110.0]
    rows = [
        ["Start", "Bad", "", "", "", "T-Pose"],
        ["Stop", "Bad", "", "", "", "T-Pose"],
        ["Start", "Good", "", "", "", "T-Pose"],
        ["Stop", "Good", "", "", "", "T-Pose"],
        ["Start", "Good", "", "", "", "N-Pose"],
        ["Stop", "Good", "", "", "", "N-Pose"],
    ]

    windows, baseline, activity = infer_analysis_windows(
        [stream_start_s + value for value in relative_times_s],
        rows,
        stream_start_s,
    )

    assert len(windows) == 2
    assert baseline.start_s == pytest.approx(10.0)
    assert baseline.end_s == pytest.approx(20.0)
    assert activity.start_s == pytest.approx(20.0)
    assert activity.end_s == pytest.approx(100.0)


def test_real_xsens_analysis_smoke(tmp_path: Path) -> None:
    """Opt-in local-data smoke test including all figures and a tiny MP4."""

    if os.environ.get("HOLOSOMA_RUN_LOCAL_DATA_TESTS") != "1":
        pytest.skip("Set HOLOSOMA_RUN_LOCAL_DATA_TESTS=1 to run the 1.5 GB local-data smoke test")
    project_root = Path(__file__).resolve().parents[2]
    hdf5_path = (
        project_root
        / "holosoma_retargeting/demo_data/xsens_tennis/2026-07-17_15-12-57_streamLog_tennis_S00.hdf5"
    )
    if not hdf5_path.exists():
        pytest.skip(f"Local Xsens recording is unavailable: {hdf5_path}")

    summary = run_analysis(
        Config(
            hdf5_path=hdf5_path,
            output_dir=tmp_path,
            animation_fps=5.0,
            animation_clip_duration_s=0.4,
        )
    )

    assert summary["sample_count"] > 100_000
    assert summary["sensor_validation"]["available"] is True
    assert np.isfinite(summary["activity_metrics"]["orientation_error_observed_baseline_deg"]["median"])
    for filename in (
        "analysis_report.md",
        "summary.json",
        "frame_metrics.csv",
        "orientation_events.csv",
        "orientation_overview.png",
        "orientation_overview.pdf",
        "relative_motion_diagnostics.png",
        "phase_distributions.png",
        "sensor_crosscheck.png",
        "orientation_keyframes.png",
        "relative_orientation_diagnostic.mp4",
    ):
        assert (tmp_path / filename).stat().st_size > 0
