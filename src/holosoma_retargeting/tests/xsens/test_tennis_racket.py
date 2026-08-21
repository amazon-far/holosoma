from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from holosoma_retargeting.config_types.retargeter import TennisRacketTrackingConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XSENS_BODY_SEGMENT_NAMES,
    XsensHdf5Motion,
)
from holosoma_retargeting.xsens.tennis_racket import (
    TENNIS_RACKET_RESULT_SCHEMA_VERSION,
    TennisRacketMotion,
    attachment_handle_intersects_palm,
    build_tennis_racket_targets,
    choose_tennis_racket_symmetry_branch,
    decide_filtered_tennis_racket_tracking,
    load_retargeting_result,
    load_tennis_racket_attachment,
    resolve_tennis_racket_attachment,
    save_tennis_racket_attachment,
    tennis_racket_target_error_rad,
)
from scipy.spatial.transform import Rotation


def _wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = np.asarray(rotation.as_quat())
    return xyzw[..., [3, 0, 1, 2]]


def _motion(frame_count: int = 6) -> XsensHdf5Motion:
    attachment = load_tennis_racket_attachment()
    names = [*XSENS_BODY_SEGMENT_NAMES, "RightHandSword"]
    positions = np.zeros((frame_count, len(names), 3))
    sword_index = names.index("RightHandSword")
    positions[:, sword_index] = attachment.source_reference_position_m
    quaternions = np.zeros((frame_count, len(names), 4))
    quaternions[..., 0] = 1.0
    return XsensHdf5Motion(
        positions_m=positions,
        times_s=np.arange(frame_count, dtype=float) / 30.0,
        stream_name="body_position_xyz_m",
        segment_names=names,
        source_indices=list(range(len(names))),
        quaternions_wijk=quaternions,
        orientation_stream_name="body_orientation_quaternion_wijk",
    )


def test_build_targets_has_exact_180_degree_racket_symmetry() -> None:
    attachment = load_tennis_racket_attachment()
    targets = build_tennis_racket_targets(_motion(), attachment)
    relative = Rotation.from_matrix(targets.candidate_racket_rotations[0, 0]).inv() * Rotation.from_matrix(
        targets.candidate_racket_rotations[0, 1]
    )
    np.testing.assert_allclose(relative.as_rotvec(), np.pi * attachment.longitudinal_axis_local, atol=1e-12)
    assert tennis_racket_target_error_rad(
        targets.candidate_racket_rotations[0, 1], targets.candidate_racket_rotations[0]
    ) == pytest.approx(0.0, abs=1e-12)


def test_branch_selection_tracks_nearest_realized_hand_and_prior_tie_break() -> None:
    targets = build_tennis_racket_targets(_motion(), load_tennis_racket_attachment())
    candidates = targets.candidate_hand_rotations[0]
    assert choose_tennis_racket_symmetry_branch(candidates[1], candidates) == 1
    first, second = Rotation.from_matrix(candidates)
    midpoint = first * (first.inv() * second) ** 0.5
    assert choose_tennis_racket_symmetry_branch(midpoint.as_matrix(), candidates, preferred_branch=1) == 1


def test_source_origin_deviation_detects_detached_prop() -> None:
    motion = _motion()
    sword_index = motion.segment_names.index("RightHandSword")
    motion.positions_m[-1, sword_index, 0] += 0.2
    targets = build_tennis_racket_targets(motion, load_tennis_racket_attachment())
    np.testing.assert_allclose(targets.source_origin_deviation_m[:-1], 0.0, atol=1e-12)
    assert targets.source_origin_deviation_m[-1] == pytest.approx(0.2)


def test_filtered_gate_uses_45_entry_60_exit_five_frames_and_wrist_margin() -> None:
    config = TennisRacketTrackingConfig()
    active = False
    streak = 0
    for expected_state in ["reentry_hysteresis"] * 4 + ["racket"]:
        decision = decide_filtered_tennis_racket_tracking(
            config,
            active=active,
            feasible_streak=streak,
            source_origin_deviation_m=0.01,
            solve_succeeded=True,
            target_error_rad=np.deg2rad(45.0),
            wrist_limit_margin_rad=np.deg2rad(6.0),
        )
        assert decision.state == expected_state
        active, streak = decision.active, decision.feasible_streak

    stays_active = decide_filtered_tennis_racket_tracking(
        config,
        active=True,
        feasible_streak=streak,
        source_origin_deviation_m=0.09,
        solve_succeeded=True,
        target_error_rad=np.deg2rad(60.0),
        wrist_limit_margin_rad=np.deg2rad(5.0),
    )
    assert stays_active.use_racket
    assert (
        decide_filtered_tennis_racket_tracking(
            config,
            active=True,
            feasible_streak=streak,
            source_origin_deviation_m=0.09,
            solve_succeeded=True,
            target_error_rad=np.deg2rad(60.1),
            wrist_limit_margin_rad=np.deg2rad(6.0),
        ).state
        == "infeasible"
    )
    assert (
        decide_filtered_tennis_racket_tracking(
            config,
            active=True,
            feasible_streak=streak,
            source_origin_deviation_m=0.01,
            solve_succeeded=True,
            target_error_rad=0.0,
            wrist_limit_margin_rad=np.deg2rad(4.9),
        ).state
        == "wrist_limit"
    )
    assert (
        decide_filtered_tennis_racket_tracking(
            config,
            active=True,
            feasible_streak=streak,
            source_origin_deviation_m=0.101,
            solve_succeeded=True,
            target_error_rad=0.0,
            wrist_limit_margin_rad=np.deg2rad(10.0),
        ).state
        == "detached"
    )


def test_global_and_observed_window_attachment_calibration() -> None:
    motion = _motion()
    global_attachment = resolve_tennis_racket_attachment(
        TennisRacketTrackingConfig(attachment_source="global"), motion=motion, hdf5_path="unused.hdf5"
    )
    assert global_attachment.calibration_source == "global"
    assert attachment_handle_intersects_palm(global_attachment)

    sword_index = motion.segment_names.index("RightHandSword")
    motion.positions_m[1:4, sword_index] += np.array([0.01, -0.02, 0.03])
    motion.quaternions_wijk[1:4, sword_index] = _wxyz(Rotation.from_euler("z", 12.0, degrees=True))
    observed = resolve_tennis_racket_attachment(
        TennisRacketTrackingConfig(
            attachment_source="observed_window",
            observed_window_s=(1.0 / 30.0, 4.0 / 30.0),
        ),
        motion=motion,
        hdf5_path="unused.hdf5",
    )
    assert observed.calibration_source == "observed_window"
    np.testing.assert_allclose(
        observed.source_reference_position_m,
        global_attachment.source_reference_position_m + np.array([0.01, -0.02, 0.03]),
    )

    # A retargeted clip may start long after the recording begins. Calibration
    # windows remain recording-relative instead of silently becoming clip-relative.
    motion = replace(
        motion,
        recording_times_s=np.arange(motion.positions_m.shape[0], dtype=float) / 30.0 + 10.0,
    )
    with pytest.raises(ValueError, match="fewer than two"):
        resolve_tennis_racket_attachment(
            TennisRacketTrackingConfig(
                attachment_source="observed_window",
                observed_window_s=(1.0 / 30.0, 4.0 / 30.0),
            ),
            motion=motion,
            hdf5_path="unused.hdf5",
        )
    with pytest.raises(ValueError, match="fewer than two"):
        resolve_tennis_racket_attachment(
            TennisRacketTrackingConfig(attachment_source="observed_window", observed_window_s=(0.0, 0.01)),
            motion=motion,
            hdf5_path="unused.hdf5",
        )


def test_embedded_tpose_calibration_and_attachment_override(monkeypatch, tmp_path) -> None:
    motion = _motion()
    calibration = SimpleNamespace(
        segment_names=tuple(motion.segment_names),
        tpose=SimpleNamespace(
            positions_m=motion.positions_m[0],
            quaternions_wijk=motion.quaternions_wijk[0],
        ),
    )
    monkeypatch.setattr(
        "holosoma_retargeting.xsens.tennis_racket.load_xsens_hdf5_calibration",
        lambda _path: calibration,
    )
    embedded = resolve_tennis_racket_attachment(
        TennisRacketTrackingConfig(),
        motion=motion,
        hdf5_path="recording.hdf5",
    )
    assert TennisRacketTrackingConfig().attachment_source == "embedded_tpose"
    assert embedded.calibration_source == "embedded_tpose"

    override_path = tmp_path / "attachment.json"
    save_tennis_racket_attachment(embedded, override_path)
    loaded = load_tennis_racket_attachment(override_path)
    np.testing.assert_allclose(loaded.position_m, embedded.position_m)
    np.testing.assert_allclose(loaded.quaternion_wxyz, embedded.quaternion_wxyz)


def test_retargeting_result_round_trip_and_legacy_compatibility(tmp_path) -> None:
    attachment = load_tennis_racket_attachment()
    frame_count = 3
    motion = TennisRacketMotion(
        position_m=np.zeros((frame_count, 3)),
        quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (frame_count, 1)),
        tracking_state=np.array(["hand", "racket", "infeasible"]),
        symmetry_branch=np.array([-1, 0, -1]),
        target_error_rad=np.arange(frame_count, dtype=float),
        source_origin_deviation_m=np.zeros(frame_count),
        min_wrist_limit_margin_rad=np.ones(frame_count),
        attachment=attachment,
        tracking_mode="filtered",
    )
    result_path = tmp_path / "result.npz"
    np.savez(result_path, qpos=np.zeros((frame_count, 36)), fps=30, **motion.as_npz_payload())
    loaded = load_retargeting_result(result_path)
    assert loaded.tennis_racket is not None
    assert loaded.tennis_racket.attachment.schema_version == TENNIS_RACKET_RESULT_SCHEMA_VERSION
    np.testing.assert_allclose(np.linalg.norm(loaded.tennis_racket.quaternion_wxyz, axis=1), 1.0)

    legacy_path = tmp_path / "legacy.npz"
    np.savez(legacy_path, qpos=np.zeros((2, 36)), fps=30)
    legacy = load_retargeting_result(legacy_path)
    assert legacy.tennis_racket is None
