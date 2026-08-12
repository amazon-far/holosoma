"""Tests for Xsens HDF5 data utilities."""

from __future__ import annotations

from collections import OrderedDict

import h5py
import numpy as np
import pytest
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XSENS_BODY_SEGMENT_NAMES,
    XSENS_Y_UP_TO_RETARGETING_Z_UP_MATRIX,
    inspect_xsens_hdf5,
    load_xsens_hdf5_calibration,
    load_xsens_hdf5_motion,
    load_xsens_hdf5_tpose,
    resolve_xsens_hdf5_path,
)
from holosoma_retargeting.xsens.orientation_tracking import quat_wijk_to_matrix
from scipy.spatial.transform import Rotation


def _headings(segment_names: list[str]) -> list[str]:
    return [f"{segment_name} ({axis})" for segment_name in segment_names for axis in ("x", "y", "z")]


def _write_stream(
    hdf5_file: h5py.File,
    stream_name: str,
    data: np.ndarray,
    times_s: np.ndarray,
    segment_names: list[str] | None = None,
) -> None:
    stream_group = hdf5_file.require_group("xsens-segments").create_group(stream_name)
    stream_group.create_dataset("data", data=data)
    stream_group.create_dataset("time_s", data=times_s.reshape(-1, 1))
    if segment_names is not None:
        stream_group.attrs["Data headings"] = str(_headings(segment_names))


def _write_identity_orientations(
    hdf5_file: h5py.File,
    times_s: np.ndarray,
    segment_names: list[str] | None = None,
) -> None:
    n_segments = len(XSENS_BODY_SEGMENT_NAMES) if segment_names is None else len(segment_names)
    quaternions_wijk = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(times_s), n_segments, 1))
    _write_stream(
        hdf5_file,
        "body_orientation_quaternion_wijk",
        quaternions_wijk,
        times_s,
        segment_names,
    )


def test_motion_data_config_registers_xsens_for_g1() -> None:
    cfg = MotionDataConfig(data_format="xsens", robot_type="g1")

    assert cfg.resolved_demo_joints == XSENS_BODY_SEGMENT_NAMES
    assert cfg.toe_names == ["Left Toe", "Right Toe"]
    assert cfg.default_human_height == 1.78
    assert cfg.target_fps == 30.0
    assert cfg.frame_start == 0
    assert cfg.max_frames is None
    assert cfg.resolved_joints_mapping["Pelvis"] == "pelvis_contour_link"
    assert cfg.resolved_joints_mapping["Right Hand"] == "right_wrist_yaw_link"


def test_loader_prefers_meter_stream_and_uses_headings_to_ignore_extra_segments(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.hdf5"
    times_s = np.arange(5, dtype=float) / 60.0
    segment_names = XSENS_BODY_SEGMENT_NAMES[:11] + ["Prop1"] + XSENS_BODY_SEGMENT_NAMES[11:]
    positions_m = np.zeros((5, len(segment_names), 3), dtype=float)
    for frame_idx in range(positions_m.shape[0]):
        for segment_idx in range(positions_m.shape[1]):
            positions_m[frame_idx, segment_idx] = [segment_idx, 100 + frame_idx, 200 + segment_idx]

    ignored_positions_cm = np.full((5, len(XSENS_BODY_SEGMENT_NAMES), 3), 999.0, dtype=float)

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "position_cm", ignored_positions_cm, times_s, XSENS_BODY_SEGMENT_NAMES)
        _write_stream(hdf5_file, "body_position_xyz_m", positions_m, times_s, segment_names)
        _write_identity_orientations(hdf5_file, times_s, segment_names)

    motion = load_xsens_hdf5_motion(hdf5_path, target_fps=30.0)

    assert motion.stream_name == "body_position_xyz_m"
    assert motion.positions_m.shape == (3, len(XSENS_BODY_SEGMENT_NAMES), 3)
    assert motion.quaternions_wijk.shape == (3, len(XSENS_BODY_SEGMENT_NAMES), 4)
    assert motion.orientation_stream_name == "body_orientation_quaternion_wijk"
    assert motion.times_s.tolist() == [times_s[0], times_s[2], times_s[4]]
    assert motion.segment_names == XSENS_BODY_SEGMENT_NAMES

    left_shoulder_source_idx = segment_names.index("Left Shoulder")
    left_shoulder_body_idx = XSENS_BODY_SEGMENT_NAMES.index("Left Shoulder")
    expected = positions_m[2, left_shoulder_source_idx]
    np.testing.assert_allclose(motion.positions_m[1, left_shoulder_body_idx], expected)


def test_loader_uses_segment_names_body_metadata_and_drops_sword_segment(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.hdf5"
    times_s = np.arange(3, dtype=float) / 30.0
    metadata_segment_names = [name.replace(" ", "") for name in XSENS_BODY_SEGMENT_NAMES] + ["RightHandSword"]
    positions_m = np.zeros((3, len(metadata_segment_names), 3), dtype=float)
    left_toe_source_idx = metadata_segment_names.index("LeftToe")
    positions_m[0, left_toe_source_idx] = [1.0, 2.0, 3.0]
    positions_m[0, -1] = [999.0, 999.0, 999.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        stream_group = hdf5_file.require_group("xsens-segments").create_group("body_position_xyz_m")
        stream_group.create_dataset("data", data=positions_m)
        stream_group.create_dataset("time_s", data=times_s.reshape(-1, 1))
        stream_group.attrs["segment_names_body"] = str(metadata_segment_names)
        _write_identity_orientations(hdf5_file, times_s, metadata_segment_names)

    motion = load_xsens_hdf5_motion(hdf5_path, target_fps=30.0)

    left_toe_body_idx = XSENS_BODY_SEGMENT_NAMES.index("Left Toe")
    assert motion.positions_m.shape == (3, len(XSENS_BODY_SEGMENT_NAMES), 3)
    assert motion.source_indices[-1] == left_toe_source_idx
    np.testing.assert_allclose(motion.positions_m[0, left_toe_body_idx], [1.0, 2.0, 3.0])
    assert not np.any(motion.positions_m == 999.0)


def test_loader_can_include_tracked_racket_pose_without_changing_default(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_racket.hdf5"
    times_s = np.arange(2, dtype=float) / 60.0
    segment_names = [name.replace(" ", "") for name in XSENS_BODY_SEGMENT_NAMES] + ["RightHandSword"]
    positions_m = np.zeros((2, len(segment_names), 3), dtype=float)
    positions_m[:, -1] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "body_position_xyz_m", positions_m, times_s, segment_names)
        _write_identity_orientations(hdf5_file, times_s, segment_names)

    body_only = load_xsens_hdf5_motion(hdf5_path, target_fps=None)
    with_racket = load_xsens_hdf5_motion(
        hdf5_path,
        target_fps=None,
        include_tracked_props=True,
    )

    assert body_only.positions_m.shape[1] == 23
    assert with_racket.positions_m.shape[1] == 24
    assert with_racket.segment_names[-1] == "RightHandSword"
    assert with_racket.source_indices[-1] == 23
    np.testing.assert_allclose(with_racket.positions_m[:, -1], positions_m[:, -1])


def test_loader_falls_back_to_cm_stream_without_headings(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.h5"
    times_s = np.arange(3, dtype=float) / 30.0
    positions_cm = np.zeros((3, len(XSENS_BODY_SEGMENT_NAMES), 3), dtype=float)
    positions_cm[0, 0] = [100.0, 200.0, 300.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "position_cm", positions_cm, times_s)
        _write_identity_orientations(hdf5_file, times_s)

    motion = load_xsens_hdf5_motion(hdf5_path, target_fps=30.0)

    assert motion.stream_name == "position_cm"
    assert motion.source_indices == list(range(len(XSENS_BODY_SEGMENT_NAMES)))
    np.testing.assert_allclose(motion.positions_m[0, 0], [1.0, 3.0, 2.0])


def test_loader_applies_frame_window_after_time_sampling(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.hdf5"
    times_s = np.arange(7, dtype=float) / 60.0
    positions_m = np.zeros((7, len(XSENS_BODY_SEGMENT_NAMES), 3), dtype=float)
    for frame_idx in range(positions_m.shape[0]):
        positions_m[frame_idx, 0] = [frame_idx, 0.0, 0.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "body_position_xyz_m", positions_m, times_s, XSENS_BODY_SEGMENT_NAMES)
        _write_identity_orientations(hdf5_file, times_s, XSENS_BODY_SEGMENT_NAMES)

    motion = load_xsens_hdf5_motion(hdf5_path, target_fps=30.0, frame_start=1, max_frames=2)

    assert motion.times_s.tolist() == [times_s[2], times_s[4]]
    np.testing.assert_allclose(motion.positions_m[:, 0, 0], [2.0, 4.0])


def test_loader_selects_sparse_post_resampling_frames_as_uniform_storyboard(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.hdf5"
    times_s = np.arange(9, dtype=float) / 60.0
    positions_m = np.zeros((9, len(XSENS_BODY_SEGMENT_NAMES), 3), dtype=float)
    positions_m[:, 0, 0] = np.arange(9)

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "body_position_xyz_m", positions_m, times_s, XSENS_BODY_SEGMENT_NAMES)
        _write_identity_orientations(hdf5_file, times_s, XSENS_BODY_SEGMENT_NAMES)

    motion = load_xsens_hdf5_motion(
        hdf5_path,
        target_fps=30.0,
        frame_indices=(0, 2, 4),
    )

    np.testing.assert_allclose(motion.times_s, np.arange(3) / 30.0)
    np.testing.assert_allclose(motion.positions_m[:, 0, 0], [0.0, 4.0, 8.0])


def test_sparse_frames_reject_windows_duplicates_and_out_of_range_indices(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.hdf5"
    times_s = np.arange(3, dtype=float) / 30.0
    positions_m = np.zeros((3, len(XSENS_BODY_SEGMENT_NAMES), 3), dtype=float)
    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "body_position_xyz_m", positions_m, times_s, XSENS_BODY_SEGMENT_NAMES)
        _write_identity_orientations(hdf5_file, times_s, XSENS_BODY_SEGMENT_NAMES)

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_xsens_hdf5_motion(hdf5_path, frame_start=1, frame_indices=(0,))
    with pytest.raises(ValueError, match="duplicates"):
        load_xsens_hdf5_motion(hdf5_path, frame_indices=(0, 0))
    with pytest.raises(ValueError, match="post-resampling range"):
        load_xsens_hdf5_motion(hdf5_path, frame_indices=(3,))


def test_motion_loader_reads_sampled_dynamic_orientations_and_ignores_extra_segment(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_orientation_sample.hdf5"
    times_s = np.arange(5, dtype=float) / 60.0
    segment_names = XSENS_BODY_SEGMENT_NAMES[:10] + ["RightHandSword"] + XSENS_BODY_SEGMENT_NAMES[10:]
    positions_m = np.zeros((5, len(segment_names), 3), dtype=float)
    quaternions_wijk = np.tile(np.array([2.0, 0.0, 0.0, 0.0]), (5, len(segment_names), 1))
    left_hand_source_idx = segment_names.index("Left Hand")
    quaternions_wijk[2, left_hand_source_idx] = [0.0, 0.0, 0.0, 3.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        position_group = hdf5_file.require_group("xsens-segments").create_group("body_position_xyz_m")
        position_group.create_dataset("data", data=positions_m)
        position_group.create_dataset("time_s", data=times_s.reshape(-1, 1))
        position_group.attrs["segment_names_body"] = str(segment_names)

        orientation_group = hdf5_file["xsens-segments"].create_group("body_orientation_quaternion_wijk")
        orientation_group.create_dataset("data", data=quaternions_wijk)
        orientation_group.create_dataset("time_s", data=times_s.reshape(-1, 1))
        orientation_group.attrs["segment_names_body"] = str(segment_names)

    motion = load_xsens_hdf5_motion(hdf5_path, target_fps=30.0)

    assert motion.orientation_stream_name == "body_orientation_quaternion_wijk"
    assert motion.quaternions_wijk.shape == (3, len(XSENS_BODY_SEGMENT_NAMES), 4)
    left_hand_body_idx = XSENS_BODY_SEGMENT_NAMES.index("Left Hand")
    np.testing.assert_allclose(motion.quaternions_wijk[1, left_hand_body_idx], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(np.linalg.norm(motion.quaternions_wijk, axis=-1), 1.0)


def test_motion_loader_converts_position_cm_flat_orientations_to_retargeting_frame(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_position_cm_orientation_sample.hdf5"
    times_s = np.array([0.0], dtype=float)
    positions_cm = np.zeros((1, len(XSENS_BODY_SEGMENT_NAMES), 3), dtype=float)
    positions_cm[0, 0] = [100.0, 200.0, 300.0]

    rot_xsens = Rotation.from_euler("y", 90.0, degrees=True).as_matrix()
    quat_xyzw = Rotation.from_matrix(rot_xsens).as_quat()
    quat_wijk = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
    quaternions_wijk = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (1, len(XSENS_BODY_SEGMENT_NAMES), 1))
    quaternions_wijk[0, 0] = quat_wijk

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "position_cm", positions_cm, times_s)
        orientation_group = hdf5_file["xsens-segments"].create_group("body_orientation_quaternion_wijk")
        orientation_group.create_dataset("data", data=quaternions_wijk.reshape(1, -1))
        orientation_group.create_dataset("time_s", data=times_s.reshape(-1, 1))

    motion = load_xsens_hdf5_motion(hdf5_path, target_fps=30.0)

    np.testing.assert_allclose(motion.positions_m[0, 0], [1.0, 3.0, 2.0])
    expected_rotation = XSENS_Y_UP_TO_RETARGETING_Z_UP_MATRIX @ rot_xsens @ XSENS_Y_UP_TO_RETARGETING_Z_UP_MATRIX.T
    actual_rotation = quat_wijk_to_matrix(motion.quaternions_wijk[0, 0])
    np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-12)


def test_motion_loader_requires_orientations(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_no_orientation.hdf5"
    times_s = np.arange(3, dtype=float) / 30.0
    positions_m = np.zeros((3, len(XSENS_BODY_SEGMENT_NAMES), 3), dtype=float)

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "body_position_xyz_m", positions_m, times_s, XSENS_BODY_SEGMENT_NAMES)

    with pytest.raises(KeyError, match="body_orientation_quaternion_wijk"):
        load_xsens_hdf5_motion(hdf5_path, target_fps=30.0)


def test_resolve_xsens_hdf5_path_accepts_task_stems_and_explicit_files(tmp_path) -> None:
    hdf5_path = tmp_path / "tennis_motion.hdf5"
    hdf5_path.touch()

    assert resolve_xsens_hdf5_path(tmp_path, "tennis_motion") == hdf5_path
    assert resolve_xsens_hdf5_path(tmp_path, "tennis_motion.hdf5") == hdf5_path


def test_tpose_loader_selects_body_segments_and_ignores_extra_segment(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_tpose.hdf5"
    segment_names = XSENS_BODY_SEGMENT_NAMES[:5] + ["RightHandSword"] + XSENS_BODY_SEGMENT_NAMES[5:]
    positions_m = np.zeros((len(segment_names), 3), dtype=float)
    quaternions_wijk = np.tile(np.array([2.0, 0.0, 0.0, 0.0]), (len(segment_names), 1))
    head_source_idx = segment_names.index("Head")
    positions_m[head_source_idx] = [1.0, 2.0, 3.0]
    positions_m[segment_names.index("RightHandSword")] = [999.0, 999.0, 999.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        group = hdf5_file.create_group("xsens-segments-tpose")
        group.attrs["segment_names_body"] = str(segment_names)
        group.create_dataset("body_position_Tpose_xyz_m", data=positions_m)
        group.create_dataset("body_orientation_Tpose_quaternion_wijk", data=quaternions_wijk)

    tpose = load_xsens_hdf5_tpose(hdf5_path)

    assert tpose.variant == "Tpose"
    assert tpose.positions_m.shape == (len(XSENS_BODY_SEGMENT_NAMES), 3)
    assert tpose.quaternions_wijk.shape == (len(XSENS_BODY_SEGMENT_NAMES), 4)
    assert tpose.segment_names == XSENS_BODY_SEGMENT_NAMES
    assert tpose.source_indices[XSENS_BODY_SEGMENT_NAMES.index("Head")] == head_source_idx
    np.testing.assert_allclose(tpose.positions_m[XSENS_BODY_SEGMENT_NAMES.index("Head")], [1.0, 2.0, 3.0])
    assert not np.any(tpose.positions_m == 999.0)
    np.testing.assert_allclose(np.linalg.norm(tpose.quaternions_wijk, axis=1), 1.0)


def test_tpose_loader_errors_on_missing_variant(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_tpose.hdf5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_group("xsens-segments-tpose")

    with pytest.raises(KeyError, match="body_position_Tpose_xyz_m"):
        load_xsens_hdf5_tpose(hdf5_path, variant="Tpose")


def test_calibration_loader_preserves_complete_source_schema(tmp_path) -> None:
    path = tmp_path / "calibration.hdf5"
    segment_names = ["Pelvis", "RightHandSword"]
    joint_names = ["RightHandSwordOrigin"]
    landmarks = OrderedDict(
        [
            ("Pelvis", OrderedDict([("pHipOrigin", [0.0, 0.0, 0.0])])),
            ("RightHandSword", OrderedDict()),
        ]
    )
    positions = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 1.0]])
    quaternions = np.tile(np.array([2.0, 0.0, 0.0, 0.0]), (2, 1))

    with h5py.File(path, "w") as hdf5_file:
        stream = hdf5_file.require_group("xsens-segments").create_group("body_position_xyz_m")
        stream.attrs["segment_names_body"] = repr(segment_names)
        stream.attrs["segment_mesh_points_body_xyz_cm"] = repr(landmarks)
        stream.attrs["mvn_version"] = "test-mvn"
        stream.attrs["mvnx_version"] = "4"
        poses = hdf5_file.create_group("xsens-segments-tpose")
        for variant in ("Tpose", "TposeISB", "identity"):
            poses.create_dataset(f"body_position_{variant}_xyz_m", data=positions)
            poses.create_dataset(f"body_orientation_{variant}_quaternion_wijk", data=quaternions)
        joints = hdf5_file.create_group("xsens-joints")
        for stream_name in (
            "body_joint_angles_eulerZXY_xyz_rad",
            "body_joint_angles_eulerXZY_xyz_rad",
        ):
            joint_stream = joints.create_group(stream_name)
            joint_stream.attrs["joint_names_body"] = repr(joint_names)
            joint_stream.attrs["joint_rotation_order_body"] = repr(
                [("RightHandSwordOrigin", ("x", "y", "z"))]
            )

    inventory = inspect_xsens_hdf5(path)
    calibration = load_xsens_hdf5_calibration(path)

    assert inventory.segment_names == tuple(segment_names)
    assert inventory.joint_names == tuple(joint_names)
    assert inventory.pose_variants == ("Tpose", "TposeISB", "identity")
    assert inventory.has_landmarks
    assert calibration.segment_names[-1] == "RightHandSword"
    assert calibration.joint_names[-1] == "RightHandSwordOrigin"
    assert calibration.tpose_isb is not None
    assert calibration.identity_pose is not None
    np.testing.assert_allclose(np.linalg.norm(calibration.tpose.quaternions_wijk, axis=1), 1.0)
    assert calibration.joint_rotation_metadata["RightHandSwordOrigin"].components == ("x", "y", "z")
