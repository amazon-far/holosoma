from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from holosoma_retargeting.examples.xsens_tennis import analyze_xsens_g1_retargeting as analysis
from scipy.spatial.transform import Rotation


def _input_tree(tmp_path: Path, stem: str = "serve_01") -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data" / "xsens_tennis"
    results_dir = tmp_path / "results"
    output_root = tmp_path / "analysis"
    data_dir.mkdir(parents=True)
    results_dir.mkdir()
    (data_dir / f"{stem}.hdf5").touch()
    (results_dir / f"{stem}.npz").touch()
    return data_dir, results_dir, output_root


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("serve_01", "serve_01"),
        ("serve_01.hdf5", "serve_01"),
        ("serve_01.H5", "serve_01"),
    ],
)
def test_sequence_name_normalization(value: str, expected: str) -> None:
    assert analysis.normalize_sequence_name(value) == expected


@pytest.mark.parametrize("value", ["", "nested/serve_01", ".hdf5"])
def test_sequence_name_normalization_rejects_invalid_tokens(value: str) -> None:
    with pytest.raises(ValueError, match="Sequence name"):
        analysis.normalize_sequence_name(value)


def test_canonical_paths_and_output_layout_are_inferred(tmp_path: Path) -> None:
    data_dir, results_dir, output_root = _input_tree(tmp_path)
    paths = analysis.resolve_sequence_paths(
        analysis.Config(
            sequence_names=("serve_01.hdf5",),
            data_dir=data_dir,
            retargeted_results_dir=results_dir,
            output_root=output_root,
        )
    )
    assert paths == (
        analysis.SequencePaths(
            "serve_01",
            (data_dir / "serve_01.hdf5").resolve(),
            (results_dir / "serve_01.npz").resolve(),
            (output_root / "serve_01").resolve(),
        ),
    )


def test_default_result_directory_uses_dataset_directory_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "input" / "custom_dataset"
    data_dir.mkdir(parents=True)
    (data_dir / "serve.h5").touch()
    demo_results = tmp_path / "demo_results"
    expected_results = demo_results / "g1" / "robot_only" / "custom_dataset"
    expected_results.mkdir(parents=True)
    (expected_results / "serve.npz").touch()
    monkeypatch.setattr(analysis, "DEMO_RESULTS_DIR", demo_results)

    (paths,) = analysis.resolve_sequence_paths(analysis.Config(sequence_names=("serve",), data_dir=data_dir))
    assert paths.qpos_npz == (expected_results / "serve.npz").resolve()
    assert paths.output_dir == (demo_results / "g1" / "analysis" / "custom_dataset" / "serve").resolve()


def test_exact_npz_matching_does_not_accept_similar_stem(tmp_path: Path) -> None:
    data_dir, results_dir, output_root = _input_tree(tmp_path)
    (results_dir / "serve_01.npz").unlink()
    (results_dir / "serve_01_staged.npz").touch()
    with pytest.raises(FileNotFoundError, match="Expected exact path"):
        analysis.resolve_sequence_paths(
            analysis.Config(
                sequence_names=("serve_01",),
                data_dir=data_dir,
                retargeted_results_dir=results_dir,
                output_root=output_root,
            )
        )


def test_missing_hdf5_lists_both_attempted_extensions(tmp_path: Path) -> None:
    data_dir, results_dir, output_root = _input_tree(tmp_path)
    (data_dir / "serve_01.hdf5").unlink()
    with pytest.raises(FileNotFoundError) as error:
        analysis.resolve_sequence_paths(
            analysis.Config(
                sequence_names=("serve_01",),
                data_dir=data_dir,
                retargeted_results_dir=results_dir,
                output_root=output_root,
            )
        )
    message = str(error.value)
    assert "serve_01.hdf5" in message
    assert "serve_01.h5" in message


def test_duplicate_hdf5_extensions_are_rejected(tmp_path: Path) -> None:
    data_dir, results_dir, output_root = _input_tree(tmp_path)
    (data_dir / "serve_01.h5").touch()
    with pytest.raises(ValueError, match=r"Both \.hdf5 and \.h5"):
        analysis.resolve_sequence_paths(
            analysis.Config(
                sequence_names=("serve_01",),
                data_dir=data_dir,
                retargeted_results_dir=results_dir,
                output_root=output_root,
            )
        )


def test_single_sequence_overrides_are_supported(tmp_path: Path) -> None:
    data_dir, results_dir, output_root = _input_tree(tmp_path)
    explicit_hdf5 = tmp_path / "explicit.h5"
    explicit_npz = tmp_path / "explicit.npz"
    explicit_hdf5.touch()
    explicit_npz.touch()
    (paths,) = analysis.resolve_sequence_paths(
        analysis.Config(
            sequence_names=("logical_name",),
            data_dir=data_dir,
            retargeted_results_dir=results_dir,
            output_root=output_root,
            hdf5_path=explicit_hdf5,
            qpos_npz=explicit_npz,
        )
    )
    assert paths.hdf5_path == explicit_hdf5.resolve()
    assert paths.qpos_npz == explicit_npz.resolve()
    assert paths.output_dir == (output_root / "logical_name").resolve()


@pytest.mark.parametrize("viser_mode", ["interactive", "record-clips"])
def test_multi_sequence_viser_is_rejected(tmp_path: Path, viser_mode: str) -> None:
    data_dir, results_dir, output_root = _input_tree(tmp_path, "one")
    (data_dir / "two.hdf5").touch()
    (results_dir / "two.npz").touch()
    with pytest.raises(ValueError, match="exactly one sequence"):
        analysis.resolve_sequence_paths(
            analysis.Config(
                sequence_names=("one", "two"),
                data_dir=data_dir,
                retargeted_results_dir=results_dir,
                output_root=output_root,
                viser_mode=viser_mode,
            )
        )


def test_multi_sequence_overrides_are_rejected(tmp_path: Path) -> None:
    data_dir, results_dir, output_root = _input_tree(tmp_path, "one")
    (data_dir / "two.hdf5").touch()
    (results_dir / "two.npz").touch()
    with pytest.raises(ValueError, match="overrides require exactly one"):
        analysis.resolve_sequence_paths(
            analysis.Config(
                sequence_names=("one", "two"),
                data_dir=data_dir,
                retargeted_results_dir=results_dir,
                output_root=output_root,
                hdf5_path=data_dir / "one.hdf5",
            )
        )


def test_cli_requires_sequence_names() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(analysis.__file__))],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "--sequence-names" in result.stderr
    assert "required" in result.stderr.casefold()


def test_timestamp_alignment_rejects_frame_and_time_mismatches() -> None:
    analysis.validate_aligned_timeline(
        {"qpos": 3, "motion": 3},
        np.array([0.0, 0.1, 0.2]),
        np.array([0.0, 0.1, 0.2]),
    )
    with pytest.raises(ValueError, match="refusing to trim"):
        analysis.validate_aligned_timeline(
            {"qpos": 3, "motion": 2},
            np.array([0.0, 0.1, 0.2]),
            np.array([0.0, 0.1, 0.2]),
        )
    with pytest.raises(ValueError, match="different timestamps"):
        analysis.validate_aligned_timeline(
            {"qpos": 3, "motion": 3},
            np.array([0.0, 0.1, 0.2]),
            np.array([0.0, 0.11, 0.2]),
        )


@pytest.mark.parametrize(
    ("source_stream", "expected_com"),
    [
        ("body_position_xyz_m", [1.0, 2.0, 3.0]),
        ("position_cm", [1.0, 3.0, 2.0]),
    ],
)
def test_auxiliary_com_uses_segment_stream_coordinate_convention(
    tmp_path: Path,
    source_stream: str,
    expected_com: list[float],
) -> None:
    path = tmp_path / "motion.hdf5"
    times = np.array([0.0, 0.1, 0.2])
    with h5py.File(path, "w") as handle:
        position = handle.create_group(f"xsens-segments/{source_stream}")
        position.create_dataset("time_s", data=times)
        com = handle.create_group("xsens-CoM/position_xyz_m")
        com.create_dataset("time_s", data=times)
        com.create_dataset("data", data=np.tile(np.array([1.0, 2.0, 3.0]), (3, 1)))
        contacts = handle.create_group("xsens-foot-contacts/is_contacting_ground")
        contacts.create_dataset("time_s", data=times)
        contacts.create_dataset("data", data=np.zeros((3, 2)))
        contacts.attrs["foot_contact_names"] = "['Left Heel', 'Right Heel']"

    _, com_values, _, contact_names, activities = analysis._load_auxiliary_xsens(
        path,
        fps=10.0,
        source_position_stream=source_stream,
    )
    np.testing.assert_allclose(com_values, np.tile(expected_com, (3, 1)))
    assert contact_names == ["Left Heel", "Right Heel"]
    assert activities == ()


def test_retargeted_npz_requires_qpos_human_joints_and_fps(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.npz"
    np.savez(missing_path, qpos=np.zeros((2, 36)), fps=30)
    with pytest.raises(KeyError, match="human_joints"):
        analysis.load_retargeted_npz(missing_path)

    valid_path = tmp_path / "valid.npz"
    np.savez(
        valid_path,
        qpos=np.zeros((2, 36)),
        human_joints=np.zeros((2, 23, 3)),
        fps=30,
    )
    qpos, joints, fps = analysis.load_retargeted_npz(valid_path)
    assert qpos.shape == (2, 36)
    assert joints.shape == (2, 23, 3)
    assert fps == 30.0


def test_retargeted_npz_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.npz"
    np.savez(
        path,
        qpos=np.zeros((2, 36)),
        human_joints=np.zeros((3, 23, 3)),
        fps=30,
    )
    with pytest.raises(ValueError, match="frame counts differ"):
        analysis.load_retargeted_npz(path)


def test_signed_support_margin_and_normalization() -> None:
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    inside, area = analysis.signed_polygon_margin(np.array([0.0, 0.0]), square)
    outside, _ = analysis.signed_polygon_margin(np.array([2.0, 0.0]), square)
    assert inside == pytest.approx(1.0)
    assert outside == pytest.approx(-1.0)
    assert area == pytest.approx(4.0)

    points = np.column_stack([square, np.zeros(4)])[None, :, :]
    margins, areas, normalized = analysis.support_metrics(
        np.array([[0.0, 0.0, 1.0]]),
        analysis.FootprintSeries(left=points, right=points),
        np.array([True]),
        np.array([False]),
    )
    assert margins == pytest.approx([1.0])
    assert areas == pytest.approx([4.0])
    assert normalized == pytest.approx([0.5])


def test_quaternion_orientation_metrics() -> None:
    reference = Rotation.from_euler("z", [0.0], degrees=True)
    target = Rotation.from_euler("z", [90.0], degrees=True)
    metrics = analysis.orientation_error_metrics(reference, target)
    assert metrics["geodesic_deg"] == pytest.approx([90.0])
    assert metrics["longitudinal_axis_deg"] == pytest.approx([90.0])
    assert metrics["face_normal_deg"] == pytest.approx([0.0])
    assert metrics["twist_deg"] == pytest.approx([0.0])


def test_world_and_root_relative_position_errors_differ() -> None:
    human_point = np.array([[1.0, 0.0, 0.0]])
    human_root = np.array([[0.0, 0.0, 0.0]])
    target_point = np.array([[11.0, 0.0, 0.0]])
    target_root = np.array([[10.0, 0.0, 0.0]])
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    world_error = np.linalg.norm(target_point - human_point, axis=1)
    human_relative = analysis.root_relative_positions(human_point, human_root, identity)
    target_relative = analysis.root_relative_positions(target_point, target_root, identity)
    assert world_error == pytest.approx([10.0])
    assert np.linalg.norm(target_relative - human_relative, axis=1) == pytest.approx([0.0])


def test_activity_label_parsing_is_inclusive() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    labels = analysis._activity_labels(
        times,
        (
            analysis.ActivityWindow("Forehand", 0.5, 1.0),
            analysis.ActivityWindow("Backhand", 2.0, 2.5),
        ),
    )
    assert labels.tolist() == ["unlabeled", "Forehand", "Backhand", "unlabeled"]


def test_clip_selection_is_non_overlapping_and_activity_aware() -> None:
    values = np.zeros(80)
    values[[10, 35, 60]] = [3.0, 2.0, 1.0]
    data = SimpleNamespace(
        fps=10.0,
        times_s=np.arange(80) / 10.0,
        activity_labels=np.array(["Forehand"] * 40 + ["Backhand"] * 40),
        metrics={
            "racket_root_position_error_m": values,
            "racket_root_orientation_error_deg": np.roll(values, 20),
            "support_margin_error_m": np.roll(values, 40),
        },
    )
    clips = analysis.select_diagnostic_clips(data, duration_s=1.0, max_clips=5)
    assert clips
    assert any(clip.label == "representative_forehand" for clip in clips)
    assert all(
        abs(left.peak_frame - right.peak_frame) > 5 for index, left in enumerate(clips) for right in clips[index + 1 :]
    )


def test_reduced_mass_proxy_conserves_mass_weighted_com() -> None:
    proxy = analysis.ProxyModel(
        points=(
            analysis.ProxyMassPoint("Pelvis", 2.0, np.array([1.0, 0.0, 0.0]), "pelvis"),
            analysis.ProxyMassPoint("Pelvis", 1.0, np.array([-1.0, 0.0, 0.0]), "contour"),
        ),
        total_mass_kg=3.0,
        reference_com_error_m=0.0,
        calibration_success=True,
        calibration_cost=0.0,
    )
    positions = np.array([[[10.0, 0.0, 0.0]], [[20.0, 0.0, 0.0]]])
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1, 1))
    com = analysis.evaluate_proxy_com(proxy, ["Pelvis"], positions, quaternions)
    np.testing.assert_allclose(
        com,
        [[10.0 + 1.0 / 3.0, 0.0, 0.0], [20.0 + 1.0 / 3.0, 0.0, 0.0]],
    )


def test_batch_summary_layout(tmp_path: Path) -> None:
    distributions = {
        metric: {"median": float(index), "p95": float(index + 1)}
        for index, metric in enumerate(
            (
                "com_root_error_m",
                "racket_root_position_error_m",
                "racket_root_orientation_error_deg",
                "support_margin_error_m",
            )
        )
    }
    summaries = [
        {
            "sequence_name": name,
            "frame_count": 10,
            "time_range_s": [0.0, 0.3],
            "windows": {"full_sequence": {"metrics": distributions}},
        }
        for name in ("one", "two")
    ]
    analysis._write_batch_summary(summaries, tmp_path)
    assert (tmp_path / "batch_summary.csv").read_text(encoding="utf-8").count("\n") == 3
    payload = (tmp_path / "batch_summary.json").read_text(encoding="utf-8")
    assert '"sequence_count": 2' in payload
    assert '"sequence_name": "one"' in payload


@pytest.mark.skipif(
    "HOLOSOMA_XSENS_ANALYSIS_SEQUENCE" not in os.environ,
    reason="Set HOLOSOMA_XSENS_ANALYSIS_SEQUENCE and local data paths for the opt-in smoke test",
)
def test_opt_in_real_data_smoke(tmp_path: Path) -> None:
    sequence = os.environ["HOLOSOMA_XSENS_ANALYSIS_SEQUENCE"]
    data_dir = Path(os.environ["HOLOSOMA_XSENS_ANALYSIS_DATA_DIR"])
    results_dir = Path(os.environ["HOLOSOMA_XSENS_ANALYSIS_RESULTS_DIR"])
    calibration = os.environ.get("HOLOSOMA_XSENS_ANALYSIS_CALIBRATION")
    summaries = analysis.run(
        analysis.Config(
            sequence_names=(sequence,),
            data_dir=data_dir,
            retargeted_results_dir=results_dir,
            output_root=tmp_path,
            frame_end=120,
            tpose_calibration_path=Path(calibration) if calibration else None,
        )
    )
    assert summaries[0]["frame_count"] == 120
    assert (tmp_path / analysis.normalize_sequence_name(sequence) / "summary.json").is_file()
