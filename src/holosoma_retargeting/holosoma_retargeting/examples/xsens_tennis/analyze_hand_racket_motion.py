#!/usr/bin/env python3
"""Analyze relative Xsens hand and tracked-racket motion in one HDF5 recording."""

from __future__ import annotations

import ast
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/holosoma-matplotlib")

import h5py
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

from holosoma_retargeting.transformation_utils import rotation_as_wxyz, rotations_from_wxyz

HAND_SEGMENT = "RightHand"
RACKET_SEGMENT = "RightHandSword"
SEGMENT_GROUP = "xsens-segments"
SENSOR_GROUP = "xsens-sensors"
POSITION_STREAM = "body_position_xyz_m"
ORIENTATION_STREAM = "body_orientation_quaternion_wijk"
SENSOR_ORIENTATION_STREAM = "sensor_orientation_quaternion_wijk"
TPOSE_GROUP = "xsens-segments-tpose"
TPOSE_POSITION_STREAM = "body_position_Tpose_xyz_m"
TPOSE_ORIENTATION_STREAM = "body_orientation_Tpose_quaternion_wijk"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


@dataclass(frozen=True)
class TimeWindow:
    """Named half-open interval in recording-relative seconds."""

    start_s: float
    end_s: float
    label: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class SequenceData:
    """The hand/racket streams and embedded reference pose needed by the analysis."""

    source_path: Path
    times_s: np.ndarray
    hand_positions_m: np.ndarray
    racket_positions_m: np.ndarray
    hand_quaternions_wxyz: np.ndarray
    racket_quaternions_wxyz: np.ndarray
    sensor_hand_quaternions_wxyz: np.ndarray | None
    sensor_racket_quaternions_wxyz: np.ndarray | None
    tpose_hand_position_m: np.ndarray
    tpose_racket_position_m: np.ndarray
    tpose_hand_quaternion_wxyz: np.ndarray
    tpose_racket_quaternion_wxyz: np.ndarray
    calibration_windows: tuple[TimeWindow, ...]
    observed_baseline: TimeWindow
    activity: TimeWindow
    native_fps: float


@dataclass(frozen=True)
class RelativePose:
    """Racket origin and axes represented in the hand segment frame."""

    translations_m: np.ndarray
    rotations: Rotation


@dataclass(frozen=True)
class OrientationResidual:
    """Quaternion-safe orientation residual measures relative to one baseline."""

    rotations: Rotation
    geodesic_deg: np.ndarray
    longitudinal_axis_misalignment_deg: np.ndarray
    twist_deg: np.ndarray
    rotvec_deg: np.ndarray


@dataclass(frozen=True)
class AnalysisResult:
    """All per-frame signals plus baselines used for exports and figures."""

    sequence: SequenceData
    relative_pose: RelativePose
    embedded_translation_m: np.ndarray
    embedded_rotation: Rotation
    observed_translation_m: np.ndarray
    observed_rotation: Rotation
    embedded_residual: OrientationResidual
    observed_residual: OrientationResidual
    translation_error_observed_m: np.ndarray
    translation_error_embedded_m: np.ndarray
    hand_angular_speed_rad_s: np.ndarray
    racket_angular_speed_rad_s: np.ndarray
    relative_angular_speed_rad_s: np.ndarray
    sensor_observed_error_deg: np.ndarray | None
    sensor_segment_error_disagreement_deg: np.ndarray | None
    phase: np.ndarray


@dataclass(frozen=True)
class OrientationEvent:
    """One sustained excursion above an orientation-error threshold."""

    start_s: float
    end_s: float
    duration_s: float
    mean_error_deg: float
    p95_error_deg: float
    max_error_deg: float
    max_relative_angular_speed_rad_s: float


@dataclass(frozen=True)
class DiagnosticClip:
    """A selected animation interval."""

    label: str
    start_s: float
    end_s: float
    selection_reason: str


@dataclass(frozen=True)
class Config:
    """CLI configuration for a hand/racket relative-pose analysis."""

    hdf5_path: Path
    output_dir: Path
    activity_start_s: float | None = None
    activity_end_s: float | None = None
    observed_baseline_start_s: float | None = None
    observed_baseline_end_s: float | None = None
    animation_fps: float = 30.0
    event_threshold_deg: float = 60.0
    event_min_duration_s: float = 0.5
    event_max_gap_s: float = 0.1
    animation_clip_duration_s: float = 6.0


def _decode(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _parse_names(value: Any, *, attribute: str) -> list[str]:
    value = _decode(value)
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{attribute} must decode to a list")
    return [str(item) for item in parsed]


def compute_relative_pose(
    hand_positions_m: np.ndarray,
    racket_positions_m: np.ndarray,
    hand_rotations: Rotation,
    racket_rotations: Rotation,
) -> RelativePose:
    """Express the racket origin and orientation in the hand frame."""

    hand_positions_m = np.asarray(hand_positions_m, dtype=float)
    racket_positions_m = np.asarray(racket_positions_m, dtype=float)
    if hand_positions_m.shape != racket_positions_m.shape or hand_positions_m.shape[-1] != 3:
        raise ValueError("Hand and racket position arrays must have equal shape (frames, 3)")
    translations_m = hand_rotations.inv().apply(racket_positions_m - hand_positions_m)
    rotations = hand_rotations.inv() * racket_rotations
    return RelativePose(translations_m=translations_m, rotations=rotations)


def compute_orientation_residual(relative_rotations: Rotation, baseline: Rotation) -> OrientationResidual:
    """Measure relative orientation changes in the baseline racket frame."""

    residual = baseline.inv() * relative_rotations
    geodesic_deg = np.atleast_1d(np.rad2deg(residual.magnitude()))
    rotated_long_axis = np.atleast_2d(residual.apply(np.array([1.0, 0.0, 0.0])))
    axis_cosine = np.clip(rotated_long_axis[:, 0], -1.0, 1.0)
    longitudinal_axis_misalignment_deg = np.rad2deg(np.arccos(axis_cosine))

    residual_xyzw = np.atleast_2d(residual.as_quat())
    residual_xyzw[residual_xyzw[:, 3] < 0] *= -1
    twist_deg = np.rad2deg(2.0 * np.arctan2(residual_xyzw[:, 0], residual_xyzw[:, 3]))
    twist_deg = (twist_deg + 180.0) % 360.0 - 180.0
    rotvec_deg = np.atleast_2d(np.rad2deg(residual.as_rotvec()))
    return OrientationResidual(
        rotations=residual,
        geodesic_deg=geodesic_deg,
        longitudinal_axis_misalignment_deg=longitudinal_axis_misalignment_deg,
        twist_deg=twist_deg,
        rotvec_deg=rotvec_deg,
    )


def angular_speed(rotations: Rotation, times_s: np.ndarray) -> np.ndarray:
    """Quaternion-safe angular speed in radians per second at every sample."""

    times_s = np.asarray(times_s, dtype=float)
    if times_s.ndim != 1 or times_s.size < 2:
        raise ValueError("At least two one-dimensional timestamps are required")
    dt = np.diff(times_s)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    increments = rotations[:-1].inv() * rotations[1:]
    interval_speed = increments.magnitude() / dt
    speed = np.empty(times_s.size, dtype=float)
    speed[0] = interval_speed[0]
    speed[1:] = interval_speed
    return speed


def infer_analysis_windows(
    calibration_times_absolute_s: Sequence[float],
    calibration_rows: Sequence[Sequence[str]],
    stream_start_absolute_s: float,
) -> tuple[tuple[TimeWindow, ...], TimeWindow, TimeWindow]:
    """Infer good calibration windows and the central activity interval."""

    open_windows: dict[tuple[str, str], float] = {}
    windows: list[TimeWindow] = []
    for absolute_time_s, raw_row in zip(calibration_times_absolute_s, calibration_rows, strict=True):
        row = [_decode(value) for value in raw_row]
        if len(row) < 6:
            continue
        action, validity, pose = row[0].strip(), row[1].strip(), row[5].strip()
        if validity.lower() != "good" or not pose:
            continue
        key = (validity.lower(), pose)
        relative_time_s = float(absolute_time_s) - stream_start_absolute_s
        if action.lower() == "start":
            open_windows[key] = relative_time_s
        elif action.lower() == "stop" and key in open_windows:
            start_s = open_windows.pop(key)
            if relative_time_s > start_s:
                windows.append(TimeWindow(start_s, relative_time_s, f"Good {pose}"))

    windows.sort(key=lambda window: window.start_s)
    good_tposes = [window for window in windows if window.label.lower() == "good t-pose"]
    if not good_tposes:
        raise ValueError(
            "Could not infer an observed baseline: provide --observed-baseline-start-s and --observed-baseline-end-s"
        )
    observed_baseline = good_tposes[0]
    next_calibration = next(
        (window for window in windows if window.start_s > observed_baseline.end_s),
        None,
    )
    if next_calibration is None:
        raise ValueError("Could not infer an activity end after the first good T-pose: provide --activity-end-s")
    activity = TimeWindow(observed_baseline.end_s, next_calibration.start_s, "Inferred tennis/activity")
    return tuple(windows), observed_baseline, activity


def _validate_window(window: TimeWindow, times_s: np.ndarray) -> None:
    if not np.isfinite([window.start_s, window.end_s]).all() or window.end_s <= window.start_s:
        raise ValueError(f"Invalid {window.label} window: {window.start_s} to {window.end_s}")
    if window.start_s < times_s[0] or window.end_s > times_s[-1]:
        raise ValueError(
            f"{window.label} window [{window.start_s:.3f}, {window.end_s:.3f}] lies outside "
            f"the recording [{times_s[0]:.3f}, {times_s[-1]:.3f}]"
        )


def _window_override(
    inferred: TimeWindow,
    start_s: float | None,
    end_s: float | None,
) -> TimeWindow:
    return TimeWindow(
        inferred.start_s if start_s is None else start_s,
        inferred.end_s if end_s is None else end_s,
        inferred.label,
    )


def load_sequence(config: Config) -> SequenceData:
    """Read only the HDF5 streams needed for the hand/racket analysis."""

    path = config.hdf5_path.expanduser().resolve()
    with h5py.File(path, "r") as hdf5_file:
        position_group = hdf5_file[f"{SEGMENT_GROUP}/{POSITION_STREAM}"]
        orientation_group = hdf5_file[f"{SEGMENT_GROUP}/{ORIENTATION_STREAM}"]
        segment_names = _parse_names(position_group.attrs["segment_names_body"], attribute="segment_names_body")
        orientation_names = _parse_names(
            orientation_group.attrs["segment_names_body"],
            attribute="segment_names_body",
        )
        if segment_names != orientation_names:
            raise ValueError("Segment position and orientation name orders differ")
        hand_index = segment_names.index(HAND_SEGMENT)
        racket_index = segment_names.index(RACKET_SEGMENT)

        absolute_times_s = np.asarray(position_group["time_s"][:, 0], dtype=float)
        if "xsens_time_since_start_s" in position_group:
            times_s = np.asarray(position_group["xsens_time_since_start_s"][:, 0], dtype=float)
            times_s -= times_s[0]
        else:
            times_s = absolute_times_s - absolute_times_s[0]
        orientation_times_s = np.asarray(orientation_group["time_s"][:, 0], dtype=float)
        if not np.allclose(orientation_times_s, absolute_times_s):
            raise ValueError("Segment position and orientation timestamps differ")

        positions = np.asarray(position_group["data"][:, [hand_index, racket_index], :], dtype=float)
        quaternions = np.asarray(orientation_group["data"][:, [hand_index, racket_index], :], dtype=float)

        sensor_path = f"{SENSOR_GROUP}/{SENSOR_ORIENTATION_STREAM}"
        sensor_hand_quaternions: np.ndarray | None = None
        sensor_racket_quaternions: np.ndarray | None = None
        if sensor_path in hdf5_file:
            sensor_group = hdf5_file[sensor_path]
            sensor_names = _parse_names(sensor_group.attrs["sensor_names"], attribute="sensor_names")
            if HAND_SEGMENT in sensor_names and RACKET_SEGMENT in sensor_names:
                sensor_times_s = np.asarray(sensor_group["time_s"][:, 0], dtype=float)
                if not np.allclose(sensor_times_s, absolute_times_s):
                    raise ValueError("Segment and raw sensor orientation timestamps differ")
                sensor_indices = [sensor_names.index(HAND_SEGMENT), sensor_names.index(RACKET_SEGMENT)]
                sensor_quaternions = np.asarray(sensor_group["data"][:, sensor_indices, :], dtype=float)
                sensor_hand_quaternions = sensor_quaternions[:, 0]
                sensor_racket_quaternions = sensor_quaternions[:, 1]

        tpose_positions = np.asarray(hdf5_file[f"{TPOSE_GROUP}/{TPOSE_POSITION_STREAM}"], dtype=float)
        tpose_quaternions = np.asarray(hdf5_file[f"{TPOSE_GROUP}/{TPOSE_ORIENTATION_STREAM}"], dtype=float)

        calibration_group = hdf5_file["experiment-calibration/body"]
        calibration_times = np.asarray(calibration_group["time_s"][:, 0], dtype=float)
        calibration_rows = [[_decode(value) for value in row] for row in calibration_group["data"][:]]

    if not np.isfinite(positions).all():
        raise ValueError("Position stream contains non-finite values")
    if not np.all(np.diff(times_s) > 0):
        raise ValueError("Xsens timestamps must be strictly increasing")
    calibration_windows, inferred_baseline, inferred_activity = infer_analysis_windows(
        calibration_times,
        calibration_rows,
        float(absolute_times_s[0]),
    )
    observed_baseline = _window_override(
        inferred_baseline,
        config.observed_baseline_start_s,
        config.observed_baseline_end_s,
    )
    activity = _window_override(inferred_activity, config.activity_start_s, config.activity_end_s)
    _validate_window(observed_baseline, times_s)
    _validate_window(activity, times_s)
    if observed_baseline.end_s > activity.start_s + 1e-9:
        raise ValueError("Observed baseline must not overlap the activity interval")

    median_dt = float(np.median(np.diff(times_s)))
    return SequenceData(
        source_path=path,
        times_s=times_s,
        hand_positions_m=positions[:, 0],
        racket_positions_m=positions[:, 1],
        hand_quaternions_wxyz=quaternions[:, 0],
        racket_quaternions_wxyz=quaternions[:, 1],
        sensor_hand_quaternions_wxyz=sensor_hand_quaternions,
        sensor_racket_quaternions_wxyz=sensor_racket_quaternions,
        tpose_hand_position_m=tpose_positions[hand_index],
        tpose_racket_position_m=tpose_positions[racket_index],
        tpose_hand_quaternion_wxyz=tpose_quaternions[hand_index],
        tpose_racket_quaternion_wxyz=tpose_quaternions[racket_index],
        calibration_windows=calibration_windows,
        observed_baseline=observed_baseline,
        activity=activity,
        native_fps=1.0 / median_dt,
    )


def _mask_for_window(times_s: np.ndarray, window: TimeWindow) -> np.ndarray:
    return (times_s >= window.start_s) & (times_s <= window.end_s)


def analyze_sequence(sequence: SequenceData) -> AnalysisResult:
    """Compute the complete relative-pose analysis without plotting or file I/O."""

    hand_rotations = rotations_from_wxyz(sequence.hand_quaternions_wxyz)
    racket_rotations = rotations_from_wxyz(sequence.racket_quaternions_wxyz)
    relative_pose = compute_relative_pose(
        sequence.hand_positions_m,
        sequence.racket_positions_m,
        hand_rotations,
        racket_rotations,
    )

    tpose_hand_rotation = rotations_from_wxyz(sequence.tpose_hand_quaternion_wxyz)
    tpose_racket_rotation = rotations_from_wxyz(sequence.tpose_racket_quaternion_wxyz)
    embedded_rotation = tpose_hand_rotation.inv() * tpose_racket_rotation
    embedded_translation_m = tpose_hand_rotation.inv().apply(
        sequence.tpose_racket_position_m - sequence.tpose_hand_position_m
    )

    baseline_mask = _mask_for_window(sequence.times_s, sequence.observed_baseline)
    if np.count_nonzero(baseline_mask) < 2:
        raise ValueError("Observed baseline window contains fewer than two frames")
    observed_rotation = relative_pose.rotations[baseline_mask].mean()
    observed_translation_m = np.mean(relative_pose.translations_m[baseline_mask], axis=0)

    embedded_residual = compute_orientation_residual(relative_pose.rotations, embedded_rotation)
    observed_residual = compute_orientation_residual(relative_pose.rotations, observed_rotation)
    translation_error_observed_m = np.linalg.norm(
        relative_pose.translations_m - observed_translation_m,
        axis=1,
    )
    translation_error_embedded_m = np.linalg.norm(
        relative_pose.translations_m - embedded_translation_m,
        axis=1,
    )

    hand_angular_speed_rad_s = angular_speed(hand_rotations, sequence.times_s)
    racket_angular_speed_rad_s = angular_speed(racket_rotations, sequence.times_s)
    relative_angular_speed_rad_s = angular_speed(relative_pose.rotations, sequence.times_s)

    sensor_observed_error_deg: np.ndarray | None = None
    sensor_segment_error_disagreement_deg: np.ndarray | None = None
    if sequence.sensor_hand_quaternions_wxyz is not None and sequence.sensor_racket_quaternions_wxyz is not None:
        sensor_hand_rotation = rotations_from_wxyz(sequence.sensor_hand_quaternions_wxyz)
        sensor_racket_rotation = rotations_from_wxyz(sequence.sensor_racket_quaternions_wxyz)
        sensor_relative_rotation = sensor_hand_rotation.inv() * sensor_racket_rotation
        sensor_baseline = sensor_relative_rotation[baseline_mask].mean()
        sensor_observed_error_deg = np.rad2deg((sensor_baseline.inv() * sensor_relative_rotation).magnitude())
        sensor_segment_error_disagreement_deg = np.abs(sensor_observed_error_deg - observed_residual.geodesic_deg)

    phase = np.full(sequence.times_s.shape, "other", dtype=object)
    phase[baseline_mask] = "observed_good_tpose"
    phase[_mask_for_window(sequence.times_s, sequence.activity)] = "inferred_activity"
    phase[sequence.times_s < sequence.observed_baseline.start_s] = "pre_baseline"
    phase[sequence.times_s > sequence.activity.end_s] = "post_activity"

    return AnalysisResult(
        sequence=sequence,
        relative_pose=relative_pose,
        embedded_translation_m=embedded_translation_m,
        embedded_rotation=embedded_rotation,
        observed_translation_m=observed_translation_m,
        observed_rotation=observed_rotation,
        embedded_residual=embedded_residual,
        observed_residual=observed_residual,
        translation_error_observed_m=translation_error_observed_m,
        translation_error_embedded_m=translation_error_embedded_m,
        hand_angular_speed_rad_s=hand_angular_speed_rad_s,
        racket_angular_speed_rad_s=racket_angular_speed_rad_s,
        relative_angular_speed_rad_s=relative_angular_speed_rad_s,
        sensor_observed_error_deg=sensor_observed_error_deg,
        sensor_segment_error_disagreement_deg=sensor_segment_error_disagreement_deg,
        phase=phase,
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return math.nan
    x_finite, y_finite = x[finite], y[finite]
    if np.std(x_finite) < 1e-12 or np.std(y_finite) < 1e-12:
        return math.nan
    return float(np.corrcoef(x_finite, y_finite)[0, 1])


def best_speed_lag(
    hand_speed: np.ndarray,
    racket_speed: np.ndarray,
    sample_period_s: float,
    max_lag_s: float = 0.25,
) -> tuple[float, float]:
    """Find scalar-speed lag; positive lag means racket motion follows hand motion."""

    max_lag_samples = max(1, round(max_lag_s / sample_period_s))
    best_lag_samples = 0
    best_correlation = -np.inf
    for lag_samples in range(-max_lag_samples, max_lag_samples + 1):
        if lag_samples > 0:
            hand_values, racket_values = hand_speed[:-lag_samples], racket_speed[lag_samples:]
        elif lag_samples < 0:
            hand_values, racket_values = hand_speed[-lag_samples:], racket_speed[:lag_samples]
        else:
            hand_values, racket_values = hand_speed, racket_speed
        correlation = _pearson(hand_values, racket_values)
        if np.isfinite(correlation) and correlation > best_correlation:
            best_correlation = correlation
            best_lag_samples = lag_samples
    return best_lag_samples * sample_period_s, float(best_correlation)


def _merge_boolean_runs(mask: np.ndarray, times_s: np.ndarray, max_gap_s: float) -> list[tuple[int, int]]:
    true_indices = np.flatnonzero(mask)
    if true_indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(true_indices[0])
    previous = start
    for index_value in true_indices[1:]:
        index = int(index_value)
        if times_s[index] - times_s[previous] > max_gap_s + 1.5 * np.median(np.diff(times_s)):
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def detect_orientation_events(
    result: AnalysisResult,
    *,
    threshold_deg: float,
    min_duration_s: float,
    max_gap_s: float,
) -> list[OrientationEvent]:
    """Find sustained activity-interval orientation residuals above a threshold."""

    times_s = result.sequence.times_s
    activity_mask = _mask_for_window(times_s, result.sequence.activity)
    high_error = activity_mask & (result.observed_residual.geodesic_deg >= threshold_deg)
    events: list[OrientationEvent] = []
    sample_period_s = float(np.median(np.diff(times_s)))
    for start, end in _merge_boolean_runs(high_error, times_s, max_gap_s):
        duration_s = float(times_s[end] - times_s[start] + sample_period_s)
        if duration_s < min_duration_s:
            continue
        event_error = result.observed_residual.geodesic_deg[start : end + 1]
        event_speed = result.relative_angular_speed_rad_s[start : end + 1]
        events.append(
            OrientationEvent(
                start_s=float(times_s[start]),
                end_s=float(times_s[end]),
                duration_s=duration_s,
                mean_error_deg=float(np.mean(event_error)),
                p95_error_deg=float(np.percentile(event_error, 95)),
                max_error_deg=float(np.max(event_error)),
                max_relative_angular_speed_rad_s=float(np.max(event_speed)),
            )
        )
    return events


def _rolling_mean(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray(values, dtype=float).copy()
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=float))])
    return (cumulative[count:] - cumulative[:-count]) / count


def select_diagnostic_clips(result: AnalysisResult, duration_s: float) -> tuple[DiagnosticClip, ...]:
    """Select baseline, coupled high-motion, and strongest-divergence clips."""

    times_s = result.sequence.times_s
    sample_period_s = float(np.median(np.diff(times_s)))
    sample_count = max(2, round(duration_s / sample_period_s))

    baseline = result.sequence.observed_baseline
    baseline_midpoint_s = 0.5 * (baseline.start_s + baseline.end_s)
    baseline_start_s = max(baseline.start_s, baseline_midpoint_s - duration_s / 2.0)
    baseline_start_s = min(baseline_start_s, baseline.end_s - duration_s)

    rolling_error = _rolling_mean(result.observed_residual.geodesic_deg, sample_count)
    rolling_racket_speed = _rolling_mean(result.racket_angular_speed_rad_s, sample_count)
    start_times_s = times_s[: rolling_error.size]
    valid_activity = (start_times_s >= result.sequence.activity.start_s) & (
        start_times_s + duration_s <= result.sequence.activity.end_s
    )
    activity_mask = _mask_for_window(times_s, result.sequence.activity)
    activity_median_error = float(np.median(result.observed_residual.geodesic_deg[activity_mask]))
    coupled_candidates = valid_activity & (rolling_error <= activity_median_error)
    if not np.any(coupled_candidates):
        coupled_candidates = valid_activity
    coupled_indices = np.flatnonzero(coupled_candidates)
    coupled_start_index = int(coupled_indices[np.argmax(rolling_racket_speed[coupled_indices])])

    divergence_indices = np.flatnonzero(valid_activity)
    divergence_start_index = int(divergence_indices[np.argmax(rolling_error[divergence_indices])])
    coupled_start_s = float(start_times_s[coupled_start_index])
    divergence_start_s = float(start_times_s[divergence_start_index])
    return (
        DiagnosticClip(
            label="Observed T-pose baseline",
            start_s=float(baseline_start_s),
            end_s=float(baseline_start_s + duration_s),
            selection_reason="Centered within the first logged good T-pose.",
        ),
        DiagnosticClip(
            label="Coupled high-motion example",
            start_s=coupled_start_s,
            end_s=coupled_start_s + duration_s,
            selection_reason=(
                "Highest mean racket angular speed among activity windows whose mean relative error does not "
                "exceed the activity median."
            ),
        ),
        DiagnosticClip(
            label="Largest sustained divergence",
            start_s=divergence_start_s,
            end_s=divergence_start_s + duration_s,
            selection_reason="Largest mean observed-baseline orientation error over the clip duration.",
        ),
    )


def build_summary(
    result: AnalysisResult,
    events: Sequence[OrientationEvent],
    clips: Sequence[DiagnosticClip],
    config: Config,
) -> dict[str, Any]:
    """Create the machine-readable summary used verbatim by the report."""

    times_s = result.sequence.times_s
    activity_mask = _mask_for_window(times_s, result.sequence.activity)
    baseline_mask = _mask_for_window(times_s, result.sequence.observed_baseline)
    dt_s = float(np.median(np.diff(times_s)))
    activity_duration_s = float(np.count_nonzero(activity_mask) * dt_s)
    observed_vs_embedded_deg = float(
        np.rad2deg((result.embedded_rotation.inv() * result.observed_rotation).magnitude())
    )
    threshold_summary: dict[str, dict[str, float]] = {}
    for threshold_deg in (30.0, 60.0, 90.0):
        threshold_mask = activity_mask & (result.observed_residual.geodesic_deg >= threshold_deg)
        runs = _merge_boolean_runs(threshold_mask, times_s, max_gap_s=dt_s * 1.5)
        longest_s = max(
            (times_s[end] - times_s[start] + dt_s for start, end in runs),
            default=0.0,
        )
        count = int(np.count_nonzero(threshold_mask))
        threshold_summary[f"ge_{int(threshold_deg)}_deg"] = {
            "frame_fraction": count / max(1, int(np.count_nonzero(activity_mask))),
            "duration_s": count * dt_s,
            "longest_contiguous_duration_s": float(longest_s),
        }

    hand_activity_speed = result.hand_angular_speed_rad_s[activity_mask]
    racket_activity_speed = result.racket_angular_speed_rad_s[activity_mask]
    zero_lag_correlation = _pearson(hand_activity_speed, racket_activity_speed)
    best_lag_s, best_lag_correlation = best_speed_lag(
        hand_activity_speed,
        racket_activity_speed,
        dt_s,
    )

    sensor_validation: dict[str, Any] = {"available": False}
    if result.sensor_observed_error_deg is not None and result.sensor_segment_error_disagreement_deg is not None:
        sensor_validation = {
            "available": True,
            "relative_angle_correlation": _pearson(
                result.observed_residual.geodesic_deg[activity_mask],
                result.sensor_observed_error_deg[activity_mask],
            ),
            "absolute_angle_disagreement_deg": _distribution(
                result.sensor_segment_error_disagreement_deg[activity_mask]
            ),
        }

    event_dicts = [event.__dict__ for event in events]
    ranked_event_indices = sorted(
        range(len(events)),
        key=lambda index: (events[index].mean_error_deg, events[index].duration_s),
        reverse=True,
    )
    return {
        "recording": result.sequence.source_path.stem,
        "source_path": str(result.sequence.source_path),
        "sample_count": int(times_s.size),
        "duration_s": float(times_s[-1] - times_s[0]),
        "native_fps": result.sequence.native_fps,
        "observed_baseline_window_s": result.sequence.observed_baseline.__dict__,
        "activity_window_s": result.sequence.activity.__dict__,
        "embedded_hand_to_racket": {
            "translation_hand_frame_m": result.embedded_translation_m.tolist(),
            "quaternion_wxyz": rotation_as_wxyz(result.embedded_rotation).tolist(),
        },
        "observed_hand_to_racket": {
            "translation_hand_frame_m": result.observed_translation_m.tolist(),
            "quaternion_wxyz": rotation_as_wxyz(result.observed_rotation).tolist(),
        },
        "observed_vs_embedded_orientation_deg": observed_vs_embedded_deg,
        "activity_metrics": {
            "duration_s": activity_duration_s,
            "orientation_error_observed_baseline_deg": _distribution(
                result.observed_residual.geodesic_deg[activity_mask]
            ),
            "orientation_error_embedded_baseline_deg": _distribution(
                result.embedded_residual.geodesic_deg[activity_mask]
            ),
            "longitudinal_axis_misalignment_deg": _distribution(
                result.observed_residual.longitudinal_axis_misalignment_deg[activity_mask]
            ),
            "absolute_twist_deg": _distribution(np.abs(result.observed_residual.twist_deg[activity_mask])),
            "translation_error_observed_baseline_m": _distribution(result.translation_error_observed_m[activity_mask]),
            "hand_angular_speed_rad_s": _distribution(hand_activity_speed),
            "racket_angular_speed_rad_s": _distribution(racket_activity_speed),
            "relative_angular_speed_rad_s": _distribution(result.relative_angular_speed_rad_s[activity_mask]),
            "angular_speed_coupling": {
                "zero_lag_correlation": zero_lag_correlation,
                "best_lag_s_positive_means_racket_follows_hand": best_lag_s,
                "best_lag_correlation": best_lag_correlation,
                "absolute_speed_difference_rad_s": _distribution(np.abs(hand_activity_speed - racket_activity_speed)),
            },
            "time_above_thresholds": threshold_summary,
        },
        "baseline_metrics": {
            "orientation_error_observed_baseline_deg": _distribution(
                result.observed_residual.geodesic_deg[baseline_mask]
            ),
            "translation_error_observed_baseline_m": _distribution(result.translation_error_observed_m[baseline_mask]),
        },
        "sensor_validation": sensor_validation,
        "event_detection": {
            "threshold_deg": config.event_threshold_deg,
            "minimum_duration_s": config.event_min_duration_s,
            "maximum_bridged_gap_s": config.event_max_gap_s,
            "event_count": len(events),
            "events": event_dicts,
            "indices_ranked_by_mean_error": ranked_event_indices,
        },
        "diagnostic_clips": [clip.__dict__ for clip in clips],
    }


def _write_frame_metrics(result: AnalysisResult, output_path: Path) -> None:
    relative_wxyz = rotation_as_wxyz(result.relative_pose.rotations)
    sensor_error = (
        result.sensor_observed_error_deg
        if result.sensor_observed_error_deg is not None
        else np.full(result.sequence.times_s.shape, np.nan)
    )
    sensor_disagreement = (
        result.sensor_segment_error_disagreement_deg
        if result.sensor_segment_error_disagreement_deg is not None
        else np.full(result.sequence.times_s.shape, np.nan)
    )
    header = [
        "time_s",
        "phase",
        "relative_translation_hand_x_m",
        "relative_translation_hand_y_m",
        "relative_translation_hand_z_m",
        "relative_quaternion_w",
        "relative_quaternion_x",
        "relative_quaternion_y",
        "relative_quaternion_z",
        "orientation_error_observed_deg",
        "orientation_error_embedded_deg",
        "longitudinal_axis_misalignment_deg",
        "twist_deg",
        "residual_rotvec_x_deg",
        "residual_rotvec_y_deg",
        "residual_rotvec_z_deg",
        "translation_error_observed_m",
        "translation_error_embedded_m",
        "hand_angular_speed_rad_s",
        "racket_angular_speed_rad_s",
        "relative_angular_speed_rad_s",
        "sensor_orientation_error_observed_deg",
        "sensor_segment_angle_disagreement_deg",
    ]
    numeric_columns = np.column_stack(
        [
            result.sequence.times_s,
            result.relative_pose.translations_m,
            relative_wxyz,
            result.observed_residual.geodesic_deg,
            result.embedded_residual.geodesic_deg,
            result.observed_residual.longitudinal_axis_misalignment_deg,
            result.observed_residual.twist_deg,
            result.observed_residual.rotvec_deg,
            result.translation_error_observed_m,
            result.translation_error_embedded_m,
            result.hand_angular_speed_rad_s,
            result.racket_angular_speed_rad_s,
            result.relative_angular_speed_rad_s,
            sensor_error,
            sensor_disagreement,
        ]
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, numeric_row in enumerate(numeric_columns):
            writer.writerow(
                [
                    f"{numeric_row[0]:.9f}",
                    result.phase[index],
                    *[f"{value:.10g}" for value in numeric_row[1:]],
                ]
            )


def _write_events(events: Sequence[OrientationEvent], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OrientationEvent.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(event.__dict__ for event in events)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "font.size": 10,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def _shade_intervals(axes: Iterable[Axes], result: AnalysisResult, *, minutes: bool) -> None:
    divisor = 60.0 if minutes else 1.0
    for axis in axes:
        axis.axvspan(
            result.sequence.activity.start_s / divisor,
            result.sequence.activity.end_s / divisor,
            color=LIGHT_GRAY,
            alpha=0.18,
            linewidth=0,
            label="Inferred activity" if axis is next(iter(axes), None) else None,
        )
        for window in result.sequence.calibration_windows:
            axis.axvspan(
                window.start_s / divisor,
                window.end_s / divisor,
                color=GREEN if "T-Pose" in window.label else BLUE,
                alpha=0.07,
                linewidth=0,
            )


def _binned_quantiles(
    times_s: np.ndarray,
    values: np.ndarray,
    bin_width_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_index = np.floor((times_s - times_s[0]) / bin_width_s).astype(int)
    unique_bins = np.unique(bin_index)
    center = np.empty(unique_bins.size)
    low = np.empty(unique_bins.size)
    median = np.empty(unique_bins.size)
    high = np.empty(unique_bins.size)
    for output_index, current_bin in enumerate(unique_bins):
        mask = bin_index == current_bin
        center[output_index] = np.mean(times_s[mask])
        low[output_index], median[output_index], high[output_index] = np.percentile(values[mask], [5, 50, 95])
    return center, low, median, high


def _plot_envelope(
    axis: Axes,
    times_s: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    bin_width_s: float,
    time_divisor: float = 1.0,
) -> None:
    center, low, median, high = _binned_quantiles(times_s, values, bin_width_s)
    axis.fill_between(center / time_divisor, low, high, color=color, alpha=0.13, linewidth=0)
    axis.plot(center / time_divisor, median, color=color, linewidth=1.05, label=label)


def _save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=180)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def plot_overview(result: AnalysisResult, events: Sequence[OrientationEvent], output_dir: Path) -> None:
    times_s = result.sequence.times_s
    figure, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    _plot_envelope(
        axes[0],
        times_s,
        result.observed_residual.geodesic_deg,
        label="Observed good T-pose baseline",
        color=BLUE,
        bin_width_s=1.0,
        time_divisor=60.0,
    )
    _plot_envelope(
        axes[0],
        times_s,
        result.embedded_residual.geodesic_deg,
        label="Embedded Xsens T-pose baseline",
        color=ORANGE,
        bin_width_s=1.0,
        time_divisor=60.0,
    )
    axes[0].set_ylabel("Orientation error [deg]")
    axes[0].legend(loc="upper left", ncols=2)

    _plot_envelope(
        axes[1],
        times_s,
        result.observed_residual.longitudinal_axis_misalignment_deg,
        label="Long-axis misalignment",
        color=GREEN,
        bin_width_s=1.0,
        time_divisor=60.0,
    )
    _plot_envelope(
        axes[1],
        times_s,
        np.abs(result.observed_residual.twist_deg),
        label="Absolute twist",
        color=PURPLE,
        bin_width_s=1.0,
        time_divisor=60.0,
    )
    axes[1].set_ylabel("Orientation component [deg]")
    axes[1].legend(loc="upper left", ncols=2)

    _plot_envelope(
        axes[2],
        times_s,
        1000.0 * result.translation_error_observed_m,
        label="Hand-relative racket-origin deviation",
        color=RED,
        bin_width_s=1.0,
        time_divisor=60.0,
    )
    axes[2].set_ylabel("Translation deviation [mm]")
    axes[2].legend(loc="upper left")

    _plot_envelope(
        axes[3],
        times_s,
        result.hand_angular_speed_rad_s,
        label="Hand",
        color=BLUE,
        bin_width_s=1.0,
        time_divisor=60.0,
    )
    _plot_envelope(
        axes[3],
        times_s,
        result.racket_angular_speed_rad_s,
        label="Racket",
        color=ORANGE,
        bin_width_s=1.0,
        time_divisor=60.0,
    )
    _plot_envelope(
        axes[3],
        times_s,
        result.relative_angular_speed_rad_s,
        label="Relative",
        color=RED,
        bin_width_s=1.0,
        time_divisor=60.0,
    )
    axes[3].set(xlabel="Time from Xsens stream start [min]", ylabel="Angular speed [rad/s]")
    axes[3].legend(loc="upper left", ncols=3)

    _shade_intervals(axes, result, minutes=True)
    for event in events:
        if event.mean_error_deg >= 120.0:
            axes[0].axvspan(event.start_s / 60.0, event.end_s / 60.0, color=RED, alpha=0.09, linewidth=0)
    figure.suptitle(
        "Hand-racket relative motion across the complete Xsens recording\n"
        "Lines show 1 s medians; bands show the 5th-95th percentile",
        fontsize=14,
    )
    _save_figure(figure, output_dir, "orientation_overview")


def plot_activity_diagnostics(
    result: AnalysisResult,
    events: Sequence[OrientationEvent],
    output_dir: Path,
    event_threshold_deg: float,
) -> None:
    mask = _mask_for_window(result.sequence.times_s, result.sequence.activity)
    times_s = result.sequence.times_s[mask]
    figure, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    for values, label, color in (
        (result.observed_residual.geodesic_deg[mask], "Observed baseline", BLUE),
        (result.embedded_residual.geodesic_deg[mask], "Embedded baseline", ORANGE),
    ):
        _plot_envelope(axes[0], times_s, values, label=label, color=color, bin_width_s=0.25)
    axes[0].axhline(
        event_threshold_deg,
        color=RED,
        linewidth=0.9,
        linestyle="--",
        label=f"{event_threshold_deg:g}° event threshold",
    )
    axes[0].set_ylabel("Orientation error [deg]")
    axes[0].legend(loc="upper left", ncols=3)

    _plot_envelope(
        axes[1],
        times_s,
        result.observed_residual.longitudinal_axis_misalignment_deg[mask],
        label="Long-axis misalignment",
        color=GREEN,
        bin_width_s=0.25,
    )
    center, _, median_twist, _ = _binned_quantiles(times_s, result.observed_residual.twist_deg[mask], 0.25)
    axes[1].plot(center, median_twist, color=PURPLE, linewidth=1.0, label="Signed twist")
    axes[1].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[1].set_ylabel("Orientation component [deg]")
    axes[1].legend(loc="upper left", ncols=2)

    for values, label, color in (
        (result.hand_angular_speed_rad_s[mask], "Hand", BLUE),
        (result.racket_angular_speed_rad_s[mask], "Racket", ORANGE),
        (result.relative_angular_speed_rad_s[mask], "Relative", RED),
    ):
        _plot_envelope(axes[2], times_s, values, label=label, color=color, bin_width_s=0.25)
    axes[2].set_ylabel("Angular speed [rad/s]")
    axes[2].legend(loc="upper left", ncols=3)

    _plot_envelope(
        axes[3],
        times_s,
        1000.0 * result.translation_error_observed_m[mask],
        label="Racket-origin deviation",
        color=RED,
        bin_width_s=0.25,
    )
    axes[3].set(xlabel="Time from Xsens stream start [s]", ylabel="Translation deviation [mm]")
    axes[3].legend(loc="upper left")
    for axis in axes:
        for event in events:
            axis.axvspan(event.start_s, event.end_s, color=RED, alpha=0.055, linewidth=0)
    figure.suptitle(
        "Relative hand-racket motion during the inferred tennis/activity interval\n"
        "Lines show 0.25 s medians; bands show the 5th-95th percentile",
        fontsize=14,
    )
    _save_figure(figure, output_dir, "relative_motion_diagnostics")


def plot_phase_distributions(
    result: AnalysisResult,
    events: Sequence[OrientationEvent],
    output_dir: Path,
    event_threshold_deg: float,
) -> None:
    phase_windows = [result.sequence.observed_baseline, result.sequence.activity]
    later_calibrations = [
        window for window in result.sequence.calibration_windows if window.start_s > result.sequence.activity.end_s
    ]
    phase_windows.extend(later_calibrations)
    labels = [window.label if index != 1 else "Inferred activity" for index, window in enumerate(phase_windows)]
    data = [
        result.observed_residual.geodesic_deg[_mask_for_window(result.sequence.times_s, window)]
        for window in phase_windows
    ]

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    box = axes[0].boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], [GREEN, LIGHT_GRAY, BLUE, GREEN, GREEN, BLUE, BLUE], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].set(title="Relative-orientation error by phase", ylabel="Observed-baseline error [deg]")

    thresholds = np.array([30.0, 60.0, 90.0])
    activity_mask = _mask_for_window(result.sequence.times_s, result.sequence.activity)
    fractions = [
        100.0 * np.mean(result.observed_residual.geodesic_deg[activity_mask] >= threshold) for threshold in thresholds
    ]
    axes[1].bar([f"≥{int(value)}°" for value in thresholds], fractions, color=[BLUE, ORANGE, RED], alpha=0.78)
    for index, fraction in enumerate(fractions):
        axes[1].text(index, fraction, f"{fraction:.1f}%", ha="center", va="bottom")
    axes[1].set(
        title=(f"Activity time above thresholds ({len(events)} sustained ≥{event_threshold_deg:g}° events)"),
        ylabel="Activity frames [%]",
        ylim=(0, max(fractions) * 1.18 if fractions else 1.0),
    )
    figure.suptitle("Phase and threshold comparison", fontsize=14)
    _save_figure(figure, output_dir, "phase_distributions")


def plot_sensor_crosscheck(result: AnalysisResult, output_dir: Path) -> None:
    if result.sensor_observed_error_deg is None or result.sensor_segment_error_disagreement_deg is None:
        return
    mask = _mask_for_window(result.sequence.times_s, result.sequence.activity)
    times_s = result.sequence.times_s[mask]
    segment_error = result.observed_residual.geodesic_deg[mask]
    sensor_error = result.sensor_observed_error_deg[mask]
    disagreement = result.sensor_segment_error_disagreement_deg[mask]

    figure, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    _plot_envelope(
        axes[0],
        times_s,
        segment_error,
        label="Xsens segment orientations",
        color=BLUE,
        bin_width_s=0.5,
    )
    _plot_envelope(
        axes[0],
        times_s,
        sensor_error,
        label="Raw sensor orientations",
        color=ORANGE,
        bin_width_s=0.5,
    )
    axes[0].set(ylabel="Relative-angle change [deg]")
    axes[0].legend(loc="upper left", ncols=2)

    stride = max(1, segment_error.size // 30000)
    hexbin = axes[1].hexbin(
        segment_error[::stride],
        sensor_error[::stride],
        gridsize=70,
        mincnt=1,
        cmap="viridis",
    )
    limit = max(float(np.max(segment_error)), float(np.max(sensor_error)))
    axes[1].plot([0.0, limit], [0.0, limit], color=RED, linestyle="--", linewidth=1.0, label="Identity")
    axes[1].set(
        xlabel="Segment relative-angle change [deg]",
        ylabel="Raw-sensor relative-angle change [deg]",
        title=(
            f"Per-frame agreement: r={_pearson(segment_error, sensor_error):.4f}; "
            f"median |difference|={np.median(disagreement):.3f}°"
        ),
    )
    axes[1].legend(loc="upper left")
    figure.colorbar(hexbin, ax=axes[1], label="Sample density")
    figure.suptitle("Raw-sensor cross-check of relative orientation", fontsize=14)
    _save_figure(figure, output_dir, "sensor_crosscheck")


def _racket_local_lines() -> list[np.ndarray]:
    shaft = np.array([[-0.09, 0.0, 0.0], [0.24, 0.0, 0.0]])
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    hoop = np.column_stack(
        [
            0.415 + 0.175 * np.cos(theta),
            np.zeros(theta.size),
            0.135 * np.sin(theta),
        ]
    )
    throat_left = np.array([[0.18, 0.0, 0.0], [0.27, 0.0, 0.075]])
    throat_right = np.array([[0.18, 0.0, 0.0], [0.27, 0.0, -0.075]])
    return [shaft, hoop, throat_left, throat_right]


def _transform_points(points: np.ndarray, origin: np.ndarray, rotation: Rotation) -> np.ndarray:
    return rotation.apply(points) + origin


def _draw_racket_pose(
    axis: Axes,
    origin: np.ndarray,
    rotation: Rotation,
    *,
    color: str,
    linestyle: str,
    label: str,
    alpha: float,
) -> None:
    for line_index, local_line in enumerate(_racket_local_lines()):
        world_line = _transform_points(local_line, origin, rotation)
        axis.plot(
            world_line[:, 0],
            world_line[:, 1],
            world_line[:, 2],
            color=color,
            linestyle=linestyle,
            linewidth=2.0 if line_index == 0 else 1.4,
            alpha=alpha,
            label=label if line_index == 0 else None,
        )


def _draw_triad(axis: Axes, origin: np.ndarray, rotation: Rotation, scale: float, alpha: float = 1.0) -> None:
    directions = rotation.as_matrix() * scale
    for direction, color, label in zip(directions.T, (RED, GREEN, BLUE), ("x", "y", "z"), strict=True):
        endpoint = origin + direction
        axis.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=color,
            linewidth=2.0,
            alpha=alpha,
        )
        axis.text(endpoint[0], endpoint[1], endpoint[2], label, color=color, fontsize=8, alpha=alpha)


def _configure_pose_axis(axis: Axes) -> None:
    axis.set(
        xlabel="Hand X [m]",
        ylabel="Hand Y [m]",
        zlabel="Hand Z [m]",
        xlim=(-0.2, 0.7),
        ylim=(-0.42, 0.42),
        zlim=(-0.42, 0.42),
    )
    axis.set_box_aspect((0.9, 0.84, 0.84))
    axis.view_init(elev=23, azim=-58)


def plot_orientation_keyframes(
    result: AnalysisResult,
    clips: Sequence[DiagnosticClip],
    output_dir: Path,
) -> None:
    key_times = [
        0.5 * (result.sequence.observed_baseline.start_s + result.sequence.observed_baseline.end_s),
        *[0.5 * (clip.start_s + clip.end_s) for clip in clips[1:]],
        float(result.sequence.times_s[np.argmax(result.observed_residual.geodesic_deg)]),
    ]
    labels = ["Observed T-pose", "Coupled high motion", "Sustained divergence", "Maximum instantaneous error"]
    figure = plt.figure(figsize=(16, 5), constrained_layout=True)
    for plot_index, (time_s, label) in enumerate(zip(key_times, labels, strict=True), start=1):
        index = int(np.argmin(np.abs(result.sequence.times_s - time_s)))
        axis = figure.add_subplot(1, 4, plot_index, projection="3d")
        _draw_racket_pose(
            axis,
            result.observed_translation_m,
            result.observed_rotation,
            color=GRAY,
            linestyle="--",
            label="Expected from observed T-pose",
            alpha=0.75,
        )
        _draw_racket_pose(
            axis,
            result.relative_pose.translations_m[index],
            result.relative_pose.rotations[index],
            color=ORANGE,
            linestyle="-",
            label="Measured",
            alpha=0.95,
        )
        _draw_triad(axis, np.zeros(3), Rotation.identity(), 0.12, alpha=0.65)
        _configure_pose_axis(axis)
        axis.set_title(
            f"{label}\nt={result.sequence.times_s[index]:.2f} s, "
            f"error={result.observed_residual.geodesic_deg[index]:.1f}°"
        )
        if plot_index == 1:
            axis.legend(loc="upper left", fontsize=8)
    figure.suptitle("Measured racket orientation in the hand coordinate frame", fontsize=14)
    _save_figure(figure, output_dir, "orientation_keyframes")


def _clip_frame_indices(times_s: np.ndarray, clip: DiagnosticClip, animation_fps: float) -> np.ndarray:
    frame_times = np.arange(clip.start_s, clip.end_s, 1.0 / animation_fps)
    right = np.searchsorted(times_s, frame_times, side="left")
    right = np.clip(right, 0, times_s.size - 1)
    left = np.clip(right - 1, 0, times_s.size - 1)
    choose_left = np.abs(times_s[left] - frame_times) <= np.abs(times_s[right] - frame_times)
    return np.where(choose_left, left, right)


def create_diagnostic_animation(
    result: AnalysisResult,
    clips: Sequence[DiagnosticClip],
    output_path: Path,
    animation_fps: float,
) -> None:
    """Render a hand-fixed 3D diagnostic montage with synchronized traces."""

    if animation_fps <= 0:
        raise ValueError("animation_fps must be positive")
    chapter_indices = [_clip_frame_indices(result.sequence.times_s, clip, animation_fps) for clip in clips]
    frames: list[tuple[int, int]] = [
        (chapter_index, int(frame_index))
        for chapter_index, indices in enumerate(chapter_indices)
        for frame_index in indices
    ]

    figure = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.45))
    pose_axis = figure.add_subplot(grid[:, 0], projection="3d")
    error_axis = figure.add_subplot(grid[0, 1])
    component_axis = figure.add_subplot(grid[1, 1], sharex=error_axis)

    def draw(frame_number: int) -> None:
        chapter_index, sample_index = frames[frame_number]
        clip = clips[chapter_index]
        indices = chapter_indices[chapter_index]
        clip_times = result.sequence.times_s[indices]
        current_time_s = result.sequence.times_s[sample_index]

        pose_axis.clear()
        _draw_racket_pose(
            pose_axis,
            result.observed_translation_m,
            result.observed_rotation,
            color=GRAY,
            linestyle="--",
            label="Expected from observed T-pose",
            alpha=0.72,
        )
        _draw_racket_pose(
            pose_axis,
            result.relative_pose.translations_m[sample_index],
            result.relative_pose.rotations[sample_index],
            color=ORANGE,
            linestyle="-",
            label="Measured racket",
            alpha=1.0,
        )
        _draw_triad(pose_axis, np.zeros(3), Rotation.identity(), 0.13, alpha=0.72)
        _configure_pose_axis(pose_axis)
        pose_axis.set_title(
            "Racket in hand-fixed coordinates\n"
            f"error={result.observed_residual.geodesic_deg[sample_index]:.1f}°, "
            f"origin deviation={1000.0 * result.translation_error_observed_m[sample_index]:.2f} mm"
        )
        pose_axis.legend(loc="upper left", fontsize=8)

        error_axis.clear()
        error_axis.plot(
            clip_times,
            result.observed_residual.geodesic_deg[indices],
            color=BLUE,
            linewidth=1.5,
            label="Observed-baseline error",
        )
        error_axis.plot(
            clip_times,
            result.embedded_residual.geodesic_deg[indices],
            color=ORANGE,
            linewidth=1.2,
            label="Embedded-baseline error",
        )
        error_axis.axvline(current_time_s, color=RED, linewidth=1.2)
        error_axis.set(ylabel="Orientation error [deg]", xlim=(clip.start_s, clip.end_s))
        error_axis.legend(loc="upper left", fontsize=8, ncols=2)

        component_axis.clear()
        component_axis.plot(
            clip_times,
            result.observed_residual.longitudinal_axis_misalignment_deg[indices],
            color=GREEN,
            linewidth=1.4,
            label="Long-axis misalignment",
        )
        component_axis.plot(
            clip_times,
            result.observed_residual.twist_deg[indices],
            color=PURPLE,
            linewidth=1.2,
            label="Signed twist",
        )
        component_axis.axhline(0.0, color=GRAY, linewidth=0.7)
        component_axis.axvline(current_time_s, color=RED, linewidth=1.2)
        component_axis.set(
            xlabel="Time from Xsens stream start [s]",
            ylabel="Orientation component [deg]",
            xlim=(clip.start_s, clip.end_s),
        )
        component_axis.legend(loc="upper left", fontsize=8, ncols=2)
        figure.suptitle(
            f"{chapter_index + 1}/3 — {clip.label}  |  t={current_time_s:.2f} s",
            fontsize=15,
        )

    movie = animation.FuncAnimation(figure, draw, frames=len(frames), interval=1000.0 / animation_fps)
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = animation.FFMpegWriter(
        fps=animation_fps,
        codec="libx264",
        bitrate=2800,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie.save(output_path, writer=writer, dpi=115)
    plt.close(figure)


def _format_event_table(events: Sequence[OrientationEvent], limit: int = 8) -> str:
    ranked = sorted(events, key=lambda event: (event.mean_error_deg, event.duration_s), reverse=True)[:limit]
    if not ranked:
        return "No sustained events met the configured threshold and duration."
    rows = [
        "| Start [s] | End [s] | Duration [s] | Mean [deg] | P95 [deg] | Max [deg] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    rows.extend(
        f"| {event.start_s:.3f} | {event.end_s:.3f} | {event.duration_s:.3f} | "
        f"{event.mean_error_deg:.2f} | {event.p95_error_deg:.2f} | {event.max_error_deg:.2f} |"
        for event in ranked
    )
    return "\n".join(rows)


def _format_clip_table(clips: Sequence[DiagnosticClip]) -> str:
    rows = [
        "| Chapter | Start [s] | End [s] | Selection |",
        "|---|---:|---:|---|",
    ]
    rows.extend(f"| {clip.label} | {clip.start_s:.3f} | {clip.end_s:.3f} | {clip.selection_reason} |" for clip in clips)
    return "\n".join(rows)


def write_report(
    result: AnalysisResult,
    summary: dict[str, Any],
    events: Sequence[OrientationEvent],
    clips: Sequence[DiagnosticClip],
    output_path: Path,
) -> None:
    activity = summary["activity_metrics"]
    observed = activity["orientation_error_observed_baseline_deg"]
    embedded = activity["orientation_error_embedded_baseline_deg"]
    translation = activity["translation_error_observed_baseline_m"]
    coupling = activity["angular_speed_coupling"]
    thresholds = activity["time_above_thresholds"]
    sensor = summary["sensor_validation"]
    event_threshold_deg = float(summary["event_detection"]["threshold_deg"])
    event_min_duration_s = float(summary["event_detection"]["minimum_duration_s"])
    sensor_text = "The raw-sensor cross-check was unavailable."
    if sensor["available"]:
        disagreement = sensor["absolute_angle_disagreement_deg"]
        sensor_text = (
            f"The raw hand/sword sensor relative-angle signal tracks the segment-derived signal with "
            f"r={sensor['relative_angle_correlation']:.4f}. Their absolute angle disagreement is "
            f"{disagreement['median']:.3f}° at the median and {disagreement['p95']:.3f}° at P95. This strongly "
            "indicates that the time-varying orientation difference is present in the sensor data rather than "
            "being caused only by the segment model or quaternion sign choices."
        )

    report = f"""# Xsens hand-racket relative motion analysis

## Scope

- Recording: `{result.sequence.source_path.name}`
- Samples: {summary["sample_count"]:,} at a median {summary["native_fps"]:.3f} Hz
- Duration: {summary["duration_s"]:.3f} s ({summary["duration_s"] / 60.0:.2f} min)
- Observed baseline: first logged good T-pose,
  {result.sequence.observed_baseline.start_s:.3f}-{result.sequence.observed_baseline.end_s:.3f} s
- Primary interval: inferred tennis/activity interval,
  {result.sequence.activity.start_s:.3f}-{result.sequence.activity.end_s:.3f} s

`RightHandSword` is treated as the tracked tennis-racket frame. The central interval is inferred from
calibration boundaries because the activity annotation stream is empty.

## Main findings

1. **The embedded and observed grip calibrations differ.** The embedded Xsens T-pose encodes the racket
   frame at {np.rad2deg(result.embedded_rotation.magnitude()):.2f}° relative to the hand (nominally the known
   -90° longitudinal roll). The mean first good T-pose is
   {summary["observed_vs_embedded_orientation_deg"]:.2f}° away from that embedded relationship.
2. **The racket orientation does not remain fixed to the observed hand-frame baseline.** During the primary
   interval, the quaternion-geodesic error is {observed["median"]:.2f}° at the median,
   {observed["p95"]:.2f}° at P95, {observed["p99"]:.2f}° at P99, and {observed["max"]:.2f}° at the maximum.
   Against the embedded T-pose the corresponding median/P95 are
   {embedded["median"]:.2f}°/{embedded["p95"]:.2f}°.
3. **Large differences are sustained, not only isolated spikes.** The observed-baseline error is at least 30°
   for {100.0 * thresholds["ge_30_deg"]["frame_fraction"]:.1f}% of the primary interval, at least 60° for
   {100.0 * thresholds["ge_60_deg"]["frame_fraction"]:.1f}%, and at least 90° for
   {100.0 * thresholds["ge_90_deg"]["frame_fraction"]:.1f}%. There are {len(events)} events above the
   configured {event_threshold_deg:g}° threshold lasting at least {event_min_duration_s:g} s.
4. **The racket origin is effectively rigid in the hand frame.** Translation deviation from the observed
   good-T-pose mean is {1000.0 * translation["median"]:.4f} mm at the median,
   {1000.0 * translation["p95"]:.4f} mm at P95, and {1000.0 * translation["max"]:.3f} mm at the maximum.
   This is consistent with the Xsens prop origin being kinematically attached to the hand; it should not be
   interpreted as an independent measurement of grip translation.
5. **Overall motion remains strongly coupled.** Hand and racket angular-speed correlation is
   {coupling["zero_lag_correlation"]:.4f} at zero lag. The best correlation within ±250 ms is
   {coupling["best_lag_correlation"]:.4f} at
   {1000.0 * coupling["best_lag_s_positive_means_racket_follows_hand"]:.1f} ms (positive means the racket
   follows the hand).
6. **The raw sensors corroborate the relative-angle change.** {sensor_text}

The combination of nearly fixed relative position and large, sometimes near-180° relative orientation changes
deserves sensor/calibration scrutiny. Xsens alone cannot distinguish intentional motion inside the grip from
prop-sensor mounting movement, magnetic disturbance, or orientation-estimation drift. The sustained extreme
episodes are larger than ordinary wrist motion transmitted through a rigid tennis grip.

## Method

For every frame, the racket origin and orientation are expressed in the hand frame:

- `p_relative = R_hand⁻¹ (p_racket - p_hand)`
- `R_relative = R_hand⁻¹ R_racket`

Orientation differences use SO(3) geodesic angles, so quaternion sign flips and Euler-angle wrapping cannot create
false discontinuities. The observed baseline is the quaternion mean across the first good T-pose. The
longitudinal-axis measure is the change of the racket +X direction; signed twist is the quaternion swing-twist
component about that axis. Primary statistics use native, unsmoothed samples. Plot lines are temporal medians with
explicitly labeled percentile bands.

The raw-sensor validation compares the **magnitude** of relative-orientation change after separate sensor and
segment baselines. This angle is invariant to the fixed sensor-to-segment mounting rotations; their vector
components are not directly comparable without those mounting transforms.

## Strongest sustained orientation events

{_format_event_table(events)}

## Diagnostic animation chapters

{_format_clip_table(clips)}

[Open the diagnostic animation](relative_orientation_diagnostic.mp4)

## Figures

![Complete recording overview](orientation_overview.png)

![Activity diagnostics](relative_motion_diagnostics.png)

![Phase distributions](phase_distributions.png)

![Raw-sensor cross-check](sensor_crosscheck.png)

![Hand-fixed orientation keyframes](orientation_keyframes.png)

## Machine-readable outputs

- [`summary.json`](summary.json): baselines, distribution summaries, coupling, thresholds, events, and clip selections.
- [`frame_metrics.csv`](frame_metrics.csv): native-rate relative pose and error signals.
- [`orientation_events.csv`](orientation_events.csv): sustained ≥{event_threshold_deg:g}° activity events.

## Limitations

- The HDF5 activity stream contains no labeled tennis strokes, so results are not segmented into forehands,
  backhands, or serves.
- Segment positions and orientations are Xsens MVN outputs, not independent optical ground truth.
- The raw sensor comparison confirms the dynamic relative-angle signal but does not identify whether its cause is
  physical grip motion, sensor mounting, magnetic disturbance, or fusion drift.
- The separate camera/composite recording is intentionally not synchronized here because its duration differs and
  the file contains no explicit cross-modal synchronization marker.
"""
    output_path.write_text(report, encoding="utf-8")


def run_analysis(config: Config) -> dict[str, Any]:
    """Run computation and write the complete requested artifact package."""

    if config.animation_fps <= 0:
        raise ValueError("animation_fps must be positive")
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    sequence = load_sequence(config)
    result = analyze_sequence(sequence)
    events = detect_orientation_events(
        result,
        threshold_deg=config.event_threshold_deg,
        min_duration_s=config.event_min_duration_s,
        max_gap_s=config.event_max_gap_s,
    )
    clips = select_diagnostic_clips(result, config.animation_clip_duration_s)
    summary = build_summary(result, events, clips, config)

    _write_frame_metrics(result, output_dir / "frame_metrics.csv")
    _write_events(events, output_dir / "orientation_events.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_overview(result, events, output_dir)
    plot_activity_diagnostics(result, events, output_dir, config.event_threshold_deg)
    plot_phase_distributions(result, events, output_dir, config.event_threshold_deg)
    plot_sensor_crosscheck(result, output_dir)
    plot_orientation_keyframes(result, clips, output_dir)
    create_diagnostic_animation(
        result,
        clips,
        output_dir / "relative_orientation_diagnostic.mp4",
        config.animation_fps,
    )
    write_report(result, summary, events, clips, output_dir / "analysis_report.md")
    return summary


def main(config: Config) -> None:
    summary = run_analysis(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
