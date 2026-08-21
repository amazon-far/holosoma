#!/usr/bin/env python3
"""Analyze achieved G1 racket orientation against the symmetry-aware Xsens target."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/holosoma-matplotlib")

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib import animation
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from holosoma_retargeting.data_utils.xsens_hdf5 import load_xsens_hdf5_motion
from holosoma_retargeting.transformation_utils import rotations_from_wxyz
from holosoma_retargeting.xsens.tennis_racket import (
    build_tennis_racket_targets,
    load_retargeting_result,
)

__all__ = ["Config", "RacketTargetErrorAnalysis", "analyze", "main"]


@dataclass(frozen=True)
class Config:
    """Inputs and rendering controls for achieved-racket target-error analysis."""

    xsens_hdf5: Path
    retargeted_npz: Path
    output_dir: Path
    source_frame_start: int = 0
    animation_fps: float = 15.0
    animation_duration_s: float = 90.0
    local_window_s: float = 10.0
    entry_error_deg: float = 45.0
    exit_error_deg: float = 60.0
    min_wrist_margin_deg: float = 5.0

    def __post_init__(self) -> None:
        if self.animation_fps <= 0.0 or self.animation_duration_s <= 0.0:
            raise ValueError("Animation FPS and duration must be positive")
        if self.local_window_s <= 0.0:
            raise ValueError("local_window_s must be positive")
        if self.exit_error_deg < self.entry_error_deg:
            raise ValueError("exit_error_deg must be at least entry_error_deg")
        if self.source_frame_start < 0:
            raise ValueError("source_frame_start must be non-negative")


@dataclass(frozen=True)
class RacketTargetErrorAnalysis:
    """Aligned source targets, achieved orientations, and saved tracker diagnostics."""

    times_s: np.ndarray
    error_deg: np.ndarray
    candidate_error_deg: np.ndarray
    nearest_symmetry_branch: np.ndarray
    target_rotations: np.ndarray
    achieved_rotations: np.ndarray
    tracking_state: np.ndarray
    selected_symmetry_branch: np.ndarray
    wrist_margin_deg: np.ndarray
    source_origin_deviation_m: np.ndarray


def _symmetry_errors(
    achieved_rotations: np.ndarray,
    candidate_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = achieved_rotations.shape[0]
    achieved_repeated = np.repeat(achieved_rotations, 2, axis=0)
    relative = (
        Rotation.from_matrix(candidate_rotations.reshape(-1, 3, 3)) * Rotation.from_matrix(achieved_repeated).inv()
    )
    candidate_error_deg = np.rad2deg(relative.magnitude()).reshape(frame_count, 2)
    nearest_branch = np.argmin(candidate_error_deg, axis=1)
    return candidate_error_deg, nearest_branch


def analyze(config: Config) -> RacketTargetErrorAnalysis:
    """Load aligned source/result motion and independently recompute symmetry-aware error."""

    result = load_retargeting_result(config.retargeted_npz)
    racket = result.tennis_racket
    if racket is None:
        raise ValueError("The retargeted NPZ does not contain saved tennis-racket motion")
    frame_count = result.qpos.shape[0]
    source = load_xsens_hdf5_motion(
        config.xsens_hdf5,
        target_fps=result.fps,
        frame_start=config.source_frame_start,
        max_frames=frame_count,
        include_tracked_props=True,
    )
    if source.positions_m.shape[0] != frame_count:
        raise ValueError(f"Source/result frame counts differ: {source.positions_m.shape[0]} vs {frame_count}")
    targets = build_tennis_racket_targets(source, racket.attachment)
    achieved_rotations = rotations_from_wxyz(racket.quaternion_wxyz).as_matrix()
    candidate_error_deg, nearest_branch = _symmetry_errors(
        achieved_rotations,
        targets.candidate_racket_rotations,
    )
    error_deg = candidate_error_deg[np.arange(frame_count), nearest_branch]
    np.testing.assert_allclose(
        error_deg,
        np.rad2deg(racket.target_error_rad),
        atol=1e-6,
        err_msg="Independently recomputed racket error differs from the saved diagnostic",
    )
    target_rotations = targets.candidate_racket_rotations[np.arange(frame_count), nearest_branch]
    return RacketTargetErrorAnalysis(
        times_s=(
            np.asarray(source.recording_times_s, dtype=float)
            if source.recording_times_s is not None
            else np.arange(frame_count, dtype=float) / result.fps
        ),
        error_deg=error_deg,
        candidate_error_deg=candidate_error_deg,
        nearest_symmetry_branch=nearest_branch,
        target_rotations=target_rotations,
        achieved_rotations=achieved_rotations,
        tracking_state=np.asarray(racket.tracking_state, dtype=str),
        selected_symmetry_branch=np.asarray(racket.symmetry_branch, dtype=np.int8),
        wrist_margin_deg=np.rad2deg(racket.min_wrist_limit_margin_rad),
        source_origin_deviation_m=np.asarray(racket.source_origin_deviation_m, dtype=float),
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
    }


def _summary(data: RacketTargetErrorAnalysis) -> dict[str, Any]:
    states, counts = np.unique(data.tracking_state, return_counts=True)
    state_metrics = {str(state): _distribution(data.error_deg[data.tracking_state == state]) for state in states}
    return {
        "frame_count": int(data.error_deg.size),
        "duration_s": float(data.times_s[-1] - data.times_s[0]) if data.times_s.size else 0.0,
        "symmetry_aware_error_deg": _distribution(data.error_deg),
        "coverage_percent": {
            str(threshold): float(100.0 * np.mean(data.error_deg <= threshold)) for threshold in (30, 45, 60, 75)
        },
        "minimum_wrist_margin_deg": float(np.min(data.wrist_margin_deg)),
        "wrist_margin_below_5deg_percent": float(100.0 * np.mean(data.wrist_margin_deg < 5.0)),
        "tracking_state_counts": {str(state): int(count) for state, count in zip(states, counts, strict=True)},
        "error_by_tracking_state_deg": state_metrics,
    }


def _write_frame_metrics(data: RacketTargetErrorAnalysis, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "frame",
                "time_s",
                "symmetry_aware_error_deg",
                "candidate_0_error_deg",
                "candidate_1_error_deg",
                "nearest_symmetry_branch",
                "selected_symmetry_branch",
                "tracking_state",
                "wrist_margin_deg",
                "source_origin_deviation_m",
            )
        )
        for frame in range(data.error_deg.size):
            writer.writerow(
                (
                    frame,
                    data.times_s[frame],
                    data.error_deg[frame],
                    data.candidate_error_deg[frame, 0],
                    data.candidate_error_deg[frame, 1],
                    data.nearest_symmetry_branch[frame],
                    data.selected_symmetry_branch[frame],
                    data.tracking_state[frame],
                    data.wrist_margin_deg[frame],
                    data.source_origin_deviation_m[frame],
                )
            )


def _downsample_indices(frame_count: int, maximum: int = 12000) -> np.ndarray:
    if frame_count <= maximum:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, maximum, dtype=int))


def _plot_static(data: RacketTargetErrorAnalysis, config: Config, path: Path) -> None:
    indices = _downsample_indices(data.error_deg.size)
    time_min = data.times_s[indices] / 60.0
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    axes[0].plot(time_min, data.error_deg[indices], linewidth=0.7, color="#0072B2")
    axes[0].axhline(config.entry_error_deg, color="#E69F00", linewidth=1.0, label="Entry threshold")
    axes[0].axhline(config.exit_error_deg, color="#D55E00", linewidth=1.0, label="Exit threshold")
    axes[0].set_ylabel("Symmetry-aware\nerror [deg]")
    axes[0].legend()
    axes[1].plot(time_min, data.wrist_margin_deg[indices], linewidth=0.7, color="#009E73")
    axes[1].axhline(config.min_wrist_margin_deg, color="#D55E00", linewidth=1.0)
    axes[1].set_ylabel("Minimum wrist\nmargin [deg]")
    axes[2].plot(time_min, 100.0 * data.source_origin_deviation_m[indices], linewidth=0.7, color="#CC79A7")
    axes[2].set_ylabel("Source-origin\ndeviation [cm]")
    axes[2].set_xlabel("Recording time [min]")
    figure.suptitle("Actual G1 tennis racket vs. Xsens racket orientation target")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _draw_triad(
    axis: Axes,
    rotation: np.ndarray,
    *,
    linestyle: str,
    label: str,
) -> None:
    colors = ("#D55E00", "#009E73", "#0072B2")
    for direction, color in zip(rotation.T, colors, strict=True):
        axis.plot(
            [0.0, direction[0]],
            [0.0, direction[1]],
            [0.0, direction[2]],
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
        )
    axis.plot([], [], [], color="black", linestyle=linestyle, label=label)


def _animation_frame_indices(data: RacketTargetErrorAnalysis, config: Config) -> np.ndarray:
    count = min(data.error_deg.size, max(2, round(config.animation_fps * config.animation_duration_s)))
    return np.unique(np.linspace(0, data.error_deg.size - 1, count, dtype=int))


def _create_animation(data: RacketTargetErrorAnalysis, config: Config, path: Path) -> None:
    frames = _animation_frame_indices(data, config)
    overview_indices = _downsample_indices(data.error_deg.size)
    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(2.2, 1.0))
    overview = figure.add_subplot(grid[0, 0])
    local = figure.add_subplot(grid[1, 0])
    triad = figure.add_subplot(grid[:, 1], projection="3d")
    time_min = data.times_s[overview_indices] / 60.0
    overview.plot(time_min, data.error_deg[overview_indices], color="#0072B2", linewidth=0.65)
    overview.axhline(config.entry_error_deg, color="#E69F00", linewidth=0.9)
    overview.axhline(config.exit_error_deg, color="#D55E00", linewidth=0.9)
    overview.set_ylabel("Symmetry-aware error [deg]")
    overview.set_xlabel("Recording time [min]")
    overview.set_xlim(float(time_min[0]), float(time_min[-1]))
    overview.set_ylim(0.0, max(90.0, float(np.percentile(data.error_deg, 99.5)) * 1.05))
    cursor = overview.axvline(0.0, color="black", linewidth=1.3)
    (local_line,) = local.plot([], [], color="#0072B2", linewidth=1.2)
    local.axhline(config.entry_error_deg, color="#E69F00", linewidth=0.9)
    local.axhline(config.exit_error_deg, color="#D55E00", linewidth=0.9)
    local.set_xlabel("Recording time [s]")
    local.set_ylabel("Local error [deg]")
    status = figure.suptitle("")

    def draw(animation_index: int) -> tuple[Any, ...]:
        frame = int(frames[animation_index])
        time_s = float(data.times_s[frame])
        cursor.set_xdata([time_s / 60.0, time_s / 60.0])
        half_window = 0.5 * config.local_window_s
        mask = (data.times_s >= time_s - half_window) & (data.times_s <= time_s + half_window)
        local_line.set_data(data.times_s[mask], data.error_deg[mask])
        local.set_xlim(
            max(float(data.times_s[0]), time_s - half_window), min(float(data.times_s[-1]), time_s + half_window)
        )
        local.set_ylim(0.0, max(90.0, float(np.max(data.error_deg[mask])) * 1.08))
        triad.cla()
        _draw_triad(triad, data.target_rotations[frame], linestyle="-", label="Nearest Xsens target")
        _draw_triad(triad, data.achieved_rotations[frame], linestyle="--", label="Achieved G1 racket")
        triad.set_xlim(-1.05, 1.05)
        triad.set_ylim(-1.05, 1.05)
        triad.set_zlim(-1.05, 1.05)
        triad.set_box_aspect((1.0, 1.0, 1.0))
        triad.set_xlabel("World X")
        triad.set_ylabel("World Y")
        triad.set_zlabel("World Z")
        triad.legend(loc="upper left", fontsize=8)
        status.set_text(
            f"t={time_s:.2f} s  |  error={data.error_deg[frame]:.1f}°  |  "
            f"state={data.tracking_state[frame]}  |  wrist margin={data.wrist_margin_deg[frame]:.1f}°  |  "
            f"target branch={data.nearest_symmetry_branch[frame]}"
        )
        return cursor, local_line, status

    movie = animation.FuncAnimation(
        figure,
        draw,
        frames=len(frames),
        interval=1000.0 / config.animation_fps,
    )
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = animation.FFMpegWriter(fps=config.animation_fps, codec="libx264", bitrate=4000)
    movie.save(path, writer=writer, dpi=110)
    plt.close(figure)


def main(config: Config) -> None:
    data = analyze(config)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(data)
    summary.update(
        {
            "xsens_hdf5": str(config.xsens_hdf5.expanduser().resolve()),
            "retargeted_npz": str(config.retargeted_npz.expanduser().resolve()),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_frame_metrics(data, output_dir / "frame_metrics.csv")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})
    _plot_static(data, config, output_dir / "racket_target_error.png")
    _create_animation(data, config, output_dir / "racket_target_error.mp4")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
