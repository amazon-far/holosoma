#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.src.recording_utils import build_record_frame_indices, record_viser_sequence  # noqa: E402
from holosoma_retargeting.src.viser_utils import QposViserApplier  # noqa: E402


@dataclass
class BatchViserRecordConfig:
    """Record every qpos .npz in a directory through one Viser browser session."""

    npz_dir: str
    """Directory containing qpos .npz files."""

    robot_urdf: str = "models/g1/g1_29dof.urdf"
    """Path to robot URDF file."""

    object_urdf: str | None = None
    """Path to object URDF file, if qpos contains an object pose."""

    assume_object_in_qpos: bool = False
    """Whether object pose is included at the end of qpos."""

    glob_pattern: str = "*.npz"
    """Glob pattern used to choose samples within npz_dir."""

    output_dir: str | None = None
    """Directory for videos. Defaults to npz_dir."""

    output_extension: str = ".mp4"
    """Video extension to append to each sample stem."""

    overwrite: bool = False
    """Overwrite existing videos instead of skipping them."""

    limit: int | None = None
    """Optional maximum number of samples to record."""

    continue_on_error: bool = False
    """Keep recording later samples if one sample fails."""

    grid_width: float = 8.0
    """Grid width for visualization."""

    grid_height: float = 8.0
    """Grid height for visualization."""

    show_meshes: bool = True
    """Whether to show mesh visualizations."""

    fps: int = 30
    """Fallback FPS when a sample does not contain an fps field."""

    record_width: int = 1280
    """Rendered recording width in pixels."""

    record_height: int = 720
    """Rendered recording height in pixels."""

    record_fps: int | None = None
    """FPS for the recorded video. Defaults to each sample's fps, or fps."""

    record_start_frame: int = 0
    """First source frame to record."""

    record_end_frame: int | None = None
    """Last source frame to record, inclusive. Defaults to the final frame."""

    record_stride: int = 1
    """Source-frame stride for recording."""

    record_connect_timeout: float = 120.0
    """Seconds to wait for a browser client before recording."""

    record_start_delay: float = 3.0
    """Seconds to wait after the first browser connection before recording."""

    record_start_delay_each: bool = False
    """Apply record_start_delay before every sample instead of only the first."""

    record_between_delay: float = 0.0
    """Seconds to wait between samples after the first."""

    record_settle_time: float = 0.0
    """Seconds to wait after each scene update before capturing a rendered frame."""

    record_warmup_renders: int = 0
    """Number of throwaway renders after each scene update before saving the frame."""

    record_transport_format: str = "jpeg"
    """Browser render transport format: jpeg or png."""

    dry_run: bool = False
    """Print the samples and target video paths without starting Viser."""


def _resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate

    package_root = Path(__file__).resolve().parents[1]
    package_candidate = package_root / candidate
    if package_candidate.exists():
        return package_candidate

    return candidate


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def _load_urdf_with_relative_assets(urdf_path: Path) -> yourdfpy.URDF:
    with _working_directory(urdf_path.parent):
        return yourdfpy.URDF.load(urdf_path.name, load_meshes=True, build_scene_graph=True)


def _load_npz(npz_path: Path, fallback_fps: int) -> tuple[np.ndarray, float]:
    data = np.load(npz_path, allow_pickle=True)
    if "qpos" not in data:
        raise KeyError(f"{npz_path} does not contain a 'qpos' array.")

    qpos = np.asarray(data["qpos"])
    if qpos.ndim != 2:
        raise ValueError(f"{npz_path} qpos must be rank 2, got shape {qpos.shape}.")
    if qpos.shape[0] <= 0:
        raise ValueError(f"{npz_path} contains no frames.")

    if "fps" not in data:
        return qpos, float(fallback_fps)

    fps_arr = np.asarray(data["fps"]).reshape(-1)
    if fps_arr.size == 0:
        return qpos, float(fallback_fps)
    return qpos, float(fps_arr[0])


def _find_samples(config: BatchViserRecordConfig) -> list[Path]:
    npz_dir = Path(config.npz_dir).expanduser()
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"npz_dir is not a directory: {npz_dir}")

    samples = sorted(p for p in npz_dir.glob(config.glob_pattern) if p.is_file())
    if config.limit is not None:
        samples = samples[: max(0, int(config.limit))]
    if not samples:
        raise FileNotFoundError(f"No samples matched {config.glob_pattern!r} in {npz_dir}")
    return samples


def _output_path(sample_path: Path, config: BatchViserRecordConfig) -> Path:
    output_dir = Path(config.output_dir).expanduser() if config.output_dir else sample_path.parent
    extension = config.output_extension
    if not extension.startswith("."):
        extension = f".{extension}"
    return output_dir / f"{sample_path.stem}{extension}"


def _print_plan(samples: list[Path], config: BatchViserRecordConfig) -> None:
    output_dir = Path(config.output_dir).expanduser() if config.output_dir else samples[0].parent
    print(f"[batch_record] Samples: {len(samples)}")
    print(f"[batch_record] Input directory: {Path(config.npz_dir).expanduser()}")
    print(f"[batch_record] Output directory: {output_dir}")
    for sample in samples:
        print(f"  {sample.name} -> {_output_path(sample, config).name}")


def main(config: BatchViserRecordConfig) -> None:
    samples = _find_samples(config)
    _print_plan(samples, config)
    if config.dry_run:
        return

    robot_urdf = _resolve_path(config.robot_urdf)
    object_urdf = _resolve_path(config.object_urdf) if config.object_urdf else None

    server = viser.ViserServer()
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)
    server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height, position=(0.0, 0.0, 0.0))

    robot_urdf_y = _load_urdf_with_relative_assets(robot_urdf)
    viser_robot = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    viser_object = None
    if object_urdf is not None:
        object_urdf_y = _load_urdf_with_relative_assets(object_urdf)
        viser_object = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

    viser_robot.show_visual = config.show_meshes
    if viser_object is not None:
        viser_object.show_visual = config.show_meshes

    with server.gui.add_folder("Batch"):
        status_text = server.gui.add_text("Status", initial_value="Waiting to record")
        current_sample_text = server.gui.add_text("Current sample", initial_value="")

    robot_dof = len(viser_robot.get_actuated_joint_limits())
    applier = QposViserApplier(
        viser_robot=viser_robot,
        robot_base_frame=robot_root,
        robot_dof=robot_dof,
        viser_object=viser_object if config.assume_object_in_qpos else None,
        object_base_frame=object_root if config.assume_object_in_qpos else None,
        contains_object_in_qpos=config.assume_object_in_qpos,
    )

    print("[batch_record] Open the Viser URL printed above once. The same browser view will record all samples.")

    recorded = 0
    skipped = 0
    failed = 0
    for index, sample_path in enumerate(samples, start=1):
        output_path = _output_path(sample_path, config)
        if output_path.exists() and not config.overwrite:
            skipped += 1
            print(f"[batch_record] Skipping existing video: {output_path}")
            continue

        try:
            qpos, sample_fps = _load_npz(sample_path, fallback_fps=config.fps)
            frame_indices = build_record_frame_indices(
                n_frames=int(qpos.shape[0]),
                start_frame=config.record_start_frame,
                end_frame=config.record_end_frame,
                stride=config.record_stride,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            applier.reset_quat_continuity()
            applier.apply_frame(qpos, frame_indices[0])
            current_sample_text.value = sample_path.name
            status_text.value = f"Recording {index}/{len(samples)}"

            if recorded > 0 and config.record_between_delay > 0.0:
                time.sleep(config.record_between_delay)

            start_delay = config.record_start_delay if (recorded == 0 or config.record_start_delay_each) else 0.0
            print(f"[batch_record] Recording {index}/{len(samples)}: {sample_path.name} -> {output_path.name}")
            record_viser_sequence(
                server=server,
                apply_frame=lambda frame_idx, qpos=qpos: applier.apply_frame(qpos, frame_idx),
                frame_indices=frame_indices,
                output_path=str(output_path),
                width=config.record_width,
                height=config.record_height,
                fps=float(config.record_fps if config.record_fps is not None else sample_fps),
                connect_timeout=config.record_connect_timeout,
                start_delay=start_delay,
                settle_time=config.record_settle_time,
                warmup_renders=config.record_warmup_renders,
                transport_format=config.record_transport_format,
            )
            recorded += 1
        except Exception as exc:
            failed += 1
            status_text.value = f"Failed on {sample_path.name}"
            print(f"[batch_record] ERROR while recording {sample_path}: {exc}")
            if not config.continue_on_error:
                raise

    status_text.value = f"Done: {recorded} recorded, {skipped} skipped, {failed} failed"
    print(f"[batch_record] Done. Recorded={recorded}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main(tyro.cli(BatchViserRecordConfig))
