"""Render synchronized G1 tennis-racket retargeting results side by side.

Example:
    MUJOCO_GL=egl python examples/xsens_tennis/compare_tennis_racket_modes.py \
        --source-hdf5 demo_data/xsens_tennis/recording.hdf5 \
        --frame-start 1000 \
        --result "Hand|demo_results/hand/recording.npz" \
        --result "Racket|demo_results/racket/recording.npz" \
        --result "Filtered|demo_results/filtered/recording.npz" \
        --output comparison.mp4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import mujoco  # type: ignore[import-not-found]
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from holosoma_retargeting.data_utils.xsens_hdf5 import load_xsens_hdf5_motion
from holosoma_retargeting.transformation_utils import rotations_from_wxyz
from holosoma_retargeting.xsens.tennis_racket import (
    RetargetingResult,
    TennisRacketTargets,
    build_tennis_racket_targets,
    load_retargeting_result,
)


@dataclass(frozen=True)
class LabeledResult:
    """One result and its source-orientation targets."""

    label: str
    path: Path
    result: RetargetingResult
    targets: TennisRacketTargets


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "|" not in value:
        raise argparse.ArgumentTypeError("Results must use LABEL|PATH syntax")
    label, raw_path = value.split("|", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label.strip() or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid result label or file: {value}")
    return label.strip(), path


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _add_connector(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    *,
    radius: float,
    rgba: Sequence[float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo visualization scene has no free geometry slots")
    geometry = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geometry,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geometry,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    scene.ngeom += 1


def _world_point(position: np.ndarray, rotation: np.ndarray, local_point: Sequence[float]) -> np.ndarray:
    return position + rotation @ np.asarray(local_point, dtype=float)


def _add_racket(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rotation: np.ndarray,
) -> None:
    frame_rgba = (0.95, 0.33, 0.08, 1.0)
    grip_rgba = (0.10, 0.10, 0.12, 1.0)
    string_rgba = (0.85, 0.87, 0.90, 0.85)

    def connector(start: Sequence[float], end: Sequence[float], radius: float, rgba: Sequence[float]) -> None:
        _add_connector(
            scene,
            _world_point(position, rotation, start),
            _world_point(position, rotation, end),
            radius=radius,
            rgba=rgba,
        )

    connector((-0.09, 0.0, 0.0), (0.09, 0.0, 0.0), 0.018, grip_rgba)
    connector((0.09, 0.0, 0.0), (0.25, 0.0, 0.0), 0.009, frame_rgba)
    connector((0.16, 0.0, 0.0), (0.27, 0.0, 0.075), 0.008, frame_rgba)
    connector((0.16, 0.0, 0.0), (0.27, 0.0, -0.075), 0.008, frame_rgba)

    center_x, radius_x, radius_z = 0.415, 0.175, 0.135
    angles = np.linspace(0.0, 2.0 * np.pi, 25)
    hoop = np.column_stack(
        (
            center_x + radius_x * np.cos(angles),
            np.zeros_like(angles),
            radius_z * np.sin(angles),
        )
    )
    for start, end in zip(hoop[:-1], hoop[1:], strict=True):
        connector(start, end, 0.009, frame_rgba)
    for x_offset in np.linspace(-0.13, 0.13, 7):
        z_extent = radius_z * np.sqrt(max(0.0, 1.0 - (x_offset / radius_x) ** 2)) * 0.90
        connector(
            (center_x + x_offset, 0.0, -z_extent),
            (center_x + x_offset, 0.0, z_extent),
            0.001,
            string_rgba,
        )
    for z_offset in np.linspace(-0.10, 0.10, 5):
        x_extent = radius_x * np.sqrt(max(0.0, 1.0 - (z_offset / radius_z) ** 2)) * 0.90
        connector(
            (center_x - x_extent, 0.0, z_offset),
            (center_x + x_extent, 0.0, z_offset),
            0.001,
            string_rgba,
        )


def _add_target_axes(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rotation: np.ndarray,
) -> None:
    colors = (
        (0.95, 0.12, 0.12, 0.95),
        (0.12, 0.90, 0.25, 0.95),
        (0.20, 0.45, 1.00, 0.95),
    )
    lengths = (0.31, 0.20, 0.20)
    for axis, (color, length) in enumerate(zip(colors, lengths, strict=True)):
        endpoint = position + rotation[:, axis] * length
        _add_connector(scene, position, endpoint, radius=0.005, rgba=color)


def _nearest_target_rotation(
    achieved_quaternion_wxyz: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    achieved = rotations_from_wxyz(achieved_quaternion_wxyz)
    candidate_rotations = Rotation.from_matrix(candidates)
    errors = (candidate_rotations * achieved.inv()).magnitude()
    return candidates[int(np.argmin(errors))]


def _panel_border_color(state: str, wrist_margin_deg: float) -> tuple[int, int, int]:
    if wrist_margin_deg < 5.0:
        return (235, 82, 82)
    if state == "racket":
        return (57, 201, 105)
    if state in {"hand", "reentry_hysteresis"}:
        return (245, 184, 65)
    return (235, 82, 82)


def _frame_right_arm_camera(
    camera: mujoco.MjvCamera,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    racket_position: np.ndarray,
    racket_rotation: np.ndarray,
) -> None:
    """Place a torso-relative front-right camera around the right arm and racket."""

    body_positions = [
        np.asarray(data.xpos[model.body(name).id], dtype=float)
        for name in (
            "torso_link",
            "right_shoulder_pitch_link",
            "right_elbow_link",
            "right_rubber_hand_link",
        )
    ]
    racket_head = _world_point(racket_position, racket_rotation, (0.59, 0.0, 0.0))
    focus_points = np.stack((*body_positions, racket_position, racket_head), axis=0)
    bounds_minimum = np.min(focus_points, axis=0)
    bounds_maximum = np.max(focus_points, axis=0)
    camera.lookat[:] = 0.5 * (bounds_minimum + bounds_maximum)

    torso_rotation = np.asarray(data.xmat[model.body("torso_link").id], dtype=float).reshape(3, 3)
    forward_xy = torso_rotation[:2, 0]
    if np.linalg.norm(forward_xy) > 1e-6:
        torso_forward_yaw_deg = float(np.rad2deg(np.arctan2(forward_xy[1], forward_xy[0])))
        # MuJoCo azimuth zero places the camera on world -X. Adding 135° to
        # the torso's forward yaw therefore gives a front-right three-quarter view.
        camera.azimuth = torso_forward_yaw_deg + 135.0
    camera.elevation = -8.0
    bounds_radius = float(np.max(np.linalg.norm(focus_points - camera.lookat[None, :], axis=1)))
    camera.distance = float(np.clip(3.0 * bounds_radius, 1.35, 2.15))


def _render_panel(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    labeled: LabeledResult,
    frame_index: int,
    *,
    panel_width: int,
    panel_height: int,
) -> Image.Image:
    racket = labeled.result.tennis_racket
    assert racket is not None
    data.qpos[:] = labeled.result.qpos[frame_index]
    mujoco.mj_forward(model, data)
    achieved_rotation = rotations_from_wxyz(racket.quaternion_wxyz[frame_index]).as_matrix()
    achieved_position = racket.position_m[frame_index]
    _frame_right_arm_camera(camera, model, data, achieved_position, achieved_rotation)
    renderer.update_scene(data, camera=camera)
    _add_racket(renderer.scene, achieved_position, achieved_rotation)
    target_rotation = _nearest_target_rotation(
        racket.quaternion_wxyz[frame_index],
        labeled.targets.candidate_racket_rotations[frame_index],
    )
    _add_target_axes(renderer.scene, achieved_position, target_rotation)
    panel = Image.fromarray(renderer.render()).resize((panel_width, panel_height))
    draw = ImageDraw.Draw(panel, "RGBA")
    state = str(racket.tracking_state[frame_index])
    error_deg = float(np.rad2deg(racket.target_error_rad[frame_index]))
    margin_deg = float(np.rad2deg(racket.min_wrist_limit_margin_rad[frame_index]))
    draw.rounded_rectangle((9, 9, panel_width - 9, 83), radius=10, fill=(0, 0, 0, 175))
    draw.text((22, 17), labeled.label, font=_font(23), fill=(255, 255, 255))
    draw.text(
        (22, 48),
        f"{state}   error {error_deg:5.1f}°   wrist margin {margin_deg:4.1f}°",
        font=_font(16),
        fill=(232, 236, 242),
    )
    border = _panel_border_color(state, margin_deg)
    draw.rectangle((2, 2, panel_width - 3, panel_height - 3), outline=border, width=5)
    return panel


def _summarize(labeled: LabeledResult) -> dict[str, Any]:
    racket = labeled.result.tennis_racket
    assert racket is not None
    error_deg = np.rad2deg(racket.target_error_rad)
    margin_deg = np.rad2deg(racket.min_wrist_limit_margin_rad)
    states, counts = np.unique(racket.tracking_state, return_counts=True)
    return {
        "label": labeled.label,
        "path": str(labeled.path),
        "frames": int(error_deg.size),
        "error_deg": {
            "mean": float(np.mean(error_deg)),
            "median": float(np.median(error_deg)),
            "p95": float(np.percentile(error_deg, 95.0)),
            "max": float(np.max(error_deg)),
        },
        "coverage_percent": {
            str(threshold): float(100.0 * np.mean(error_deg <= threshold)) for threshold in (30, 45, 60, 75)
        },
        "minimum_wrist_margin_deg": float(np.min(margin_deg)),
        "wrist_margin_below_5deg_percent": float(100.0 * np.mean(margin_deg < 5.0)),
        "tracking_state_counts": {str(state): int(count) for state, count in zip(states, counts, strict=True)},
    }


def render_comparison(
    labeled_results: Sequence[LabeledResult],
    output_path: Path,
    *,
    title: str,
    fps: float,
    frame_step: int,
    panel_size: int,
    render_size: int,
    model_path: Path,
) -> None:
    frame_count = labeled_results[0].result.qpos.shape[0]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=render_size, width=render_size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    header_height = 66
    canvas_size = (panel_size * len(labeled_results), panel_size + header_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps / frame_step,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )
    title_font = _font(23)
    try:
        for frame_index in range(0, frame_count, frame_step):
            canvas = Image.new("RGB", canvas_size, color=(13, 17, 23))
            draw = ImageDraw.Draw(canvas)
            elapsed_s = frame_index / fps
            draw.text(
                (18, 15),
                f"{title}   •   t = {elapsed_s:4.2f} s   •   target XYZ = red / green / blue",
                font=title_font,
                fill=(240, 243, 247),
            )
            for panel_index, labeled in enumerate(labeled_results):
                panel = _render_panel(
                    renderer,
                    model,
                    data,
                    camera,
                    labeled,
                    frame_index,
                    panel_width=panel_size,
                    panel_height=panel_size,
                )
                canvas.paste(panel, (panel_index * panel_size, header_height))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hdf5", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--result", action="append", type=_parse_labeled_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Tennis-racket retargeting comparison")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models/g1/g1_29dof.xml",
    )
    parser.add_argument("--panel-size", type=int, default=480)
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args()

    loaded_results = [(label, path, load_retargeting_result(path)) for label, path in args.result]
    frame_counts = {result.qpos.shape[0] for _, _, result in loaded_results}
    if len(frame_counts) != 1:
        raise ValueError("All comparison results must have the same frame count")
    if any(result.tennis_racket is None for _, _, result in loaded_results):
        raise ValueError("All comparison results must contain saved tennis-racket motion")
    fps_values = {round(result.fps, 8) for _, _, result in loaded_results}
    if len(fps_values) != 1:
        raise ValueError("All comparison results must have the same FPS")
    if args.frame_step <= 0:
        raise ValueError("frame-step must be positive")
    frame_count = frame_counts.pop()
    fps = float(fps_values.pop())
    source_motion = load_xsens_hdf5_motion(
        args.source_hdf5.expanduser().resolve(),
        target_fps=fps,
        frame_start=args.frame_start,
        max_frames=frame_count,
        include_tracked_props=True,
    )
    labeled_results = [
        LabeledResult(
            label=label,
            path=path,
            result=result,
            targets=build_tennis_racket_targets(source_motion, result.tennis_racket.attachment),
        )
        for label, path, result in loaded_results
        if result.tennis_racket is not None
    ]
    output_path = args.output.expanduser().resolve()
    render_comparison(
        labeled_results,
        output_path,
        title=args.title,
        fps=fps,
        frame_step=args.frame_step,
        panel_size=args.panel_size,
        render_size=args.render_size,
        model_path=args.model.expanduser().resolve(),
    )
    summary = {
        "title": args.title,
        "source_hdf5": str(args.source_hdf5.expanduser().resolve()),
        "frame_start": args.frame_start,
        "fps": fps,
        "render_fps": fps / args.frame_step,
        "results": [_summarize(item) for item in labeled_results],
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved comparison video to {output_path}")
    print(f"Saved metrics to {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
