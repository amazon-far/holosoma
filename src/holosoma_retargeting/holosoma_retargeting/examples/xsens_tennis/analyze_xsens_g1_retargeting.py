#!/usr/bin/env python3
"""Analyze Xsens-to-G1 retargeting quality for one or more named sequences."""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/holosoma-matplotlib")

import h5py
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import tyro
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.data_utils.xsens_hdf5 import (  # noqa: E402
    XSENS_BODY_SEGMENT_NAMES,
    XsensHdf5Motion,
    load_xsens_hdf5_motion,
    sample_indices_by_time,
    transform_xsens_stream_to_retargeting,
)
from holosoma_retargeting.kinematics import KinematicTree, rotate_vectors  # noqa: E402
from holosoma_retargeting.src.paths import DEMO_RESULTS_DIR, PACKAGE_ROOT  # noqa: E402
from holosoma_retargeting.viser_player import (  # noqa: E402
    G1_RACKET_FRAME_WXYZ,
    G1_RACKET_GRIP_OFFSET_M,
    G1_RACKET_ORIENTATION_LINK,
    G1_RACKET_POSITION_LINK,
)
from holosoma_retargeting.xsens.g1_kinematic_reduction import (  # noqa: E402
    G1XsensReductionConfig,
    build_g1_proportioned_xsens_tree,
    extract_g1_anthropometry,
)
from holosoma_retargeting.xsens.morphology_adaptation import (  # noqa: E402
    adapt_xsens_tpose_to_g1,
    build_subject_xsens_reference_model,
)
from holosoma_retargeting.xsens.tpose_calibration import (  # noqa: E402
    XsensTposeCalibrationConfig,
    save_xsens_tpose_calibration,
    solve_xsens_tpose_calibration_from_data,
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
ACTOR_RGB = {"human": (0, 114, 178), "g1_xsens": (0, 158, 115), "g1": (230, 159, 0)}
RACKET_SOURCE_NAME = "RightHandSword"
DEFAULT_DATA_DIR = Path("demo_data/xsens_tennis")
DEFAULT_ROBOT_MODEL = Path("models/g1/g1_29dof.xml")


@dataclass(frozen=True)
class Config:
    """Configuration for sequence-name-based Xsens-to-G1 analysis."""

    sequence_names: tuple[str, ...]
    data_dir: Path = DEFAULT_DATA_DIR
    retargeted_results_dir: Path | None = None
    output_root: Path | None = None
    hdf5_path: Path | None = None
    qpos_npz: Path | None = None
    robot_model_path: Path = DEFAULT_ROBOT_MODEL
    tpose_calibration_path: Path | None = None
    frame_start: int = 0
    frame_end: int | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None
    clip_duration_s: float = 6.0
    max_clips: int = 6
    max_speed_lag_s: float = 0.5
    sole_contact_height_threshold_m: float = 0.03
    viser_mode: Literal["none", "interactive", "record-clips"] = "none"
    viser_port: int = 8080
    actor_spacing_m: float = 2.0
    trail_duration_s: float = 1.0
    record_width: int = 1280
    record_height: int = 720
    record_connect_timeout_s: float = 120.0


@dataclass(frozen=True)
class SequencePaths:
    sequence_name: str
    hdf5_path: Path
    qpos_npz: Path
    output_dir: Path


@dataclass(frozen=True)
class ActivityWindow:
    label: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class DiagnosticClip:
    label: str
    metric: str
    peak_frame: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class FootprintSeries:
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class ProxyMassPoint:
    segment_name: str
    mass_kg: float
    local_offset_m: np.ndarray
    source_body_name: str


@dataclass(frozen=True)
class ProxyModel:
    points: tuple[ProxyMassPoint, ...]
    total_mass_kg: float
    reference_com_error_m: float
    calibration_success: bool
    calibration_cost: float


@dataclass(frozen=True)
class AnalysisData:
    paths: SequencePaths
    fps: float
    frame_indices: np.ndarray
    times_s: np.ndarray
    activity_labels: np.ndarray
    activity_windows: tuple[ActivityWindow, ...]
    contact_phase: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    qpos: np.ndarray
    segment_names: tuple[str, ...]
    human_positions_m: np.ndarray
    human_quaternions_wxyz: np.ndarray
    g1_xsens_positions_m: np.ndarray
    g1_xsens_quaternions_wxyz: np.ndarray
    human_com_m: np.ndarray
    g1_xsens_com_m: np.ndarray
    g1_com_m: np.ndarray
    human_root_position_m: np.ndarray
    human_root_quaternion_wxyz: np.ndarray
    g1_root_position_m: np.ndarray
    g1_root_quaternion_wxyz: np.ndarray
    human_racket_position_m: np.ndarray
    human_racket_quaternion_wxyz: np.ndarray
    g1_xsens_racket_position_m: np.ndarray
    g1_xsens_racket_quaternion_wxyz: np.ndarray
    g1_racket_position_m: np.ndarray
    g1_racket_quaternion_wxyz: np.ndarray
    human_footprints: FootprintSeries
    g1_xsens_footprints: FootprintSeries
    g1_footprints: FootprintSeries
    human_support_margin_m: np.ndarray
    g1_xsens_support_margin_m: np.ndarray
    g1_support_margin_m: np.ndarray
    human_support_area_m2: np.ndarray
    g1_xsens_support_area_m2: np.ndarray
    g1_support_area_m2: np.ndarray
    human_normalized_margin: np.ndarray
    g1_xsens_normalized_margin: np.ndarray
    g1_normalized_margin: np.ndarray
    g1_left_sole_height_m: np.ndarray
    g1_right_sole_height_m: np.ndarray
    metrics: Mapping[str, np.ndarray]
    proxy_model: ProxyModel
    human_model: KinematicTree
    g1_xsens_model: KinematicTree
    robot_model_path: Path


def _resolve_package_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    candidate = PACKAGE_ROOT / expanded
    return candidate.resolve()


def _resolve_robot_model_path(path: Path) -> Path:
    resolved = _resolve_package_path(path)
    if resolved.suffix.casefold() == ".urdf":
        resolved = resolved.with_suffix(".xml")
    if resolved.suffix.casefold() != ".xml" or not resolved.is_file():
        raise FileNotFoundError(f"Expected an existing MuJoCo XML robot model, got: {resolved}")
    if not resolved.with_suffix(".urdf").is_file():
        urdf_path = resolved.with_suffix(".urdf")
        raise FileNotFoundError(f"The matching G1 URDF is required for calibration and Viser: {urdf_path}")
    return resolved


def normalize_sequence_name(value: str) -> str:
    """Normalize one basename-like sequence token to an extension-free stem."""

    raw = value.strip()
    if not raw:
        raise ValueError("Sequence names must not be empty")
    if raw.startswith("."):
        raise ValueError(f"Sequence names must have a non-hidden stem: {value!r}")
    if Path(raw).name != raw:
        raise ValueError(f"Sequence names must be basenames, not paths: {value!r}")
    suffix = Path(raw).suffix.casefold()
    if suffix in {".hdf5", ".h5"}:
        raw = raw[: -len(suffix)]
    if not raw:
        raise ValueError(f"Sequence name has no stem: {value!r}")
    return raw


def resolve_sequence_paths(config: Config) -> tuple[SequencePaths, ...]:
    """Resolve exact HDF5/NPZ/output paths from required sequence names."""

    if not config.sequence_names:
        raise ValueError("At least one --sequence-names value is required")
    names = tuple(normalize_sequence_name(value) for value in config.sequence_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate sequence names after normalization: {duplicates}")
    if len(names) != 1 and (config.hdf5_path is not None or config.qpos_npz is not None):
        raise ValueError("--hdf5-path and --qpos-npz overrides require exactly one sequence name")
    if config.viser_mode != "none" and len(names) != 1:
        raise ValueError("Interactive and recorded Viser modes require exactly one sequence name")

    data_dir = _resolve_package_path(config.data_dir)
    results_dir = (
        _resolve_package_path(config.retargeted_results_dir)
        if config.retargeted_results_dir is not None
        else (DEMO_RESULTS_DIR / "g1" / "robot_only" / data_dir.name).resolve()
    )
    output_root = (
        _resolve_package_path(config.output_root)
        if config.output_root is not None
        else (DEMO_RESULTS_DIR / "g1" / "analysis" / data_dir.name).resolve()
    )

    resolved: list[SequencePaths] = []
    for name in names:
        if config.hdf5_path is not None:
            hdf5_path = _resolve_package_path(config.hdf5_path)
            if not hdf5_path.is_file():
                raise FileNotFoundError(f"Explicit Xsens HDF5 file does not exist: {hdf5_path}")
        else:
            candidates = [data_dir / f"{name}.hdf5", data_dir / f"{name}.h5"]
            matches = [path for path in candidates if path.is_file()]
            if len(matches) != 1:
                attempted = "\n  ".join(str(path) for path in candidates)
                if not matches:
                    raise FileNotFoundError(
                        f"No Xsens HDF5 file found for sequence '{name}'. Attempted:\n  {attempted}"
                    )
                raise ValueError(f"Both .hdf5 and .h5 files exist for sequence '{name}': {matches}")
            hdf5_path = matches[0].resolve()

        qpos_path = (
            _resolve_package_path(config.qpos_npz)
            if config.qpos_npz is not None
            else (results_dir / f"{name}.npz").resolve()
        )
        if not qpos_path.is_file():
            raise FileNotFoundError(
                f"Retargeted NPZ not found for sequence '{name}'. Expected exact path:\n  {qpos_path}\n"
                "Use --retargeted-results-dir for staged or experimental results."
            )
        resolved.append(SequencePaths(name, hdf5_path, qpos_path, (output_root / name).resolve()))
    return tuple(resolved)


def _rotations_from_wxyz(quaternions: np.ndarray) -> Rotation:
    values = np.asarray(quaternions, dtype=float)
    if values.shape[-1] != 4 or not np.isfinite(values).all():
        raise ValueError(f"Invalid scalar-first quaternion array: {values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Quaternion array contains a zero-length quaternion")
    normalized = values / norms
    return Rotation.from_quat(normalized[..., [1, 2, 3, 0]])


def _wxyz(rotations: Rotation) -> np.ndarray:
    xyzw = np.asarray(rotations.as_quat(), dtype=float)
    xyzw = xyzw.copy()
    xyzw = np.where((xyzw[..., 3] < 0.0)[..., None], -xyzw, xyzw)
    return xyzw[..., [3, 0, 1, 2]]


def root_relative_positions(
    points_m: np.ndarray,
    root_positions_m: np.ndarray,
    root_quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    """Express world positions in each frame's full six-DoF root frame."""

    points = np.asarray(points_m, dtype=float)
    roots = np.asarray(root_positions_m, dtype=float)
    if points.shape != roots.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point and root positions must both have shape (frames, 3)")
    return _rotations_from_wxyz(root_quaternions_wxyz).inv().apply(points - roots)


def root_relative_rotations(
    rotations_wxyz: np.ndarray,
    root_quaternions_wxyz: np.ndarray,
) -> Rotation:
    return _rotations_from_wxyz(root_quaternions_wxyz).inv() * _rotations_from_wxyz(rotations_wxyz)


def orientation_error_metrics(reference: Rotation, target: Rotation) -> dict[str, np.ndarray]:
    """Return quaternion-safe global error, two axis errors, and longitudinal twist."""

    residual = reference.inv() * target
    geodesic_deg = np.atleast_1d(np.rad2deg(residual.magnitude()))
    reference_long = np.atleast_2d(reference.apply(np.array([1.0, 0.0, 0.0])))
    target_long = np.atleast_2d(target.apply(np.array([1.0, 0.0, 0.0])))
    reference_normal = np.atleast_2d(reference.apply(np.array([0.0, 0.0, 1.0])))
    target_normal = np.atleast_2d(target.apply(np.array([0.0, 0.0, 1.0])))
    longitudinal_axis_deg = np.rad2deg(np.arccos(np.clip(np.sum(reference_long * target_long, axis=1), -1.0, 1.0)))
    face_normal_deg = np.rad2deg(np.arccos(np.clip(np.sum(reference_normal * target_normal, axis=1), -1.0, 1.0)))
    q = np.atleast_2d(residual.as_quat()).copy()
    q[q[:, 3] < 0.0] *= -1.0
    twist_deg = np.rad2deg(2.0 * np.arctan2(q[:, 0], q[:, 3]))
    twist_deg = (twist_deg + 180.0) % 360.0 - 180.0
    return {
        "geodesic_deg": geodesic_deg,
        "longitudinal_axis_deg": longitudinal_axis_deg,
        "face_normal_deg": face_normal_deg,
        "twist_deg": twist_deg,
    }


def angular_speed(rotations: Rotation, times_s: np.ndarray) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size != len(rotations):
        raise ValueError("Rotation and timestamp counts must match")
    if times.size == 1:
        return np.zeros(1)
    increments = (rotations[:-1].inv() * rotations[1:]).magnitude()
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError("Timestamps must be strictly increasing")
    values = increments / intervals
    return np.concatenate([values[:1], values])


def linear_speed(positions_m: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=float)
    if positions.shape != (len(times_s), 3):
        raise ValueError("Positions must have shape (frames, 3)")
    if len(times_s) == 1:
        return np.zeros(1)
    values = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times_s)
    return np.concatenate([values[:1], values])


def best_lag(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    fps: float,
    max_lag_s: float,
) -> tuple[float, float]:
    """Return lag applied to target and the corresponding Pearson correlation."""

    ref = np.asarray(reference, dtype=float)
    tgt = np.asarray(target, dtype=float)
    max_frames = max(0, round(max_lag_s * fps))
    best = (0, -np.inf)
    for lag in range(-max_frames, max_frames + 1):
        if lag < 0:
            x, y = ref[-lag:], tgt[:lag]
        elif lag > 0:
            x, y = ref[:-lag], tgt[lag:]
        else:
            x, y = ref, tgt
        if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(corr) and corr > best[1]:
            best = (lag, corr)
    return best[0] / fps, best[1]


def signed_polygon_margin(point_xy: np.ndarray, polygon_xy: np.ndarray) -> tuple[float, float]:
    """Return signed distance to a convex polygon boundary and polygon area."""

    point = np.asarray(point_xy, dtype=float)
    points = np.unique(np.asarray(polygon_xy, dtype=float), axis=0)
    if point.shape != (2,) or points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return float("nan"), float("nan")
    try:
        hull = ConvexHull(points)
    except Exception:
        return float("nan"), float("nan")
    polygon = points[hull.vertices]
    next_polygon = np.roll(polygon, -1, axis=0)
    edges = next_polygon - polygon
    relative = point - polygon
    edge_norm_sq = np.sum(edges * edges, axis=1)
    fractions = np.clip(np.sum(relative * edges, axis=1) / np.maximum(edge_norm_sq, 1e-15), 0.0, 1.0)
    closest = polygon + fractions[:, None] * edges
    distance = float(np.min(np.linalg.norm(point - closest, axis=1)))
    equations = hull.equations
    inside = bool(np.all(equations[:, :2] @ point + equations[:, 2] <= 1e-10))
    return distance if inside else -distance, float(hull.volume)


def support_metrics(
    com_m: np.ndarray,
    footprints: FootprintSeries,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(com_m)
    margins = np.full(frame_count, np.nan)
    areas = np.full(frame_count, np.nan)
    for index in range(frame_count):
        active: list[np.ndarray] = []
        if left_contact[index]:
            active.append(footprints.left[index, :, :2])
        if right_contact[index]:
            active.append(footprints.right[index, :, :2])
        if not active:
            continue
        margins[index], areas[index] = signed_polygon_margin(com_m[index, :2], np.concatenate(active))
    normalized = margins / np.sqrt(areas)
    return margins, areas, normalized


def _normalize_source_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _body_source_name(model: KinematicTree, body_name: str) -> str:
    body = model.body_map()[body_name]
    return str(body.metadata.get("xsens:sourceSegmentName", body.name))


def footprint_series(
    model: KinematicTree,
    segment_names: Sequence[str],
    positions_m: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> FootprintSeries:
    """Transform outsole mesh vertices for both feet through a motion."""

    source_indices = {_normalize_source_name(name): index for index, name in enumerate(segment_names)}

    def side_points(side: str) -> np.ndarray:
        transformed: list[np.ndarray] = []
        for body in model.bodies:
            source_name = _body_source_name(model, body.name)
            normalized = _normalize_source_name(source_name)
            if not normalized.startswith(side.casefold()) or not normalized.endswith(("foot", "toe")):
                continue
            meshes = [mesh for mesh in body.meshes if "outsole" in mesh.name.casefold()]
            if not meshes:
                continue
            index = source_indices[normalized]
            vertices = np.concatenate([mesh.vertices_m for mesh in meshes])
            rotated = rotate_vectors(
                quaternions_wxyz[:, index, None, :],
                vertices[None, :, :],
            )
            transformed.append(positions_m[:, index, None, :] + rotated)
        if not transformed:
            raise ValueError(f"Model has no {side} outsole meshes")
        return np.concatenate(transformed, axis=1)

    return FootprintSeries(left=side_points("left"), right=side_points("right"))


def reconstruct_target_racket(
    model: KinematicTree,
    segment_names: Sequence[str],
    positions_m: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    """Reconstruct the G1-sized tracked-racket origin from its virtual joint."""

    source_indices = {_normalize_source_name(name): index for index, name in enumerate(segment_names)}
    racket_joint = next(
        joint
        for joint in model.joints
        if _normalize_source_name(str(joint.metadata.get("xsens:sourceJointName", joint.name)))
        == _normalize_source_name("RightHandSwordOrigin")
    )
    parent_source = _body_source_name(model, racket_joint.parent_body)
    child_source = _body_source_name(model, racket_joint.child_body)
    parent_index = source_indices[_normalize_source_name(parent_source)]
    child_index = source_indices[_normalize_source_name(child_source)]
    parent_anchor = rotate_vectors(
        quaternions_wxyz[:, parent_index, :],
        np.broadcast_to(racket_joint.parent_frame.translation_m, (len(positions_m), 3)),
    )
    child_anchor = rotate_vectors(
        quaternions_wxyz[:, child_index, :],
        np.broadcast_to(racket_joint.child_frame.translation_m, (len(positions_m), 3)),
    )
    return positions_m[:, parent_index, :] + parent_anchor - child_anchor


def _g1_proxy_segment(body_name: str) -> str:
    """Assign each physical G1 inertial body to one reduced Xsens segment."""

    if body_name in {"pelvis", "pelvis_contour_link"}:
        return "Pelvis"
    if body_name == "waist_yaw_link":
        return "L5"
    if body_name == "waist_roll_link":
        return "L3"
    if body_name in {"torso_link", "waist_support_link"}:
        return "T8"
    for side, title in (("left", "Left"), ("right", "Right")):
        if not body_name.startswith(f"{side}_"):
            continue
        suffix = body_name[len(side) + 1 :]
        if suffix.startswith(("hip_pitch", "hip_roll", "hip_yaw")):
            return f"{title}UpperLeg"
        if suffix.startswith(("knee", "ankle_intermediate")):
            return f"{title}LowerLeg"
        if suffix.startswith("ankle_roll_sphere_"):
            sphere = int(re.search(r"sphere_(\d+)", suffix).group(1))  # type: ignore[union-attr]
            return f"{title}Toe" if sphere in {3, 4, 5} else f"{title}Foot"
        if suffix.startswith(("ankle_pitch", "ankle_roll")):
            return f"{title}Foot"
        if suffix.startswith("shoulder_pitch"):
            return f"{title}Shoulder"
        if suffix.startswith(("shoulder_roll", "shoulder_yaw")):
            return f"{title}UpperArm"
        if suffix.startswith(("elbow", "wrist_roll", "wrist_pitch")):
            return f"{title}ForeArm"
        if suffix.startswith(("wrist_yaw", "rubber_hand", "thumb", "pinky")):
            return f"{title}Hand"
    raise KeyError(f"No reduced-Xsens mass assignment for G1 body '{body_name}'")


def build_proxy_model(
    hdf5_path: Path,
    robot_model_path: Path,
    target_segment_names: Sequence[str],
    calibration_path: Path | None,
) -> ProxyModel:
    """Map physical G1 inertial centroids into reduced-Xsens segment frames."""

    tpose = adapt_xsens_tpose_to_g1(
        hdf5_path=hdf5_path,
        g1_model_path=robot_model_path,
        grounding="match_lowest_soles",
    )
    tpose_indices = {_normalize_source_name(name): index for index, name in enumerate(tpose.segment_names)}
    resolved_calibration = _resolve_package_path(calibration_path) if calibration_path is not None else None
    if resolved_calibration is not None and resolved_calibration.is_file():
        resolved = resolved_calibration
        with np.load(resolved, allow_pickle=False) as saved:
            qpos = np.asarray(saved["qpos"], dtype=float)[0]
            calibration_success = bool(saved["solver_success"]) if "solver_success" in saved else True
            calibration_cost = float(saved["solver_cost"]) if "solver_cost" in saved else float("nan")
    else:
        result = solve_xsens_tpose_calibration_from_data(
            tpose,
            config=XsensTposeCalibrationConfig(
                robot_type="g1",
                variant="Tpose",
                robot_urdf_file=str(robot_model_path.with_suffix(".urdf")),
                default_human_height=1.78,
                max_nfev=400,
                verbose=0,
            ),
            position_scale_factor=1.0,
        )
        qpos = np.asarray(result.qpos[0], dtype=float)
        calibration_success = bool(result.solver_success)
        calibration_cost = float(result.solver_cost)
        if resolved_calibration is not None:
            save_xsens_tpose_calibration(result, resolved_calibration)

    model = mujoco.MjModel.from_xml_path(str(robot_model_path))
    data = mujoco.MjData(model)
    if qpos.shape != (model.nq,):
        raise ValueError(f"T-pose calibration qpos has shape {qpos.shape}; expected {(model.nq,)}")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    target_name_by_normalized = {_normalize_source_name(name): name for name in target_segment_names}
    points: list[ProxyMassPoint] = []
    weighted_reference = np.zeros(3)
    total_mass = 0.0
    for body_id in range(1, model.nbody):
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue
        body_name = model.body(body_id).name
        proxy_segment = _g1_proxy_segment(body_name)
        normalized_segment = _normalize_source_name(proxy_segment)
        if normalized_segment not in target_name_by_normalized:
            raise KeyError(f"Proxy segment '{proxy_segment}' is absent from target motion")
        segment_name = target_name_by_normalized[normalized_segment]
        segment_index = tpose_indices[_normalize_source_name(segment_name)]
        segment_rotation = _rotations_from_wxyz(tpose.quaternions_wijk[segment_index])
        local_offset = segment_rotation.inv().apply(data.xipos[body_id] - tpose.positions_m[segment_index])
        points.append(ProxyMassPoint(segment_name, mass, local_offset, body_name))
        weighted_reference += mass * (tpose.positions_m[segment_index] + segment_rotation.apply(local_offset))
        total_mass += mass
    proxy_reference = weighted_reference / total_mass
    exact_reference = np.asarray(data.subtree_com[model.body("pelvis").id], dtype=float)
    return ProxyModel(
        points=tuple(points),
        total_mass_kg=total_mass,
        reference_com_error_m=float(np.linalg.norm(proxy_reference - exact_reference)),
        calibration_success=calibration_success,
        calibration_cost=calibration_cost,
    )


def evaluate_proxy_com(
    proxy: ProxyModel,
    segment_names: Sequence[str],
    positions_m: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    indices = {_normalize_source_name(name): index for index, name in enumerate(segment_names)}
    weighted = np.zeros((len(positions_m), 3))
    for point in proxy.points:
        index = indices[_normalize_source_name(point.segment_name)]
        offsets = rotate_vectors(
            quaternions_wxyz[:, index, :],
            np.broadcast_to(point.local_offset_m, (len(positions_m), 3)),
        )
        weighted += point.mass_kg * (positions_m[:, index, :] + offsets)
    return weighted / proxy.total_mass_kg


def evaluate_g1_motion(
    model_path: Path,
    qpos: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    FootprintSeries,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate exact G1 CoM, root, racket, and sole points using MuJoCo."""

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Retargeted qpos width is {qpos.shape[1]}; robot model expects {model.nq}")
    data = mujoco.MjData(model)
    pelvis_id = model.body("pelvis").id
    position_body = model.body(G1_RACKET_POSITION_LINK).id
    orientation_body = model.body(G1_RACKET_ORIENTATION_LINK).id
    side_ids = {
        side: [model.body(f"{side}_ankle_roll_sphere_{index}_link").id for index in range(1, 6)]
        for side in ("left", "right")
    }
    count = len(qpos)
    com = np.empty((count, 3))
    root_position = np.empty((count, 3))
    root_quaternion = np.empty((count, 4))
    racket_position = np.empty((count, 3))
    racket_quaternion = np.empty((count, 4))
    soles = {side: np.empty((count, 5, 3)) for side in side_ids}
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        com[frame] = data.subtree_com[pelvis_id]
        root_position[frame] = data.xpos[pelvis_id]
        root_quaternion[frame] = data.xquat[pelvis_id]
        position_rotation = data.xmat[position_body].reshape(3, 3)
        racket_position[frame] = data.xpos[position_body] + position_rotation @ G1_RACKET_GRIP_OFFSET_M
        hand_rotation = Rotation.from_matrix(data.xmat[orientation_body].reshape(3, 3))
        racket_rotation = hand_rotation * _rotations_from_wxyz(G1_RACKET_FRAME_WXYZ)
        racket_quaternion[frame] = _wxyz(racket_rotation)
        for side, body_ids in side_ids.items():
            soles[side][frame] = data.xpos[body_ids]
    return (
        com,
        root_position,
        root_quaternion,
        racket_position,
        FootprintSeries(left=soles["left"], right=soles["right"]),
        np.min(soles["left"][:, :, 2], axis=1),
        np.min(soles["right"][:, :, 2], axis=1),
        racket_quaternion,
    )


def _decode(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _parse_list_attribute(value: Any, attribute: str) -> list[str]:
    parsed = ast.literal_eval(_decode(value))
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{attribute} must decode to a list")
    return [str(item) for item in parsed]


def _load_auxiliary_xsens(
    hdf5_path: Path,
    *,
    fps: float,
    source_position_stream: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], tuple[ActivityWindow, ...]]:
    with h5py.File(hdf5_path, "r") as handle:
        position_stream = handle[f"xsens-segments/{source_position_stream}"]
        source_times = np.asarray(position_stream["time_s"], dtype=float).reshape(-1)
        indices = sample_indices_by_time(source_times, fps)
        com_stream = handle["xsens-CoM/position_xyz_m"]
        contact_stream = handle["xsens-foot-contacts/is_contacting_ground"]
        for stream in (com_stream, contact_stream):
            times = np.asarray(stream["time_s"], dtype=float).reshape(-1)
            if times.shape != source_times.shape or not np.allclose(times, source_times):
                raise ValueError("Xsens CoM/contact streams do not share the segment timeline")
        com = transform_xsens_stream_to_retargeting(
            np.asarray(com_stream["data"], dtype=float)[indices],
            source_position_stream,
        )
        contacts = np.asarray(contact_stream["data"], dtype=float)[indices]
        contact_names = _parse_list_attribute(
            contact_stream.attrs["foot_contact_names"],
            "foot_contact_names",
        )
        windows: list[ActivityWindow] = []
        if "experiment-activities/activities" in handle:
            activity = handle["experiment-activities/activities"]
            rows = np.asarray(activity["data"])
            activity_times = np.asarray(activity["time_s"], dtype=float).reshape(-1)
            starts: dict[str, float] = {}
            for timestamp, row in zip(activity_times, rows, strict=True):
                values = [_decode(value).strip() for value in row]
                if len(values) < 2:
                    continue
                label, marker = values[0], values[1].casefold()
                valid = len(values) < 3 or values[2].casefold() in {"", "good", "valid", "true"}
                if not label or not valid:
                    continue
                if marker == "start":
                    starts[label] = float(timestamp - source_times[0])
                elif marker == "stop" and label in starts:
                    windows.append(ActivityWindow(label, starts.pop(label), float(timestamp - source_times[0])))
    return source_times[indices] - source_times[0], com, contacts, contact_names, tuple(windows)


def _contact_masks(contacts: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_columns = [index for index, name in enumerate(names) if name.casefold().startswith("left")]
    right_columns = [index for index, name in enumerate(names) if name.casefold().startswith("right")]
    if not left_columns or not right_columns:
        raise ValueError(f"Could not identify left/right Xsens contact columns: {names}")
    left = np.any(contacts[:, left_columns] > 0.5, axis=1)
    right = np.any(contacts[:, right_columns] > 0.5, axis=1)
    phase = np.full(len(contacts), "flight", dtype="U16")
    phase[left & ~right] = "left_support"
    phase[right & ~left] = "right_support"
    phase[left & right] = "double_support"
    return left, right, phase


def _activity_labels(times_s: np.ndarray, windows: Sequence[ActivityWindow]) -> np.ndarray:
    labels = np.full(len(times_s), "unlabeled", dtype="U64")
    for window in windows:
        labels[(times_s >= window.start_s) & (times_s <= window.end_s)] = window.label
    return labels


def _select_frames(
    config: Config,
    full_times_s: np.ndarray,
) -> np.ndarray:
    if config.frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    end = len(full_times_s) if config.frame_end is None else config.frame_end
    if end <= config.frame_start or end > len(full_times_s):
        raise ValueError(f"Invalid frame range [{config.frame_start}, {end}) for {len(full_times_s)} frames")
    mask = np.zeros(len(full_times_s), dtype=bool)
    mask[config.frame_start : end] = True
    if config.start_time_s is not None:
        mask &= full_times_s >= config.start_time_s
    if config.end_time_s is not None:
        mask &= full_times_s <= config.end_time_s
    frames = np.flatnonzero(mask)
    if frames.size == 0:
        raise ValueError("Selected frame/time window is empty")
    return frames


def validate_aligned_timeline(
    counts: Mapping[str, int],
    motion_times_s: np.ndarray,
    auxiliary_times_s: np.ndarray,
    *,
    tolerance_s: float = 1e-6,
) -> None:
    """Reject frame-count or timestamp differences instead of silently trimming."""

    if len(set(counts.values())) != 1:
        raise ValueError(f"Aligned input frame counts differ; refusing to trim: {dict(counts)}")
    motion_times = np.asarray(motion_times_s, dtype=float)
    auxiliary_times = np.asarray(auxiliary_times_s, dtype=float)
    if motion_times.shape != auxiliary_times.shape or not np.allclose(
        motion_times,
        auxiliary_times,
        rtol=0.0,
        atol=tolerance_s,
    ):
        raise ValueError("Resampled Xsens motion and auxiliary streams have different timestamps")


def load_retargeted_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load the required retargeting arrays and reject incomplete or invalid files."""

    with np.load(path, allow_pickle=False) as saved:
        required = {"qpos", "human_joints", "fps"}
        missing = sorted(required.difference(saved.files))
        if missing:
            raise KeyError(f"Retargeted NPZ {path} is missing required arrays: {missing}")
        qpos = np.asarray(saved["qpos"], dtype=float)
        target_positions = np.asarray(saved["human_joints"], dtype=float)
        fps = float(saved["fps"])
    if qpos.ndim != 2 or target_positions.ndim != 3 or target_positions.shape[2] != 3:
        raise ValueError("Expected qpos (frames, nq) and human_joints (frames, segments, 3)")
    if qpos.shape[0] != target_positions.shape[0]:
        raise ValueError("qpos and human_joints frame counts differ")
    if not np.isfinite(qpos).all() or not np.isfinite(target_positions).all() or fps <= 0.0:
        raise ValueError("Retargeted NPZ contains invalid values or FPS")
    return qpos, target_positions, fps


def load_and_analyze(config: Config, paths: SequencePaths) -> AnalysisData:
    robot_model_path = _resolve_robot_model_path(config.robot_model_path)
    qpos_full, target_positions_full, fps = load_retargeted_npz(paths.qpos_npz)

    motion: XsensHdf5Motion = load_xsens_hdf5_motion(
        paths.hdf5_path,
        target_fps=fps,
        include_tracked_props=True,
    )
    full_times_s, human_com_full, contacts_full, contact_names, activity_windows = _load_auxiliary_xsens(
        paths.hdf5_path,
        fps=fps,
        source_position_stream=motion.stream_name,
    )
    frame_count = qpos_full.shape[0]
    counts = {
        "qpos": frame_count,
        "human_joints": target_positions_full.shape[0],
        "Xsens motion": motion.positions_m.shape[0],
        "Xsens CoM": human_com_full.shape[0],
        "Xsens contacts": contacts_full.shape[0],
    }
    validate_aligned_timeline(counts, motion.times_s - motion.times_s[0], full_times_s)

    frames = _select_frames(config, full_times_s)
    times_s = full_times_s[frames]
    qpos = qpos_full[frames]
    target_body_positions = target_positions_full[frames]
    human_positions = np.asarray(motion.positions_m, dtype=float)[frames]
    human_quaternions = np.asarray(motion.quaternions_wijk, dtype=float)[frames]
    human_com = human_com_full[frames]
    contacts = contacts_full[frames]
    segment_names = tuple(motion.segment_names)
    body_names = tuple(XSENS_BODY_SEGMENT_NAMES)
    normalized_segments = {_normalize_source_name(name) for name in segment_names}
    missing_segments = [name for name in body_names if _normalize_source_name(name) not in normalized_segments]
    if missing_segments:
        raise ValueError(f"Resampled Xsens motion is missing canonical body segments: {missing_segments}")
    if target_body_positions.shape[1] != len(XSENS_BODY_SEGMENT_NAMES):
        raise ValueError(
            f"human_joints has {target_body_positions.shape[1]} segments; "
            f"expected {len(XSENS_BODY_SEGMENT_NAMES)} canonical Xsens bodies"
        )
    source_index = {_normalize_source_name(name): index for index, name in enumerate(segment_names)}
    if "righthandsword" not in source_index:
        raise ValueError("Xsens recording must contain the tracked RightHandSword racket segment")
    target_positions = np.zeros_like(human_positions)
    for body_index, body_name in enumerate(XSENS_BODY_SEGMENT_NAMES):
        target_positions[:, source_index[_normalize_source_name(body_name)]] = target_body_positions[:, body_index]
    target_quaternions = human_quaternions.copy()

    anthropometry = extract_g1_anthropometry(robot_model_path)
    target_model = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(include_visuals=True, include_tennis_racket=True),
    )
    target_positions[:, source_index["righthandsword"]] = reconstruct_target_racket(
        target_model,
        segment_names,
        target_positions,
        target_quaternions,
    )
    human_model = build_subject_xsens_reference_model(paths.hdf5_path, include_tennis_racket=True)

    proxy = build_proxy_model(
        paths.hdf5_path,
        robot_model_path,
        body_names,
        config.tpose_calibration_path or paths.output_dir / "tpose_calibration.npz",
    )
    target_com = evaluate_proxy_com(
        proxy,
        segment_names,
        target_positions,
        target_quaternions,
    )
    (
        g1_com,
        g1_root_position,
        g1_root_quaternion,
        g1_racket_position,
        g1_footprints,
        g1_left_height,
        g1_right_height,
        g1_racket_quaternion,
    ) = evaluate_g1_motion(robot_model_path, qpos)

    left_contact, right_contact, contact_phase = _contact_masks(contacts, contact_names)
    human_footprints = footprint_series(
        human_model,
        segment_names,
        human_positions,
        human_quaternions,
    )
    target_footprints = footprint_series(
        target_model,
        segment_names,
        target_positions,
        target_quaternions,
    )
    human_margin, human_area, human_normalized = support_metrics(
        human_com, human_footprints, left_contact, right_contact
    )
    target_margin, target_area, target_normalized = support_metrics(
        target_com, target_footprints, left_contact, right_contact
    )
    g1_margin, g1_area, g1_normalized = support_metrics(g1_com, g1_footprints, left_contact, right_contact)

    pelvis_index = source_index["pelvis"]
    racket_index = source_index["righthandsword"]
    human_root_position = human_positions[:, pelvis_index]
    human_root_quaternion = human_quaternions[:, pelvis_index]
    human_racket_position = human_positions[:, racket_index]
    human_racket_quaternion = human_quaternions[:, racket_index]
    target_racket_position = target_positions[:, racket_index]
    target_racket_quaternion = target_quaternions[:, racket_index]

    human_com_root = root_relative_positions(human_com, human_root_position, human_root_quaternion)
    g1_com_root = root_relative_positions(g1_com, g1_root_position, g1_root_quaternion)
    human_racket_root = root_relative_positions(human_racket_position, human_root_position, human_root_quaternion)
    g1_racket_root = root_relative_positions(g1_racket_position, g1_root_position, g1_root_quaternion)
    global_orientation = orientation_error_metrics(
        _rotations_from_wxyz(human_racket_quaternion),
        _rotations_from_wxyz(g1_racket_quaternion),
    )
    root_orientation = orientation_error_metrics(
        root_relative_rotations(human_racket_quaternion, human_root_quaternion),
        root_relative_rotations(g1_racket_quaternion, g1_root_quaternion),
    )
    human_linear_speed = linear_speed(human_racket_position, times_s)
    g1_linear_speed = linear_speed(g1_racket_position, times_s)
    human_angular_speed = angular_speed(_rotations_from_wxyz(human_racket_quaternion), times_s)
    g1_angular_speed = angular_speed(_rotations_from_wxyz(g1_racket_quaternion), times_s)
    metrics: dict[str, np.ndarray] = {
        "com_world_error_m": np.linalg.norm(g1_com - human_com, axis=1),
        "com_root_error_m": np.linalg.norm(g1_com_root - human_com_root, axis=1),
        "support_margin_error_m": np.abs(g1_margin - human_margin),
        "racket_world_position_error_m": np.linalg.norm(g1_racket_position - human_racket_position, axis=1),
        "racket_root_position_error_m": np.linalg.norm(g1_racket_root - human_racket_root, axis=1),
        "racket_global_orientation_error_deg": global_orientation["geodesic_deg"],
        "racket_root_orientation_error_deg": root_orientation["geodesic_deg"],
        "racket_longitudinal_axis_error_deg": global_orientation["longitudinal_axis_deg"],
        "racket_face_normal_error_deg": global_orientation["face_normal_deg"],
        "racket_twist_error_deg": global_orientation["twist_deg"],
        "human_racket_linear_speed_m_s": human_linear_speed,
        "g1_racket_linear_speed_m_s": g1_linear_speed,
        "racket_linear_speed_error_m_s": np.abs(g1_linear_speed - human_linear_speed),
        "human_racket_angular_speed_rad_s": human_angular_speed,
        "g1_racket_angular_speed_rad_s": g1_angular_speed,
        "racket_angular_speed_error_rad_s": np.abs(g1_angular_speed - human_angular_speed),
        "g1_left_contact_agreement": (
            left_contact == (g1_left_height <= config.sole_contact_height_threshold_m)
        ).astype(float),
        "g1_right_contact_agreement": (
            right_contact == (g1_right_height <= config.sole_contact_height_threshold_m)
        ).astype(float),
    }
    return AnalysisData(
        paths=paths,
        fps=fps,
        frame_indices=frames,
        times_s=times_s,
        activity_labels=_activity_labels(times_s, activity_windows),
        activity_windows=activity_windows,
        contact_phase=contact_phase,
        left_contact=left_contact,
        right_contact=right_contact,
        qpos=qpos,
        segment_names=segment_names,
        human_positions_m=human_positions,
        human_quaternions_wxyz=human_quaternions,
        g1_xsens_positions_m=target_positions,
        g1_xsens_quaternions_wxyz=target_quaternions,
        human_com_m=human_com,
        g1_xsens_com_m=target_com,
        g1_com_m=g1_com,
        human_root_position_m=human_root_position,
        human_root_quaternion_wxyz=human_root_quaternion,
        g1_root_position_m=g1_root_position,
        g1_root_quaternion_wxyz=g1_root_quaternion,
        human_racket_position_m=human_racket_position,
        human_racket_quaternion_wxyz=human_racket_quaternion,
        g1_xsens_racket_position_m=target_racket_position,
        g1_xsens_racket_quaternion_wxyz=target_racket_quaternion,
        g1_racket_position_m=g1_racket_position,
        g1_racket_quaternion_wxyz=g1_racket_quaternion,
        human_footprints=human_footprints,
        g1_xsens_footprints=target_footprints,
        g1_footprints=g1_footprints,
        human_support_margin_m=human_margin,
        g1_xsens_support_margin_m=target_margin,
        g1_support_margin_m=g1_margin,
        human_support_area_m2=human_area,
        g1_xsens_support_area_m2=target_area,
        g1_support_area_m2=g1_area,
        human_normalized_margin=human_normalized,
        g1_xsens_normalized_margin=target_normalized,
        g1_normalized_margin=g1_normalized,
        g1_left_sole_height_m=g1_left_height,
        g1_right_sole_height_m=g1_right_height,
        metrics=metrics,
        proxy_model=proxy,
        human_model=human_model,
        g1_xsens_model=target_model,
        robot_model_path=robot_model_path,
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _window_mask(data: AnalysisData, label: str) -> np.ndarray:
    if label == "full_sequence":
        return np.ones(len(data.times_s), dtype=bool)
    if label in {"flight", "left_support", "right_support", "double_support"}:
        return data.contact_phase == label
    return data.activity_labels == label


def _window_labels(data: AnalysisData) -> list[str]:
    labels = ["full_sequence"]
    labels.extend(label for label in dict.fromkeys(data.activity_labels.tolist()) if label != "unlabeled")
    labels.extend(["left_support", "right_support", "double_support", "flight"])
    return labels


def select_diagnostic_clips(data: AnalysisData, duration_s: float, max_clips: int) -> tuple[DiagnosticClip, ...]:
    if duration_s <= 0.0 or max_clips <= 0:
        raise ValueError("clip_duration_s and max_clips must be positive")
    half = max(1, round(duration_s * data.fps / 2.0))
    candidates = [
        ("worst_racket_position", "racket_root_position_error_m"),
        ("worst_racket_orientation", "racket_root_orientation_error_deg"),
        ("worst_stability_margin", "support_margin_error_m"),
    ]
    candidates.extend(
        (f"representative_{_slug(label)}", "racket_root_orientation_error_deg")
        for label in dict.fromkeys(data.activity_labels.tolist())
        if label != "unlabeled"
    )
    selected: list[DiagnosticClip] = []
    for label, metric_name in candidates:
        values = np.asarray(data.metrics[metric_name], dtype=float).copy()
        if label.startswith("representative_"):
            original_label = next(
                value
                for value in dict.fromkeys(data.activity_labels.tolist())
                if _slug(value) == label.removeprefix("representative_")
            )
            values[data.activity_labels != original_label] = np.nan
        if not np.isfinite(values).any():
            continue
        order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1]
        peak = next(
            (
                int(index)
                for index in order
                if all(abs(int(index) - existing.peak_frame) > half for existing in selected)
            ),
            None,
        )
        if peak is None:
            continue
        start = max(0, peak - half)
        end = min(len(values), start + 2 * half + 1)
        start = max(0, end - (2 * half + 1))
        selected.append(
            DiagnosticClip(
                label=label,
                metric=metric_name,
                peak_frame=peak,
                start_frame=start,
                end_frame=end,
                start_s=float(data.times_s[start]),
                end_s=float(data.times_s[end - 1]),
            )
        )
        if len(selected) >= max_clips:
            break
    return tuple(selected)


def build_summary(
    data: AnalysisData,
    clips: Sequence[DiagnosticClip],
    max_lag_s: float,
    sole_contact_height_threshold_m: float,
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for label in _window_labels(data):
        mask = _window_mask(data, label)
        if not np.any(mask):
            continue
        windows[label] = {
            "frame_count": int(np.count_nonzero(mask)),
            "duration_s": float(np.count_nonzero(mask) / data.fps),
            "metrics": {name: _distribution(values[mask]) for name, values in data.metrics.items()},
            "support": {
                actor: {
                    "margin_m": _distribution(margin[mask]),
                    "normalized_margin": _distribution(normalized[mask]),
                    "polygon_area_m2": _distribution(area[mask]),
                    "outside_fraction": float(np.mean(margin[mask][np.isfinite(margin[mask])] < 0.0))
                    if np.isfinite(margin[mask]).any()
                    else None,
                    "outside_duration_s": float(
                        np.count_nonzero(margin[mask][np.isfinite(margin[mask])] < 0.0) / data.fps
                    ),
                }
                for actor, margin, normalized, area in (
                    (
                        "human",
                        data.human_support_margin_m,
                        data.human_normalized_margin,
                        data.human_support_area_m2,
                    ),
                    (
                        "g1_xsens",
                        data.g1_xsens_support_margin_m,
                        data.g1_xsens_normalized_margin,
                        data.g1_xsens_support_area_m2,
                    ),
                    (
                        "g1",
                        data.g1_support_margin_m,
                        data.g1_normalized_margin,
                        data.g1_support_area_m2,
                    ),
                )
            },
        }
    linear_lag, linear_corr = best_lag(
        data.metrics["human_racket_linear_speed_m_s"],
        data.metrics["g1_racket_linear_speed_m_s"],
        fps=data.fps,
        max_lag_s=max_lag_s,
    )
    angular_lag, angular_corr = best_lag(
        data.metrics["human_racket_angular_speed_rad_s"],
        data.metrics["g1_racket_angular_speed_rad_s"],
        fps=data.fps,
        max_lag_s=max_lag_s,
    )
    return {
        "sequence_name": data.paths.sequence_name,
        "source_hdf5": str(data.paths.hdf5_path),
        "retargeted_npz": str(data.paths.qpos_npz),
        "fps": data.fps,
        "frame_count": len(data.times_s),
        "source_frame_range": [int(data.frame_indices[0]), int(data.frame_indices[-1])],
        "time_range_s": [float(data.times_s[0]), float(data.times_s[-1])],
        "error_reference": "human-subject Xsens",
        "root_relative_definition": "R_root^T * (p - p_root), with the full root orientation",
        "support_contact_source": "Xsens heel/toe contact flags shared across actor-specific footprints",
        "proxy_com": {
            "method": "G1 inertial masses and calibrated centroids assigned to reduced Xsens segments",
            "racket_included": False,
            "point_count": len(data.proxy_model.points),
            "total_mass_kg": data.proxy_model.total_mass_kg,
            "reference_com_error_m": data.proxy_model.reference_com_error_m,
            "calibration_success": data.proxy_model.calibration_success,
            "calibration_cost": data.proxy_model.calibration_cost,
            "assignments": [
                {
                    "source_body": point.source_body_name,
                    "segment": point.segment_name,
                    "mass_kg": point.mass_kg,
                }
                for point in data.proxy_model.points
            ],
        },
        "speed_alignment": {
            "linear_speed_best_lag_s": linear_lag,
            "linear_speed_correlation": linear_corr,
            "angular_speed_best_lag_s": angular_lag,
            "angular_speed_correlation": angular_corr,
        },
        "g1_contact_agreement": {
            "height_threshold_m": sole_contact_height_threshold_m,
            "left_fraction": float(np.mean(data.metrics["g1_left_contact_agreement"])),
            "right_fraction": float(np.mean(data.metrics["g1_right_contact_agreement"])),
        },
        "activity_windows": [asdict(window) for window in data.activity_windows],
        "windows": windows,
        "selected_clips": [asdict(clip) for clip in clips],
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "window"


def _write_frame_metrics(data: AnalysisData, path: Path) -> None:
    metric_names = list(data.metrics)
    header = [
        "frame",
        "source_frame",
        "time_s",
        "activity",
        "contact_phase",
        "left_contact",
        "right_contact",
    ]
    for actor in ("human", "g1_xsens", "g1"):
        header.extend([f"{actor}_com_x_m", f"{actor}_com_y_m", f"{actor}_com_z_m"])
        header.extend([f"{actor}_support_margin_m", f"{actor}_normalized_margin", f"{actor}_support_area_m2"])
        header.extend([f"{actor}_racket_x_m", f"{actor}_racket_y_m", f"{actor}_racket_z_m"])
    header.extend(["g1_left_sole_height_m", "g1_right_sole_height_m", *metric_names])
    actor_arrays = {
        "human": (
            data.human_com_m,
            data.human_support_margin_m,
            data.human_normalized_margin,
            data.human_support_area_m2,
            data.human_racket_position_m,
        ),
        "g1_xsens": (
            data.g1_xsens_com_m,
            data.g1_xsens_support_margin_m,
            data.g1_xsens_normalized_margin,
            data.g1_xsens_support_area_m2,
            data.g1_xsens_racket_position_m,
        ),
        "g1": (
            data.g1_com_m,
            data.g1_support_margin_m,
            data.g1_normalized_margin,
            data.g1_support_area_m2,
            data.g1_racket_position_m,
        ),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(len(data.times_s)):
            row: list[Any] = [
                index,
                int(data.frame_indices[index]),
                float(data.times_s[index]),
                data.activity_labels[index],
                data.contact_phase[index],
                int(data.left_contact[index]),
                int(data.right_contact[index]),
            ]
            for actor in ("human", "g1_xsens", "g1"):
                com, margin, normalized, area, racket = actor_arrays[actor]
                row.extend(com[index].tolist())
                row.extend([margin[index], normalized[index], area[index]])
                row.extend(racket[index].tolist())
            row.extend([data.g1_left_sole_height_m[index], data.g1_right_sole_height_m[index]])
            row.extend(data.metrics[name][index] for name in metric_names)
            writer.writerow(row)


def _write_window_metrics(summary: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for window, values in summary["windows"].items():
        for metric, distribution in values["metrics"].items():
            rows.append({"window": window, "metric": metric, **distribution})
        for actor, support in values["support"].items():
            rows.append(
                {
                    "window": window,
                    "metric": f"{actor}_support_margin_m",
                    **support["margin_m"],
                    "outside_fraction": support["outside_fraction"],
                }
            )
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=180)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def _shade_activities(axes: Sequence[Any], data: AnalysisData) -> None:
    colors = ["#DDEEFF", "#FFF1CC", "#E5F5E0", "#F2E5FF"]
    for window_index, window in enumerate(data.activity_windows):
        for axis in axes:
            axis.axvspan(
                window.start_s / 60.0,
                window.end_s / 60.0,
                color=colors[window_index % len(colors)],
                alpha=0.45,
                label=window.label if axis is axes[0] else None,
            )


def plot_overview(data: AnalysisData, output_dir: Path) -> None:
    x = data.times_s / 60.0
    figure, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    _shade_activities(axes, data)
    for label, values, color in (
        ("Human Xsens", data.human_support_margin_m, BLUE),
        ("G1-sized Xsens proxy", data.g1_xsens_support_margin_m, GREEN),
        ("Physical G1", data.g1_support_margin_m, ORANGE),
    ):
        axes[0].plot(x, values, color=color, linewidth=0.6, alpha=0.8, label=label)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("CoM stability\nmargin [m]")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].plot(x, data.metrics["com_world_error_m"], color=RED, linewidth=0.7, label="World")
    axes[1].plot(x, data.metrics["com_root_error_m"], color=PURPLE, linewidth=0.7, label="Root-relative")
    axes[1].set_ylabel("G1-human CoM\nerror [m]")
    axes[1].legend()
    axes[2].plot(x, data.metrics["racket_world_position_error_m"], color=RED, linewidth=0.7, label="World")
    axes[2].plot(x, data.metrics["racket_root_position_error_m"], color=PURPLE, linewidth=0.7, label="Root-relative")
    axes[2].set_ylabel("Racket position\nerror [m]")
    axes[2].legend()
    axes[3].plot(
        x,
        data.metrics["racket_global_orientation_error_deg"],
        color=RED,
        linewidth=0.7,
        label="Global",
    )
    axes[3].plot(
        x,
        data.metrics["racket_root_orientation_error_deg"],
        color=PURPLE,
        linewidth=0.7,
        label="Root-relative",
    )
    axes[3].set_ylabel("Racket orientation\nerror [deg]")
    axes[3].set_xlabel("Recording time [min]")
    axes[3].legend()
    figure.suptitle(f"Xsens-to-G1 retargeting overview: {data.paths.sequence_name}")
    _save_figure(figure, output_dir, "retargeting_overview")


def plot_distributions(data: AnalysisData, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        ("com_root_error_m", "Root-relative CoM error [m]"),
        ("racket_root_position_error_m", "Root-relative racket position error [m]"),
        ("racket_root_orientation_error_deg", "Root-relative racket orientation error [deg]"),
        ("support_margin_error_m", "Absolute stability-margin error [m]"),
    ]
    labels = [label for label in dict.fromkeys(data.activity_labels.tolist()) if label != "unlabeled"]
    if not labels:
        labels = ["full_sequence"]
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for label in labels:
            mask = _window_mask(data, label)
            values = data.metrics[metric][mask]
            values = values[np.isfinite(values)]
            if values.size:
                axis.hist(values, bins=60, density=True, histtype="step", linewidth=1.5, label=label)
        axis.set_xlabel(title)
        axis.set_ylabel("Density")
        axis.legend()
    figure.suptitle("Error distributions by labeled activity")
    _save_figure(figure, output_dir, "error_distributions")


def _active_polygon(
    footprints: FootprintSeries,
    data: AnalysisData,
    frame: int,
) -> np.ndarray:
    active: list[np.ndarray] = []
    if data.left_contact[frame]:
        active.append(footprints.left[frame, :, :2])
    if data.right_contact[frame]:
        active.append(footprints.right[frame, :, :2])
    if not active:
        return np.empty((0, 2))
    points = np.unique(np.concatenate(active), axis=0)
    if len(points) < 3:
        return points
    try:
        return points[ConvexHull(points).vertices]
    except Exception:
        return points


def plot_support_keyframes(
    data: AnalysisData,
    clips: Sequence[DiagnosticClip],
    output_dir: Path,
) -> None:
    selected = list(clips[:3])
    if not selected:
        return
    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=(13, 4 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )
    actor_values = [
        ("Human Xsens", data.human_footprints, data.human_com_m, BLUE),
        ("G1-sized Xsens", data.g1_xsens_footprints, data.g1_xsens_com_m, GREEN),
        ("Physical G1", data.g1_footprints, data.g1_com_m, ORANGE),
    ]
    for row, clip in enumerate(selected):
        frame = clip.peak_frame
        for column, (label, footprints, com, color) in enumerate(actor_values):
            axis = axes[row, column]
            polygon = _active_polygon(footprints, data, frame)
            if len(polygon):
                closed = np.vstack([polygon, polygon[0]])
                axis.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.2)
                axis.plot(closed[:, 0], closed[:, 1], color=color)
            axis.scatter(com[frame, 0], com[frame, 1], color=RED, marker="x", s=70, label="CoM projection")
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_title(f"{label}\n{clip.label}, t={data.times_s[frame]:.2f} s")
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
    figure.suptitle("CoM projection and support polygons at diagnostic keyframes")
    _save_figure(figure, output_dir, "support_polygon_keyframes")


def plot_racket_trajectories(
    data: AnalysisData,
    clips: Sequence[DiagnosticClip],
    output_dir: Path,
) -> None:
    selected = list(clips[:4])
    if not selected:
        return
    figure = plt.figure(figsize=(15, 4 * len(selected)), constrained_layout=True)
    human_root = root_relative_positions(
        data.human_racket_position_m,
        data.human_root_position_m,
        data.human_root_quaternion_wxyz,
    )
    target_root = root_relative_positions(
        data.g1_xsens_racket_position_m,
        data.g1_xsens_positions_m[:, 0],
        data.g1_xsens_quaternions_wxyz[:, 0],
    )
    g1_root = root_relative_positions(
        data.g1_racket_position_m,
        data.g1_root_position_m,
        data.g1_root_quaternion_wxyz,
    )
    for row, clip in enumerate(selected, start=1):
        axis = figure.add_subplot(len(selected), 1, row, projection="3d")
        interval = slice(clip.start_frame, clip.end_frame)
        for label, values, color in (
            ("Human Xsens", human_root, BLUE),
            ("G1-sized Xsens", target_root, GREEN),
            ("Physical G1", g1_root, ORANGE),
        ):
            axis.plot(*values[interval].T, color=color, label=label)
        axis.set_title(f"{clip.label}: {clip.start_s:.2f}-{clip.end_s:.2f} s")
        axis.set_xlabel("root x [m]")
        axis.set_ylabel("root y [m]")
        axis.set_zlabel("root z [m]")
        axis.legend()
    figure.suptitle("Root-relative racket trajectories in selected windows")
    _save_figure(figure, output_dir, "racket_trajectory_windows")


def write_report(data: AnalysisData, summary: Mapping[str, Any], path: Path) -> None:
    full = summary["windows"]["full_sequence"]
    metrics = full["metrics"]
    headline_rows = []
    for label, metric_name, precision in (
        ("Root-relative CoM error [m]", "com_root_error_m", 4),
        ("Root-relative racket position error [m]", "racket_root_position_error_m", 4),
        ("Root-relative racket orientation error [deg]", "racket_root_orientation_error_deg", 2),
        ("Stability-margin error [m]", "support_margin_error_m", 4),
    ):
        values = metrics[metric_name]
        headline_rows.append(
            f"| {label} | {values['median']:.{precision}f} | "
            f"{values['p95']:.{precision}f} | {values['max']:.{precision}f} |"
        )
    headline_table = "\n".join(headline_rows)
    report = f"""# Xsens-to-G1 retargeting analysis

## Inputs

- Sequence: `{data.paths.sequence_name}`
- Xsens HDF5: `{data.paths.hdf5_path}`
- Retargeted G1 NPZ: `{data.paths.qpos_npz}`
- Samples: {len(data.times_s):,} at {data.fps:g} Hz
- Time interval: {data.times_s[0]:.3f}-{data.times_s[-1]:.3f} s

## Interpretation

Human-subject Xsens is the error reference. World errors include the intentional G1 root-motion scaling.
Root-relative errors express each signal in its actor's full six-DoF pelvis/root frame and therefore isolate
body-relative reproduction. The G1-sized Xsens CoM is a model-based proxy: physical G1 inertial masses and their
calibrated T-pose centroids are attached to the reduced Xsens segments. The racket is excluded from every CoM.

Support polygons use the measured human heel/toe contact state for all actors, but actor-specific outsole geometry.
This makes stability margins comparable while the exported G1 sole heights expose lost or penetrated contacts.

## Full-sequence headline results

| Quantity | Median | P95 | Maximum |
|---|---:|---:|---:|
{headline_table}

## Generated artifacts

- `summary.json`: complete methods, distributions, activities, support statistics, proxy mapping, and clips.
- `frame_metrics.csv`: aligned per-frame signals and errors.
- `window_metrics.csv`: compact full/activity/contact-phase statistics.
- `selected_clips.json`: automatically selected diagnostic windows.
- `retargeting_overview.*`, `error_distributions.*`, `support_polygon_keyframes.*`, and
  `racket_trajectory_windows.*`: PNG/PDF figures.

## Limitations

- The native Xsens CoM is an MVN estimate rather than independent force-plate ground truth.
- The reduced G1-sized Xsens avatar has no native inertial model; its reported CoM is explicitly a proxy.
- Root-relative metrics remove root translation and orientation errors, so they must be interpreted alongside
  the world-frame metrics.
- The support state comes from Xsens for all actors; G1 sole-height columns should be checked for contact mismatch.
"""
    path.write_text(report, encoding="utf-8")


def export_analysis(
    data: AnalysisData,
    config: Config,
) -> dict[str, Any]:
    output_dir = data.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = select_diagnostic_clips(data, config.clip_duration_s, config.max_clips)
    summary = build_summary(
        data,
        clips,
        config.max_speed_lag_s,
        config.sole_contact_height_threshold_m,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "selected_clips.json").write_text(
        json.dumps([asdict(clip) for clip in clips], indent=2),
        encoding="utf-8",
    )
    _write_frame_metrics(data, output_dir / "frame_metrics.csv")
    _write_window_metrics(summary, output_dir / "window_metrics.csv")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})
    plot_overview(data, output_dir)
    plot_distributions(data, output_dir)
    plot_support_keyframes(data, clips, output_dir)
    plot_racket_trajectories(data, clips, output_dir)
    write_report(data, summary, output_dir / "analysis_report.md")
    return summary


def _write_batch_summary(summaries: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {"sequence_count": len(summaries), "sequences": list(summaries)}
    (output_root / "batch_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row: dict[str, Any] = {
            "sequence_name": summary["sequence_name"],
            "frame_count": summary["frame_count"],
            "duration_s": summary["time_range_s"][1] - summary["time_range_s"][0],
        }
        headline = summary["windows"]["full_sequence"]["metrics"]
        for metric in (
            "com_root_error_m",
            "racket_root_position_error_m",
            "racket_root_orientation_error_deg",
            "support_margin_error_m",
        ):
            row[f"{metric}_median"] = headline[metric]["median"]
            row[f"{metric}_p95"] = headline[metric]["p95"]
        rows.append(row)
    with (output_root / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _viser_polygon_points(polygon_xy: np.ndarray, z: float = 0.01) -> np.ndarray:
    if len(polygon_xy) < 2:
        return np.zeros((0, 2, 3))
    closed = np.vstack([polygon_xy, polygon_xy[0]])
    points = np.column_stack([closed, np.full(len(closed), z)])
    return np.stack([points[:-1], points[1:]], axis=1)


def launch_viser(
    data: AnalysisData,
    clips: Sequence[DiagnosticClip],
    config: Config,
) -> None:
    """Launch a purpose-built three-actor Viser diagnostic or record clips."""

    import viser  # type: ignore[import-not-found]  # noqa: PLC0415
    import yourdfpy  # type: ignore[import-untyped]  # noqa: PLC0415
    from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: PLC0415

    from holosoma_retargeting.src.recording_utils import record_viser_sequence  # noqa: PLC0415
    from holosoma_retargeting.src.viser_utils import (  # noqa: PLC0415
        QposViserApplier,
        create_timed_motion_control_sliders,
    )
    from holosoma_retargeting.viser_player import (  # noqa: PLC0415
        add_g1_tennis_racket,
        update_g1_tennis_racket_pose,
    )

    server = viser.ViserServer(port=config.viser_port)

    class TreeActor:
        def __init__(self, model: KinematicTree, root: str) -> None:
            self.root = server.scene.add_frame(root, show_axes=False)
            self.frames: dict[str, Any] = {}
            self.source_to_body: dict[str, str] = {}
            self.meshes: list[Any] = []
            for body in model.bodies:
                source = _body_source_name(model, body.name)
                self.source_to_body[_normalize_source_name(source)] = body.name
                frame = server.scene.add_frame(f"{root}/{_slug(body.name)}", show_axes=False)
                self.frames[body.name] = frame
                for mesh_index, mesh in enumerate(body.meshes):
                    self.meshes.append(
                        server.scene.add_mesh_simple(
                            f"{root}/{_slug(body.name)}/mesh_{mesh_index}",
                            vertices=np.asarray(mesh.vertices_m),
                            faces=np.asarray(mesh.faces, dtype=np.int32),
                            color=mesh.color_rgb,
                            side="double",
                        )
                    )

        def apply(self, names: Sequence[str], positions: np.ndarray, quaternions: np.ndarray) -> None:
            for index, source in enumerate(names):
                body = self.source_to_body.get(_normalize_source_name(source))
                if body is not None:
                    self.frames[body].position = positions[index]
                    self.frames[body].wxyz = quaternions[index]

        def set_visible(self, visible: bool) -> None:
            for mesh in self.meshes:
                mesh.visible = visible

    human_actor = TreeActor(data.human_model, "/human")
    target_actor = TreeActor(data.g1_xsens_model, "/g1_xsens")
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    urdf_path = data.robot_model_path.with_suffix(".urdf")
    robot_urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(urdf_path.parent), load_meshes=True)
    robot_actor = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot")
    robot_applier = QposViserApplier(
        viser_robot=robot_actor,
        robot_base_frame=robot_root,
        robot_dof=29,
        contains_object_in_qpos=False,
    )
    robot_racket, _ = add_g1_tennis_racket(server)
    server.scene.add_grid("/grid", width=20.0, height=20.0)
    com_handles = {
        actor: server.scene.add_point_cloud(
            f"/analysis/{actor}/com",
            points=np.zeros((1, 3)),
            colors=np.asarray([ACTOR_RGB[actor]], dtype=np.uint8),
            point_size=0.035,
            point_shape="circle",
        )
        for actor in ACTOR_RGB
    }
    projection_handles = {
        actor: server.scene.add_line_segments(
            f"/analysis/{actor}/projection",
            points=np.zeros((1, 2, 3)),
            colors=ACTOR_RGB[actor],
            line_width=2.0,
        )
        for actor in ACTOR_RGB
    }
    polygon_handles = {
        actor: server.scene.add_line_segments(
            f"/analysis/{actor}/support",
            points=np.zeros((1, 2, 3)),
            colors=ACTOR_RGB[actor],
            line_width=4.0,
        )
        for actor in ACTOR_RGB
    }
    trail_handles = {
        actor: server.scene.add_line_segments(
            f"/analysis/{actor}/racket_trail",
            points=np.zeros((1, 2, 3)),
            colors=ACTOR_RGB[actor],
            line_width=3.0,
        )
        for actor in ACTOR_RGB
    }
    with server.gui.add_folder("Analysis"):
        overlay_layout = server.gui.add_checkbox("Overlay actors", initial_value=config.actor_spacing_m == 0.0)
        show_com = server.gui.add_checkbox("Show CoM and support", initial_value=True)
        show_trails = server.gui.add_checkbox("Show racket trails", initial_value=True)
        status = server.gui.add_markdown("Metrics will update during playback.")

    trail_frames = max(2, round(config.trail_duration_s * data.fps))

    def offsets() -> dict[str, np.ndarray]:
        spacing = 0.0 if overlay_layout.value else config.actor_spacing_m
        return {
            "human": np.array([0.0, -spacing, 0.0]),
            "g1_xsens": np.zeros(3),
            "g1": np.array([0.0, spacing, 0.0]),
        }

    def apply_frame(frame: int) -> None:
        index = int(np.clip(frame, 0, len(data.times_s) - 1))
        actor_offsets = offsets()
        human_actor.root.position = actor_offsets["human"]
        target_actor.root.position = actor_offsets["g1_xsens"]
        human_actor.apply(
            data.segment_names,
            data.human_positions_m[index],
            data.human_quaternions_wxyz[index],
        )
        target_actor.apply(
            data.segment_names,
            data.g1_xsens_positions_m[index],
            data.g1_xsens_quaternions_wxyz[index],
        )
        q = data.qpos[index].copy()
        q[:3] += actor_offsets["g1"]
        robot_applier.apply_qpos(q, has_object_input=False)
        update_g1_tennis_racket_pose(robot_racket, robot_urdf)
        actor_data = {
            "human": (data.human_com_m, data.human_footprints, data.human_racket_position_m),
            "g1_xsens": (data.g1_xsens_com_m, data.g1_xsens_footprints, data.g1_xsens_racket_position_m),
            "g1": (data.g1_com_m, data.g1_footprints, data.g1_racket_position_m),
        }
        for actor, (com, footprints, racket) in actor_data.items():
            offset = actor_offsets[actor]
            point = com[index] + offset
            com_handles[actor].points = point[None, :]
            projection_handles[actor].points = np.array([[point, [point[0], point[1], 0.01]]])
            polygon = _active_polygon(footprints, data, index)
            polygon_handles[actor].points = _viser_polygon_points(polygon + offset[:2])
            start = max(0, index - trail_frames)
            trail = racket[start : index + 1] + offset
            if len(trail) >= 2:
                trail_handles[actor].points = np.stack([trail[:-1], trail[1:]], axis=1)
            com_handles[actor].visible = bool(show_com.value)
            projection_handles[actor].visible = bool(show_com.value)
            polygon_handles[actor].visible = bool(show_com.value)
            trail_handles[actor].visible = bool(show_trails.value)
        status.content = (
            f"**t={data.times_s[index]:.2f} s**  \n"
            f"Root-relative racket position error: "
            f"{data.metrics['racket_root_position_error_m'][index]:.3f} m  \n"
            f"Root-relative racket orientation error: "
            f"{data.metrics['racket_root_orientation_error_deg'][index]:.1f} deg  \n"
            f"CoM stability-margin error: {data.metrics['support_margin_error_m'][index]:.3f} m"
        )

    apply_frame(0)
    if config.viser_mode == "record-clips":
        for clip in clips:
            indices = np.arange(clip.start_frame, clip.end_frame, dtype=int).tolist()
            record_viser_sequence(
                server=server,
                apply_frame=apply_frame,
                frame_indices=indices,
                output_path=str(data.paths.output_dir / f"viser_{clip.label}.mp4"),
                width=config.record_width,
                height=config.record_height,
                fps=data.fps,
                connect_timeout=config.record_connect_timeout_s,
                start_delay=1.0,
                settle_time=0.0,
                warmup_renders=0,
                transport_format="jpeg",
            )
        return
    create_timed_motion_control_sliders(
        server,
        data.times_s,
        lambda current_time: apply_frame(int(np.searchsorted(data.times_s, current_time, side="right") - 1)),
        initial_fps=data.fps,
        initial_interp_mult=1,
        initial_playback_speed=1.0,
        loop=False,
    )
    print(f"Open the Viser URL above for {data.paths.sequence_name}. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping Viser player.")


def run(config: Config) -> list[dict[str, Any]]:
    paths = resolve_sequence_paths(config)
    summaries: list[dict[str, Any]] = []
    last_data: AnalysisData | None = None
    for sequence_path in paths:
        print(f"[analysis] processing {sequence_path.sequence_name}")
        data = load_and_analyze(config, sequence_path)
        summary = export_analysis(data, config)
        summaries.append(summary)
        last_data = data
        print(f"[analysis] wrote {sequence_path.output_dir}")
    output_root = paths[0].output_dir.parent
    _write_batch_summary(summaries, output_root)
    if config.viser_mode != "none":
        assert last_data is not None
        clips = tuple(DiagnosticClip(**values) for values in summaries[0]["selected_clips"])
        launch_viser(last_data, clips, config)
    return summaries


def main(config: Config) -> None:
    summaries = run(config)
    print(json.dumps({"processed_sequences": [summary["sequence_name"] for summary in summaries]}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
