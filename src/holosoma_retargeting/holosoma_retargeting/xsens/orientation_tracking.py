"""Xsens orientation and segment-axis target construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from holosoma_retargeting.config_types.data_type import XSENS_DEMO_JOINTS
from holosoma_retargeting.transformation_utils import (
    rotation_matrices_as_wxyz,
    rotation_matrices_from_wxyz,
)

quat_wijk_to_matrix = rotation_matrices_from_wxyz


def matrix_to_quat_wijk(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to scalar-first quaternions without changing their sign branch."""

    return rotation_matrices_as_wxyz(matrix, canonical=False)


@dataclass(frozen=True)
class XsensAxisSpec:
    """Specification for one direction-only Xsens-to-robot axis target."""

    name: str
    xsens_segment: str
    xsens_axis_start: str
    xsens_axis_end: str
    robot_axis_start: str
    robot_axis_end: str | None
    weight: float = 1.0
    robot_local_axis: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        has_end_frame = self.robot_axis_end is not None
        has_local_axis = self.robot_local_axis is not None
        if has_end_frame == has_local_axis:
            raise ValueError("An axis spec must define exactly one robot end frame or body-fixed local axis")


@dataclass(frozen=True)
class XsensOrientationTargets:
    """Per-frame orientation and segment-axis targets for retargeting."""

    orientation_names: list[str]
    orientation_robot_link_names: list[str]
    orientation_offsets_wijk: np.ndarray
    orientation_target_rotations: np.ndarray
    axis_names: list[str]
    axis_xsens_segment_names: list[str]
    axis_robot_start_link_names: list[str]
    axis_robot_end_link_names: list[str]
    axis_robot_local_vectors: np.ndarray
    axis_target_vectors: np.ndarray
    axis_weights: np.ndarray


class XsensOrientationCalibration(Protocol):
    """Calibration metadata required to construct dynamic orientation targets."""

    active_orientation_mapping_names: list[str]
    robot_link_names: list[str]
    orientation_offsets_wijk: np.ndarray
    axis_names: list[str]
    axis_xsens_segment_names: list[str]
    axis_local_tpose_xyz: np.ndarray
    axis_robot_start_link_names: list[str]
    axis_robot_end_link_names: list[str]
    axis_robot_local_vectors: np.ndarray
    axis_weights: np.ndarray


def describe_xsens_orientation_correspondences(
    orientation_names: Sequence[str],
    robot_link_names: Sequence[str],
    orientation_offsets_wijk: np.ndarray,
) -> tuple[str, ...]:
    """Describe the calibrated Xsens-segment to robot-link rotations."""

    offsets = np.asarray(orientation_offsets_wijk, dtype=float)
    if len(orientation_names) != len(robot_link_names) or len(orientation_names) != len(offsets):
        raise ValueError("Orientation segment names, robot link names, and offsets must have the same length")

    lines = [
        "R_G1_target_world(t) = R_Xsens_segment_world(t) @ R_offset",
        "R_offset = R_Xsens_segment_Tpose_world^T @ R_G1_link_Tpose_world",
    ]
    for xsens_name, robot_link, offset in zip(
        orientation_names,
        robot_link_names,
        offsets,
        strict=True,
    ):
        normalized_offset = np.asarray(offset, dtype=float)
        normalized_offset /= max(float(np.linalg.norm(normalized_offset)), 1e-12)
        offset_angle_deg = float(np.degrees(2.0 * np.arccos(np.clip(abs(normalized_offset[0]), 0.0, 1.0))))
        offset_text = ", ".join(f"{value:+.6f}" for value in normalized_offset)
        lines.append(
            f"{xsens_name} -> {robot_link}: offset_wxyz=({offset_text}), offset_angle={offset_angle_deg:.2f} deg"
        )
    return tuple(lines)


XSENS_AXIS_SPECS = (
    XsensAxisSpec(
        "pelvis_lateral",
        "Pelvis",
        "Right Upper Leg",
        "Left Upper Leg",
        "right_hip_pitch_link",
        "left_hip_pitch_link",
    ),
    XsensAxisSpec(
        "shoulder_lateral",
        "L5",
        "Right Shoulder",
        "Left Shoulder",
        "right_shoulder_pitch_link",
        "left_shoulder_pitch_link",
    ),
    XsensAxisSpec("torso_up", "L5", "Pelvis", "L5", "pelvis_contour_link", "torso_link"),
    XsensAxisSpec(
        "left_upper_arm",
        "Left Upper Arm",
        "Left Upper Arm",
        "Left Forearm",
        "left_shoulder_yaw_link",
        "left_elbow_link",
    ),
    XsensAxisSpec(
        "right_upper_arm",
        "Right Upper Arm",
        "Right Upper Arm",
        "Right Forearm",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ),
    XsensAxisSpec(
        "left_forearm",
        "Left Forearm",
        "Left Forearm",
        "Left Hand",
        "left_elbow_link",
        "left_wrist_yaw_link",
    ),
    XsensAxisSpec(
        "right_forearm",
        "Right Forearm",
        "Right Forearm",
        "Right Hand",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ),
    # Upper Leg positions use the distal hip-yaw origins, but the direction
    # spans the complete spatial hip cluster from hip pitch to knee. Starting
    # at hip yaw over-rotates this target and regresses distal leg/foot fit.
    XsensAxisSpec(
        "left_thigh",
        "Left Upper Leg",
        "Left Upper Leg",
        "Left Lower Leg",
        "left_hip_pitch_link",
        "left_knee_link",
    ),
    XsensAxisSpec(
        "right_thigh",
        "Right Upper Leg",
        "Right Upper Leg",
        "Right Lower Leg",
        "right_hip_pitch_link",
        "right_knee_link",
    ),
    XsensAxisSpec(
        "left_shank",
        "Left Lower Leg",
        "Left Lower Leg",
        "Left Foot",
        "left_knee_link",
        "left_ankle_pitch_link",
    ),
    XsensAxisSpec(
        "right_shank",
        "Right Lower Leg",
        "Right Lower Leg",
        "Right Foot",
        "right_knee_link",
        "right_ankle_pitch_link",
    ),
    XsensAxisSpec(
        "left_foot_forward",
        "Left Foot",
        "Left Foot",
        "Left Toe",
        "left_ankle_roll_link",
        None,
        robot_local_axis=(1.0, 0.0, 0.0),
    ),
    XsensAxisSpec(
        "right_foot_forward",
        "Right Foot",
        "Right Foot",
        "Right Toe",
        "right_ankle_roll_link",
        None,
        robot_local_axis=(1.0, 0.0, 0.0),
    ),
)


def normalize_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalize a vector with a deterministic fallback for degenerate inputs."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is None:
        fallback = np.zeros_like(vector)
    return np.asarray(fallback, dtype=float)


def segment_index(name: str) -> int:
    """Return the standard Xsens body segment index."""
    return XSENS_DEMO_JOINTS.index(name)


def build_xsens_axis_calibration_metadata(
    *,
    tpose_positions_m: np.ndarray,
    tpose_quaternions_wijk: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build serializable axis metadata from an Xsens T-pose."""
    positions = np.asarray(tpose_positions_m, dtype=float)
    rotations = quat_wijk_to_matrix(tpose_quaternions_wijk)
    axis_names = []
    axis_segments = []
    axis_start_links = []
    axis_end_links = []
    robot_local_vectors = []
    local_axes = []
    weights = []

    for spec in XSENS_AXIS_SPECS:
        start = positions[segment_index(spec.xsens_axis_start)]
        end = positions[segment_index(spec.xsens_axis_end)]
        world_axis = normalize_vector(end - start, np.array([1.0, 0.0, 0.0]))
        local_axis = rotations[segment_index(spec.xsens_segment)].T @ world_axis
        axis_names.append(spec.name)
        axis_segments.append(spec.xsens_segment)
        axis_start_links.append(spec.robot_axis_start)
        axis_end_links.append(spec.robot_axis_end or "")
        robot_local_vectors.append(spec.robot_local_axis or (0.0, 0.0, 0.0))
        local_axes.append(normalize_vector(local_axis, np.array([1.0, 0.0, 0.0])))
        weights.append(spec.weight)

    return {
        "axis_names": np.asarray(axis_names, dtype=str),
        "axis_xsens_segment_names": np.asarray(axis_segments, dtype=str),
        "axis_local_tpose_xyz": np.asarray(local_axes, dtype=float),
        "axis_robot_start_link_names": np.asarray(axis_start_links, dtype=str),
        "axis_robot_end_link_names": np.asarray(axis_end_links, dtype=str),
        "axis_robot_local_vectors": np.asarray(robot_local_vectors, dtype=float),
        "axis_weights": np.asarray(weights, dtype=float),
    }


def build_xsens_orientation_targets(
    *,
    orientation_names: list[str],
    orientation_robot_link_names: list[str],
    orientation_offsets_wijk: np.ndarray,
    axis_names: list[str],
    axis_xsens_segment_names: list[str],
    axis_local_tpose_xyz: np.ndarray,
    axis_robot_start_link_names: list[str],
    axis_robot_end_link_names: list[str],
    axis_robot_local_vectors: np.ndarray,
    axis_weights: np.ndarray,
    motion_quaternions_wijk: np.ndarray,
    segment_names: list[str],
) -> XsensOrientationTargets:
    """Create dynamic orientation and axis targets from calibration metadata."""

    motion_quaternions = np.asarray(motion_quaternions_wijk, dtype=float)
    if motion_quaternions.ndim != 3 or motion_quaternions.shape[-1] != 4:
        raise ValueError(f"Expected motion quaternions with shape (T, J, 4), got {motion_quaternions.shape}")
    if motion_quaternions.shape[1] != len(segment_names):
        raise ValueError(
            "Motion quaternion segment count does not match segment names: "
            f"{motion_quaternions.shape[1]} vs {len(segment_names)}"
        )

    rotations = quat_wijk_to_matrix(motion_quaternions)
    segment_to_index = {name: idx for idx, name in enumerate(segment_names)}

    orientation_offsets = np.asarray(orientation_offsets_wijk, dtype=float)
    if len(orientation_names) != len(orientation_robot_link_names) or len(orientation_names) != len(
        orientation_offsets
    ):
        raise ValueError("Calibration orientation mapping names, robot links, and offsets must have the same length")
    offset_rotations = quat_wijk_to_matrix(orientation_offsets)
    orientation_targets = np.zeros((motion_quaternions.shape[0], len(orientation_names), 3, 3), dtype=float)
    for mapping_idx, xsens_name in enumerate(orientation_names):
        if xsens_name not in segment_to_index:
            raise ValueError(f"Calibration orientation segment is not present in motion data: {xsens_name}")
        source_idx = segment_to_index[xsens_name]
        orientation_targets[:, mapping_idx] = rotations[:, source_idx] @ offset_rotations[mapping_idx]

    axis_targets = np.zeros((motion_quaternions.shape[0], len(axis_names), 3), dtype=float)
    robot_local_vectors = np.asarray(axis_robot_local_vectors, dtype=float).copy()
    if robot_local_vectors.shape != (len(axis_names), 3):
        raise ValueError(
            f"axis_robot_local_vectors must have shape ({len(axis_names)}, 3), got {robot_local_vectors.shape}"
        )
    if len(axis_robot_start_link_names) != len(axis_names) or len(axis_robot_end_link_names) != len(axis_names):
        raise ValueError("Calibration axis names, start frames, and end frames must have the same length")
    for axis_idx, (end_frame, local_vector) in enumerate(
        zip(axis_robot_end_link_names, robot_local_vectors, strict=True)
    ):
        has_end_frame = bool(end_frame)
        has_local_axis = float(np.linalg.norm(local_vector)) > 1e-12
        if has_end_frame == has_local_axis:
            raise ValueError(
                f"Axis '{axis_names[axis_idx]}' must define exactly one robot end frame or body-fixed local vector"
            )
        if has_local_axis:
            robot_local_vectors[axis_idx] = normalize_vector(local_vector)
    for axis_idx, xsens_name in enumerate(axis_xsens_segment_names):
        if xsens_name not in segment_to_index:
            raise ValueError(f"Calibration axis-driving segment is not present in motion data: {xsens_name}")
        source_idx = segment_to_index[xsens_name]
        axis_targets[:, axis_idx] = rotations[:, source_idx] @ axis_local_tpose_xyz[axis_idx]
        norms = np.linalg.norm(axis_targets[:, axis_idx], axis=-1, keepdims=True)
        axis_targets[:, axis_idx] = axis_targets[:, axis_idx] / np.maximum(norms, 1e-12)

    return XsensOrientationTargets(
        orientation_names=orientation_names,
        orientation_robot_link_names=orientation_robot_link_names,
        orientation_offsets_wijk=orientation_offsets,
        orientation_target_rotations=orientation_targets,
        axis_names=axis_names,
        axis_xsens_segment_names=axis_xsens_segment_names,
        axis_robot_start_link_names=axis_robot_start_link_names,
        axis_robot_end_link_names=axis_robot_end_link_names,
        axis_robot_local_vectors=robot_local_vectors,
        axis_target_vectors=axis_targets,
        axis_weights=np.asarray(axis_weights, dtype=float),
    )


def build_xsens_orientation_targets_from_calibration(
    calibration: XsensOrientationCalibration,
    *,
    motion_quaternions_wijk: np.ndarray,
    segment_names: list[str],
) -> XsensOrientationTargets:
    """Create motion targets from shared Xsens orientation-calibration metadata."""

    return build_xsens_orientation_targets(
        orientation_names=calibration.active_orientation_mapping_names,
        orientation_robot_link_names=calibration.robot_link_names,
        orientation_offsets_wijk=calibration.orientation_offsets_wijk,
        axis_names=calibration.axis_names,
        axis_xsens_segment_names=calibration.axis_xsens_segment_names,
        axis_local_tpose_xyz=calibration.axis_local_tpose_xyz,
        axis_robot_start_link_names=calibration.axis_robot_start_link_names,
        axis_robot_end_link_names=calibration.axis_robot_end_link_names,
        axis_robot_local_vectors=calibration.axis_robot_local_vectors,
        axis_weights=calibration.axis_weights,
        motion_quaternions_wijk=motion_quaternions_wijk,
        segment_names=segment_names,
    )


def load_xsens_orientation_targets(
    *,
    calibration_path: str | Path,
    motion_quaternions_wijk: np.ndarray,
    segment_names: list[str],
) -> XsensOrientationTargets:
    """Load a calibration artifact and create per-frame orientation/axis targets."""
    calibration_path = Path(calibration_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Xsens orientation calibration artifact not found: {calibration_path}")

    with np.load(calibration_path, allow_pickle=True) as data:
        required = (
            "active_orientation_mapping_names",
            "robot_link_names",
            "orientation_offsets_wijk",
            "axis_names",
            "axis_xsens_segment_names",
            "axis_local_tpose_xyz",
            "axis_robot_start_link_names",
            "axis_robot_end_link_names",
            "axis_robot_local_vectors",
            "axis_weights",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "Xsens calibration artifact is missing orientation/axis metadata. "
                f"Regenerate it with examples/xsens_tennis/calibrate_tpose.py. Missing: {missing}"
            )

        return build_xsens_orientation_targets(
            orientation_names=[str(value) for value in data["active_orientation_mapping_names"]],
            orientation_robot_link_names=[str(value) for value in data["robot_link_names"]],
            orientation_offsets_wijk=np.asarray(data["orientation_offsets_wijk"], dtype=float),
            axis_names=[str(value) for value in data["axis_names"]],
            axis_xsens_segment_names=[str(value) for value in data["axis_xsens_segment_names"]],
            axis_local_tpose_xyz=np.asarray(data["axis_local_tpose_xyz"], dtype=float),
            axis_robot_start_link_names=[str(value) for value in data["axis_robot_start_link_names"]],
            axis_robot_end_link_names=[str(value) for value in data["axis_robot_end_link_names"]],
            axis_robot_local_vectors=np.asarray(data["axis_robot_local_vectors"], dtype=float),
            axis_weights=np.asarray(data["axis_weights"], dtype=float),
            motion_quaternions_wijk=motion_quaternions_wijk,
            segment_names=segment_names,
        )
