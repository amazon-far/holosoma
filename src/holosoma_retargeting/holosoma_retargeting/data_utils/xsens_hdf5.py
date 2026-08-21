"""Utilities for loading ActionNet-style Xsens HDF5 segment data."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from holosoma_retargeting.transformation_utils import (
    rotation_matrices_as_wxyz,
    rotation_matrices_from_wxyz,
)

XSENS_DEVICE_NAME = "xsens-segments"
XSENS_TPOSE_DEVICE_NAME = "xsens-segments-tpose"
XSENS_POSITION_STREAM_NAMES = ("body_position_xyz_m", "position_cm")
XSENS_ORIENTATION_STREAM_NAME = "body_orientation_quaternion_wijk"
XSENS_JOINT_DEVICE_NAME = "xsens-joints"
XSENS_JOINT_STREAM_NAMES = (
    "body_joint_angles_eulerZXY_xyz_rad",
    "body_joint_angles_eulerXZY_xyz_rad",
)
XSENS_Y_UP_TO_RETARGETING_Z_UP_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

XSENS_BODY_SEGMENT_NAMES = [
    "Pelvis",
    "L5",
    "L3",
    "T12",
    "T8",
    "Neck",
    "Head",
    "Right Shoulder",
    "Right Upper Arm",
    "Right Forearm",
    "Right Hand",
    "Left Shoulder",
    "Left Upper Arm",
    "Left Forearm",
    "Left Hand",
    "Right Upper Leg",
    "Right Lower Leg",
    "Right Foot",
    "Right Toe",
    "Left Upper Leg",
    "Left Lower Leg",
    "Left Foot",
    "Left Toe",
]
XSENS_TRACKED_PROP_NAMES = ["RightHandSword"]


@dataclass(frozen=True)
class XsensHdf5Motion:
    """Loaded Xsens segment positions and orientations in retargeting coordinates."""

    positions_m: np.ndarray
    times_s: np.ndarray
    stream_name: str
    segment_names: list[str]
    source_indices: list[int]
    quaternions_wijk: np.ndarray
    orientation_stream_name: str
    recording_times_s: np.ndarray | None = None


@dataclass(frozen=True)
class XsensHdf5Tpose:
    """Loaded Xsens static T-pose in retargeting coordinates."""

    positions_m: np.ndarray
    quaternions_wijk: np.ndarray
    variant: str
    segment_names: list[str]
    source_indices: list[int]


@dataclass(frozen=True)
class SegmentPoseSet:
    """Static world poses for every segment in one Xsens reference pose."""

    positions_m: np.ndarray
    quaternions_wijk: np.ndarray
    variant: str


@dataclass(frozen=True)
class JointRotationMetadata:
    """Semantic Xsens rotation channels for one articulated joint."""

    joint_name: str
    components: tuple[str, ...]
    available_euler_streams: tuple[str, ...]


@dataclass(frozen=True)
class XsensHdf5Inventory:
    """Lightweight description of the Xsens content available in an HDF5 file."""

    path: Path
    segment_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    pose_variants: tuple[str, ...]
    joint_stream_names: tuple[str, ...]
    has_landmarks: bool


@dataclass(frozen=True)
class XsensHdf5Calibration:
    """Complete subject calibration embedded in one Xsens HDF5 recording."""

    source_path: Path
    source_stream_name: str
    segment_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    tpose: SegmentPoseSet
    tpose_isb: SegmentPoseSet | None
    identity_pose: SegmentPoseSet | None
    landmarks_m: Mapping[str, Mapping[str, np.ndarray]]
    joint_rotation_metadata: Mapping[str, JointRotationMetadata]
    joint_stream_names: tuple[str, ...]
    mvn_version: str | None
    mvnx_version: str | None


def _import_h5py():
    try:
        import h5py  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Loading Xsens HDF5 files requires the optional dependency 'h5py'.") from exc
    return h5py


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _parse_ordered_dict(value: Any) -> OrderedDict[str, OrderedDict[str, list[float]]]:
    """Parse Xsens's stringified ``OrderedDict`` landmark metadata safely."""

    value = _decode_attr(value)
    if not isinstance(value, str):
        raise TypeError("segment_mesh_points_body_xyz_cm must be stored as text")

    expression = ast.parse(value, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.List,
        ast.Tuple,
        ast.Constant,
        ast.UnaryOp,
        ast.USub,
    )
    for node in ast.walk(expression):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported syntax in Xsens landmark metadata: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id != "OrderedDict":
            raise ValueError(f"Unsupported name in Xsens landmark metadata: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "OrderedDict" or node.keywords:
                raise ValueError("Only OrderedDict(...) calls are allowed in Xsens landmark metadata")

    parsed = eval(  # noqa: S307 - the AST is strictly whitelisted above.
        compile(expression, filename="<xsens-landmarks>", mode="eval"),
        {"__builtins__": {}, "OrderedDict": OrderedDict},
        {},
    )
    if not isinstance(parsed, OrderedDict):
        raise ValueError("Xsens landmark metadata did not decode to an OrderedDict")
    return parsed


def _parse_string_sequence(value: Any, *, attribute_name: str) -> tuple[str, ...]:
    value = _decode_attr(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if not isinstance(value, str):
        raise TypeError(f"{attribute_name} must be stored as text or a sequence")
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{attribute_name} must decode to a sequence")
    return tuple(str(item) for item in parsed)


def parse_data_headings(data_headings_attr: Any) -> list[str] | None:
    """Parse an ActionNet `Data headings` attribute into heading strings."""
    if data_headings_attr is None:
        return None

    data_headings_attr = _decode_attr(data_headings_attr)
    if isinstance(data_headings_attr, np.ndarray):
        data_headings_attr = data_headings_attr.tolist()
    if isinstance(data_headings_attr, list):
        return [str(item) for item in data_headings_attr]
    if not isinstance(data_headings_attr, str):
        return None

    parsed = ast.literal_eval(data_headings_attr)
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed]


def segment_names_from_headings(headings: list[str] | None) -> list[str] | None:
    """Extract segment names from flattened `Segment (axis)` headings."""
    if headings is None:
        return None
    if len(headings) % 3 != 0:
        return None

    segment_names: list[str] = []
    for i in range(0, len(headings), 3):
        names = []
        for heading in headings[i : i + 3]:
            if "(" in heading:
                names.append(heading.rsplit("(", 1)[0].strip())
            else:
                names.append(heading.strip())
        if len(set(names)) != 1:
            return None
        segment_names.append(names[0])
    return segment_names


def parse_segment_names(segment_names_attr: Any) -> list[str] | None:
    """Parse an ActionNet `segment_names_body` attribute into segment names."""
    if segment_names_attr is None:
        return None

    segment_names_attr = _decode_attr(segment_names_attr)
    if isinstance(segment_names_attr, np.ndarray):
        segment_names_attr = segment_names_attr.tolist()
    if isinstance(segment_names_attr, list):
        return [str(item) for item in segment_names_attr]
    if not isinstance(segment_names_attr, str):
        return None

    parsed = ast.literal_eval(segment_names_attr)
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed]


def _segment_names_from_attrs(hdf5_obj: Any) -> list[str] | None:
    headings = parse_data_headings(hdf5_obj.attrs.get("Data headings"))
    segment_names = segment_names_from_headings(headings)
    if segment_names is None:
        segment_names = parse_segment_names(hdf5_obj.attrs.get("segment_names_body"))
    return segment_names


def transform_xsens_y_up_to_retargeting_z_up(positions: np.ndarray) -> np.ndarray:
    """Convert Xsens coordinates `[x, y_up, z]` to retargeting `[x, y, z_up]`."""
    return positions[..., [0, 2, 1]]


def transform_xsens_stream_to_retargeting(positions: np.ndarray, stream_name: str) -> np.ndarray:
    """Convert a known Xsens position stream into retargeting coordinates."""
    if stream_name == "body_position_xyz_m":
        return positions
    return transform_xsens_y_up_to_retargeting_z_up(positions)


def transform_xsens_orientation_stream_to_retargeting(
    quaternions_wijk: np.ndarray,
    stream_name: str,
) -> np.ndarray:
    """Convert Xsens world-frame orientations to the position stream's retargeting convention."""
    if stream_name == "body_position_xyz_m":
        return quaternions_wijk
    rotations = rotation_matrices_from_wxyz(quaternions_wijk)
    basis = XSENS_Y_UP_TO_RETARGETING_Z_UP_MATRIX
    return rotation_matrices_as_wxyz(basis @ rotations @ basis.T, canonical=False)


def _get_position_stream(hdf5_file: Any) -> tuple[str, Any]:
    if XSENS_DEVICE_NAME not in hdf5_file:
        raise KeyError(f"No {XSENS_DEVICE_NAME} group found in the HDF5 file")

    streams = hdf5_file[XSENS_DEVICE_NAME]
    for stream_name in XSENS_POSITION_STREAM_NAMES:
        if stream_name not in streams:
            continue
        stream_group = streams[stream_name]
        if "data" in stream_group and "time_s" in stream_group:
            return stream_name, stream_group

    expected = ", ".join(XSENS_POSITION_STREAM_NAMES)
    raise KeyError(f"No Xsens segment position stream found. Expected one of: {expected}")


def _get_orientation_stream(hdf5_file: Any) -> Any:
    if XSENS_DEVICE_NAME not in hdf5_file:
        raise KeyError(f"No {XSENS_DEVICE_NAME} group found in the HDF5 file")
    streams = hdf5_file[XSENS_DEVICE_NAME]
    if XSENS_ORIENTATION_STREAM_NAME not in streams:
        raise KeyError(f"No Xsens segment orientation stream found: {XSENS_ORIENTATION_STREAM_NAME}")
    stream_group = streams[XSENS_ORIENTATION_STREAM_NAME]
    if "data" not in stream_group or "time_s" not in stream_group:
        raise KeyError(
            f"Xsens orientation stream is missing required 'data' or 'time_s': {XSENS_ORIENTATION_STREAM_NAME}"
        )
    return stream_group


def _reshape_position_data(position_data: np.ndarray, segment_names: list[str] | None) -> np.ndarray:
    if position_data.ndim == 3 and position_data.shape[-1] == 3:
        return position_data
    if position_data.ndim == 2 and position_data.shape[1] % 3 == 0:
        return position_data.reshape(position_data.shape[0], -1, 3)

    if segment_names is None:
        raise ValueError(f"Unsupported Xsens position data shape: {position_data.shape}")
    expected_flat_width = len(segment_names) * 3
    if position_data.ndim == 2 and position_data.shape[1] == expected_flat_width:
        return position_data.reshape(position_data.shape[0], len(segment_names), 3)

    raise ValueError(f"Unsupported Xsens position data shape: {position_data.shape}")


def _reshape_static_position_data(position_data: np.ndarray, segment_names: list[str] | None) -> np.ndarray:
    if position_data.ndim == 2 and position_data.shape[-1] == 3:
        return position_data
    if position_data.ndim == 3 and position_data.shape[-1] == 3 and position_data.shape[0] == 1:
        return position_data[0]
    if position_data.ndim == 1 and position_data.shape[0] % 3 == 0:
        return position_data.reshape(-1, 3)

    if segment_names is None:
        raise ValueError(f"Unsupported Xsens static position data shape: {position_data.shape}")
    expected_flat_width = len(segment_names) * 3
    if position_data.ndim == 1 and position_data.shape[0] == expected_flat_width:
        return position_data.reshape(len(segment_names), 3)

    raise ValueError(f"Unsupported Xsens static position data shape: {position_data.shape}")


def _reshape_quaternion_data(quaternion_data: np.ndarray, segment_names: list[str] | None) -> np.ndarray:
    if quaternion_data.ndim == 3 and quaternion_data.shape[-1] == 4:
        return quaternion_data
    if quaternion_data.ndim == 2 and quaternion_data.shape[-1] == 4:
        return quaternion_data
    if quaternion_data.ndim == 3 and quaternion_data.shape[-1] == 4 and quaternion_data.shape[0] == 1:
        return quaternion_data[0]
    if quaternion_data.ndim == 1 and quaternion_data.shape[0] % 4 == 0:
        return quaternion_data.reshape(-1, 4)
    if quaternion_data.ndim == 2 and quaternion_data.shape[1] % 4 == 0:
        return quaternion_data.reshape(quaternion_data.shape[0], -1, 4)

    if segment_names is None:
        raise ValueError(f"Unsupported Xsens quaternion data shape: {quaternion_data.shape}")
    expected_flat_width = len(segment_names) * 4
    if quaternion_data.ndim == 2 and quaternion_data.shape[1] == expected_flat_width:
        return quaternion_data.reshape(quaternion_data.shape[0], len(segment_names), 4)
    if quaternion_data.ndim == 1 and quaternion_data.shape[0] == expected_flat_width:
        return quaternion_data.reshape(len(segment_names), 4)

    raise ValueError(f"Unsupported Xsens quaternion data shape: {quaternion_data.shape}")


def _normalize_quaternions_wijk(quaternions: np.ndarray) -> np.ndarray:
    quat_norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return quaternions / np.maximum(quat_norms, 1e-12)


def _normalize_segment_name(name: str) -> str:
    return "".join(name.lower().split())


def _body_segment_indices(
    segment_names: list[str] | None,
    n_segments: int,
    *,
    include_tracked_props: bool = False,
) -> tuple[list[str], list[int]]:
    if segment_names is None:
        if n_segments < len(XSENS_BODY_SEGMENT_NAMES):
            raise ValueError(f"Expected at least {len(XSENS_BODY_SEGMENT_NAMES)} Xsens body segments, got {n_segments}")
        selected_names = list(XSENS_BODY_SEGMENT_NAMES)
        source_indices = list(range(len(XSENS_BODY_SEGMENT_NAMES)))
        if include_tracked_props and n_segments >= len(XSENS_BODY_SEGMENT_NAMES) + 1:
            # Common tennis layout: the tracked racket frame follows the 23 body segments.
            selected_names.append(XSENS_TRACKED_PROP_NAMES[0])
            source_indices.append(len(XSENS_BODY_SEGMENT_NAMES))
        return selected_names, source_indices

    normalized_to_index = {_normalize_segment_name(name): i for i, name in enumerate(segment_names)}
    missing = [name for name in XSENS_BODY_SEGMENT_NAMES if _normalize_segment_name(name) not in normalized_to_index]
    if missing:
        raise ValueError(f"Xsens stream is missing body segments: {missing}")
    selected_names = list(XSENS_BODY_SEGMENT_NAMES)
    source_indices = [normalized_to_index[_normalize_segment_name(name)] for name in selected_names]
    if include_tracked_props:
        for prop_name in XSENS_TRACKED_PROP_NAMES:
            source_index = normalized_to_index.get(_normalize_segment_name(prop_name))
            if source_index is not None:
                selected_names.append(prop_name)
                source_indices.append(source_index)
    return selected_names, source_indices


def _unit_scale_for_stream(stream_name: str) -> float:
    if stream_name.endswith("_m"):
        return 1.0
    if stream_name.endswith("_cm"):
        return 0.01
    return 1.0


def sample_indices_by_time(times_s: np.ndarray, target_fps: float | None) -> np.ndarray:
    """Sample monotonically increasing timestamps at `target_fps` using previous-frame hold."""
    times_s = np.asarray(times_s, dtype=float).reshape(-1)
    if times_s.size == 0:
        raise ValueError("Xsens stream has no timestamps")
    if target_fps is None:
        return np.arange(times_s.size)
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")

    period_s = 1.0 / target_fps
    sample_times_s = np.arange(times_s[0], times_s[-1] + period_s * 0.5, period_s)
    indices = np.searchsorted(times_s, sample_times_s, side="right") - 1
    indices = np.clip(indices, 0, times_s.size - 1)
    return np.unique(indices)


def _slice_sample_indices(
    sample_indices: np.ndarray,
    frame_start: int,
    max_frames: int | None,
    frame_indices: tuple[int, ...] | None,
) -> np.ndarray:
    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    if frame_indices is not None:
        if frame_start != 0 or max_frames is not None:
            raise ValueError("frame_indices is mutually exclusive with frame_start and max_frames")
        sparse_indices = np.asarray(frame_indices, dtype=int)
        if sparse_indices.ndim != 1 or sparse_indices.size == 0:
            raise ValueError("frame_indices must contain at least one index")
        if np.any(sparse_indices < 0) or np.any(sparse_indices >= sample_indices.size):
            raise ValueError(f"frame_indices must be within the post-resampling range [0, {sample_indices.size - 1}]")
        if np.unique(sparse_indices).size != sparse_indices.size:
            raise ValueError("frame_indices must not contain duplicates")
        return sample_indices[sparse_indices]

    sample_indices = sample_indices[frame_start:]
    if max_frames is not None:
        sample_indices = sample_indices[:max_frames]
    if sample_indices.size == 0:
        raise ValueError("Selected Xsens frame window is empty")
    return sample_indices


def load_xsens_hdf5_motion(
    path: str | Path,
    target_fps: float | None = 30.0,
    frame_start: int = 0,
    max_frames: int | None = None,
    frame_indices: tuple[int, ...] | None = None,
    *,
    include_tracked_props: bool = False,
) -> XsensHdf5Motion:
    """Load synchronized Xsens segment poses and optional tracked props."""
    h5py = _import_h5py()
    path = Path(path)

    with h5py.File(path, "r") as hdf5_file:
        stream_name, stream_group = _get_position_stream(hdf5_file)
        segment_names = _segment_names_from_attrs(stream_group)

        positions = np.asarray(stream_group["data"], dtype=float)
        times_s = np.asarray(stream_group["time_s"], dtype=float).reshape(-1)
        orientation_group = _get_orientation_stream(hdf5_file)
        orientation_segment_names = _segment_names_from_attrs(orientation_group)
        quaternions = np.asarray(orientation_group["data"], dtype=float)
        orientation_times_s = np.asarray(orientation_group["time_s"], dtype=float).reshape(-1)

    positions = _reshape_position_data(positions, segment_names)
    selected_segment_names, source_indices = _body_segment_indices(
        segment_names,
        positions.shape[1],
        include_tracked_props=include_tracked_props,
    )
    positions = positions[:, source_indices, :] * _unit_scale_for_stream(stream_name)
    positions = transform_xsens_stream_to_retargeting(positions, stream_name)

    sample_indices = sample_indices_by_time(times_s, target_fps)
    sample_indices = _slice_sample_indices(sample_indices, frame_start, max_frames, frame_indices)
    quaternions = _reshape_quaternion_data(quaternions, orientation_segment_names)
    selected_orientation_names, orientation_source_indices = _body_segment_indices(
        orientation_segment_names,
        quaternions.shape[1],
        include_tracked_props=include_tracked_props,
    )
    if selected_orientation_names != selected_segment_names:
        raise ValueError("Xsens position and orientation body segment selections differ")
    if source_indices != orientation_source_indices:
        raise ValueError("Xsens position and orientation source segment indices differ")
    if orientation_times_s.shape != times_s.shape or not np.allclose(orientation_times_s, times_s):
        raise ValueError("Xsens position and orientation streams must share timestamps")
    quaternions = _normalize_quaternions_wijk(quaternions[:, orientation_source_indices, :])
    quaternions = transform_xsens_orientation_stream_to_retargeting(quaternions, stream_name)

    selected_times_s = times_s[sample_indices]
    recording_times_s = selected_times_s - float(times_s[0])
    if frame_indices is not None:
        if target_fps is not None:
            storyboard_period_s = 1.0 / target_fps
        else:
            source_intervals_s = np.diff(times_s)
            positive_intervals_s = source_intervals_s[source_intervals_s > 0.0]
            storyboard_period_s = float(np.median(positive_intervals_s)) if positive_intervals_s.size else 1.0 / 30.0
        selected_times_s = np.arange(sample_indices.size, dtype=float) * storyboard_period_s

    return XsensHdf5Motion(
        positions_m=positions[sample_indices],
        times_s=selected_times_s,
        stream_name=stream_name,
        segment_names=selected_segment_names,
        source_indices=source_indices,
        quaternions_wijk=quaternions[sample_indices],
        orientation_stream_name=XSENS_ORIENTATION_STREAM_NAME,
        recording_times_s=recording_times_s,
    )


def load_xsens_hdf5_tpose(path: str | Path, variant: str = "Tpose") -> XsensHdf5Tpose:
    """Load Xsens static T-pose body segment positions and orientations.

    The ActionNet tennis files store T-pose data as direct datasets under
    `xsens-segments-tpose`, not as time series. Quaternions are returned in the
    source `w, i, j, k` order.
    """
    h5py = _import_h5py()
    path = Path(path)
    position_dataset_name = f"body_position_{variant}_xyz_m"
    quaternion_dataset_name = f"body_orientation_{variant}_quaternion_wijk"

    with h5py.File(path, "r") as hdf5_file:
        if XSENS_TPOSE_DEVICE_NAME not in hdf5_file:
            raise KeyError(f"No {XSENS_TPOSE_DEVICE_NAME} group found in the HDF5 file")

        tpose_group = hdf5_file[XSENS_TPOSE_DEVICE_NAME]
        if position_dataset_name not in tpose_group:
            raise KeyError(f"No Xsens T-pose position dataset found: {position_dataset_name}")
        if quaternion_dataset_name not in tpose_group:
            raise KeyError(f"No Xsens T-pose quaternion dataset found: {quaternion_dataset_name}")

        position_dataset = tpose_group[position_dataset_name]
        quaternion_dataset = tpose_group[quaternion_dataset_name]
        segment_names = _segment_names_from_attrs(position_dataset)
        if segment_names is None:
            segment_names = _segment_names_from_attrs(quaternion_dataset)
        if segment_names is None:
            segment_names = _segment_names_from_attrs(tpose_group)
        if segment_names is None and XSENS_DEVICE_NAME in hdf5_file:
            xsens_streams = hdf5_file[XSENS_DEVICE_NAME]
            for stream_name in XSENS_POSITION_STREAM_NAMES:
                if stream_name in xsens_streams:
                    segment_names = _segment_names_from_attrs(xsens_streams[stream_name])
                    if segment_names is not None:
                        break

        positions = np.asarray(position_dataset, dtype=float)
        quaternions = np.asarray(quaternion_dataset, dtype=float)

    positions = _reshape_static_position_data(positions, segment_names)
    quaternions = _reshape_quaternion_data(quaternions, segment_names)

    if positions.shape[0] != quaternions.shape[0]:
        raise ValueError(
            f"Xsens T-pose position/quaternion segment counts differ: {positions.shape[0]} vs {quaternions.shape[0]}"
        )

    selected_segment_names, source_indices = _body_segment_indices(segment_names, positions.shape[0])
    positions = positions[source_indices]
    quaternions = quaternions[source_indices]
    quaternions = _normalize_quaternions_wijk(quaternions)

    return XsensHdf5Tpose(
        positions_m=positions,
        quaternions_wijk=quaternions,
        variant=variant,
        segment_names=selected_segment_names,
        source_indices=source_indices,
    )


def _calibration_position_stream(hdf5_file: Any) -> tuple[str, Any]:
    if XSENS_DEVICE_NAME not in hdf5_file:
        raise KeyError(f"No {XSENS_DEVICE_NAME} group found in the HDF5 file")
    streams = hdf5_file[XSENS_DEVICE_NAME]
    for stream_name in XSENS_POSITION_STREAM_NAMES:
        if stream_name in streams:
            return stream_name, streams[stream_name]
    expected = ", ".join(XSENS_POSITION_STREAM_NAMES)
    raise KeyError(f"No Xsens segment metadata stream found. Expected one of: {expected}")


def _load_pose_variant(tpose_group: Any, variant: str, n_segments: int) -> SegmentPoseSet | None:
    position_name = f"body_position_{variant}_xyz_m"
    quaternion_name = f"body_orientation_{variant}_quaternion_wijk"
    has_position = position_name in tpose_group
    has_orientation = quaternion_name in tpose_group
    if not has_position and not has_orientation:
        return None
    if not has_position or not has_orientation:
        missing = quaternion_name if has_position else position_name
        raise KeyError(f"Xsens reference pose '{variant}' is incomplete; missing {missing}")

    positions = _reshape_static_position_data(np.asarray(tpose_group[position_name], dtype=float), None)
    quaternions = _reshape_quaternion_data(np.asarray(tpose_group[quaternion_name], dtype=float), None)
    if positions.shape != (n_segments, 3):
        raise ValueError(f"Unexpected {variant} position shape: {positions.shape}; expected {(n_segments, 3)}")
    if quaternions.shape != (n_segments, 4):
        raise ValueError(f"Unexpected {variant} quaternion shape: {quaternions.shape}; expected {(n_segments, 4)}")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError(f"Xsens reference pose '{variant}' contains non-finite values")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"Xsens reference pose '{variant}' contains a zero-length quaternion")
    return SegmentPoseSet(
        positions_m=positions,
        quaternions_wijk=_normalize_quaternions_wijk(quaternions),
        variant=variant,
    )


def _joint_metadata(hdf5_file: Any) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, JointRotationMetadata]]:
    if XSENS_JOINT_DEVICE_NAME not in hdf5_file:
        raise KeyError(f"No {XSENS_JOINT_DEVICE_NAME} group found in the HDF5 file")
    joint_group = hdf5_file[XSENS_JOINT_DEVICE_NAME]
    available_streams = tuple(name for name in XSENS_JOINT_STREAM_NAMES if name in joint_group)
    if not available_streams:
        expected = ", ".join(XSENS_JOINT_STREAM_NAMES)
        raise KeyError(f"No Xsens body joint-angle stream found. Expected one of: {expected}")

    primary = joint_group[available_streams[0]]
    joint_names = _parse_string_sequence(primary.attrs.get("joint_names_body"), attribute_name="joint_names_body")
    rotation_order_value = primary.attrs.get("joint_rotation_order_body")
    rotation_components: dict[str, tuple[str, ...]] = {}
    if rotation_order_value is not None:
        parsed = ast.literal_eval(str(_decode_attr(rotation_order_value)))
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("joint_rotation_order_body must decode to a sequence")
        for item in parsed:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each joint_rotation_order_body entry must contain a joint and its components")
            rotation_components[str(item[0])] = tuple(str(component) for component in item[1])

    metadata = {
        name: JointRotationMetadata(
            joint_name=name,
            components=rotation_components.get(name, ()),
            available_euler_streams=available_streams,
        )
        for name in joint_names
    }
    return joint_names, available_streams, metadata


def inspect_xsens_hdf5(path: str | Path) -> XsensHdf5Inventory:
    """Return a lightweight inventory without loading motion arrays."""

    h5py = _import_h5py()
    path = Path(path)
    with h5py.File(path, "r") as hdf5_file:
        _, stream = _calibration_position_stream(hdf5_file)
        segment_names = _segment_names_from_attrs(stream)
        if segment_names is None:
            raise KeyError("Xsens segment stream is missing segment_names_body/Data headings metadata")
        joint_names, joint_stream_names, _ = _joint_metadata(hdf5_file)
        tpose_group = hdf5_file.get(XSENS_TPOSE_DEVICE_NAME)
        if tpose_group is None:
            pose_variants: tuple[str, ...] = ()
        else:
            pose_variants = tuple(
                variant
                for variant in ("Tpose", "TposeISB", "identity")
                if f"body_position_{variant}_xyz_m" in tpose_group
                and f"body_orientation_{variant}_quaternion_wijk" in tpose_group
            )
        return XsensHdf5Inventory(
            path=path,
            segment_names=tuple(segment_names),
            joint_names=joint_names,
            pose_variants=pose_variants,
            joint_stream_names=joint_stream_names,
            has_landmarks="segment_mesh_points_body_xyz_cm" in stream.attrs,
        )


def load_xsens_hdf5_calibration(path: str | Path) -> XsensHdf5Calibration:
    """Load the complete subject-specific kinematic calibration from one recording."""

    h5py = _import_h5py()
    path = Path(path)
    with h5py.File(path, "r") as hdf5_file:
        stream_name, stream = _calibration_position_stream(hdf5_file)
        segment_names_list = _segment_names_from_attrs(stream)
        if segment_names_list is None:
            raise KeyError("Xsens segment stream is missing segment_names_body/Data headings metadata")
        segment_names = tuple(segment_names_list)

        landmark_value = stream.attrs.get("segment_mesh_points_body_xyz_cm")
        if landmark_value is None:
            raise KeyError("Xsens segment stream is missing segment_mesh_points_body_xyz_cm metadata")
        raw_landmarks = _parse_ordered_dict(landmark_value)
        landmarks_m: dict[str, dict[str, np.ndarray]] = {
            str(segment): {
                str(name): np.asarray(value, dtype=float) * 0.01 for name, value in segment_landmarks.items()
            }
            for segment, segment_landmarks in raw_landmarks.items()
        }
        missing_landmarks = [segment for segment in segment_names if segment not in landmarks_m]
        if missing_landmarks:
            raise ValueError(f"Xsens landmark metadata is missing segments: {missing_landmarks}")

        if XSENS_TPOSE_DEVICE_NAME not in hdf5_file:
            raise KeyError(f"No {XSENS_TPOSE_DEVICE_NAME} group found in the HDF5 file")
        tpose_group = hdf5_file[XSENS_TPOSE_DEVICE_NAME]
        tpose = _load_pose_variant(tpose_group, "Tpose", len(segment_names))
        if tpose is None:
            raise KeyError("Xsens calibration is missing the required Tpose position/orientation datasets")
        tpose_isb = _load_pose_variant(tpose_group, "TposeISB", len(segment_names))
        identity_pose = _load_pose_variant(tpose_group, "identity", len(segment_names))

        joint_names, joint_stream_names, joint_rotation_metadata = _joint_metadata(hdf5_file)
        mvn_version = stream.attrs.get("mvn_version")
        mvnx_version = stream.attrs.get("mvnx_version")

    return XsensHdf5Calibration(
        source_path=path,
        source_stream_name=stream_name,
        segment_names=segment_names,
        joint_names=joint_names,
        tpose=tpose,
        tpose_isb=tpose_isb,
        identity_pose=identity_pose,
        landmarks_m=landmarks_m,
        joint_rotation_metadata=joint_rotation_metadata,
        joint_stream_names=joint_stream_names,
        mvn_version=None if mvn_version is None else str(_decode_attr(mvn_version)),
        mvnx_version=None if mvnx_version is None else str(_decode_attr(mvnx_version)),
    )


def resolve_xsens_hdf5_path(data_path: str | Path, task_name: str) -> Path:
    """Resolve a task name to an Xsens HDF5 file under `data_path`."""
    data_path = Path(data_path)
    candidate = data_path / task_name
    if candidate.is_file() and candidate.suffix.lower() in {".hdf5", ".h5"}:
        return candidate

    for suffix in (".hdf5", ".h5"):
        candidate_with_suffix = data_path / f"{task_name}{suffix}"
        if candidate_with_suffix.is_file():
            return candidate_with_suffix

    raise FileNotFoundError(f"Xsens HDF5 data file not found for task '{task_name}' in {data_path}")
