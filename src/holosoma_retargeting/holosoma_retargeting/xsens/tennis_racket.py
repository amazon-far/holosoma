"""Shared tennis-racket attachment, target, and achieved-motion utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from holosoma_retargeting.config_types.retargeter import TennisRacketTrackingConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XsensHdf5Motion,
    load_xsens_hdf5_calibration,
    load_xsens_hdf5_motion,
)
from holosoma_retargeting.transformation_utils import (
    normalize_quaternions_wxyz,
    position_quaternion_from_transform,
    rotation_as_wxyz,
    rotations_from_wxyz,
    transform_from_position_quaternion,
)
from holosoma_retargeting.xsens.kinematic_model import XSENS_RACKET_SOURCE_SEGMENT

if TYPE_CHECKING:
    import mujoco  # type: ignore[import-not-found]


TENNIS_RACKET_RESULT_SCHEMA_VERSION = 1
TENNIS_RACKET_HAND_SEGMENT = "Right Hand"
TENNIS_RACKET_HAND_LINK = "right_rubber_hand_link"
TENNIS_RACKET_LONGITUDINAL_AXIS_LOCAL = np.array([1.0, 0.0, 0.0], dtype=float)
DEFAULT_TENNIS_RACKET_ATTACHMENT_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "g1" / "tennis_racket_attachment.json"
)

TennisRacketAttachmentSource = Literal["global", "embedded_tpose", "observed_window"]

__all__ = [
    "DEFAULT_TENNIS_RACKET_ATTACHMENT_PATH",
    "TENNIS_RACKET_HAND_LINK",
    "TENNIS_RACKET_HAND_SEGMENT",
    "TENNIS_RACKET_LONGITUDINAL_AXIS_LOCAL",
    "TENNIS_RACKET_RESULT_SCHEMA_VERSION",
    "RetargetingResult",
    "TennisRacketAttachment",
    "TennisRacketFilterDecision",
    "TennisRacketMotion",
    "TennisRacketTargets",
    "achieved_tennis_racket_pose",
    "attachment_handle_intersects_palm",
    "build_tennis_racket_targets",
    "choose_tennis_racket_symmetry_branch",
    "decide_filtered_tennis_racket_tracking",
    "load_retargeting_result",
    "load_tennis_racket_attachment",
    "resolve_tennis_racket_attachment",
    "save_tennis_racket_attachment",
    "tennis_racket_motion_from_npz",
    "tennis_racket_target_error_rad",
]


@dataclass(frozen=True)
class TennisRacketAttachment:
    """One rigid transform from a physical G1 hand link to the racket frame."""

    hand_link: str
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    longitudinal_axis_local: np.ndarray
    palm_bounds_min_m: np.ndarray
    palm_bounds_max_m: np.ndarray
    source_reference_position_m: np.ndarray
    source_reference_quaternion_wxyz: np.ndarray
    calibration_source: TennisRacketAttachmentSource = "global"
    schema_version: int = TENNIS_RACKET_RESULT_SCHEMA_VERSION
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.hand_link:
            raise ValueError("A tennis-racket attachment must name its hand link")
        position = np.asarray(self.position_m, dtype=float).reshape(3)
        axis = np.asarray(self.longitudinal_axis_local, dtype=float).reshape(3)
        source_position = np.asarray(self.source_reference_position_m, dtype=float).reshape(3)
        palm_minimum = np.asarray(self.palm_bounds_min_m, dtype=float).reshape(3)
        palm_maximum = np.asarray(self.palm_bounds_max_m, dtype=float).reshape(3)
        if not np.isfinite(position).all() or not np.isfinite(source_position).all():
            raise ValueError("Tennis-racket attachment positions must be finite")
        if not np.isfinite(palm_minimum).all() or not np.all(palm_maximum > palm_minimum):
            raise ValueError("Tennis-racket palm bounds must be finite and nonempty")
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis).all() or axis_norm <= 1e-12:
            raise ValueError("The tennis-racket longitudinal axis must be finite and nonzero")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_wxyz", normalize_quaternions_wxyz(self.quaternion_wxyz))
        object.__setattr__(self, "longitudinal_axis_local", axis / axis_norm)
        object.__setattr__(self, "palm_bounds_min_m", palm_minimum)
        object.__setattr__(self, "palm_bounds_max_m", palm_maximum)
        object.__setattr__(self, "source_reference_position_m", source_position)
        object.__setattr__(
            self,
            "source_reference_quaternion_wxyz",
            normalize_quaternions_wxyz(self.source_reference_quaternion_wxyz),
        )


@dataclass(frozen=True)
class TennisRacketTargets:
    """Per-frame racket-equivalent right-hand orientation candidates."""

    attachment: TennisRacketAttachment
    candidate_hand_rotations: np.ndarray
    candidate_racket_rotations: np.ndarray
    source_origin_deviation_m: np.ndarray
    source_times_s: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.candidate_hand_rotations.shape[0])


@dataclass(frozen=True)
class TennisRacketMotion:
    """Achieved world-space tennis-racket motion and retargeting diagnostics."""

    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    tracking_state: np.ndarray
    symmetry_branch: np.ndarray
    target_error_rad: np.ndarray
    source_origin_deviation_m: np.ndarray
    min_wrist_limit_margin_rad: np.ndarray
    attachment: TennisRacketAttachment
    tracking_mode: str

    def __post_init__(self) -> None:
        positions = np.asarray(self.position_m, dtype=float)
        quaternions = np.asarray(self.quaternion_wxyz, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Tennis-racket positions must have shape [T, 3], got {positions.shape}")
        if quaternions.shape != (positions.shape[0], 4):
            raise ValueError("Tennis-racket quaternions must have shape [T, 4]")
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        if not np.isfinite(positions).all() or not np.isfinite(quaternions).all() or np.any(norms <= 1e-12):
            raise ValueError("Tennis-racket poses must contain finite values and valid quaternions")
        frame_count = positions.shape[0]
        for name in (
            "tracking_state",
            "symmetry_branch",
            "target_error_rad",
            "source_origin_deviation_m",
            "min_wrist_limit_margin_rad",
        ):
            if np.asarray(getattr(self, name)).shape != (frame_count,):
                raise ValueError(f"{name} must have shape [T]")
        object.__setattr__(self, "position_m", positions)
        object.__setattr__(self, "quaternion_wxyz", quaternions / norms)

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "tennis_racket_schema_version": np.asarray(self.attachment.schema_version, dtype=np.int64),
            "tennis_racket_position_m": np.asarray(self.position_m, dtype=float),
            "tennis_racket_quaternion_wxyz": np.asarray(self.quaternion_wxyz, dtype=float),
            "tennis_racket_tracking_state": np.asarray(self.tracking_state, dtype=str),
            "tennis_racket_symmetry_branch": np.asarray(self.symmetry_branch, dtype=np.int8),
            "tennis_racket_target_error_rad": np.asarray(self.target_error_rad, dtype=float),
            "tennis_racket_source_origin_deviation_m": np.asarray(self.source_origin_deviation_m, dtype=float),
            "tennis_racket_min_wrist_limit_margin_rad": np.asarray(self.min_wrist_limit_margin_rad, dtype=float),
            "tennis_racket_tracking_mode": np.asarray(self.tracking_mode),
            "tennis_racket_attachment_source": np.asarray(self.attachment.calibration_source),
            "tennis_racket_attachment_hand_link": np.asarray(self.attachment.hand_link),
            "tennis_racket_attachment_position_m": np.asarray(self.attachment.position_m, dtype=float),
            "tennis_racket_attachment_quaternion_wxyz": np.asarray(self.attachment.quaternion_wxyz, dtype=float),
            "tennis_racket_longitudinal_axis_local": np.asarray(self.attachment.longitudinal_axis_local, dtype=float),
            "tennis_racket_palm_bounds_min_m": np.asarray(self.attachment.palm_bounds_min_m, dtype=float),
            "tennis_racket_palm_bounds_max_m": np.asarray(self.attachment.palm_bounds_max_m, dtype=float),
        }


@dataclass(frozen=True)
class RetargetingResult:
    """Common retargeting result arrays with optional achieved racket motion."""

    qpos: np.ndarray
    fps: float
    tennis_racket: TennisRacketMotion | None = None


@dataclass(frozen=True)
class TennisRacketFilterDecision:
    """One deterministic transition of filtered racket-tracking hysteresis."""

    active: bool
    feasible_streak: int
    use_racket: bool
    state: str


def decide_filtered_tennis_racket_tracking(
    config: TennisRacketTrackingConfig,
    *,
    active: bool,
    feasible_streak: int,
    source_origin_deviation_m: float,
    solve_succeeded: bool,
    target_error_rad: float,
    wrist_limit_margin_rad: float,
) -> TennisRacketFilterDecision:
    """Apply detachment, feasibility, wrist-margin, and re-entry hysteresis policy."""

    detach_threshold = config.detach_exit_threshold_m if active else config.detach_reentry_threshold_m
    if source_origin_deviation_m > detach_threshold:
        return TennisRacketFilterDecision(False, 0, False, "detached")
    residual_limit = config.feasible_exit_error_rad if active else config.feasible_entry_error_rad
    if not solve_succeeded:
        return TennisRacketFilterDecision(False, 0, False, "infeasible")
    if wrist_limit_margin_rad < config.min_wrist_limit_margin_rad:
        return TennisRacketFilterDecision(False, 0, False, "wrist_limit")
    if not np.isfinite(target_error_rad) or target_error_rad > residual_limit:
        return TennisRacketFilterDecision(False, 0, False, "infeasible")
    if active:
        return TennisRacketFilterDecision(True, config.reentry_frames, True, "racket")
    next_streak = feasible_streak + 1
    if next_streak >= config.reentry_frames:
        return TennisRacketFilterDecision(True, next_streak, True, "racket")
    return TennisRacketFilterDecision(False, next_streak, False, "reentry_hysteresis")


def _scalar_text(value: np.ndarray | str) -> str:
    return str(np.asarray(value).reshape(()).item())


def tennis_racket_motion_from_npz(data: Mapping[str, np.ndarray]) -> TennisRacketMotion | None:
    """Load optional achieved racket data while accepting legacy qpos-only results."""

    if "tennis_racket_position_m" not in data:
        return None
    base_attachment = load_tennis_racket_attachment()
    attachment = replace(
        base_attachment,
        hand_link=_scalar_text(data["tennis_racket_attachment_hand_link"]),
        position_m=np.asarray(data["tennis_racket_attachment_position_m"], dtype=float),
        quaternion_wxyz=np.asarray(data["tennis_racket_attachment_quaternion_wxyz"], dtype=float),
        longitudinal_axis_local=np.asarray(data["tennis_racket_longitudinal_axis_local"], dtype=float),
        palm_bounds_min_m=np.asarray(
            data.get("tennis_racket_palm_bounds_min_m", base_attachment.palm_bounds_min_m), dtype=float
        ),
        palm_bounds_max_m=np.asarray(
            data.get("tennis_racket_palm_bounds_max_m", base_attachment.palm_bounds_max_m), dtype=float
        ),
        calibration_source=_scalar_text(data["tennis_racket_attachment_source"]),
        schema_version=int(np.asarray(data["tennis_racket_schema_version"]).reshape(())),
        artifact_path=None,
    )
    return TennisRacketMotion(
        position_m=np.asarray(data["tennis_racket_position_m"], dtype=float),
        quaternion_wxyz=np.asarray(data["tennis_racket_quaternion_wxyz"], dtype=float),
        tracking_state=np.asarray(data["tennis_racket_tracking_state"], dtype=str),
        symmetry_branch=np.asarray(data["tennis_racket_symmetry_branch"], dtype=np.int8),
        target_error_rad=np.asarray(data["tennis_racket_target_error_rad"], dtype=float),
        source_origin_deviation_m=np.asarray(data["tennis_racket_source_origin_deviation_m"], dtype=float),
        min_wrist_limit_margin_rad=np.asarray(data["tennis_racket_min_wrist_limit_margin_rad"], dtype=float),
        attachment=attachment,
        tracking_mode=_scalar_text(data["tennis_racket_tracking_mode"]),
    )


def load_retargeting_result(path: str | Path) -> RetargetingResult:
    """Load a raw result, retaining optional first-class racket motion."""

    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=float)
        fps = float(np.asarray(data["fps"]).reshape(())) if "fps" in data else 30.0
        racket = tennis_racket_motion_from_npz(data)
    if racket is not None and racket.position_m.shape[0] != qpos.shape[0]:
        raise ValueError("Saved tennis-racket motion is not aligned with qpos")
    return RetargetingResult(qpos=qpos, fps=fps, tennis_racket=racket)


def load_tennis_racket_attachment(path: str | Path | None = None) -> TennisRacketAttachment:
    """Load and validate a model-specific G1 hand-to-racket JSON artifact."""

    artifact_path = DEFAULT_TENNIS_RACKET_ATTACHMENT_PATH if path is None else Path(path).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Tennis-racket attachment artifact not found: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != TENNIS_RACKET_RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported tennis-racket attachment schema in {artifact_path}: {payload.get('schema_version')}"
        )
    return TennisRacketAttachment(
        hand_link=str(payload["hand_link"]),
        position_m=np.asarray(payload["position_m"], dtype=float),
        quaternion_wxyz=np.asarray(payload["quaternion_wxyz"], dtype=float),
        longitudinal_axis_local=np.asarray(payload["longitudinal_axis_local"], dtype=float),
        palm_bounds_min_m=np.asarray(payload["palm_bounds_min_m"], dtype=float),
        palm_bounds_max_m=np.asarray(payload["palm_bounds_max_m"], dtype=float),
        source_reference_position_m=np.asarray(payload["source_reference_position_m"], dtype=float),
        source_reference_quaternion_wxyz=np.asarray(payload["source_reference_quaternion_wxyz"], dtype=float),
        artifact_path=artifact_path.resolve(),
    )


def save_tennis_racket_attachment(attachment: TennisRacketAttachment, path: str | Path) -> None:
    """Write a readable, versioned G1 hand-to-racket attachment artifact."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": attachment.schema_version,
        "hand_link": attachment.hand_link,
        "position_m": attachment.position_m.tolist(),
        "quaternion_wxyz": attachment.quaternion_wxyz.tolist(),
        "longitudinal_axis_local": attachment.longitudinal_axis_local.tolist(),
        "palm_bounds_min_m": attachment.palm_bounds_min_m.tolist(),
        "palm_bounds_max_m": attachment.palm_bounds_max_m.tolist(),
        "source_reference_position_m": attachment.source_reference_position_m.tolist(),
        "source_reference_quaternion_wxyz": attachment.source_reference_quaternion_wxyz.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _segment_index(names: list[str] | tuple[str, ...], name: str) -> int:
    normalized = {value.replace(" ", "").casefold(): index for index, value in enumerate(names)}
    key = name.replace(" ", "").casefold()
    if key not in normalized:
        raise ValueError(f"Xsens motion does not contain required segment '{name}'")
    return normalized[key]


def _relative_pose(
    hand_position_m: np.ndarray,
    hand_quaternion_wxyz: np.ndarray,
    racket_position_m: np.ndarray,
    racket_quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hand_rotation = rotations_from_wxyz(np.atleast_2d(hand_quaternion_wxyz))
    racket_rotation = rotations_from_wxyz(np.atleast_2d(racket_quaternion_wxyz))
    positions = hand_rotation.inv().apply(np.atleast_2d(racket_position_m) - np.atleast_2d(hand_position_m))
    rotations = hand_rotation.inv() * racket_rotation
    return positions, rotation_as_wxyz(rotations)


def _selected_source_reference(
    config: TennisRacketTrackingConfig,
    motion: XsensHdf5Motion,
    hdf5_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    hand_index = _segment_index(motion.segment_names, TENNIS_RACKET_HAND_SEGMENT)
    racket_index = _segment_index(motion.segment_names, XSENS_RACKET_SOURCE_SEGMENT)
    if config.attachment_source == "embedded_tpose":
        calibration = load_xsens_hdf5_calibration(hdf5_path)
        hand_tpose_index = _segment_index(calibration.segment_names, TENNIS_RACKET_HAND_SEGMENT)
        racket_tpose_index = _segment_index(calibration.segment_names, XSENS_RACKET_SOURCE_SEGMENT)
        positions, quaternions = _relative_pose(
            calibration.tpose.positions_m[hand_tpose_index],
            calibration.tpose.quaternions_wijk[hand_tpose_index],
            calibration.tpose.positions_m[racket_tpose_index],
            calibration.tpose.quaternions_wijk[racket_tpose_index],
        )
        return positions[0], quaternions[0]
    if config.attachment_source == "observed_window":
        if config.observed_window_s is None:
            raise ValueError("observed_window attachment calibration requires observed_window_s")
        start_s, end_s = config.observed_window_s
        if not np.isfinite([start_s, end_s]).all() or start_s < 0.0 or end_s <= start_s:
            raise ValueError("observed_window_s must be a finite [start, end) interval with 0 <= start < end")
        relative_times = (
            np.asarray(motion.recording_times_s, dtype=float)
            if motion.recording_times_s is not None
            else np.asarray(motion.times_s, dtype=float) - float(motion.times_s[0])
        )
        mask = (relative_times >= start_s) & (relative_times < end_s)
        if np.count_nonzero(mask) < 2 and hdf5_path.is_file():
            positive_steps_s = np.diff(relative_times)
            positive_steps_s = positive_steps_s[positive_steps_s > 0.0]
            target_fps = 30.0 if positive_steps_s.size == 0 else 1.0 / float(np.median(positive_steps_s))
            pad_frames = 2
            calibration_motion = load_xsens_hdf5_motion(
                hdf5_path,
                target_fps=target_fps,
                frame_start=max(0, int(np.floor(start_s * target_fps)) - pad_frames),
                max_frames=max(2, int(np.ceil((end_s - start_s) * target_fps)) + 2 * pad_frames),
                include_tracked_props=True,
            )
            relative_times = np.asarray(calibration_motion.recording_times_s, dtype=float)
            mask = (relative_times >= start_s) & (relative_times < end_s)
            motion = calibration_motion
            hand_index = _segment_index(motion.segment_names, TENNIS_RACKET_HAND_SEGMENT)
            racket_index = _segment_index(motion.segment_names, XSENS_RACKET_SOURCE_SEGMENT)
        if np.count_nonzero(mask) < 2:
            raise ValueError("observed_window_s contains fewer than two resampled Xsens frames")
        positions, quaternions = _relative_pose(
            motion.positions_m[mask, hand_index],
            motion.quaternions_wijk[mask, hand_index],
            motion.positions_m[mask, racket_index],
            motion.quaternions_wijk[mask, racket_index],
        )
        mean_rotation = rotations_from_wxyz(quaternions).mean()
        return np.mean(positions, axis=0), rotation_as_wxyz(mean_rotation)
    raise ValueError(f"Unknown recording-derived tennis-racket attachment source: {config.attachment_source}")


def resolve_tennis_racket_attachment(
    config: TennisRacketTrackingConfig,
    *,
    motion: XsensHdf5Motion,
    hdf5_path: str | Path,
) -> TennisRacketAttachment:
    """Resolve the global grasp and apply an optional recording-derived six-DoF correction."""

    attachment = load_tennis_racket_attachment(config.attachment_path)
    if config.attachment_source == "global":
        return attachment
    selected_position, selected_quaternion = _selected_source_reference(config, motion, Path(hdf5_path).expanduser())
    global_transform = transform_from_position_quaternion(attachment.position_m, attachment.quaternion_wxyz)
    reference_transform = transform_from_position_quaternion(
        attachment.source_reference_position_m,
        attachment.source_reference_quaternion_wxyz,
    )
    selected_transform = transform_from_position_quaternion(selected_position, selected_quaternion)
    corrected_position, corrected_quaternion = position_quaternion_from_transform(
        global_transform @ np.linalg.inv(reference_transform) @ selected_transform
    )
    return replace(
        attachment,
        position_m=corrected_position,
        quaternion_wxyz=corrected_quaternion,
        source_reference_position_m=selected_position,
        source_reference_quaternion_wxyz=selected_quaternion,
        calibration_source=config.attachment_source,
    )


def build_tennis_racket_targets(
    motion: XsensHdf5Motion,
    attachment: TennisRacketAttachment,
) -> TennisRacketTargets:
    """Construct 0/180-degree-equivalent racket and G1-hand targets."""

    hand_index = _segment_index(motion.segment_names, TENNIS_RACKET_HAND_SEGMENT)
    racket_index = _segment_index(motion.segment_names, XSENS_RACKET_SOURCE_SEGMENT)
    hand_rotations = rotations_from_wxyz(motion.quaternions_wijk[:, hand_index])
    racket_rotations = rotations_from_wxyz(motion.quaternions_wijk[:, racket_index])
    half_turn = Rotation.from_rotvec(np.pi * attachment.longitudinal_axis_local)
    attachment_rotation = rotations_from_wxyz(attachment.quaternion_wxyz)
    racket_candidates = (racket_rotations, racket_rotations * half_turn)
    hand_candidates = tuple(candidate * attachment_rotation.inv() for candidate in racket_candidates)
    candidate_racket_rotations = np.stack([candidate.as_matrix() for candidate in racket_candidates], axis=1)
    candidate_hand_rotations = np.stack([candidate.as_matrix() for candidate in hand_candidates], axis=1)
    source_relative_positions = hand_rotations.inv().apply(
        motion.positions_m[:, racket_index] - motion.positions_m[:, hand_index]
    )
    origin_deviation = np.linalg.norm(
        source_relative_positions - attachment.source_reference_position_m[None, :],
        axis=1,
    )
    return TennisRacketTargets(
        attachment=attachment,
        candidate_hand_rotations=candidate_hand_rotations,
        candidate_racket_rotations=candidate_racket_rotations,
        source_origin_deviation_m=origin_deviation,
        source_times_s=np.asarray(motion.times_s, dtype=float).copy(),
    )


def choose_tennis_racket_symmetry_branch(
    current_hand_rotation: np.ndarray,
    candidate_hand_rotations: np.ndarray,
    *,
    preferred_branch: int | None = None,
) -> int:
    """Choose the equivalent target nearest the current hand, using the prior branch as a tie-breaker."""

    current = Rotation.from_matrix(np.asarray(current_hand_rotation, dtype=float).reshape(3, 3))
    candidates = Rotation.from_matrix(np.asarray(candidate_hand_rotations, dtype=float).reshape(2, 3, 3))
    errors = np.asarray((candidates * current.inv()).magnitude(), dtype=float)
    if preferred_branch in (0, 1) and abs(float(errors[0] - errors[1])) <= np.deg2rad(1.0):
        return int(preferred_branch)
    return int(np.argmin(errors))


def tennis_racket_target_error_rad(
    achieved_racket_rotation: np.ndarray,
    candidate_racket_rotations: np.ndarray,
) -> float:
    """Return the minimum SO(3) error over the racket's 180-degree symmetry class."""

    achieved = Rotation.from_matrix(np.asarray(achieved_racket_rotation, dtype=float).reshape(3, 3))
    candidates = Rotation.from_matrix(np.asarray(candidate_racket_rotations, dtype=float).reshape(2, 3, 3))
    return float(np.min((candidates * achieved.inv()).magnitude()))


def attachment_handle_intersects_palm(
    attachment: TennisRacketAttachment,
    *,
    inset_fraction: float = 0.15,
) -> bool:
    """Check that the grasp origin lies inside an inset palm volume.

    The racket frame origin is the handle centerline at the grasp. Requiring it
    inside the inset rubber-hand bounds prevents surface-resting calibrations.
    """

    if not 0.0 <= inset_fraction < 0.5:
        raise ValueError("inset_fraction must be in [0, 0.5)")
    span = attachment.palm_bounds_max_m - attachment.palm_bounds_min_m
    minimum = attachment.palm_bounds_min_m + inset_fraction * span
    maximum = attachment.palm_bounds_max_m - inset_fraction * span
    return bool(np.all(attachment.position_m >= minimum) and np.all(attachment.position_m <= maximum))


def achieved_tennis_racket_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    attachment: TennisRacketAttachment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the achieved world-space racket pose from the shared rigid hand attachment."""

    hand_body_id = model.body(attachment.hand_link).id
    hand_position = np.asarray(data.xpos[hand_body_id], dtype=float)
    hand_rotation = np.asarray(data.xmat[hand_body_id], dtype=float).reshape(3, 3)
    racket_rotation = hand_rotation @ rotations_from_wxyz(attachment.quaternion_wxyz).as_matrix()
    racket_position = hand_position + hand_rotation @ attachment.position_m
    return racket_position, racket_rotation, hand_rotation
