"""XSens naming and grounding adapters for generic morphology transfer."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XsensHdf5Motion,
    XsensHdf5Tpose,
    load_xsens_hdf5_calibration,
    load_xsens_hdf5_tpose,
)
from holosoma_retargeting.kinematics import (
    GroundingSurface,
    KinematicMorphologyAdapter,
    KinematicMotion,
    KinematicPose,
    KinematicTree,
    LowestSurfaceGrounding,
    SurfacePoseEvaluator,
    with_body_attachments,
)
from holosoma_retargeting.kinematics.model import rotate_vector
from holosoma_retargeting.xsens.g1_kinematic_reduction import (
    G1XsensReductionConfig,
    build_g1_proportioned_xsens_tree,
    extract_g1_anthropometry,
)
from holosoma_retargeting.xsens.geometry_attachments import build_xsens_avatar_mesh_attachments
from holosoma_retargeting.xsens.kinematic_model import build_xsens_kinematic_tree, normalize_xsens_name

XsensGroundingMode = Literal["none", "match_lowest_soles"]
XsensRootMotionMode = Literal[
    "preserve_world",
    "scale_by_leg_length",
    "scale_by_leg_length_contact_aware",
]
_SOLE_SOURCE_BODY_NAMES = ("LeftFoot", "LeftToe", "RightFoot", "RightToe")
_ROOT_MOTION_MODES = {
    "preserve_world",
    "scale_by_leg_length",
    "scale_by_leg_length_contact_aware",
}
_SIDES = ("Left", "Right")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XsensRootMotionConfig:
    """Configure floating-base translation during Xsens morphology transfer."""

    mode: XsensRootMotionMode = "scale_by_leg_length"
    ground_height_m: float | None = None
    contact_height_tolerance_m: float = 0.03
    contact_speed_threshold_m_s: float = 0.3
    contact_min_duration_s: float = 0.10
    contact_max_gap_s: float = 0.067


@dataclass(frozen=True)
class XsensRootMotionReport:
    """Resolved measurements and contact counts for one transferred motion."""

    mode: XsensRootMotionMode
    source_leg_length_m: float
    target_leg_length_m: float
    scale: float
    ground_height_m: float
    left_contact_frames: int
    right_contact_frames: int


@dataclass(frozen=True)
class PreparedG1XsensMorphology:
    """Reusable subject/target models and adapter for G1-proportioned Xsens poses."""

    source_model: KinematicTree
    target_model: KinematicTree
    adapter: KinematicMorphologyAdapter


def _body_to_source_mapping(
    model: KinematicTree,
    source_body_names: tuple[str, ...],
    *,
    model_label: str,
) -> dict[str, str]:
    normalized_sources: dict[str, str] = {}
    for name in source_body_names:
        normalized = normalize_xsens_name(name)
        if normalized in normalized_sources:
            raise ValueError(f"XSens source body names contain a duplicate semantic name '{name}'")
        normalized_sources[normalized] = name

    model_source_names = tuple(str(body.metadata.get("xsens:sourceSegmentName", body.name)) for body in model.bodies)
    normalized_model_order = tuple(normalize_xsens_name(name) for name in model_source_names)
    normalized_source_order = tuple(normalize_xsens_name(name) for name in source_body_names)
    if normalized_model_order != normalized_source_order:
        raise ValueError(
            f"{model_label} XSens semantic body order does not match the source motion: "
            f"model={model_source_names}, source={source_body_names}"
        )

    return {
        body.name: normalized_sources[normalize_xsens_name(source_name)]
        for body, source_name in zip(model.bodies, model_source_names, strict=True)
    }


def _outsole_surfaces(model: KinematicTree, *, side: str | None = None) -> tuple[GroundingSurface, ...]:
    if side is not None and side not in _SIDES:
        raise ValueError(f"Unknown Xsens body side '{side}'")
    source_to_body = {
        normalize_xsens_name(str(body.metadata.get("xsens:sourceSegmentName", body.name))): body
        for body in model.bodies
    }
    surfaces: list[GroundingSurface] = []
    for source_name in _SOLE_SOURCE_BODY_NAMES:
        if side is not None and not source_name.startswith(side):
            continue
        body = source_to_body.get(normalize_xsens_name(source_name))
        if body is None:
            raise KeyError(f"XSens grounding model is missing body '{source_name}'")
        mesh_names = tuple(mesh.name for mesh in body.meshes if "outsole" in mesh.name.lower())
        if not mesh_names:
            raise ValueError(f"XSens grounding body '{body.name}' has no outsole mesh")
        surfaces.append(GroundingSurface(body.name, mesh_names))
    return tuple(surfaces)


def build_xsens_morphology_adapter(
    target_model: KinematicTree,
    source_segment_names: Sequence[str],
    *,
    source_model: KinematicTree | None = None,
    grounding: XsensGroundingMode = "none",
) -> KinematicMorphologyAdapter:
    """Build a generic adapter using XSens semantic body metadata and order."""

    source_names = tuple(source_segment_names)
    target_mapping = _body_to_source_mapping(target_model, source_names, model_label="Target")
    grounding_policy = None
    if grounding == "match_lowest_soles":
        if source_model is None:
            raise ValueError("XSens lowest-sole grounding requires a source model")
        source_mapping = _body_to_source_mapping(source_model, source_names, model_label="Source")
        grounding_policy = LowestSurfaceGrounding(
            source_model,
            target_model,
            source_names,
            source_body_to_pose_body=source_mapping,
            target_body_to_pose_body=target_mapping,
            source_surfaces=_outsole_surfaces(source_model),
            target_surfaces=_outsole_surfaces(target_model),
        )
    elif grounding != "none":
        raise ValueError(f"Unknown XSens grounding mode '{grounding}'")

    return KinematicMorphologyAdapter(
        target_model,
        source_names,
        target_body_to_source_body=target_mapping,
        grounding=grounding_policy,
    )


def _validate_root_motion_config(config: XsensRootMotionConfig) -> None:
    if config.mode not in _ROOT_MOTION_MODES:
        raise ValueError(f"Unknown Xsens root-motion mode '{config.mode}'")
    if config.ground_height_m is not None and not np.isfinite(config.ground_height_m):
        raise ValueError("ground_height_m must be finite when provided")
    non_negative = {
        "contact_height_tolerance_m": config.contact_height_tolerance_m,
        "contact_speed_threshold_m_s": config.contact_speed_threshold_m_s,
        "contact_max_gap_s": config.contact_max_gap_s,
    }
    for name, value in non_negative.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(config.contact_min_duration_s) or config.contact_min_duration_s <= 0.0:
        raise ValueError("contact_min_duration_s must be finite and positive")


def _reference_pose(model: KinematicTree) -> KinematicPose:
    return KinematicPose(
        tuple(body.name for body in model.bodies),
        np.asarray([body.reference_pose.translation_m for body in model.bodies], dtype=float),
        np.asarray([body.reference_pose.rotation_wxyz for body in model.bodies], dtype=float),
    )


def _surface_evaluator(
    model: KinematicTree,
    pose_body_names: tuple[str, ...],
    body_to_pose_body: Mapping[str, str],
    *,
    side: str | None = None,
) -> SurfacePoseEvaluator:
    return SurfacePoseEvaluator(
        model,
        pose_body_names,
        body_to_pose_body,
        _outsole_surfaces(model, side=side),
    )


def _functional_leg_length_m(model: KinematicTree) -> float:
    """Measure mean neutral hip-to-lowest-outsole vertical distance."""

    body_names = tuple(body.name for body in model.bodies)
    identity_mapping = {name: name for name in body_names}
    reference_pose = _reference_pose(model)
    body_map = model.body_map()
    joint_map = model.joint_map()
    lengths: list[float] = []
    for side in _SIDES:
        joint_name = f"{side}Hip"
        if joint_name not in joint_map:
            raise KeyError(f"Xsens morphology model is missing joint '{joint_name}'")
        hip = joint_map[joint_name]
        parent = body_map[hip.parent_body]
        hip_world = parent.reference_pose.translation_m + rotate_vector(
            parent.reference_pose.rotation_wxyz,
            hip.parent_frame.translation_m,
        )
        sole_height = _surface_evaluator(
            model,
            body_names,
            identity_mapping,
            side=side,
        ).minimum_height_m(reference_pose)
        length = float(hip_world[2] - sole_height)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(f"{side} hip-to-outsole leg length must be finite and positive, got {length}")
        lengths.append(length)
    return float(np.mean(lengths))


def _surface_references(
    motion: KinematicMotion,
    evaluator: SurfacePoseEvaluator,
) -> np.ndarray:
    return evaluator.support_references_m(motion)


def _resolved_ground_height_m(
    left_references: np.ndarray,
    right_references: np.ndarray,
    configured_height_m: float | None,
) -> float:
    if configured_height_m is not None:
        return float(configured_height_m)
    heights = np.concatenate([left_references[:, 2], right_references[:, 2]])
    sample_count = max(1, int(np.ceil(0.05 * heights.size)))
    lowest = np.partition(heights, sample_count - 1)[:sample_count]
    return float(np.median(lowest))


def _boolean_runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, item in enumerate(mask):
        if bool(item) == value and start is None:
            start = index
        elif bool(item) != value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _run_duration_s(times_s: np.ndarray, start: int, end: int, nominal_dt_s: float) -> float:
    return float(times_s[end] - times_s[start] + nominal_dt_s)


def _clean_contact_mask(
    mask: np.ndarray,
    times_s: np.ndarray,
    *,
    max_gap_s: float,
    min_duration_s: float,
) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool).copy()
    if cleaned.size == 0:
        return cleaned
    intervals = np.diff(times_s)
    nominal_dt_s = float(np.median(intervals)) if intervals.size else min_duration_s
    for start, end in _boolean_runs(cleaned, False):
        if start == 0 or end == cleaned.size - 1:
            continue
        if _run_duration_s(times_s, start, end, nominal_dt_s) <= max_gap_s:
            cleaned[start : end + 1] = True
    for start, end in _boolean_runs(cleaned, True):
        if _run_duration_s(times_s, start, end, nominal_dt_s) < min_duration_s:
            cleaned[start : end + 1] = False
    return cleaned


def _detect_contacts(
    references: np.ndarray,
    times_s: np.ndarray,
    *,
    ground_height_m: float,
    config: XsensRootMotionConfig,
) -> np.ndarray:
    if references.shape[0] <= 1:
        planar_speeds = np.zeros(references.shape[0], dtype=float)
    else:
        planar_velocity = np.gradient(references[:, :2], times_s, axis=0)
        planar_speeds = np.linalg.norm(planar_velocity, axis=1)
    raw = (references[:, 2] <= ground_height_m + config.contact_height_tolerance_m) & (
        planar_speeds <= config.contact_speed_threshold_m_s
    )
    return _clean_contact_mask(
        raw,
        times_s,
        max_gap_s=config.contact_max_gap_s,
        min_duration_s=config.contact_min_duration_s,
    )


def _contact_aware_xy_correction(
    target_references: Mapping[str, np.ndarray],
    contacts: Mapping[str, np.ndarray],
) -> np.ndarray:
    frame_count = next(iter(target_references.values())).shape[0]
    corrections = np.zeros((frame_count, 2), dtype=float)
    locks: dict[str, np.ndarray] = {}
    previous = np.zeros(2, dtype=float)
    for frame in range(frame_count):
        active: list[str] = []
        for side in _SIDES:
            side_contacts = contacts[side]
            if side_contacts[frame]:
                active.append(side)
                if frame == 0 or not side_contacts[frame - 1]:
                    locks[side] = target_references[side][frame, :2] + previous
        if active:
            previous = np.mean(
                [locks[side] - target_references[side][frame, :2] for side in active],
                axis=0,
            )
        corrections[frame] = previous
    return corrections


def apply_xsens_root_motion(
    source: KinematicMotion,
    raw_target: KinematicMotion,
    *,
    source_model: KinematicTree,
    target_model: KinematicTree,
    grounding: XsensGroundingMode,
    config: XsensRootMotionConfig,
) -> tuple[KinematicMotion, XsensRootMotionReport]:
    """Translate a reconstructed target motion according to one root-motion policy."""

    _validate_root_motion_config(config)
    if grounding not in {"none", "match_lowest_soles"}:
        raise ValueError(f"Unknown XSens grounding mode '{grounding}'")
    if source.body_names != raw_target.body_names:
        raise ValueError("Source and target body names/order must match for root-motion transfer")
    if source.positions_m.shape != raw_target.positions_m.shape:
        raise ValueError("Source and target position arrays must have the same shape")
    if not np.array_equal(source.times_s, raw_target.times_s):
        raise ValueError("Source and target timestamps must match for root-motion transfer")

    source_mapping = _body_to_source_mapping(source_model, source.body_names, model_label="Source")
    target_mapping = _body_to_source_mapping(target_model, source.body_names, model_label="Target")
    source_all = _surface_evaluator(source_model, source.body_names, source_mapping)
    target_all = _surface_evaluator(target_model, source.body_names, target_mapping)
    source_sides = {
        side: _surface_evaluator(source_model, source.body_names, source_mapping, side=side) for side in _SIDES
    }
    needs_contact_references = config.mode == "scale_by_leg_length_contact_aware"
    source_references = (
        {side: _surface_references(source, source_sides[side]) for side in _SIDES}
        if config.ground_height_m is None or needs_contact_references
        else {}
    )
    ground_height_m = (
        _resolved_ground_height_m(
            source_references["Left"],
            source_references["Right"],
            None,
        )
        if config.ground_height_m is None
        else float(config.ground_height_m)
    )
    source_leg_length_m = _functional_leg_length_m(source_model)
    target_leg_length_m = _functional_leg_length_m(target_model)
    measured_scale = target_leg_length_m / source_leg_length_m
    scale = 1.0 if config.mode == "preserve_world" else measured_scale
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Xsens root-motion scale must be finite and positive, got {scale}")

    source_root_name = source_mapping[source_model.root_body]
    root_index = source.body_names.index(source_root_name)
    source_root = np.asarray(source.positions_m[:, root_index], dtype=float)
    positions = np.asarray(raw_target.positions_m, dtype=float).copy()
    initial_xy = source_root[0, :2].copy()
    translation = np.zeros((positions.shape[0], 3), dtype=float)
    desired_xy = initial_xy + scale * (source_root[:, :2] - initial_xy)
    translation[:, :2] = desired_xy - positions[:, root_index, :2]
    if grounding == "match_lowest_soles":
        source_heights = source_all.minimum_heights_m(source)
        desired_heights = ground_height_m + scale * (source_heights - ground_height_m)
        target_heights = target_all.minimum_heights_m(raw_target)
        translation[:, 2] = desired_heights - target_heights
    else:
        desired_root_z = ground_height_m + scale * (source_root[:, 2] - ground_height_m)
        translation[:, 2] = desired_root_z - positions[:, root_index, 2]
    positions += translation[:, None, :]

    contacts = {side: np.zeros(positions.shape[0], dtype=bool) for side in _SIDES}
    if needs_contact_references:
        contacts = {
            side: _detect_contacts(
                source_references[side],
                np.asarray(source.times_s, dtype=float),
                ground_height_m=ground_height_m,
                config=config,
            )
            for side in _SIDES
        }
        baseline = KinematicMotion(
            raw_target.body_names,
            positions,
            np.asarray(raw_target.orientations_wxyz, dtype=float),
            np.asarray(raw_target.times_s, dtype=float),
        )
        target_sides = {
            side: _surface_evaluator(target_model, source.body_names, target_mapping, side=side)
            for side in _SIDES
        }
        target_references = {
            side: _surface_references(baseline, target_sides[side]) for side in _SIDES
        }
        correction_xy = _contact_aware_xy_correction(target_references, contacts)
        positions[:, :, :2] += correction_xy[:, None, :]

    motion = KinematicMotion(
        raw_target.body_names,
        positions,
        np.asarray(raw_target.orientations_wxyz).copy(),
        np.asarray(raw_target.times_s).copy(),
    )
    report = XsensRootMotionReport(
        mode=config.mode,
        source_leg_length_m=source_leg_length_m,
        target_leg_length_m=target_leg_length_m,
        scale=float(scale),
        ground_height_m=ground_height_m,
        left_contact_frames=int(np.count_nonzero(contacts["Left"])),
        right_contact_frames=int(np.count_nonzero(contacts["Right"])),
    )
    return motion, report


def build_subject_xsens_reference_model(
    hdf5_path: str | Path,
    *,
    include_tennis_racket: bool = True,
) -> KinematicTree:
    """Build the calibrated subject model and visuals needed for grounding."""

    calibration = load_xsens_hdf5_calibration(Path(hdf5_path).expanduser())
    model = build_xsens_kinematic_tree(
        calibration,
        include_tennis_racket=include_tennis_racket,
    )
    body_names = {body.name for body in model.bodies}
    meshes = {
        name: attachments
        for name, attachments in build_xsens_avatar_mesh_attachments(calibration).items()
        if name in body_names
    }
    return with_body_attachments(model, meshes=meshes)


def prepare_g1_xsens_morphology(
    source_segment_names: Sequence[str],
    *,
    hdf5_path: str | Path,
    g1_model_path: str | Path | None = None,
    grounding: XsensGroundingMode = "match_lowest_soles",
    preserve_joint_offsets: bool = False,
) -> PreparedG1XsensMorphology:
    """Build the direct subject-to-G1 Xsens morphology path used by retargeting.

    The resulting adapter performs no IK or optimization: it preserves every
    source segment orientation (and thus every relative joint rotation) while
    reconstructing body origins from the G1-sized tree's joint anchors.
    """

    source_names = tuple(source_segment_names)
    source_model = build_subject_xsens_reference_model(
        hdf5_path,
        include_tennis_racket=False,
    )
    target_model = build_g1_proportioned_xsens_tree(
        extract_g1_anthropometry(g1_model_path),
        G1XsensReductionConfig(
            preserve_joint_offsets=preserve_joint_offsets,
            include_visuals=True,
            include_tennis_racket=False,
        ),
    )
    adapter = build_xsens_morphology_adapter(
        target_model,
        source_names,
        source_model=source_model if grounding == "match_lowest_soles" else None,
        grounding=grounding,
    )
    return PreparedG1XsensMorphology(source_model, target_model, adapter)


def adapt_xsens_motion_to_g1(
    motion: XsensHdf5Motion,
    *,
    hdf5_path: str | Path,
    g1_model_path: str | Path | None = None,
    grounding: XsensGroundingMode = "match_lowest_soles",
    root_motion: XsensRootMotionConfig | None = None,
    preserve_joint_offsets: bool = False,
) -> KinematicMotion:
    """Reconstruct one body-only Xsens motion using G1-derived proportions."""

    root_motion = XsensRootMotionConfig() if root_motion is None else root_motion
    prepared = prepare_g1_xsens_morphology(
        motion.segment_names,
        hdf5_path=hdf5_path,
        g1_model_path=g1_model_path,
        grounding="none",
        preserve_joint_offsets=preserve_joint_offsets,
    )
    source = KinematicMotion(
        tuple(motion.segment_names),
        motion.positions_m,
        motion.quaternions_wijk,
        motion.times_s,
    )
    adapted, report = apply_xsens_root_motion(
        source,
        prepared.adapter.adapt_motion(source),
        source_model=prepared.source_model,
        target_model=prepared.target_model,
        grounding=grounding,
        config=root_motion,
    )
    logger.info(
        "Mapped Xsens root motion (mode=%s, grounding=%s, source_leg=%.4f m, "
        "target_leg=%.4f m, scale=%.5f, ground=%.4f m, contacts_left=%d, contacts_right=%d)",
        report.mode,
        grounding,
        report.source_leg_length_m,
        report.target_leg_length_m,
        report.scale,
        report.ground_height_m,
        report.left_contact_frames,
        report.right_contact_frames,
    )
    return adapted


def adapt_xsens_tpose_to_g1(
    *,
    hdf5_path: str | Path,
    g1_model_path: str | Path | None = None,
    grounding: XsensGroundingMode = "match_lowest_soles",
    preserve_joint_offsets: bool = False,
    variant: str = "Tpose",
) -> XsensHdf5Tpose:
    """Reconstruct a recorded Xsens T-pose with G1-derived proportions.

    Only positions are morphology-adapted. The recorded global segment
    orientations remain byte-for-byte identical so the sensor-frame side of
    orientation-offset calibration is unchanged.
    """

    tpose = load_xsens_hdf5_tpose(hdf5_path, variant=variant)
    prepared = prepare_g1_xsens_morphology(
        tpose.segment_names,
        hdf5_path=hdf5_path,
        g1_model_path=g1_model_path,
        grounding=grounding,
        preserve_joint_offsets=preserve_joint_offsets,
    )
    adapted = prepared.adapter.adapt_pose(
        KinematicPose(
            tuple(tpose.segment_names),
            tpose.positions_m,
            tpose.quaternions_wijk,
        )
    )
    return XsensHdf5Tpose(
        positions_m=adapted.positions_m,
        quaternions_wijk=adapted.orientations_wxyz,
        variant=f"G1Proportioned{variant}",
        segment_names=list(tpose.segment_names),
        source_indices=list(tpose.source_indices),
    )


def xsens_body_to_source_mapping(
    model: KinematicTree,
    source_segment_names: Sequence[str],
) -> Mapping[str, str]:
    """Return the validated model-body to source-segment mapping."""

    return _body_to_source_mapping(model, tuple(source_segment_names), model_label="Model")


__all__ = [
    "PreparedG1XsensMorphology",
    "XsensGroundingMode",
    "XsensRootMotionConfig",
    "XsensRootMotionMode",
    "XsensRootMotionReport",
    "adapt_xsens_motion_to_g1",
    "adapt_xsens_tpose_to_g1",
    "apply_xsens_root_motion",
    "build_subject_xsens_reference_model",
    "build_xsens_morphology_adapter",
    "prepare_g1_xsens_morphology",
    "xsens_body_to_source_mapping",
]
