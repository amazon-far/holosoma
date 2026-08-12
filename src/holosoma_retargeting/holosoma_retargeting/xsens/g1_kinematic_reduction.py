"""Configuration-independent reduction of the G1 kinematic tree to Xsens topology."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mujoco  # type: ignore[import-not-found]
import numpy as np

from holosoma_retargeting.kinematics import (
    KinematicTree,
    MeshAttachment,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
    compute_reference_joint_positions,
    validate_kinematic_tree,
)
from holosoma_retargeting.usd import create_usd_stage, validate_usd_kinematic_tree, write_kinematic_tree_to_stage
from holosoma_retargeting.xsens.avatar_mesh import (
    LIGHT_GRAY,
    AvatarMeshPart,
    XsensAvatarProportions,
    build_tennis_racket_meshes,
    build_xsens_avatar_meshes,
    cylinder_between,
)
from holosoma_retargeting.xsens.kinematic_model import (
    TENNIS_RACKET_BODY,
    XSENS_JOINT_SPECS,
    XSENS_RACKET_SOURCE_SEGMENT,
    canonical_xsens_joint_name,
    canonical_xsens_segment_name,
)

G1_XSENS_REDUCTION_VERSION = "10"
XSENS_JOINT_STREAM_NAMES = (
    "body_joint_angles_eulerZXY_xyz_rad",
    "body_joint_angles_eulerXZY_xyz_rad",
)
SPINE_FRACTIONS = (0.1875, 0.375, 0.5625, 0.75, 1.0)


@dataclass(frozen=True)
class G1XsensReductionConfig:
    """Options controlling the idealized G1-to-Xsens reduction."""

    preserve_joint_offsets: bool = False
    include_visuals: bool = True
    include_tennis_racket: bool = True


@dataclass(frozen=True)
class G1Anthropometry:
    """Pose-independent dimensions extracted from fixed G1 model transforms."""

    model_path: Path
    model_sha256: str
    lengths_m: Mapping[str, float]
    side_lengths_m: Mapping[str, float]
    widths_m: Mapping[str, float]
    root_anchors_m: Mapping[str, np.ndarray]
    compound_offsets_m: Mapping[str, np.ndarray]
    compound_offset_edges_m: Mapping[str, tuple[np.ndarray, ...]]
    span_vectors_m: Mapping[str, np.ndarray]
    region_extents_m: Mapping[str, np.ndarray]
    region_centers_m: Mapping[str, np.ndarray]
    segment_radii_m: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _ShoulderMorphologyFit:
    """Fixed virtual-joint anchors fitted to canonical T- and N-pose endpoints."""

    parent_anchor_m: np.ndarray
    child_anchor_m: np.ndarray
    tpose_target_offset_m: np.ndarray
    npose_target_offset_m: np.ndarray
    npose_child_rotation: np.ndarray
    tpose_error_m: float
    npose_error_m: float


@dataclass(frozen=True)
class G1XsensProportionReport:
    """Machine-readable diagnostics for one generated G1-proportioned Xsens USD."""

    source_path: Path
    output_path: Path
    report_path: Path
    source_sha256: str
    preserve_joint_offsets: bool
    body_count: int
    joint_count: int
    target_lengths_m: Mapping[str, float]
    generated_lengths_m: Mapping[str, float]
    widths_m: Mapping[str, float]
    root_anchors_m: Mapping[str, tuple[float, float, float]]
    raw_offsets_m: Mapping[str, tuple[float, float, float]]
    raw_offset_edges_m: Mapping[str, tuple[tuple[float, float, float], ...]]
    collapsed_adapter_offsets_m: Mapping[str, tuple[float, float, float]]
    applied_offsets_m: Mapping[str, tuple[float, float, float]]
    max_length_error_m: float
    max_joint_residual_m: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "report_path": str(self.report_path),
            "source_sha256": self.source_sha256,
            "generator_version": G1_XSENS_REDUCTION_VERSION,
            "preserve_joint_offsets": self.preserve_joint_offsets,
            "body_count": self.body_count,
            "joint_count": self.joint_count,
            "target_lengths_m": dict(self.target_lengths_m),
            "generated_lengths_m": dict(self.generated_lengths_m),
            "widths_m": dict(self.widths_m),
            "root_anchors_m": {name: list(value) for name, value in self.root_anchors_m.items()},
            "raw_offsets_m": {name: list(value) for name, value in self.raw_offsets_m.items()},
            "raw_offset_edge_frame": "parent_body_local",
            "raw_offset_edges_m": {
                name: [list(edge) for edge in edges] for name, edges in self.raw_offset_edges_m.items()
            },
            "collapsed_adapter_offsets_m": {
                name: list(value) for name, value in self.collapsed_adapter_offsets_m.items()
            },
            "applied_offsets_m": {name: list(value) for name, value in self.applied_offsets_m.items()},
            "max_length_error_m": self.max_length_error_m,
            "max_joint_residual_m": self.max_joint_residual_m,
            "warnings": list(self.warnings),
        }


def _default_g1_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "g1" / "g1_29dof.xml"


def _resolve_model_path(path: str | Path | None) -> Path:
    model_path = _default_g1_model_path() if path is None else Path(path)
    if not model_path.is_file():
        package_relative = Path(__file__).resolve().parents[1] / model_path
        if package_relative.is_file():
            model_path = package_relative
        else:
            raise FileNotFoundError(f"G1 model not found: {path}")
    return model_path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quat_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _static_body_transforms(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Compose fixed body transforms without creating MjData or evaluating qpos."""

    positions = np.zeros((model.nbody, 3), dtype=float)
    rotations = np.tile(np.eye(3), (model.nbody, 1, 1))
    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        parent_rotation = rotations[parent_id]
        rotations[body_id] = parent_rotation @ _quat_matrix(model.body_quat[body_id])
        positions[body_id] = positions[parent_id] + parent_rotation @ np.asarray(model.body_pos[body_id])
    return positions, rotations


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise KeyError(f"G1 model is missing {object_type.name} '{name}'")
    return int(object_id)


def _joint_positions(
    model: mujoco.MjModel,
    body_positions: np.ndarray,
    body_rotations: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        if not name:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        result[name] = body_positions[body_id] + body_rotations[body_id] @ np.asarray(model.jnt_pos[joint_id])
    return result


def _body_point(
    model: mujoco.MjModel,
    body_positions: np.ndarray,
    name: str,
) -> np.ndarray:
    return body_positions[_named_id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def _local_body_translation(model: mujoco.MjModel, name: str) -> np.ndarray:
    """Return a fixed child-body translation expressed in its parent body frame."""

    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.asarray(model.body_pos[body_id], dtype=float).copy()


def _geom_mesh_points(
    model: mujoco.MjModel,
    body_positions: np.ndarray,
    body_rotations: np.ndarray,
    geom_name: str,
) -> np.ndarray:
    geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        center = body_positions[int(model.geom_bodyid[geom_id])]
        radius = float(np.max(model.geom_size[geom_id]))
        return np.asarray([center - radius, center + radius])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    geom_rotation = _quat_matrix(model.geom_quat[geom_id])
    geom_position = np.asarray(model.geom_pos[geom_id], dtype=float)
    body_id = int(model.geom_bodyid[geom_id])
    local_points = vertices @ geom_rotation.T + geom_position
    return local_points @ body_rotations[body_id].T + body_positions[body_id]


def _combined_geom_points(
    model: mujoco.MjModel,
    body_positions: np.ndarray,
    body_rotations: np.ndarray,
    names: tuple[str, ...],
) -> np.ndarray:
    return np.vstack([_geom_mesh_points(model, body_positions, body_rotations, name) for name in names])


def _aabb_extent(points: np.ndarray) -> np.ndarray:
    return np.asarray(points.max(axis=0) - points.min(axis=0), dtype=float)


def _transverse_radii(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    axis = np.asarray(end - start, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-10:
        return np.array([0.025, 0.025])
    axis /= norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_a = reference - float(np.dot(reference, axis)) * axis
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(axis, basis_a)
    centered = points - start
    extents = np.array(
        [
            np.ptp(centered @ basis_a) * 0.5,
            np.ptp(centered @ basis_b) * 0.5,
        ]
    )
    return np.maximum(extents, 0.008)


def _mean_sides(values: Mapping[str, float], metric: str) -> float:
    return 0.5 * (float(values[f"left_{metric}"]) + float(values[f"right_{metric}"]))


def extract_g1_anthropometry(robot_model_path: str | Path | None = None) -> G1Anthropometry:
    """Extract G1 dimensions from static model transforms; no kinematic state is accepted or evaluated."""

    model_path = _resolve_model_path(robot_model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    body_positions, body_rotations = _static_body_transforms(model)
    joints = _joint_positions(model, body_positions, body_rotations)
    pelvis = _body_point(model, body_positions, "pelvis")

    side_lengths: dict[str, float] = {}
    root_anchors: dict[str, np.ndarray] = {}
    offsets: dict[str, np.ndarray] = {}
    offset_edges: dict[str, tuple[np.ndarray, ...]] = {}
    spans: dict[str, np.ndarray] = {}
    radii_by_side: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        shoulder_pitch = joints[f"{side}_shoulder_pitch_joint"]
        shoulder_yaw = joints[f"{side}_shoulder_yaw_joint"]
        elbow = joints[f"{side}_elbow_joint"]
        wrist_roll = joints[f"{side}_wrist_roll_joint"]
        wrist_yaw = joints[f"{side}_wrist_yaw_joint"]
        hip_pitch = joints[f"{side}_hip_pitch_joint"]
        hip_yaw = joints[f"{side}_hip_yaw_joint"]
        knee = joints[f"{side}_knee_joint"]
        ankle_pitch = joints[f"{side}_ankle_pitch_joint"]
        ankle_roll = joints[f"{side}_ankle_roll_joint"]
        hand_tip = _body_point(model, body_positions, f"{side}_pinky_link")
        metatarsal = 0.5 * (
            _body_point(model, body_positions, f"{side}_ankle_roll_sphere_3_link")
            + _body_point(model, body_positions, f"{side}_ankle_roll_sphere_4_link")
        )
        toe_tip = _body_point(model, body_positions, f"{side}_ankle_roll_sphere_5_link")

        points = {
            "upper_arm": (shoulder_yaw, elbow),
            "forearm": (elbow, wrist_roll),
            "hand": (wrist_yaw, hand_tip),
            "thigh": (hip_yaw, knee),
            "shank": (knee, ankle_pitch),
            "foot": (ankle_roll, metatarsal),
            "toe": (metatarsal, toe_tip),
        }
        geom_names = {
            "upper_arm": (f"{side}_shoulder_yaw_link",),
            "forearm": (f"{side}_elbow_link",),
            "hand": (f"{side}_rubber_hand_link",),
            "thigh": (f"{side}_hip_yaw_link",),
            "shank": (f"{side}_knee_link",),
            "foot": (f"{side}_ankle_roll_link",),
        }
        for metric, (start, end) in points.items():
            side_lengths[f"{side}_{metric}"] = float(np.linalg.norm(end - start))
            if metric in geom_names:
                mesh_points = _combined_geom_points(
                    model,
                    body_positions,
                    body_rotations,
                    geom_names[metric],
                )
                radii_by_side[f"{side}_{metric}"] = _transverse_radii(mesh_points, start, end)

        offsets[f"{side}_shoulder"] = shoulder_yaw - shoulder_pitch
        offsets[f"{side}_hip"] = hip_yaw - hip_pitch
        offsets[f"{side}_wrist"] = wrist_yaw - wrist_roll
        offsets[f"{side}_ankle"] = ankle_roll - ankle_pitch
        offset_edges[f"{side}_shoulder"] = (
            _local_body_translation(model, f"{side}_shoulder_roll_link"),
            _local_body_translation(model, f"{side}_shoulder_yaw_link"),
        )
        offset_edges[f"{side}_hip"] = (
            _local_body_translation(model, f"{side}_hip_roll_link"),
            _local_body_translation(model, f"{side}_hip_yaw_link"),
        )
        offset_edges[f"{side}_wrist"] = (
            _local_body_translation(model, f"{side}_wrist_pitch_link"),
            _local_body_translation(model, f"{side}_wrist_yaw_link"),
        )
        offset_edges[f"{side}_ankle"] = (_local_body_translation(model, f"{side}_ankle_roll_link"),)
        spans[f"{side}_arm"] = elbow - shoulder_yaw
        spans[f"{side}_forearm"] = wrist_roll - elbow
        spans[f"{side}_leg"] = knee - hip_yaw
        spans[f"{side}_shank"] = ankle_pitch - knee
        root_anchors[f"{side}_hip"] = hip_pitch - pelvis

    shoulder_center = 0.5 * (joints["left_shoulder_pitch_joint"] + joints["right_shoulder_pitch_joint"])
    pelvis_points = _combined_geom_points(
        model,
        body_positions,
        body_rotations,
        ("pelvis_contour_link",),
    )
    torso_points = _geom_mesh_points(model, body_positions, body_rotations, "torso_link")
    head_points = _geom_mesh_points(model, body_positions, body_rotations, "head_link")
    head_min = head_points.min(axis=0)
    head_max = head_points.max(axis=0)
    head_extent = head_max - head_min
    offsets["waist"] = joints["waist_pitch_joint"] - joints["waist_yaw_joint"]
    offset_edges["waist"] = (
        _local_body_translation(model, "waist_roll_link"),
        _local_body_translation(model, "torso_link"),
    )
    spans["torso"] = shoulder_center - pelvis

    lengths = {
        metric: _mean_sides(side_lengths, metric)
        for metric in ("upper_arm", "forearm", "hand", "thigh", "shank", "foot", "toe")
    }
    lengths["torso"] = float(np.linalg.norm(spans["torso"]))
    lengths["neck"] = max(0.02, float(head_min[2] - shoulder_center[2]))
    lengths["head"] = float(head_extent[2])
    widths = {
        "shoulder": float(np.linalg.norm(joints["left_shoulder_pitch_joint"] - joints["right_shoulder_pitch_joint"])),
        "hip": float(np.linalg.norm(joints["left_hip_pitch_joint"] - joints["right_hip_pitch_joint"])),
    }
    region_extents = {
        "pelvis": _aabb_extent(pelvis_points),
        "torso": _aabb_extent(torso_points),
        "head": head_extent,
    }
    region_centers = {
        "pelvis": 0.5 * (pelvis_points.min(axis=0) + pelvis_points.max(axis=0)) - pelvis,
        "torso": 0.5 * (torso_points.min(axis=0) + torso_points.max(axis=0)) - pelvis,
        "head": np.array(
            [
                0.5 * (head_min[0] + head_max[0]) - shoulder_center[0],
                0.5 * (head_min[1] + head_max[1]) - shoulder_center[1],
                0.5 * head_extent[2],
            ]
        ),
    }
    segment_radii = {
        metric: 0.5 * (radii_by_side[f"left_{metric}"] + radii_by_side[f"right_{metric}"])
        for metric in ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")
    }
    segment_radii["toe"] = np.maximum(segment_radii["foot"] * np.array([0.85, 0.75]), 0.008)

    return G1Anthropometry(
        model_path=model_path,
        model_sha256=_sha256(model_path),
        lengths_m=lengths,
        side_lengths_m=side_lengths,
        widths_m=widths,
        root_anchors_m=root_anchors,
        compound_offsets_m=offsets,
        compound_offset_edges_m=offset_edges,
        span_vectors_m=spans,
        region_extents_m=region_extents,
        region_centers_m=region_centers,
        segment_radii_m=segment_radii,
    )


def _align_vector(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, source))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, reference)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _canonical_offsets(anthropometry: G1Anthropometry, preserve: bool) -> dict[str, np.ndarray]:
    """Reduce fixed compound edges into the canonical Xsens reference frames.

    Shoulder and wrist clusters retain their net endpoint displacement rather
    than their internal edge-path length. Hip and waist aggregates already live
    in canonical root frames. The single ankle edge is expressed in the
    canonical shank frame, rather than being incorrectly rotated with the
    thigh.
    """

    result = {name: np.zeros(3, dtype=float) for name in anthropometry.compound_offsets_m}
    if not preserve:
        return result
    result.update(_canonical_upper_limb_offsets(anthropometry))
    for side in ("left", "right"):
        result[f"{side}_hip"] = anthropometry.compound_offsets_m[f"{side}_hip"].copy()
        shank_rotation = _align_vector(
            anthropometry.span_vectors_m[f"{side}_shank"],
            np.array([0.0, 0.0, -1.0]),
        )
        result[f"{side}_ankle"] = shank_rotation @ anthropometry.compound_offsets_m[f"{side}_ankle"]
    torso_rotation = _align_vector(anthropometry.span_vectors_m["torso"], np.array([0.0, 0.0, 1.0]))
    result["waist"] = torso_rotation @ anthropometry.compound_offsets_m["waist"]
    return result


def _canonical_upper_limb_offsets(anthropometry: G1Anthropometry) -> dict[str, np.ndarray]:
    """Return endpoint-equivalent shoulder and wrist offsets in Xsens frames."""

    torso_rotation = _align_vector(anthropometry.span_vectors_m["torso"], np.array([0.0, 0.0, 1.0]))
    result: dict[str, np.ndarray] = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm_direction = np.array([0.0, sign, 0.0])
        forearm_rotation = _align_vector(
            anthropometry.span_vectors_m[f"{side}_forearm"],
            arm_direction,
        )
        result[f"{side}_shoulder"] = torso_rotation @ anthropometry.compound_offsets_m[f"{side}_shoulder"]
        result[f"{side}_wrist"] = forearm_rotation @ anthropometry.compound_offsets_m[f"{side}_wrist"]
    return result


def _fit_collapsed_shoulder_morphology(
    anthropometry: G1Anthropometry,
    side: str,
    sign: float,
) -> _ShoulderMorphologyFit:
    """Fit one virtual shoulder joint equally to canonical T- and N-poses.

    With parent/child segment rotations ``R_s`` and ``R_u``, the reconstructed
    shoulder-to-upper-arm offset is ``R_s p - R_u c``.  In the shoulder frame,
    the two equally weighted calibration objectives are therefore::

        p - c       = d_T
        p - R_N c   = d_N

    ``d_T`` is the full G1 shoulder edge-path length along the horizontal arm
    axis. ``d_N`` is the neutral G1 shoulder-chain endpoint, and ``R_N`` rotates
    the Xsens upper arm from horizontal to hanging vertically.  The least-
    squares solution is fixed for the morphology and requires no per-frame IK.
    """

    shoulder_path_length = sum(
        float(np.linalg.norm(edge))
        for edge in anthropometry.compound_offset_edges_m[f"{side}_shoulder"]
    )
    tpose_target = np.array([0.0, sign * shoulder_path_length, 0.0])
    npose_target = _canonical_upper_limb_offsets(anthropometry)[f"{side}_shoulder"].copy()
    # The extracted model is symmetric to sub-micron precision. Enforce exact
    # sagittal symmetry instead of retaining numerical mesh-transform noise.
    npose_target[0] = 0.0

    angle = -sign * 0.5 * np.pi
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    npose_child_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ]
    )
    identity = np.eye(3)
    design = np.vstack(
        [
            np.hstack([identity, -identity]),
            np.hstack([identity, -npose_child_rotation]),
        ]
    )
    targets = np.concatenate([tpose_target, npose_target])
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
    parent_anchor = solution[:3]
    child_anchor = solution[3:]
    tpose_error = float(np.linalg.norm(parent_anchor - child_anchor - tpose_target))
    npose_error = float(
        np.linalg.norm(parent_anchor - npose_child_rotation @ child_anchor - npose_target)
    )
    return _ShoulderMorphologyFit(
        parent_anchor_m=parent_anchor,
        child_anchor_m=child_anchor,
        tpose_target_offset_m=tpose_target,
        npose_target_offset_m=npose_target,
        npose_child_rotation=npose_child_rotation,
        tpose_error_m=tpose_error,
        npose_error_m=npose_error,
    )


def _collapsed_adapter_offsets(anthropometry: G1Anthropometry, preserve: bool) -> dict[str, np.ndarray]:
    """Keep cluster extent without retaining its separated-axis geometry.

    The Xsens shoulder joint uses fitted parent/child anchors so its endpoint
    agrees with both canonical T- and N-pose G1 configurations.  Its reference
    T-pose displacement is the G1 shoulder chain's full edge-path length along
    the canonical arm axis.  The joint frames introduce the N-pose dependence.
    Wrist clusters retain their endpoint-equivalent span on the forearm side of
    the virtual joint.  Co-locating that joint with the hand origin matches the
    Xsens topology and prevents the collapsed span from orbiting with the hand
    orientation.  The pelvis-to-hip adapter retains its scalar extent along the
    idealized leg axis.  Independently measured rigid spans remain available
    separately from these fixed adapter spans.
    """

    result = {name: np.zeros(3, dtype=float) for name in anthropometry.compound_offsets_m}
    if preserve:
        return result
    endpoint_offsets = _canonical_upper_limb_offsets(anthropometry)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        shoulder_fit = _fit_collapsed_shoulder_morphology(anthropometry, side, sign)
        result[f"{side}_shoulder"] = shoulder_fit.tpose_target_offset_m
        result[f"{side}_wrist"] = endpoint_offsets[f"{side}_wrist"]
        result[f"{side}_hip"] = np.array([0.0, 0.0, -np.linalg.norm(anthropometry.compound_offsets_m[f"{side}_hip"])])
    return result


def _avatar_part_attachment(
    part: AvatarMeshPart,
    *,
    scale: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> MeshAttachment:
    vertices = np.asarray(part.mesh.vertices, dtype=float).copy()
    if scale is not None:
        scale_array = np.asarray(scale, dtype=float)
        if scale_array.shape == (3,):
            vertices *= scale_array
        elif scale_array.shape == (3, 3):
            vertices = vertices @ scale_array.T
        else:
            raise ValueError(f"Mesh scale must have shape (3,) or (3, 3), got {scale_array.shape}")
    if translation is not None:
        vertices += np.asarray(translation, dtype=float)
    return MeshAttachment(
        name=part.name,
        vertices_m=vertices,
        faces=np.asarray(part.mesh.faces, dtype=np.int64),
        color_rgb=part.color,
        category=part.category,
    )


def _build_reference_layout(
    anthropometry: G1Anthropometry,
    config: G1XsensReductionConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    lengths = anthropometry.lengths_m
    offsets = _canonical_offsets(anthropometry, config.preserve_joint_offsets)
    adapters = _collapsed_adapter_offsets(anthropometry, config.preserve_joint_offsets)
    positions: dict[str, np.ndarray] = {"Pelvis": np.zeros(3, dtype=float)}
    joint_centers: dict[str, np.ndarray] = {}

    neck_target = np.array([0.0, 0.0, lengths["torso"]])
    if config.preserve_joint_offsets:
        waist = offsets["waist"]
        if np.linalg.norm(waist) >= lengths["torso"] * 0.8:
            waist *= lengths["torso"] * 0.8 / np.linalg.norm(waist)
        positions["L5"] = 0.5 * waist
        positions["L3"] = waist
        remainder = neck_target - waist
        positions["T12"] = waist + remainder / 3.0
        positions["T8"] = waist + 2.0 * remainder / 3.0
        positions["Neck"] = neck_target
    else:
        for name, fraction in zip(("L5", "L3", "T12", "T8", "Neck"), SPINE_FRACTIONS, strict=True):
            positions[name] = np.array([0.0, 0.0, fraction * lengths["torso"]])
    positions["Head"] = positions["Neck"] + np.array([0.0, 0.0, lengths["neck"]])

    for side, title, sign in (("left", "Left", 1.0), ("right", "Right", -1.0)):
        arm_direction = np.array([0.0, sign, 0.0])
        shoulder_root = positions["Neck"] + np.array([0.0, sign * anthropometry.widths_m["shoulder"] * 0.5, 0.0])
        positions[f"{title}Shoulder"] = shoulder_root
        positions[f"{title}UpperArm"] = shoulder_root + offsets[f"{side}_shoulder"] + adapters[f"{side}_shoulder"]
        if not config.preserve_joint_offsets:
            shoulder_fit = _fit_collapsed_shoulder_morphology(anthropometry, side, sign)
            joint_centers[f"{title}Shoulder"] = shoulder_root + shoulder_fit.parent_anchor_m
        positions[f"{title}ForeArm"] = positions[f"{title}UpperArm"] + arm_direction * lengths["upper_arm"]
        wrist_roll_center = positions[f"{title}ForeArm"] + arm_direction * lengths["forearm"]
        positions[f"{title}Hand"] = wrist_roll_center + offsets[f"{side}_wrist"] + adapters[f"{side}_wrist"]

        hip_root = np.asarray(anthropometry.root_anchors_m[f"{side}_hip"], dtype=float)
        hip_center = hip_root + offsets[f"{side}_hip"] + adapters[f"{side}_hip"]
        positions[f"{title}UpperLeg"] = hip_center
        positions[f"{title}LowerLeg"] = positions[f"{title}UpperLeg"] + np.array([0.0, 0.0, -lengths["thigh"]])
        ankle_center = positions[f"{title}LowerLeg"] + np.array([0.0, 0.0, -lengths["shank"]])
        positions[f"{title}Foot"] = ankle_center + offsets[f"{side}_ankle"]
        positions[f"{title}Toe"] = positions[f"{title}Foot"] + np.array([lengths["foot"], 0.0, 0.0])

        # Hip preserved mode keeps its joint at the proximal compound axis.
        # The virtual wrist is always distal to the collapsed G1 wrist span and
        # co-located with the Xsens hand origin.  This keeps the span attached to
        # the forearm instead of rotating it with the hand.
        joint_centers[f"{title}Hip"] = hip_root if config.preserve_joint_offsets else hip_center
        joint_centers[f"{title}Wrist"] = positions[f"{title}Hand"]
        joint_centers[f"{title}Ankle"] = ankle_center

    minimum_z = min(positions["LeftFoot"][2], positions["RightFoot"][2])
    shift = np.array([0.0, 0.0, -minimum_z])
    positions = {name: value + shift for name, value in positions.items()}
    joint_centers = {name: value + shift for name, value in joint_centers.items()}

    for spec in XSENS_JOINT_SPECS:
        if spec.source_joint.endswith("SwordOrigin"):
            continue
        joint_centers.setdefault(spec.source_joint, positions[spec.child_segment])
    return positions, joint_centers, offsets


def _aabb_landmarks(center: np.ndarray, extents: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    center = np.asarray(center, dtype=float)
    half = 0.5 * np.asarray(extents, dtype=float)
    return {
        f"{prefix}_{x_index}{y_index}{z_index}": center + half * np.array([x_sign, y_sign, z_sign], dtype=float)
        for x_index, x_sign in enumerate((-1.0, 1.0))
        for y_index, y_sign in enumerate((-1.0, 1.0))
        for z_index, z_sign in enumerate((-1.0, 1.0))
    }


def _g1_xsens_avatar_proportions_from_layout(
    anthropometry: G1Anthropometry,
    positions: Mapping[str, np.ndarray],
    joint_centers: Mapping[str, np.ndarray],
) -> XsensAvatarProportions:
    """Adapt G1 measurements to the exact input contract of the shared Xsens visual factory."""

    body_names = ("Pelvis",) + tuple(
        spec.child_segment for spec in XSENS_JOINT_SPECS if not spec.source_joint.endswith("SwordOrigin")
    )
    landmarks: dict[str, dict[str, np.ndarray]] = {name: {} for name in body_names}

    pelvis_origin = positions["Pelvis"]
    landmarks["Pelvis"] = {
        "jLeftHip": positions["LeftUpperLeg"] - pelvis_origin,
        "jRightHip": positions["RightUpperLeg"] - pelvis_origin,
        "jL5S1": positions["L5"] - pelvis_origin,
        **_aabb_landmarks(
            anthropometry.region_centers_m["pelvis"],
            anthropometry.region_extents_m["pelvis"],
            "pPelvisEnvelope",
        ),
    }

    distal_specs = {
        "L5": ("jL4L3", "L3"),
        "L3": ("jL1T12", "T12"),
        "T12": ("jT9T8", "T8"),
        "T8": ("jT1C7", "Neck"),
        "Neck": ("jC1Head", "Head"),
        "RightShoulder": ("jRightShoulder", "RightUpperArm"),
        "RightUpperArm": ("jRightElbow", "RightForeArm"),
        "LeftShoulder": ("jLeftShoulder", "LeftUpperArm"),
        "LeftUpperArm": ("jLeftElbow", "LeftForeArm"),
        "RightUpperLeg": ("jRightKnee", "RightLowerLeg"),
        "LeftUpperLeg": ("jLeftKnee", "LeftLowerLeg"),
    }
    for segment, (landmark_name, child) in distal_specs.items():
        if segment.endswith("Shoulder"):
            landmarks[segment][landmark_name] = joint_centers[segment] - positions[segment]
        else:
            landmarks[segment][landmark_name] = positions[child] - positions[segment]
    for side in ("Right", "Left"):
        landmarks[f"{side}ForeArm"][f"j{side}Wrist"] = joint_centers[f"{side}Wrist"] - positions[f"{side}ForeArm"]
        landmarks[f"{side}LowerLeg"][f"j{side}Ankle"] = joint_centers[f"{side}Ankle"] - positions[f"{side}LowerLeg"]

    landmarks["Head"] = _aabb_landmarks(
        anthropometry.region_centers_m["head"],
        anthropometry.region_extents_m["head"],
        "pHeadEnvelope",
    )

    hand_radii = np.asarray(anthropometry.segment_radii_m["hand"], dtype=float)
    thigh_radii = np.asarray(anthropometry.segment_radii_m["thigh"], dtype=float)
    shank_radii = np.asarray(anthropometry.segment_radii_m["shank"], dtype=float)
    foot_radii = np.asarray(anthropometry.segment_radii_m["foot"], dtype=float)
    for side, sign in (("Right", -1.0), ("Left", 1.0)):
        hand_length = float(anthropometry.lengths_m["hand"])
        landmarks[f"{side}Hand"] = {
            f"p{side}TopOfHand": np.array([0.0, sign * hand_length, 0.0]),
            f"p{side}Pinky": np.array([-hand_radii[1] * 0.55, sign * hand_length * 0.64, 0.0]),
            f"p{side}HandPalm": np.array([hand_radii[1] * 0.55, sign * hand_length * 0.58, hand_radii[0] * 0.35]),
        }

        upper_leg = f"{side}UpperLeg"
        thigh_end = landmarks[upper_leg][f"j{side}Knee"]
        landmarks[upper_leg].update(
            {
                f"p{side}GreaterTrochanter": np.array([0.0, sign * thigh_radii[1] / 0.78, 0.0]),
                f"p{side}KneeLatEpicondyle": thigh_end + np.array([0.0, sign * thigh_radii[1] * 0.82, 0.0]),
                f"p{side}KneeMedEpicondyle": thigh_end - np.array([0.0, sign * thigh_radii[1] * 0.82, 0.0]),
                f"p{side}Patella": thigh_end + np.array([thigh_radii[0] / 1.25, 0.0, 0.0]),
            }
        )

        lower_leg = f"{side}LowerLeg"
        shank_end = landmarks[lower_leg][f"j{side}Ankle"]
        landmarks[lower_leg].update(
            {
                f"p{side}LatMalleolus": shank_end + np.array([0.0, sign * shank_radii[1] * 0.72, 0.0]),
                f"p{side}MedMalleolus": shank_end - np.array([0.0, sign * shank_radii[1] * 0.72, 0.0]),
                f"p{side}TibialTub": np.array([shank_radii[0] / 0.72, 0.0, shank_end[2] * 0.18]),
                f"p{side}Fibula": np.array([0.0, sign * shank_radii[1] / 1.02, shank_end[2] * 0.22]),
            }
        )

        foot = f"{side}Foot"
        foot_length = float(anthropometry.lengths_m["foot"])
        foot_width = foot_radii[0]
        foot_height = foot_radii[1]
        landmarks[foot] = {
            f"j{side}BallFoot": positions[f"{side}Toe"] - positions[foot],
            f"p{side}HeelCenter": np.array([-foot_length * 0.24, 0.0, -foot_height * 0.62]),
            f"p{side}FirstMetatarsal": np.array([foot_length * 0.88, sign * foot_width, -foot_height * 0.62]),
            f"p{side}FifthMetatarsal": np.array([foot_length * 0.88, -sign * foot_width, -foot_height * 0.62]),
            f"p{side}TopOfFoot": np.array([foot_length * 0.42, 0.0, foot_height * 0.72]),
        }
        landmarks[f"{side}Toe"] = {f"p{side}Toe": np.array([anthropometry.lengths_m["toe"], 0.0, 0.0])}

    tpose_positions = np.vstack([positions[name] for name in body_names])
    return XsensAvatarProportions(
        segment_names=body_names,
        tpose_positions_m=tpose_positions,
        tpose_quaternions_wijk=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(body_names), 1)),
        landmarks_m=landmarks,
    )


def g1_anthropometry_to_xsens_avatar_proportions(
    anthropometry: G1Anthropometry,
    config: G1XsensReductionConfig | None = None,
) -> XsensAvatarProportions:
    """Expose G1 dimensions through the shared calibrated-Xsens visual input schema."""

    config = config or G1XsensReductionConfig()
    positions, joint_centers, _ = _build_reference_layout(anthropometry, config)
    return _g1_xsens_avatar_proportions_from_layout(anthropometry, positions, joint_centers)


def _parts_bounds(parts: tuple[AvatarMeshPart, ...]) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack([np.asarray(part.mesh.vertices, dtype=float) for part in parts])
    return vertices.min(axis=0), vertices.max(axis=0)


def _axis_preserving_transverse_scale(
    parts: tuple[AvatarMeshPart, ...],
    axis: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Scale two transverse directions while preserving the segment axis."""

    direction = np.asarray(axis, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-10:
        raise ValueError("Cannot scale a zero-length segment")
    direction /= length
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(direction, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(direction, basis_u)

    vertices = np.vstack([np.asarray(part.mesh.vertices, dtype=float) for part in parts])
    extent_u = float(np.ptp(vertices @ basis_u))
    extent_v = float(np.ptp(vertices @ basis_v))
    if extent_u <= 1e-10 or extent_v <= 1e-10:
        raise ValueError("Cannot scale a segment with zero transverse extent")
    target_radii = np.asarray(radii, dtype=float)
    scale_u = 2.0 * float(target_radii[0]) / extent_u
    scale_v = 2.0 * float(target_radii[1]) / extent_v
    return np.outer(direction, direction) + scale_u * np.outer(basis_u, basis_u) + scale_v * np.outer(basis_v, basis_v)


def _scaled_part_attachments(
    parts: tuple[AvatarMeshPart, ...],
    scale: np.ndarray,
    translation: np.ndarray | None = None,
) -> tuple[MeshAttachment, ...]:
    translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    return tuple(_avatar_part_attachment(part, scale=scale, translation=translation) for part in parts)


def _shared_visual_attachments(
    anthropometry: G1Anthropometry,
    positions: Mapping[str, np.ndarray],
    joint_centers: Mapping[str, np.ndarray],
) -> dict[str, tuple[MeshAttachment, ...]]:
    proportions = _g1_xsens_avatar_proportions_from_layout(anthropometry, positions, joint_centers)
    shared_parts = build_xsens_avatar_meshes(proportions)
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (np.ones(3), np.zeros(3)) for name in proportions.segment_names
    }

    spine_names = ("L5", "L3", "T12", "T8")
    spine_vertices = np.vstack(
        [np.asarray(part.mesh.vertices, dtype=float) for name in spine_names for part in shared_parts[name]]
    )
    spine_min = spine_vertices.min(axis=0)
    spine_max = spine_vertices.max(axis=0)
    torso_extent = np.asarray(anthropometry.region_extents_m["torso"], dtype=float)
    torso_center = np.asarray(anthropometry.region_centers_m["torso"], dtype=float)
    spine_scale = np.array(
        [torso_extent[0] / (spine_max[0] - spine_min[0]), torso_extent[1] / (spine_max[1] - spine_min[1]), 1.0]
    )
    spine_translation = np.array(
        [
            torso_center[0] - 0.5 * (spine_min[0] + spine_max[0]) * spine_scale[0],
            torso_center[1] - 0.5 * (spine_min[1] + spine_max[1]) * spine_scale[1],
            0.0,
        ]
    )
    for name in spine_names:
        transforms[name] = (spine_scale, spine_translation)

    pelvis_min, pelvis_max = _parts_bounds(shared_parts["Pelvis"])
    extracted_center = np.asarray(anthropometry.region_centers_m["pelvis"], dtype=float)
    extracted_half = 0.5 * np.asarray(anthropometry.region_extents_m["pelvis"], dtype=float)
    target_min = extracted_center - extracted_half
    target_max = extracted_center + extracted_half
    target_max = np.maximum(target_max, positions["L5"] - positions["Pelvis"])
    pelvis_scale = (target_max - target_min) / (pelvis_max - pelvis_min)
    pelvis_translation = target_min - pelvis_min * pelvis_scale
    transforms["Pelvis"] = (pelvis_scale, pelvis_translation)

    head_min, head_max = _parts_bounds(shared_parts["Head"])
    head_extent = np.asarray(anthropometry.region_extents_m["head"], dtype=float)
    head_center = np.asarray(anthropometry.region_centers_m["head"], dtype=float)
    head_scale = head_extent / (head_max - head_min)
    transforms["Head"] = (head_scale, head_center - 0.5 * (head_min + head_max) * head_scale)

    transverse_targets = {
        "Shoulder": anthropometry.segment_radii_m["upper_arm"],
        "UpperArm": anthropometry.segment_radii_m["upper_arm"],
        "ForeArm": anthropometry.segment_radii_m["forearm"],
    }
    # Shoulder and compound-wrist spans make some arm axes diagonal. Scale in each
    # segment's local transverse basis so its authored joint endpoint remains
    # unchanged; componentwise world-X/Z scaling would shorten those segments.
    for side in ("Left", "Right"):
        for kind, radii_values in transverse_targets.items():
            name = f"{side}{kind}"
            radii = np.asarray(radii_values, dtype=float)
            distal_landmark = {
                "Shoulder": f"j{side}Shoulder",
                "UpperArm": f"j{side}Elbow",
                "ForeArm": f"j{side}Wrist",
            }[kind]
            axis = proportions.landmarks_m[name][distal_landmark]
            scale = _axis_preserving_transverse_scale(shared_parts[name], axis, radii)
            transforms[name] = (scale, np.zeros(3))

        for kind, metric in (("UpperLeg", "thigh"), ("LowerLeg", "shank")):
            name = f"{side}{kind}"
            part_min, part_max = _parts_bounds(shared_parts[name])
            radii = np.asarray(anthropometry.segment_radii_m[metric], dtype=float)
            scale = np.ones(3)
            scale[0] = 2.0 * radii[0] / (part_max[0] - part_min[0])
            scale[1] = 2.0 * radii[1] / (part_max[1] - part_min[1])
            translation = np.zeros(3)
            if kind == "UpperLeg":
                pelvis_bottom = extracted_center[2] - extracted_half[2]
                side_key = side.lower()
                collapsed_hip_z = float(anthropometry.root_anchors_m[f"{side_key}_hip"][2]) - float(
                    np.linalg.norm(anthropometry.compound_offsets_m[f"{side_key}_hip"])
                )
                target_max_z = pelvis_bottom - collapsed_hip_z
                scale[2] = (target_max_z - part_min[2]) / (part_max[2] - part_min[2])
                translation[2] = part_min[2] - part_min[2] * scale[2]
            transforms[name] = (scale, translation)

        hand_name = f"{side}Hand"
        hand_min, hand_max = _parts_bounds(shared_parts[hand_name])
        hand_radii = np.asarray(anthropometry.segment_radii_m["hand"], dtype=float)
        hand_target = np.array([2.0 * hand_radii[1], anthropometry.lengths_m["hand"], 2.0 * hand_radii[0]])
        hand_scale = hand_target / (hand_max - hand_min)
        hand_scaled_min = hand_min * hand_scale
        hand_scaled_max = hand_max * hand_scale
        hand_translation = np.array(
            [
                0.0,
                -hand_scaled_min[1] if side == "Left" else -hand_scaled_max[1],
                -0.5 * (hand_scaled_min[2] + hand_scaled_max[2]),
            ]
        )
        transforms[hand_name] = (hand_scale, hand_translation)

        for kind, metric in (("Foot", "foot"), ("Toe", "toe")):
            name = f"{side}{kind}"
            part_min, part_max = _parts_bounds(shared_parts[name])
            radii = np.asarray(anthropometry.segment_radii_m[metric], dtype=float)
            scale = np.ones(3)
            scale[1] = 2.0 * radii[0] / (part_max[1] - part_min[1])
            scale[2] = 2.0 * radii[1] / (part_max[2] - part_min[2])
            transforms[name] = (scale, np.zeros(3))

    neck_min, neck_max = _parts_bounds(shared_parts["Neck"])
    head_extent = np.asarray(anthropometry.region_extents_m["head"], dtype=float)
    neck_target = head_extent[:2] * 0.46
    neck_scale = np.ones(3)
    neck_scale[:2] = neck_target / (neck_max[:2] - neck_min[:2])
    transforms["Neck"] = (neck_scale, np.zeros(3))

    attachments = {
        name: _scaled_part_attachments(parts, *transforms[name])
        for name, parts in shared_parts.items()
    }
    adapter_radius = 0.72 * float(np.min(anthropometry.segment_radii_m["upper_arm"]))
    for side in ("Left", "Right"):
        upper_arm_name = f"{side}UpperArm"
        child_anchor = joint_centers[f"{side}Shoulder"] - positions[upper_arm_name]
        if np.linalg.norm(child_anchor) <= 1e-12:
            continue
        adapter_part = AvatarMeshPart(
            f"{side.lower()}_shoulder_child_adapter",
            cylinder_between(child_anchor, np.zeros(3), adapter_radius, sections=14),
            LIGHT_GRAY,
        )
        attachments[upper_arm_name] += (_avatar_part_attachment(adapter_part),)
    return attachments


def build_g1_proportioned_xsens_tree(
    anthropometry: G1Anthropometry,
    config: G1XsensReductionConfig | None = None,
) -> KinematicTree:
    """Build a canonical Xsens tree using invariant G1 measurements."""

    config = config or G1XsensReductionConfig()
    positions, joint_centers, applied_offsets = _build_reference_layout(anthropometry, config)
    joint_specs = tuple(
        spec
        for spec in XSENS_JOINT_SPECS
        if config.include_tennis_racket or not spec.source_joint.endswith("SwordOrigin")
    )
    body_names = ("Pelvis",) + tuple(canonical_xsens_segment_name(spec.child_segment) for spec in joint_specs)
    if config.include_tennis_racket:
        positions[TENNIS_RACKET_BODY] = positions["RightHand"].copy()
        joint_centers["RightHandSwordOrigin"] = positions[TENNIS_RACKET_BODY].copy()
    visual_attachments = (
        _shared_visual_attachments(anthropometry, positions, joint_centers) if config.include_visuals else {}
    )
    if config.include_visuals and config.include_tennis_racket:
        visual_attachments[TENNIS_RACKET_BODY] = tuple(
            _avatar_part_attachment(part) for part in build_tennis_racket_meshes()
        )
    bodies = []
    for index, name in enumerate(body_names):
        source_name = XSENS_RACKET_SOURCE_SEGMENT if name == TENNIS_RACKET_BODY else name
        bodies.append(
            RigidBodyDefinition(
                name=name,
                reference_pose=Transform(positions[name], np.array([1.0, 0.0, 0.0, 0.0])),
                meshes=visual_attachments.get(name, ()),
                metadata={
                    "xsens:sourceSegmentName": source_name,
                    "xsens:sourceSegmentIndex": index,
                    "model:proportionedFrom": "g1_29dof",
                },
            )
        )

    joints = []
    for index, spec in enumerate(joint_specs):
        center = joint_centers[spec.source_joint]
        parent_body = canonical_xsens_segment_name(spec.parent_segment)
        child_body = canonical_xsens_segment_name(spec.child_segment)
        joints.append(
            SphericalJointDefinition(
                name=canonical_xsens_joint_name(spec.source_joint),
                parent_body=parent_body,
                child_body=child_body,
                parent_frame=Transform(center - positions[parent_body], np.array([1.0, 0.0, 0.0, 0.0])),
                child_frame=Transform(center - positions[child_body], np.array([1.0, 0.0, 0.0, 0.0])),
                metadata={
                    "xsens:sourceJointName": spec.source_joint,
                    "xsens:sourceJointIndex": index,
                    "xsens:childSegmentIndex": body_names.index(child_body),
                    "xsens:eulerStreams": XSENS_JOINT_STREAM_NAMES,
                    "xsens:rotationComponents": ("x", "y", "z"),
                },
            )
        )

    model = KinematicTree(
        name="G1ProportionedXsensAvatar",
        root_body="Pelvis",
        bodies=tuple(bodies),
        joints=tuple(joints),
        metadata={
            "model:proportionSource": "g1_29dof",
            "model:proportionedFrom": "g1_29dof",
            "model:sourceSha256": anthropometry.model_sha256,
            "model:generator": "holosoma_retargeting.xsens.g1_kinematic_reduction",
            "model:generatorVersion": G1_XSENS_REDUCTION_VERSION,
            "model:preserveJointOffsets": config.preserve_joint_offsets,
            "xsens:referencePose": "Tpose",
            "xsens:coordinateConvention": "+X forward, +Y left, +Z up",
            "xsens:jointOrder": tuple(canonical_xsens_joint_name(spec.source_joint) for spec in joint_specs),
            "model:appliedOffsetNames": tuple(applied_offsets),
        },
    )
    validate_kinematic_tree(model).raise_if_invalid()
    return model


def _generated_lengths(model: KinematicTree, anthropometry: G1Anthropometry) -> dict[str, float]:
    body = model.body_map()
    joints = compute_reference_joint_positions(model)
    wrist_offsets = _canonical_upper_limb_offsets(anthropometry)

    def body_distance(first: str, second: str) -> float:
        return float(
            np.linalg.norm(body[first].reference_pose.translation_m - body[second].reference_pose.translation_m)
        )

    def terminal_mesh_length(body_name: str, axis: int) -> float:
        if not body[body_name].meshes:
            fallback = "hand" if body_name.endswith("Hand") else "toe" if body_name.endswith("Toe") else "head"
            return float(anthropometry.lengths_m[fallback])
        vertices = np.vstack([mesh.vertices_m for mesh in body[body_name].meshes])
        return float(np.ptp(vertices[:, axis]))

    return {
        "upper_arm": 0.5
        * (body_distance("LeftForeArm", "LeftUpperArm") + body_distance("RightForeArm", "RightUpperArm")),
        "forearm": 0.5
        * sum(
            np.linalg.norm(
                body[f"{title}Hand"].reference_pose.translation_m
                - body[f"{title}ForeArm"].reference_pose.translation_m
                - wrist_offsets[f"{side}_wrist"]
            )
            for side, title in (("left", "Left"), ("right", "Right"))
        ),
        "thigh": 0.5
        * (body_distance("LeftLowerLeg", "LeftUpperLeg") + body_distance("RightLowerLeg", "RightUpperLeg")),
        "shank": 0.5
        * (
            np.linalg.norm(joints["LeftAnkle"] - body["LeftLowerLeg"].reference_pose.translation_m)
            + np.linalg.norm(joints["RightAnkle"] - body["RightLowerLeg"].reference_pose.translation_m)
        ),
        "hand": 0.5 * (terminal_mesh_length("LeftHand", 1) + terminal_mesh_length("RightHand", 1)),
        "foot": 0.5 * (body_distance("LeftFoot", "LeftToe") + body_distance("RightFoot", "RightToe")),
        "toe": 0.5 * (terminal_mesh_length("LeftToe", 0) + terminal_mesh_length("RightToe", 0)),
        "torso": body_distance("Pelvis", "Neck"),
        "neck": body_distance("Neck", "Head"),
        "head": terminal_mesh_length("Head", 2),
    }


def export_g1_proportioned_xsens_usd(
    output_path: str | Path,
    *,
    robot_model_path: str | Path | None = None,
    report_path: str | Path | None = None,
    config: G1XsensReductionConfig | None = None,
) -> G1XsensProportionReport:
    """Generate a source-independent G1-proportioned Xsens USD and JSON report."""

    config = config or G1XsensReductionConfig()
    output_path = Path(output_path)
    report_path = Path(report_path) if report_path is not None else output_path.with_suffix(".json")
    anthropometry = extract_g1_anthropometry(robot_model_path)
    model = build_g1_proportioned_xsens_tree(anthropometry, config)
    stage = create_usd_stage(output_path)
    write_kinematic_tree_to_stage(stage, model)
    usd_validation = validate_usd_kinematic_tree(stage)
    usd_validation.raise_if_invalid()
    stage.GetRootLayer().Save()

    generated = _generated_lengths(model, anthropometry)
    compared = ("upper_arm", "forearm", "thigh", "shank", "hand", "foot", "toe", "torso", "neck", "head")
    max_length_error = max(abs(float(generated[name]) - float(anthropometry.lengths_m[name])) for name in compared)
    applied = _canonical_offsets(anthropometry, config.preserve_joint_offsets)
    collapsed_adapters = _collapsed_adapter_offsets(anthropometry, config.preserve_joint_offsets)
    report = G1XsensProportionReport(
        source_path=anthropometry.model_path,
        output_path=output_path,
        report_path=report_path,
        source_sha256=anthropometry.model_sha256,
        preserve_joint_offsets=config.preserve_joint_offsets,
        body_count=len(model.bodies),
        joint_count=len(model.joints),
        target_lengths_m=anthropometry.lengths_m,
        generated_lengths_m=generated,
        widths_m=anthropometry.widths_m,
        root_anchors_m={name: tuple(map(float, value)) for name, value in anthropometry.root_anchors_m.items()},
        raw_offsets_m={name: tuple(map(float, value)) for name, value in anthropometry.compound_offsets_m.items()},
        raw_offset_edges_m={
            name: tuple(tuple(map(float, edge)) for edge in edges)
            for name, edges in anthropometry.compound_offset_edges_m.items()
        },
        collapsed_adapter_offsets_m={name: tuple(map(float, value)) for name, value in collapsed_adapters.items()},
        applied_offsets_m={name: tuple(map(float, value)) for name, value in applied.items()},
        max_length_error_m=max_length_error,
        max_joint_residual_m=usd_validation.max_joint_residual_m,
        warnings=usd_validation.warnings,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "G1Anthropometry",
    "G1XsensProportionReport",
    "G1XsensReductionConfig",
    "build_g1_proportioned_xsens_tree",
    "export_g1_proportioned_xsens_usd",
    "extract_g1_anthropometry",
    "g1_anthropometry_to_xsens_avatar_proportions",
]
