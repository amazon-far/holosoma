"""Format- and simulator-independent kinematic tree data structures."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, TypeAlias

import numpy as np

MetadataScalar: TypeAlias = str | int | float | bool
MetadataValue: TypeAlias = MetadataScalar | tuple[MetadataScalar, ...]


@dataclass(frozen=True)
class Transform:
    """A translation in metres and a scalar-first quaternion."""

    translation_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    rotation_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def identity(cls) -> Transform:
        return cls()


@dataclass(frozen=True)
class PointSetAttachment:
    """Named points expressed in their owning body's local frame."""

    name: str
    points_m: np.ndarray
    point_names: tuple[str, ...]
    width_m: float = 0.008
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class MeshAttachment:
    """A triangular render mesh expressed in an owning body's local frame."""

    name: str
    vertices_m: np.ndarray
    faces: np.ndarray
    color_rgb: tuple[int, int, int] = (180, 180, 180)
    category: str = "render"
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class RigidBodyDefinition:
    """One rigid segment in a kinematic tree."""

    name: str
    reference_pose: Transform
    point_sets: tuple[PointSetAttachment, ...] = ()
    meshes: tuple[MeshAttachment, ...] = ()
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class SphericalJointDefinition:
    """An unrestricted spherical joint between two rigid bodies."""

    name: str
    parent_body: str
    child_body: str
    parent_frame: Transform
    child_frame: Transform
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class KinematicTree:
    """A floating-base rigid-body tree with optional local geometry."""

    name: str
    root_body: str
    bodies: tuple[RigidBodyDefinition, ...]
    joints: tuple[SphericalJointDefinition, ...]
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def body_map(self) -> dict[str, RigidBodyDefinition]:
        return {body.name: body for body in self.bodies}

    def joint_map(self) -> dict[str, SphericalJointDefinition]:
        return {joint.name: joint for joint in self.joints}


@dataclass(frozen=True)
class ValidationReport:
    """Validation result shared by in-memory and serialized models."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    max_joint_residual_m: float = 0.0

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise ValueError("Invalid kinematic tree:\n- " + "\n- ".join(self.errors))


def _normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be positive")
    return quaternion / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = _normalized_quaternion(left)
    rw, rx, ry, rz = _normalized_quaternion(right)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    q = _normalized_quaternion(quaternion)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _normalized_quaternion(_quat_multiply(left, right))


def rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = _normalized_quaternion(quaternion)
    vector = np.asarray(vector, dtype=float)
    pure = np.array([0.0, vector[0], vector[1], vector[2]])
    rotated = _quat_multiply_raw(_quat_multiply_raw(q, pure), quaternion_conjugate(q))
    return rotated[1:]


def _quat_multiply_raw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=float)
    rw, rx, ry, rz = np.asarray(right, dtype=float)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def transform_point(transform: Transform, point_m: np.ndarray) -> np.ndarray:
    return np.asarray(transform.translation_m, dtype=float) + rotate_vector(transform.rotation_wxyz, point_m)


def compute_joint_positions(
    model: KinematicTree,
    body_poses: Mapping[str, Transform],
) -> dict[str, np.ndarray]:
    """Compute joint centers from child frames for arbitrary world body poses."""

    positions: dict[str, np.ndarray] = {}
    for joint in model.joints:
        if joint.child_body not in body_poses:
            raise KeyError(f"Missing pose for child body '{joint.child_body}'")
        positions[joint.name] = transform_point(body_poses[joint.child_body], joint.child_frame.translation_m)
    return positions


def compute_reference_joint_positions(model: KinematicTree) -> dict[str, np.ndarray]:
    return compute_joint_positions(model, {body.name: body.reference_pose for body in model.bodies})


def validate_kinematic_tree(model: KinematicTree, *, anchor_tolerance_m: float = 5e-6) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    body_names = [body.name for body in model.bodies]
    joint_names = [joint.name for joint in model.joints]
    body_set = set(body_names)

    if len(body_names) != len(body_set):
        errors.append("Body names must be unique")
    if len(joint_names) != len(set(joint_names)):
        errors.append("Joint names must be unique")
    if model.root_body not in body_set:
        errors.append(f"Root body '{model.root_body}' does not exist")

    children: dict[str, str] = {}
    adjacency: dict[str, list[str]] = {name: [] for name in body_names}
    max_residual = 0.0
    body_map = model.body_map()
    for joint in model.joints:
        if joint.parent_body not in body_set:
            errors.append(f"Joint '{joint.name}' has unknown parent '{joint.parent_body}'")
            continue
        if joint.child_body not in body_set:
            errors.append(f"Joint '{joint.name}' has unknown child '{joint.child_body}'")
            continue
        if joint.child_body in children:
            errors.append(
                f"Body '{joint.child_body}' has multiple parent joints: "
                f"'{children[joint.child_body]}' and '{joint.name}'"
            )
        children[joint.child_body] = joint.name
        adjacency[joint.parent_body].append(joint.child_body)
        parent_position = transform_point(
            body_map[joint.parent_body].reference_pose,
            joint.parent_frame.translation_m,
        )
        child_position = transform_point(
            body_map[joint.child_body].reference_pose,
            joint.child_frame.translation_m,
        )
        residual = float(np.linalg.norm(parent_position - child_position))
        max_residual = max(max_residual, residual)
        if residual > anchor_tolerance_m:
            errors.append(
                f"Joint '{joint.name}' reference anchors differ by {residual:.9g} m "
                f"(tolerance {anchor_tolerance_m:.9g} m)"
            )

    if model.root_body in children:
        errors.append(f"Root body '{model.root_body}' must not have a parent joint")
    errors.extend(
        f"Non-root body '{body_name}' has no parent joint"
        for body_name in body_names
        if body_name != model.root_body and body_name not in children
    )

    visited: set[str] = set()
    active: set[str] = set()

    def visit(body_name: str) -> None:
        if body_name in active:
            errors.append(f"Kinematic tree contains a cycle through '{body_name}'")
            return
        if body_name in visited:
            return
        active.add(body_name)
        for child_name in adjacency.get(body_name, []):
            visit(child_name)
        active.remove(body_name)
        visited.add(body_name)

    if model.root_body in body_set:
        visit(model.root_body)
        unreachable = sorted(body_set - visited)
        if unreachable:
            errors.append(f"Bodies are unreachable from root '{model.root_body}': {unreachable}")

    for body in model.bodies:
        translation = np.asarray(body.reference_pose.translation_m)
        quaternion = np.asarray(body.reference_pose.rotation_wxyz)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            errors.append(f"Body '{body.name}' has an invalid reference translation")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            errors.append(f"Body '{body.name}' has an invalid reference quaternion")
        elif abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-6:
            errors.append(f"Body '{body.name}' reference quaternion is not normalized")
        errors.extend(
            f"Point set '{body.name}/{point_set.name}' names and positions differ"
            for point_set in body.point_sets
            if point_set.points_m.shape != (len(point_set.point_names), 3)
        )
        for mesh in body.meshes:
            if mesh.vertices_m.ndim != 2 or mesh.vertices_m.shape[1] != 3:
                errors.append(f"Mesh '{body.name}/{mesh.name}' has invalid vertices")
            if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
                errors.append(f"Mesh '{body.name}/{mesh.name}' must contain triangular faces")

    return ValidationReport(tuple(errors), tuple(warnings), max_residual)


def with_body_attachments(
    model: KinematicTree,
    *,
    point_sets: Mapping[str, tuple[PointSetAttachment, ...]] | None = None,
    meshes: Mapping[str, tuple[MeshAttachment, ...]] | None = None,
) -> KinematicTree:
    """Return a model with body-local attachments replaced for supplied bodies."""

    point_sets = point_sets or {}
    meshes = meshes or {}
    unknown = (set(point_sets) | set(meshes)) - {body.name for body in model.bodies}
    if unknown:
        raise KeyError(f"Attachments reference unknown bodies: {sorted(unknown)}")
    bodies = tuple(
        replace(
            body,
            point_sets=point_sets.get(body.name, body.point_sets),
            meshes=meshes.get(body.name, body.meshes),
        )
        for body in model.bodies
    )
    return replace(model, bodies=bodies)
