"""Backend-independent kinematic morphology adaptation and grounding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .model import KinematicTree, rotate_vector, validate_kinematic_tree


@dataclass(frozen=True)
class KinematicPose:
    """Named global rigid-body poses for one instant."""

    body_names: tuple[str, ...]
    positions_m: np.ndarray
    orientations_wxyz: np.ndarray


@dataclass(frozen=True)
class KinematicMotion:
    """Named global rigid-body poses over a timestamped motion."""

    body_names: tuple[str, ...]
    positions_m: np.ndarray
    orientations_wxyz: np.ndarray
    times_s: np.ndarray


@dataclass(frozen=True)
class GroundingSurface:
    """Explicit render meshes whose lowest vertices define a ground surface."""

    body_name: str
    mesh_names: tuple[str, ...]


@dataclass(frozen=True)
class _SurfaceMesh:
    pose_index: int
    vertices_m: np.ndarray


class SurfacePoseEvaluator:
    """Evaluate selected model surfaces in the coordinates of a named pose."""

    def __init__(
        self,
        model: KinematicTree,
        pose_body_names: tuple[str, ...],
        model_body_to_pose_body: Mapping[str, str],
        surfaces: Sequence[GroundingSurface],
    ) -> None:
        pose_indices = _unique_name_indices(pose_body_names, label="pose")
        body_map = model.body_map()
        meshes: list[_SurfaceMesh] = []
        if not surfaces:
            raise ValueError("Grounding requires at least one surface")
        for surface in surfaces:
            if surface.body_name not in body_map:
                raise KeyError(f"Grounding surface references unknown body '{surface.body_name}'")
            if not surface.mesh_names:
                raise ValueError(f"Grounding surface '{surface.body_name}' must select at least one mesh")
            pose_body_name = model_body_to_pose_body.get(surface.body_name)
            if pose_body_name is None:
                raise KeyError(f"No pose-body mapping for grounding body '{surface.body_name}'")
            if pose_body_name not in pose_indices:
                raise KeyError(f"Grounding body '{surface.body_name}' maps to unknown pose body '{pose_body_name}'")
            body_meshes = {mesh.name: mesh for mesh in body_map[surface.body_name].meshes}
            for mesh_name in surface.mesh_names:
                if mesh_name not in body_meshes:
                    raise KeyError(f"Grounding body '{surface.body_name}' has no mesh '{mesh_name}'")
                vertices = np.asarray(body_meshes[mesh_name].vertices_m, dtype=float)
                if vertices.ndim != 2 or vertices.shape[1:] != (3,) or vertices.shape[0] == 0:
                    raise ValueError(
                        f"Grounding mesh '{surface.body_name}/{mesh_name}' must have shape [N, 3]"
                    )
                if not np.isfinite(vertices).all():
                    raise ValueError(f"Grounding mesh '{surface.body_name}/{mesh_name}' contains non-finite vertices")
                meshes.append(_SurfaceMesh(pose_indices[pose_body_name], vertices.copy()))
        self._body_names = pose_body_names
        self._meshes = tuple(meshes)

    def minimum_height_m(self, pose: KinematicPose) -> float:
        _validate_pose(pose, expected_body_names=self._body_names)
        minimum_z = np.inf
        for surface_mesh in self._meshes:
            position = pose.positions_m[surface_mesh.pose_index]
            orientation = pose.orientations_wxyz[surface_mesh.pose_index]
            for vertex_m in surface_mesh.vertices_m:
                minimum_z = min(minimum_z, float(position[2] + rotate_vector(orientation, vertex_m)[2]))
        return float(minimum_z)

    def support_reference_m(self, pose: KinematicPose) -> np.ndarray:
        """Return a stable surface-center XY position and the lowest surface Z."""

        _validate_pose(pose, expected_body_names=self._body_names)
        world_vertices: list[np.ndarray] = []
        for surface_mesh in self._meshes:
            position = pose.positions_m[surface_mesh.pose_index]
            orientation = pose.orientations_wxyz[surface_mesh.pose_index]
            world_vertices.extend(
                position + rotate_vector(orientation, vertex_m) for vertex_m in surface_mesh.vertices_m
            )
        vertices = np.asarray(world_vertices, dtype=float)
        return np.array(
            [
                float(np.mean(vertices[:, 0])),
                float(np.mean(vertices[:, 1])),
                float(np.min(vertices[:, 2])),
            ]
        )


class LowestSurfaceGrounding:
    """Align selected target surfaces with selected source surfaces."""

    def __init__(
        self,
        source_model: KinematicTree,
        target_model: KinematicTree,
        pose_body_names: Sequence[str],
        *,
        source_body_to_pose_body: Mapping[str, str],
        target_body_to_pose_body: Mapping[str, str],
        source_surfaces: Sequence[GroundingSurface],
        target_surfaces: Sequence[GroundingSurface],
    ) -> None:
        validate_kinematic_tree(source_model).raise_if_invalid()
        validate_kinematic_tree(target_model).raise_if_invalid()
        body_names = tuple(pose_body_names)
        self.body_names = body_names
        self._source = SurfacePoseEvaluator(
            source_model,
            body_names,
            source_body_to_pose_body,
            source_surfaces,
        )
        self._target = SurfacePoseEvaluator(
            target_model,
            body_names,
            target_body_to_pose_body,
            target_surfaces,
        )

    def vertical_offset_m(self, source: KinematicPose, target: KinematicPose) -> float:
        """Return one Z translation that aligns the selected lowest surfaces."""

        return self._source.minimum_height_m(source) - self._target.minimum_height_m(target)

    def apply(self, source: KinematicPose, target: KinematicPose) -> KinematicPose:
        offset_m = self.vertical_offset_m(source, target)
        positions = np.asarray(target.positions_m, dtype=float).copy()
        positions[:, 2] += offset_m
        return KinematicPose(target.body_names, positions, np.asarray(target.orientations_wxyz).copy())


@dataclass(frozen=True)
class _AdaptationStep:
    parent_index: int
    child_index: int
    parent_anchor_m: np.ndarray
    child_anchor_m: np.ndarray


class KinematicMorphologyAdapter:
    """Apply global body orientations to a target model's calibrated morphology."""

    def __init__(
        self,
        target_model: KinematicTree,
        source_body_names: Sequence[str],
        *,
        target_body_to_source_body: Mapping[str, str],
        grounding: LowestSurfaceGrounding | None = None,
    ) -> None:
        validate_kinematic_tree(target_model).raise_if_invalid()
        body_names = tuple(source_body_names)
        source_indices = _unique_name_indices(body_names, label="source")
        target_body_names = tuple(body.name for body in target_model.bodies)
        mapping = dict(target_body_to_source_body)
        missing_target_bodies = sorted(set(target_body_names) - set(mapping))
        unknown_target_bodies = sorted(set(mapping) - set(target_body_names))
        if missing_target_bodies or unknown_target_bodies:
            raise ValueError(
                "Target/source mapping must cover every target body exactly; "
                f"missing={missing_target_bodies}, unknown={unknown_target_bodies}"
            )
        mapped_sources = tuple(mapping[name] for name in target_body_names)
        unknown_sources = sorted(set(mapped_sources) - set(body_names))
        duplicated_sources = sorted(name for name in set(mapped_sources) if mapped_sources.count(name) > 1)
        unused_sources = sorted(set(body_names) - set(mapped_sources))
        if unknown_sources or duplicated_sources or unused_sources:
            raise ValueError(
                "Target/source mapping must be bijective; "
                f"unknown={unknown_sources}, duplicated={duplicated_sources}, unused={unused_sources}"
            )

        body_indices = {target: source_indices[source] for target, source in mapping.items()}
        children: dict[str, list] = {name: [] for name in target_body_names}
        for joint in target_model.joints:
            children[joint.parent_body].append(joint)
        steps: list[_AdaptationStep] = []
        visited = {target_model.root_body}
        queue = [target_model.root_body]
        while queue:
            parent_body = queue.pop(0)
            for joint in children[parent_body]:
                if joint.child_body in visited:
                    raise ValueError(f"Kinematic model contains a cycle through '{joint.child_body}'")
                visited.add(joint.child_body)
                queue.append(joint.child_body)
                steps.append(
                    _AdaptationStep(
                        parent_index=body_indices[joint.parent_body],
                        child_index=body_indices[joint.child_body],
                        parent_anchor_m=np.asarray(joint.parent_frame.translation_m, dtype=float).copy(),
                        child_anchor_m=np.asarray(joint.child_frame.translation_m, dtype=float).copy(),
                    )
                )
        unreachable = sorted(set(target_body_names) - visited)
        if unreachable:
            raise ValueError(f"Target bodies are unreachable from '{target_model.root_body}': {unreachable}")
        if grounding is not None and grounding.body_names != body_names:
            raise ValueError("Grounding pose-body order must equal the morphology adapter source-body order")

        self.target_model = target_model
        self.source_body_names = body_names
        self.target_body_to_source_body = mapping
        self.root_index = body_indices[target_model.root_body]
        self.steps = tuple(steps)
        self.grounding = grounding

    def adapt_pose(self, source: KinematicPose) -> KinematicPose:
        """Reconstruct target body origins while preserving root and orientations."""

        _validate_pose(source, expected_body_names=self.source_body_names)
        positions = np.asarray(source.positions_m, dtype=float).copy()
        orientations = np.asarray(source.orientations_wxyz, dtype=float).copy()
        for step in self.steps:
            parent_anchor_world = rotate_vector(orientations[step.parent_index], step.parent_anchor_m)
            child_anchor_world = rotate_vector(orientations[step.child_index], step.child_anchor_m)
            positions[step.child_index] = positions[step.parent_index] + parent_anchor_world - child_anchor_world
        target = KinematicPose(self.source_body_names, positions, orientations)
        return target if self.grounding is None else self.grounding.apply(source, target)

    def adapt_motion(self, source: KinematicMotion) -> KinematicMotion:
        """Adapt every frame using the same pose kernel and preserve timestamps."""

        _validate_motion(source, expected_body_names=self.source_body_names)
        positions = np.empty_like(np.asarray(source.positions_m, dtype=float))
        for frame_index in range(positions.shape[0]):
            adapted = self.adapt_pose(
                KinematicPose(
                    source.body_names,
                    source.positions_m[frame_index],
                    source.orientations_wxyz[frame_index],
                )
            )
            positions[frame_index] = adapted.positions_m
        return KinematicMotion(
            self.source_body_names,
            positions,
            np.asarray(source.orientations_wxyz).copy(),
            np.asarray(source.times_s).copy(),
        )


def reference_root_floor_clearance_m(model: KinematicTree) -> float:
    """Return the authored vertical distance from the root origin to the visual floor."""

    validate_kinematic_tree(model).raise_if_invalid()
    body_names = tuple(body.name for body in model.bodies)
    surfaces = tuple(
        GroundingSurface(body.name, tuple(mesh.name for mesh in body.meshes))
        for body in model.bodies
        if body.meshes
    )
    if not surfaces:
        raise ValueError("Cannot compute visual floor clearance for a model without meshes")
    mapping = {body_name: body_name for body_name in body_names}
    evaluator = SurfacePoseEvaluator(model, body_names, mapping, surfaces)
    reference_pose = KinematicPose(
        body_names,
        np.array([body.reference_pose.translation_m for body in model.bodies]),
        np.array([body.reference_pose.rotation_wxyz for body in model.bodies]),
    )
    minimum_z = evaluator.minimum_height_m(reference_pose)
    root_z = float(model.body_map()[model.root_body].reference_pose.translation_m[2])
    return root_z - float(minimum_z)


def reference_grounding_offset_m(source_model: KinematicTree, target_model: KinematicTree) -> float:
    """Return the target root-z correction that preserves the source model's floor."""

    return reference_root_floor_clearance_m(target_model) - reference_root_floor_clearance_m(source_model)


def _unique_name_indices(names: tuple[str, ...], *, label: str) -> dict[str, int]:
    if not names:
        raise ValueError(f"{label.capitalize()} body names must not be empty")
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(f"{label.capitalize()} body names must be unique: {duplicates}")
    return {name: index for index, name in enumerate(names)}


def _validate_pose(pose: KinematicPose, *, expected_body_names: tuple[str, ...]) -> None:
    if tuple(pose.body_names) != expected_body_names:
        raise ValueError("Pose body names/order do not match the morphology adapter")
    positions = np.asarray(pose.positions_m, dtype=float)
    orientations = np.asarray(pose.orientations_wxyz, dtype=float)
    expected_positions = (len(expected_body_names), 3)
    expected_orientations = (len(expected_body_names), 4)
    if positions.shape != expected_positions or orientations.shape != expected_orientations:
        raise ValueError(
            f"Pose arrays must have shapes {expected_positions} and {expected_orientations}; "
            f"got {positions.shape} and {orientations.shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise ValueError("Pose arrays must contain only finite values")
    if np.any(np.linalg.norm(orientations, axis=1) <= 1e-12):
        raise ValueError("Pose contains a zero-length quaternion")


def _validate_motion(motion: KinematicMotion, *, expected_body_names: tuple[str, ...]) -> None:
    if tuple(motion.body_names) != expected_body_names:
        raise ValueError("Motion body names/order do not match the morphology adapter")
    positions = np.asarray(motion.positions_m, dtype=float)
    orientations = np.asarray(motion.orientations_wxyz, dtype=float)
    times = np.asarray(motion.times_s, dtype=float).reshape(-1)
    expected_positions_tail = (len(expected_body_names), 3)
    expected_orientations_tail = (len(expected_body_names), 4)
    if positions.ndim != 3 or positions.shape[1:] != expected_positions_tail:
        raise ValueError(f"Motion positions must have shape [F, {len(expected_body_names)}, 3]")
    if orientations.shape != positions.shape[:1] + expected_orientations_tail:
        raise ValueError(f"Motion orientations must have shape [F, {len(expected_body_names)}, 4]")
    if positions.shape[0] == 0 or times.shape != (positions.shape[0],):
        raise ValueError("Motion timestamps must match a non-empty frame dimension")
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all() or not np.isfinite(times).all():
        raise ValueError("Motion arrays must contain only finite values")
    if np.any(np.linalg.norm(orientations, axis=2) <= 1e-12):
        raise ValueError("Motion contains a zero-length quaternion")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("Motion timestamps must be strictly increasing")
