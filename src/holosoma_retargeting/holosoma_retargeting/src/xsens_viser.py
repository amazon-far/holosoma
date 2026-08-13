"""Viser rendering and timestamp sampling for rigid Xsens USDA avatars."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import viser  # type: ignore[import-not-found]

from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XSENS_BODY_SEGMENT_NAMES,
    XSENS_TRACKED_PROP_NAMES,
    XsensHdf5Motion,
    load_xsens_hdf5_calibration,
)
from holosoma_retargeting.kinematics import KinematicTree
from holosoma_retargeting.src.viser_utils import interpolation_window, quat_slerp
from holosoma_retargeting.src.paths import DEMO_RESULTS_DIR, PACKAGE_ROOT
from holosoma_retargeting.usd import open_usd_stage, read_kinematic_tree_from_stage
from holosoma_retargeting.xsens.g1_kinematic_reduction import G1_XSENS_REDUCTION_VERSION
from holosoma_retargeting.xsens.kinematic_model import (
    TENNIS_RACKET_BODY,
    calibration_fingerprint,
    normalize_xsens_name,
)

DEFAULT_G1_XSENS_USD = DEMO_RESULTS_DIR / "g1/models/g1_proportioned_xsens.usda"


def _package_dir() -> Path:
    return PACKAGE_ROOT


def resolve_package_path(path: str | Path) -> Path:
    """Resolve a path from the working directory or retargeting package root."""

    value = Path(path).expanduser()
    candidates = (value, _package_dir() / value) if not value.is_absolute() else (value,)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[-1].resolve()


def resolve_subject_xsens_usd(hdf5_path: str | Path, usd_path: str | Path | None = None) -> Path:
    hdf5 = resolve_package_path(hdf5_path)
    model_path = (
        hdf5.with_name(f"{hdf5.stem}_xsens_model.usda")
        if usd_path is None
        else resolve_package_path(usd_path)
    )
    if not model_path.is_file():
        command = f"python examples/xsens_tennis/export_xsens_usd.py --hdf5-path {hdf5}"
        raise FileNotFoundError(
            f"Recording-specific Xsens USDA not found: {model_path}\nGenerate it with:\n{command}"
        )
    return model_path


def resolve_g1_xsens_usd(usd_path: str | Path | None = None) -> Path:
    model_path = resolve_package_path(DEFAULT_G1_XSENS_USD if usd_path is None else usd_path)
    if not model_path.is_file():
        command = (
            "python examples/xsens_tennis/generate_g1_xsens_usd.py "
            f"--output-path {model_path}"
        )
        raise FileNotFoundError(f"G1-proportioned Xsens USDA not found: {model_path}\nGenerate it with:\n{command}")
    return model_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_xsens_usd_model(path: str | Path) -> KinematicTree:
    model_path = resolve_package_path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Xsens USDA not found: {model_path}")
    return read_kinematic_tree_from_stage(open_usd_stage(model_path))


def _source_names(model: KinematicTree) -> dict[str, str]:
    result: dict[str, str] = {}
    for body in model.bodies:
        source_name = str(body.metadata.get("xsens:sourceSegmentName", body.name))
        normalized = normalize_xsens_name(source_name)
        if normalized in result:
            raise ValueError(f"Xsens USDA maps multiple bodies to source segment '{source_name}'")
        result[normalized] = body.name
    return result


def validate_xsens_body_mapping(model: KinematicTree, *, require_racket: bool = True) -> None:
    mapped = _source_names(model)
    expected = list(XSENS_BODY_SEGMENT_NAMES)
    if require_racket:
        expected.extend(XSENS_TRACKED_PROP_NAMES)
    missing = [name for name in expected if normalize_xsens_name(name) not in mapped]
    if missing:
        raise ValueError(f"Xsens USDA is missing source segment mappings: {missing}")
    if require_racket and TENNIS_RACKET_BODY not in model.body_map():
        raise ValueError(f"Xsens USDA is missing canonical body '{TENNIS_RACKET_BODY}'")


def validate_subject_xsens_usd(model: KinematicTree, hdf5_path: str | Path) -> None:
    """Ensure a calibrated USDA was exported from this exact HDF5 recording."""

    hdf5 = resolve_package_path(hdf5_path)
    validate_xsens_body_mapping(model)
    expected_hash = _file_sha256(hdf5)
    actual_hash = str(model.metadata.get("xsens:sourceFileSha256", ""))
    calibration = load_xsens_hdf5_calibration(hdf5)
    expected_fingerprint = calibration_fingerprint(calibration)
    actual_fingerprint = str(model.metadata.get("xsens:calibrationFingerprint", ""))
    errors: list[str] = []
    if actual_hash != expected_hash:
        errors.append("source file SHA256 does not match")
    if actual_fingerprint != expected_fingerprint:
        errors.append("calibration fingerprint does not match")
    if errors:
        command = f"python examples/xsens_tennis/export_xsens_usd.py --hdf5-path {hdf5}"
        raise ValueError(
            f"Recording-specific Xsens USDA is stale or belongs to another recording: {', '.join(errors)}.\n"
            f"Regenerate it with:\n{command}"
        )


def validate_g1_xsens_usd(model: KinematicTree) -> None:
    """Validate the source-independent G1-proportioned Xsens model contract."""

    validate_xsens_body_mapping(model)
    root_marker = model.metadata.get("model:proportionedFrom", model.metadata.get("model:proportionSource"))
    body_markers = {body.metadata.get("model:proportionedFrom") for body in model.bodies}
    if root_marker != "g1_29dof" or body_markers != {"g1_29dof"}:
        raise ValueError("G1 Xsens USDA must declare model:proportionedFrom='g1_29dof' on its model and bodies")
    generator_version = str(model.metadata.get("model:generatorVersion", ""))
    if generator_version != G1_XSENS_REDUCTION_VERSION:
        command = (
            "python examples/xsens_tennis/generate_g1_xsens_usd.py "
            f"--output-path {DEFAULT_G1_XSENS_USD}"
        )
        raise ValueError(
            "G1 Xsens USDA was generated by an incompatible model generator "
            f"(found version {generator_version or 'missing'}, expected {G1_XSENS_REDUCTION_VERSION}).\n"
            f"Regenerate it with:\n{command}"
        )


@dataclass(frozen=True)
class XsensPoseSample:
    positions_m: np.ndarray
    quaternions_wxyz: np.ndarray


class XsensMotionSampler:
    """Interpolate global Xsens segment poses on their native timestamp timeline."""

    def __init__(self, motion: XsensHdf5Motion):
        positions = np.asarray(motion.positions_m, dtype=float)
        quaternions = np.asarray(motion.quaternions_wijk, dtype=float)
        times = np.asarray(motion.times_s, dtype=float).reshape(-1)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("Xsens positions must have shape [frames, segments, 3]")
        if quaternions.shape != positions.shape[:2] + (4,):
            raise ValueError("Xsens quaternions must have shape [frames, segments, 4]")
        if times.shape[0] != positions.shape[0] or times.size == 0:
            raise ValueError("Xsens timestamps must match the non-empty motion frame count")
        self.positions_m = positions
        self.quaternions_wxyz = quaternions
        self.times_s = times - times[0]
        interpolation_window(self.times_s, float(self.times_s[0]))
        self.segment_names = tuple(motion.segment_names)
        self.source_indices = tuple(motion.source_indices)

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])

    def sample(self, time_s: float) -> XsensPoseSample:
        lower, upper, weight = interpolation_window(self.times_s, time_s)
        if lower == upper:
            return XsensPoseSample(
                self.positions_m[lower].copy(),
                self.quaternions_wxyz[lower].copy(),
            )
        positions = (1.0 - weight) * self.positions_m[lower] + weight * self.positions_m[upper]
        quaternions = np.stack(
            [
                quat_slerp(self.quaternions_wxyz[lower, index], self.quaternions_wxyz[upper, index], weight)
                for index in range(self.quaternions_wxyz.shape[1])
            ]
        )
        return XsensPoseSample(positions, quaternions)


def _scene_token(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in value)


class XsensUsdActor:
    """Static USDA meshes driven by global Xsens segment poses."""

    def __init__(
        self,
        server: viser.ViserServer,
        usd_path: str | Path,
        *,
        root_node_name: str = "/xsens",
        model: KinematicTree | None = None,
        show_meshes: bool = True,
        show_landmarks: bool = False,
    ) -> None:
        self.usd_path = resolve_package_path(usd_path)
        self.model = load_xsens_usd_model(self.usd_path) if model is None else model
        self.root_node_name = root_node_name.rstrip("/")
        self.root = server.scene.add_frame(self.root_node_name, show_axes=False)
        self.body_frames: dict[str, Any] = {}
        self.mesh_handles: list[Any] = []
        self.landmark_handles: list[Any] = []
        self.source_to_body: dict[str, str] = {}

        for body in self.model.bodies:
            source_name = str(body.metadata.get("xsens:sourceSegmentName", body.name))
            normalized_source = normalize_xsens_name(source_name)
            self.source_to_body[normalized_source] = body.name
            body_path = f"{self.root_node_name}/bodies/{_scene_token(body.name)}"
            frame = server.scene.add_frame(
                body_path,
                show_axes=False,
                position=np.asarray(body.reference_pose.translation_m, dtype=float),
                wxyz=np.asarray(body.reference_pose.rotation_wxyz, dtype=float),
            )
            self.body_frames[body.name] = frame
            for mesh_index, mesh in enumerate(body.meshes):
                handle = server.scene.add_mesh_simple(
                    f"{body_path}/meshes/{mesh_index:02d}_{_scene_token(mesh.name)}",
                    vertices=np.asarray(mesh.vertices_m, dtype=float),
                    faces=np.asarray(mesh.faces, dtype=np.int32),
                    color=mesh.color_rgb,
                    side="double",
                    visible=show_meshes,
                )
                self.mesh_handles.append(handle)
            for point_index, point_set in enumerate(body.point_sets):
                colors = np.tile(np.array([[255, 214, 10]], dtype=np.uint8), (len(point_set.points_m), 1))
                handle = server.scene.add_point_cloud(
                    f"{body_path}/landmarks/{point_index:02d}_{_scene_token(point_set.name)}",
                    points=np.asarray(point_set.points_m, dtype=float),
                    colors=colors,
                    point_size=max(0.002, float(point_set.width_m)),
                    point_shape="circle",
                    visible=show_landmarks,
                )
                self.landmark_handles.append(handle)

    def set_mesh_visibility(self, visible: bool) -> None:
        for handle in self.mesh_handles:
            handle.visible = bool(visible)

    def set_landmark_visibility(self, visible: bool) -> None:
        for handle in self.landmark_handles:
            handle.visible = bool(visible)

    def apply_pose(
        self,
        segment_names: tuple[str, ...] | list[str],
        positions_m: np.ndarray,
        quaternions_wxyz: np.ndarray,
    ) -> None:
        """Apply HDF5 global transforms directly, without FK or scaling."""

        positions = np.asarray(positions_m, dtype=float)
        quaternions = np.asarray(quaternions_wxyz, dtype=float)
        if positions.shape != (len(segment_names), 3) or quaternions.shape != (len(segment_names), 4):
            raise ValueError("Xsens pose arrays do not match segment_names")
        missing: list[str] = []
        for index, source_name in enumerate(segment_names):
            body_name = self.source_to_body.get(normalize_xsens_name(source_name))
            if body_name is None:
                missing.append(source_name)
                continue
            frame = self.body_frames[body_name]
            frame.position = positions[index]
            frame.wxyz = quaternions[index]
        if missing:
            raise KeyError(f"Xsens USDA has no body mapping for motion segments: {missing}")
