#!/usr/bin/env python3
"""Compare canonical Xsens and G1 reference poses side-by-side in Viser.

The source avatar is loaded from ``--calibrated-xsens-usd-path`` when supplied.
Otherwise the script exports a calibrated USD from ``--hdf5-path`` first. The
recording's calibrated T-pose is also used to construct a canonical N-pose, and
the sidebar switches all three columns between the two configurations.

Example:
    python examples/xsens_tennis/compare_xsens_g1_poses.py \
        --hdf5-path demo_data/xsens_tennis/2026-06-14_tennis_S02_xsens_myo_data_02.hdf5
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.data_utils.xsens_hdf5 import (  # noqa: E402
    XsensHdf5Tpose,
    load_xsens_hdf5_tpose,
)
from holosoma_retargeting.kinematics import KinematicPose, KinematicTree, Transform  # noqa: E402
from holosoma_retargeting.src.mujoco_utils import (  # noqa: E402
    MujocoFramePoseSet,
    evaluate_mujoco_frame_poses,
    replace_named_joint_qpos,
)
from holosoma_retargeting.src.paths import DEMO_RESULTS_DIR  # noqa: E402
from holosoma_retargeting.usd import open_usd_stage, read_kinematic_tree_from_stage  # noqa: E402
from holosoma_retargeting.xsens.kinematic_model import normalize_xsens_name  # noqa: E402
from holosoma_retargeting.xsens.morphology_adaptation import (  # noqa: E402
    build_xsens_morphology_adapter,
    prepare_g1_xsens_morphology,
    xsens_body_to_source_mapping,
)
from holosoma_retargeting.xsens.orientation_tracking import (  # noqa: E402
    describe_xsens_orientation_correspondences,
)
from holosoma_retargeting.xsens.tpose_calibration import (  # noqa: E402
    XsensTposeCalibrationConfig,
    XsensTposeCalibrationResult,
    solve_xsens_tpose_calibration_from_data,
)
from holosoma_retargeting.xsens.usd_conversion import convert_xsens_hdf5_to_usd  # noqa: E402

ReferencePoseName = Literal["tpose", "npose"]
REFERENCE_POSE_LABELS: Mapping[ReferencePoseName, str] = {
    "tpose": "T-pose",
    "npose": "N-pose",
}


@dataclass(frozen=True)
class XsensG1PoseComparisonConfig:
    """Inputs and display options for the three-model reference-pose comparison."""

    hdf5_path: Path = Path("demo_data/xsens_tennis/2026-06-14_tennis_S02_xsens_myo_data_02.hdf5")
    """Xsens recording providing the calibrated avatar and T-pose targets."""

    calibrated_xsens_usd_path: Path | None = None
    """Existing human-subject Xsens avatar USD; generated from hdf5_path when omitted."""

    generated_usd_dir: Path = DEMO_RESULTS_DIR / "g1/models/xsens"
    """Destination for the generated human-subject Xsens avatar USD."""

    g1_urdf_path: Path = Path("models/g1/g1_29dof.urdf")
    """G1 model rendered by Viser and used by the T-pose calibration."""

    preserve_joint_offsets: bool = False
    """Preserve compound G1 offsets in the G1-proportioned Xsens avatar."""

    include_tennis_racket: bool = False
    """Include the calibrated recording's tracked tennis-racket segment."""

    spacing_m: float = 2.0
    """Center-to-center lateral spacing between columns."""

    port: int = 8080
    """Viser server port."""

    tpose_max_nfev: int = 400
    """Maximum evaluations for the physical G1 T-pose calibration."""

    default_human_height: float = 1.78
    """Human height used by the existing Xsens-to-G1 T-pose calibration."""

    show_orientation_correspondences: bool = True
    """Initially show calibrated segment/link axes used for orientation tracking."""

    initial_pose: ReferencePoseName = "tpose"
    """Reference pose selected when the viewer opens."""


# Backward-compatible type name for callers importing the old T-pose viewer.
XsensG1TposeComparisonConfig = XsensG1PoseComparisonConfig


@dataclass(frozen=True)
class ReferencePoseAssets:
    """The three synchronized actors for one canonical reference pose."""

    human_xsens_pose: KinematicPose
    g1_xsens_pose: KinematicPose
    g1_qpos: np.ndarray
    g1_correspondence_link_poses: MujocoFramePoseSet


@dataclass(frozen=True)
class ComparisonAssets:
    calibrated_xsens_model: KinematicTree
    g1_xsens_model: KinematicTree
    calibrated_xsens_usd_path: Path
    g1_urdf_path: Path
    calibration: XsensTposeCalibrationResult
    reference_poses: Mapping[ReferencePoseName, ReferencePoseAssets]

    @property
    def g1_xsens_tpose(self) -> KinematicPose:
        """Compatibility view of the original T-pose-only asset."""

        return self.reference_poses["tpose"].g1_xsens_pose

    @property
    def g1_correspondence_link_poses(self) -> MujocoFramePoseSet:
        """Compatibility view of the original T-pose-only frame poses."""

        return self.reference_poses["tpose"].g1_correspondence_link_poses


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_existing_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    package_relative = _package_root() / path
    if package_relative.is_file():
        return package_relative.resolve()
    raise FileNotFoundError(path)


def _resolve_g1_xml(urdf_path: Path) -> Path:
    sibling_xml = urdf_path.with_suffix(".xml")
    return sibling_xml if sibling_xml.is_file() else urdf_path


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


def _matrix_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def _x_rotation(angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ]
    )


def canonical_xsens_npose_orientations(tpose: KinematicPose) -> np.ndarray:
    """Rotate each complete T-pose arm chain downward in its shoulder frame."""

    indices = {normalize_xsens_name(name): index for index, name in enumerate(tpose.body_names)}
    orientations = np.asarray(tpose.orientations_wxyz, dtype=float).copy()
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        shoulder_index = indices[normalize_xsens_name(f"{side} Shoulder")]
        shoulder_rotation = _quat_matrix(orientations[shoulder_index])
        local_arm_rotation = _x_rotation(-sign * 0.5 * np.pi)
        world_arm_rotation = shoulder_rotation @ local_arm_rotation @ shoulder_rotation.T
        for segment in ("Upper Arm", "Forearm", "Hand"):
            segment_index = indices[normalize_xsens_name(f"{side} {segment}")]
            orientations[segment_index] = _matrix_quat_wxyz(
                world_arm_rotation @ _quat_matrix(orientations[segment_index])
            )
    return orientations


def build_canonical_xsens_npose(
    model: KinematicTree,
    tpose: KinematicPose,
) -> KinematicPose:
    """Reconstruct a canonical hanging-arm pose using one model's own anchors."""

    adapter = build_xsens_morphology_adapter(model, tpose.body_names, grounding="none")
    return adapter.adapt_pose(
        KinematicPose(
            tpose.body_names,
            np.asarray(tpose.positions_m, dtype=float),
            canonical_xsens_npose_orientations(tpose),
        )
    )


def g1_npose_qpos_from_tpose(
    model_path: str | Path,
    tpose_qpos: np.ndarray,
) -> np.ndarray:
    """Lower both calibrated G1 arms by neutralizing shoulder roll only."""

    return replace_named_joint_qpos(
        model_path,
        tpose_qpos,
        {
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
        },
    )


def tree_vertical_bounds(
    model: KinematicTree,
    body_poses: Mapping[str, Transform] | None = None,
) -> tuple[float, float]:
    """Return bounds for reference or supplied body poses, including meshes."""

    z_values: list[float] = []
    for body in model.bodies:
        pose = body.reference_pose if body_poses is None else body_poses[body.name]
        rotation = _quat_matrix(pose.rotation_wxyz)
        if body.meshes:
            for mesh in body.meshes:
                world_vertices = np.asarray(mesh.vertices_m, dtype=float) @ rotation.T + pose.translation_m
                z_values.extend((float(world_vertices[:, 2].min()), float(world_vertices[:, 2].max())))
        else:
            z_values.append(float(pose.translation_m[2]))
    if not z_values:
        raise ValueError("Cannot compute bounds for an empty kinematic tree")
    return min(z_values), max(z_values)


def side_by_side_offsets(spacing_m: float) -> tuple[float, float, float]:
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    return (-float(spacing_m), 0.0, float(spacing_m))


def body_poses_from_xsens_pose(
    model: KinematicTree,
    pose: KinematicPose,
    *,
    preserve_unmapped_bodies: bool = False,
) -> dict[str, Transform]:
    """Map one ordered Xsens pose onto a validated Xsens kinematic tree."""

    if preserve_unmapped_bodies:
        source_indices = {normalize_xsens_name(name): index for index, name in enumerate(pose.body_names)}
        result: dict[str, Transform] = {}
        for body in model.bodies:
            source_name = normalize_xsens_name(str(body.metadata.get("xsens:sourceSegmentName", body.name)))
            if source_name not in source_indices:
                result[body.name] = body.reference_pose
                continue
            source_index = source_indices[source_name]
            result[body.name] = Transform(
                np.asarray(pose.positions_m[source_index], dtype=float),
                np.asarray(pose.orientations_wxyz[source_index], dtype=float),
            )
        return result

    body_to_source = xsens_body_to_source_mapping(model, pose.body_names)
    source_indices = {name: index for index, name in enumerate(pose.body_names)}
    return {
        body.name: Transform(
            np.asarray(pose.positions_m[source_indices[body_to_source[body.name]]], dtype=float),
            np.asarray(pose.orientations_wxyz[source_indices[body_to_source[body.name]]], dtype=float),
        )
        for body in model.bodies
    }


@dataclass(frozen=True)
class _KinematicTreeHandles:
    root: viser.FrameHandle
    bodies: Mapping[str, viser.FrameHandle]
    axes: Mapping[str, viser.FrameHandle]
    lateral_offset_m: float


def orientation_correspondence_body_names(
    model: KinematicTree,
    orientation_names: list[str],
) -> set[str]:
    """Return model bodies participating in calibrated orientation tracking."""

    active_names = {normalize_xsens_name(name) for name in orientation_names}
    return {
        body.name
        for body in model.bodies
        if normalize_xsens_name(str(body.metadata.get("xsens:sourceSegmentName", body.name))) in active_names
    }


def _calibrated_usd_path(config: XsensG1PoseComparisonConfig, hdf5_path: Path) -> Path:
    if config.calibrated_xsens_usd_path is not None:
        return _resolve_existing_path(config.calibrated_xsens_usd_path)
    output_dir = config.generated_usd_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{hdf5_path.stem}_xsens_model.usda"
    convert_xsens_hdf5_to_usd(
        hdf5_path,
        output_path,
        include_visuals=True,
        include_landmarks=False,
        include_tennis_racket=config.include_tennis_racket,
    )
    return output_path.resolve()


def prepare_comparison_assets(config: XsensG1PoseComparisonConfig) -> ComparisonAssets:
    """Load/generate all three comparison assets without starting Viser."""

    hdf5_path = _resolve_existing_path(config.hdf5_path)
    g1_urdf_path = _resolve_existing_path(config.g1_urdf_path)
    calibrated_usd_path = _calibrated_usd_path(config, hdf5_path)
    calibrated_xsens_model = read_kinematic_tree_from_stage(open_usd_stage(calibrated_usd_path))

    tpose = load_xsens_hdf5_tpose(hdf5_path)
    prepared_morphology = prepare_g1_xsens_morphology(
        tpose.segment_names,
        hdf5_path=hdf5_path,
        g1_model_path=_resolve_g1_xml(g1_urdf_path),
        grounding="match_lowest_soles",
        preserve_joint_offsets=config.preserve_joint_offsets,
    )
    human_xsens_tpose = KinematicPose(
        tuple(tpose.segment_names),
        tpose.positions_m,
        tpose.quaternions_wijk,
    )
    human_xsens_npose = build_canonical_xsens_npose(
        prepared_morphology.source_model,
        human_xsens_tpose,
    )
    g1_xsens_tpose = prepared_morphology.adapter.adapt_pose(human_xsens_tpose)
    g1_xsens_npose = prepared_morphology.adapter.adapt_pose(human_xsens_npose)

    calibration_config = XsensTposeCalibrationConfig(
        robot_type="g1",
        variant="Tpose",
        robot_urdf_file=str(g1_urdf_path),
        default_human_height=config.default_human_height,
        max_nfev=config.tpose_max_nfev,
        verbose=0,
    )
    calibration = solve_xsens_tpose_calibration_from_data(
        XsensHdf5Tpose(
            positions_m=g1_xsens_tpose.positions_m,
            quaternions_wijk=tpose.quaternions_wijk,
            variant="G1ProportionedTpose",
            segment_names=list(tpose.segment_names),
            source_indices=list(tpose.source_indices),
        ),
        config=calibration_config,
        position_scale_factor=1.0,
    )
    g1_model_path = _resolve_g1_xml(g1_urdf_path)
    g1_tpose_qpos = np.asarray(calibration.qpos[0], dtype=float)
    g1_npose_qpos = g1_npose_qpos_from_tpose(g1_model_path, g1_tpose_qpos)
    reference_poses: dict[ReferencePoseName, ReferencePoseAssets] = {
        "tpose": ReferencePoseAssets(
            human_xsens_pose=human_xsens_tpose,
            g1_xsens_pose=g1_xsens_tpose,
            g1_qpos=g1_tpose_qpos,
            g1_correspondence_link_poses=evaluate_mujoco_frame_poses(
                g1_model_path,
                g1_tpose_qpos,
                calibration.robot_link_names,
            ),
        ),
        "npose": ReferencePoseAssets(
            human_xsens_pose=human_xsens_npose,
            g1_xsens_pose=g1_xsens_npose,
            g1_qpos=g1_npose_qpos,
            g1_correspondence_link_poses=evaluate_mujoco_frame_poses(
                g1_model_path,
                g1_npose_qpos,
                calibration.robot_link_names,
            ),
        ),
    }
    return ComparisonAssets(
        calibrated_xsens_model=calibrated_xsens_model,
        g1_xsens_model=prepared_morphology.target_model,
        calibrated_xsens_usd_path=calibrated_usd_path,
        g1_urdf_path=g1_urdf_path,
        calibration=calibration,
        reference_poses=reference_poses,
    )


def _add_kinematic_tree(
    server: viser.ViserServer,
    *,
    root_path: str,
    model: KinematicTree,
    lateral_offset_m: float,
    body_poses: Mapping[str, Transform] | None = None,
    include_tennis_racket: bool = False,
) -> tuple[_KinematicTreeHandles, float]:
    minimum_z, maximum_z = tree_vertical_bounds(model, body_poses)
    root = server.scene.add_frame(
        root_path,
        show_axes=False,
        position=np.array([0.0, lateral_offset_m, -minimum_z]),
    )
    body_frames: dict[str, viser.FrameHandle] = {}
    axes: dict[str, viser.FrameHandle] = {}
    for body in model.bodies:
        if body.name == "TennisRacket" and not include_tennis_racket:
            continue
        pose = body.reference_pose if body_poses is None else body_poses[body.name]
        body_path = f"{root_path}/bodies/{body.name}"
        body_frames[body.name] = server.scene.add_frame(
            body_path,
            show_axes=False,
            position=pose.translation_m,
            wxyz=pose.rotation_wxyz,
        )
        axes[body.name] = server.scene.add_frame(
            f"{body_path}/axes",
            show_axes=True,
            axes_length=0.07,
            axes_radius=0.0025,
            visible=False,
        )
        for mesh in body.meshes:
            server.scene.add_mesh_simple(
                f"{body_path}/{mesh.name}",
                vertices=np.asarray(mesh.vertices_m, dtype=float),
                faces=np.asarray(mesh.faces, dtype=np.int64),
                color=mesh.color_rgb,
                material="toon5",
                flat_shading=False,
                side="double",
            )
    return (
        _KinematicTreeHandles(
            root=root,
            bodies=body_frames,
            axes=axes,
            lateral_offset_m=lateral_offset_m,
        ),
        maximum_z - minimum_z,
    )


def _update_kinematic_tree_pose(
    handles: _KinematicTreeHandles,
    model: KinematicTree,
    pose: KinematicPose,
    *,
    preserve_unmapped_bodies: bool = False,
) -> None:
    body_poses = body_poses_from_xsens_pose(
        model,
        pose,
        preserve_unmapped_bodies=preserve_unmapped_bodies,
    )
    minimum_z, _ = tree_vertical_bounds(model, body_poses)
    handles.root.position = np.array([0.0, handles.lateral_offset_m, -minimum_z])
    for body_name, frame in handles.bodies.items():
        body_pose = body_poses[body_name]
        frame.position = body_pose.translation_m
        frame.wxyz = body_pose.rotation_wxyz


def make_comparison_viewer(
    config: XsensG1PoseComparisonConfig,
    assets: ComparisonAssets | None = None,
) -> viser.ViserServer:
    assets = assets or prepare_comparison_assets(config)
    calibrated_offset, generated_offset, robot_offset = side_by_side_offsets(config.spacing_m)
    initial_reference_pose = assets.reference_poses[config.initial_pose]

    server = viser.ViserServer(port=config.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=5.0,
        height=3.0 * config.spacing_m + 2.0,
        position=(0.0, 0.0, 0.0),
    )

    calibrated_handles, calibrated_height = _add_kinematic_tree(
        server,
        root_path="/comparison/calibrated_xsens",
        model=assets.calibrated_xsens_model,
        lateral_offset_m=calibrated_offset,
        body_poses=body_poses_from_xsens_pose(
            assets.calibrated_xsens_model,
            initial_reference_pose.human_xsens_pose,
            preserve_unmapped_bodies=True,
        ),
        include_tennis_racket=config.include_tennis_racket,
    )
    generated_handles, generated_height = _add_kinematic_tree(
        server,
        root_path="/comparison/g1_xsens",
        model=assets.g1_xsens_model,
        lateral_offset_m=generated_offset,
        body_poses=body_poses_from_xsens_pose(
            assets.g1_xsens_model,
            initial_reference_pose.g1_xsens_pose,
        ),
    )

    g1_qpos = np.asarray(initial_reference_pose.g1_qpos, dtype=float)
    robot_root = server.scene.add_frame(
        "/comparison/actual_g1",
        show_axes=False,
        position=np.array([g1_qpos[0], robot_offset + g1_qpos[1], g1_qpos[2]]),
        wxyz=g1_qpos[3:7],
    )
    robot_axes = server.scene.add_frame(
        "/comparison/actual_g1/axes",
        show_axes=True,
        axes_length=0.1,
        axes_radius=0.003,
        visible=False,
    )
    robot_urdf = yourdfpy.URDF.load(
        str(assets.g1_urdf_path),
        load_meshes=True,
        build_scene_graph=True,
    )
    viser_robot = ViserUrdf(
        server,
        urdf_or_path=robot_urdf,
        root_node_name="/comparison/actual_g1",
    )
    robot_dof = len(viser_robot.get_actuated_joint_limits())
    viser_robot.update_cfg(g1_qpos[7 : 7 + robot_dof])

    robot_correspondence_axes: dict[str, viser.FrameHandle] = {}
    for link_name, position, quaternion in zip(
        initial_reference_pose.g1_correspondence_link_poses.names,
        initial_reference_pose.g1_correspondence_link_poses.positions_m,
        initial_reference_pose.g1_correspondence_link_poses.quaternions_wxyz,
        strict=True,
    ):
        robot_correspondence_axes[link_name] = server.scene.add_frame(
            f"/orientation_correspondence/actual_g1/{link_name}",
            position=np.asarray(position, dtype=float) + np.array([0.0, robot_offset, 0.0]),
            wxyz=np.asarray(quaternion, dtype=float),
            show_axes=True,
            axes_length=0.1,
            axes_radius=0.003,
            visible=config.show_orientation_correspondences,
        )

    calibrated_correspondence_bodies = orientation_correspondence_body_names(
        assets.calibrated_xsens_model,
        assets.calibration.active_orientation_mapping_names,
    )
    generated_correspondence_bodies = orientation_correspondence_body_names(
        assets.g1_xsens_model,
        assets.calibration.active_orientation_mapping_names,
    )

    label_height = max(calibrated_height, generated_height, 1.5) + 0.12
    label_specs = (
        ("calibrated_xsens", "Human-subject Xsens avatar", calibrated_offset),
        (
            "g1_xsens",
            (
                "G1 Xsens avatar (compound offsets preserved)"
                if config.preserve_joint_offsets
                else "G1 Xsens avatar (compound axes collapsed)"
            ),
            generated_offset,
        ),
        ("actual_g1", "Actual G1", robot_offset),
    )
    for name, text, lateral_offset in label_specs:
        server.scene.add_label(
            f"/labels/{name}",
            text,
            position=(0.0, lateral_offset, label_height),
            anchor="bottom-center",
            font_size_mode="scene",
            font_scene_height=0.075,
        )

    pose_labels = tuple(REFERENCE_POSE_LABELS.values())
    pose_names_by_label = {label: name for name, label in REFERENCE_POSE_LABELS.items()}
    reference_pose_selector = server.gui.add_dropdown(
        "Reference pose",
        options=pose_labels,
        initial_value=REFERENCE_POSE_LABELS[config.initial_pose],
    )

    with server.gui.add_folder("Models"):
        show_calibrated = server.gui.add_checkbox("Human-subject Xsens avatar", initial_value=True)
        show_generated = server.gui.add_checkbox("G1 Xsens avatar", initial_value=True)
        show_robot = server.gui.add_checkbox("Actual G1", initial_value=True)
        show_axes = server.gui.add_checkbox("Show all Xsens frames", initial_value=False)
        show_correspondences = server.gui.add_checkbox(
            "Show orientation correspondences",
            initial_value=config.show_orientation_correspondences,
        )

    @show_calibrated.on_update
    def _(_) -> None:
        calibrated_handles.root.visible = bool(show_calibrated.value)

    @show_generated.on_update
    def _(_) -> None:
        generated_handles.root.visible = bool(show_generated.value)

    @show_robot.on_update
    def _(_) -> None:
        viser_robot.show_visual = bool(show_robot.value)
        robot_root.visible = bool(show_robot.value)
        _update_axes_visibility()

    def _update_axes_visibility() -> None:
        show_all = bool(show_axes.value)
        show_mapped = bool(show_correspondences.value)
        for body_name, handle in calibrated_handles.axes.items():
            handle.visible = show_all or (show_mapped and body_name in calibrated_correspondence_bodies)
        for body_name, handle in generated_handles.axes.items():
            handle.visible = show_all or (show_mapped and body_name in generated_correspondence_bodies)
        robot_axes.visible = show_all
        for handle in robot_correspondence_axes.values():
            handle.visible = show_mapped and bool(show_robot.value)

    def _apply_reference_pose(pose_name: ReferencePoseName) -> None:
        reference_pose = assets.reference_poses[pose_name]
        _update_kinematic_tree_pose(
            calibrated_handles,
            assets.calibrated_xsens_model,
            reference_pose.human_xsens_pose,
            preserve_unmapped_bodies=True,
        )
        _update_kinematic_tree_pose(
            generated_handles,
            assets.g1_xsens_model,
            reference_pose.g1_xsens_pose,
        )
        if "TennisRacket" in calibrated_handles.bodies:
            calibrated_handles.bodies["TennisRacket"].visible = pose_name == "tpose"

        qpos = np.asarray(reference_pose.g1_qpos, dtype=float)
        robot_root.position = np.array([qpos[0], robot_offset + qpos[1], qpos[2]])
        robot_root.wxyz = qpos[3:7]
        viser_robot.update_cfg(qpos[7 : 7 + robot_dof])
        for link_name, position, quaternion in zip(
            reference_pose.g1_correspondence_link_poses.names,
            reference_pose.g1_correspondence_link_poses.positions_m,
            reference_pose.g1_correspondence_link_poses.quaternions_wxyz,
            strict=True,
        ):
            handle = robot_correspondence_axes[link_name]
            handle.position = np.asarray(position, dtype=float) + np.array([0.0, robot_offset, 0.0])
            handle.wxyz = np.asarray(quaternion, dtype=float)
        print(f"[compare_xsens_g1_poses] Reference pose: {REFERENCE_POSE_LABELS[pose_name]}")

    @reference_pose_selector.on_update
    def _(_) -> None:
        _apply_reference_pose(pose_names_by_label[str(reference_pose_selector.value)])

    @show_axes.on_update
    def _(_) -> None:
        _update_axes_visibility()

    @show_correspondences.on_update
    def _(_) -> None:
        _update_axes_visibility()

    _apply_reference_pose(config.initial_pose)
    _update_axes_visibility()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (5.2, 0.0, 1.55)
        client.camera.look_at = (0.0, 0.0, 0.78)
        client.camera.up = (0.0, 0.0, 1.0)

    print(f"[compare_xsens_g1_poses] Human-subject Xsens avatar USD: {assets.calibrated_xsens_usd_path}")
    print(
        "[compare_xsens_g1_poses] G1 T-pose "
        f"solver_success={assets.calibration.solver_success} cost={assets.calibration.solver_cost:.4f}"
    )
    print(
        "[compare_xsens_g1_poses] Human Xsens -> G1-sized Xsens is a direct one-to-one joint-configuration "
        "transfer (no IK/optimization): global segment orientations are copied exactly and positions are "
        "reconstructed only from G1-sized joint anchors."
    )
    print(
        "[compare_xsens_g1_poses] N-pose lowers each complete Xsens arm chain by 90 degrees in the "
        "shoulder frame and neutralizes only the physical G1 shoulder-roll joints."
    )
    print("[compare_xsens_g1_poses] Rotational correspondence:")
    for line in describe_xsens_orientation_correspondences(
        assets.calibration.active_orientation_mapping_names,
        assets.calibration.robot_link_names,
        assets.calibration.orientation_offsets_wijk,
    ):
        print(f"[compare_xsens_g1_poses]   {line}")
    print("[compare_xsens_g1_poses] Column order: human-subject Xsens avatar | G1 Xsens avatar | actual G1")
    return server


def main(config: XsensG1PoseComparisonConfig) -> None:
    assets = prepare_comparison_assets(config)
    make_comparison_viewer(config, assets)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(XsensG1PoseComparisonConfig))
