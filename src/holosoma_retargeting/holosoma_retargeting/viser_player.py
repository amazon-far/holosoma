#!/usr/bin/env python3
"""Play G1 qpos and global Xsens HDF5 motion in one Viser scene."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig, XsensViserConfig  # noqa: E402
from holosoma_retargeting.data_utils.xsens_hdf5 import (  # noqa: E402
    XsensHdf5Motion,
    load_xsens_hdf5_motion,
)
from holosoma_retargeting.kinematics import KinematicMorphologyAdapter, KinematicMotion  # noqa: E402
from holosoma_retargeting.src.recording_utils import (  # noqa: E402
    build_record_frame_indices,
    record_viser_sequence,
)
from holosoma_retargeting.src.viser_utils import (  # noqa: E402
    CameraFollowController,
    QposViserApplier,
    create_motion_control_sliders,
    create_timed_motion_control_sliders,
    interpolation_window,
    sample_qpos_at_time,
)
from holosoma_retargeting.src.xsens_viser import (  # noqa: E402
    XsensMotionSampler,
    XsensUsdActor,
    load_xsens_usd_model,
    resolve_g1_xsens_usd,
    resolve_package_path,
    resolve_subject_xsens_usd,
    validate_g1_xsens_usd,
    validate_subject_xsens_usd,
)
from holosoma_retargeting.xsens.kinematic_model import TENNIS_RACKET_BODY  # noqa: E402
from holosoma_retargeting.xsens.morphology_adaptation import (  # noqa: E402
    apply_xsens_root_motion,
    build_subject_xsens_reference_model,
    build_xsens_morphology_adapter,
)


def load_npz(npz_path: str) -> tuple[np.ndarray, int]:
    data = np.load(resolve_package_path(npz_path), allow_pickle=True)
    qpos = np.asarray(data["qpos"])
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def _nominal_fps(times_s: np.ndarray, fallback: float) -> float:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size <= 1:
        return float(fallback)
    intervals = np.diff(times)
    intervals = intervals[intervals > 0.0]
    return float(1.0 / np.median(intervals)) if intervals.size else float(fallback)


ActorMode = Literal["robot", "xsens", "g1_xsens"]
ACTOR_MODE_ORDER: tuple[ActorMode, ...] = ("robot", "xsens", "g1_xsens")
ACTOR_LAYOUT_ORDER: tuple[ActorMode, ...] = ("xsens", "g1_xsens", "robot")


def resolve_actor_modes(actor_modes: Sequence[str]) -> tuple[ActorMode, ...]:
    """Expand aliases, validate actor modes, and return a stable unique ordering."""

    requested = tuple(actor_modes)
    if not requested:
        raise ValueError("actor_modes must contain at least one actor")
    allowed = {*ACTOR_MODE_ORDER, "all"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"Unknown actor modes: {unknown}. Expected values from {sorted(allowed)}")
    if "all" in requested:
        return ACTOR_MODE_ORDER
    return tuple(mode for mode in ACTOR_MODE_ORDER if mode in requested)


def resolve_actor_offsets(
    actor_modes: Sequence[str],
    spacing_m: float,
) -> dict[ActorMode, np.ndarray]:
    """Center active actors laterally in human, G1-avatar, physical-G1 order."""

    spacing = float(spacing_m)
    if not np.isfinite(spacing) or spacing < 0.0:
        raise ValueError("actor_spacing_m must be a finite non-negative value")
    active_modes = set(resolve_actor_modes(actor_modes))
    layout_modes = tuple(mode for mode in ACTOR_LAYOUT_ORDER if mode in active_modes)
    center_index = 0.5 * (len(layout_modes) - 1)
    return {
        mode: np.array([0.0, (index - center_index) * spacing, 0.0])
        for index, mode in enumerate(layout_modes)
    }


def offset_qpos_positions(
    qpos: np.ndarray,
    offset_m: Sequence[float],
    *,
    has_object_input: bool,
) -> np.ndarray:
    """Translate displayed robot and object roots without changing source motion."""

    motion = np.asarray(qpos, dtype=float)
    offset = np.asarray(offset_m, dtype=float)
    if motion.ndim != 2 or motion.shape[1] < 7:
        raise ValueError("qpos must have shape [frames, values] with a floating base")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Actor offset must contain three finite xyz values")
    result = motion.copy()
    result[:, 0:3] += offset
    if has_object_input:
        result[:, -7:-4] += offset
    return result


def resolve_record_output_path(config: ViserConfig) -> str:
    """Resolve an explicit recording path or derive one from the source motion."""

    if config.record_path is not None:
        return config.record_path
    if isinstance(config, XsensViserConfig) and config.xsens_hdf5 is not None:
        actor_modes = resolve_actor_modes(config.actor_modes)
        if "xsens" in actor_modes or "g1_xsens" in actor_modes:
            return str(resolve_package_path(config.xsens_hdf5).with_suffix(".mp4"))
    return str(resolve_package_path(config.qpos_npz).with_suffix(".mp4"))


def compute_camera_follow_target(
    *,
    robot_position_m: np.ndarray | None = None,
    avatar_positions_m: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Return the center of the displayed robot and avatar actor positions."""

    actor_centers: list[np.ndarray] = []
    if robot_position_m is not None:
        robot_position = np.asarray(robot_position_m, dtype=float)
        if robot_position.shape != (3,) or not np.isfinite(robot_position).all():
            raise ValueError("robot_position_m must contain three finite xyz values")
        actor_centers.append(robot_position)
    for positions_m in avatar_positions_m:
        positions = np.asarray(positions_m, dtype=float)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or positions.shape[0] == 0:
            raise ValueError("Each avatar position array must have shape [segments, 3]")
        finite_positions = positions[np.isfinite(positions).all(axis=1)]
        if finite_positions.shape[0] == 0:
            raise ValueError("Each avatar must contain at least one finite segment position")
        actor_centers.append(finite_positions.mean(axis=0))
    if not actor_centers:
        raise ValueError("At least one robot or avatar position is required")
    return np.mean(actor_centers, axis=0)


def add_tennis_racket_control(
    server: viser.ViserServer,
    actors: Sequence[XsensUsdActor],
    *,
    initial_visible: bool,
) -> Any | None:
    """Add one tennis-only control for every Xsens actor that has a racket body."""

    racket_frames = tuple(
        actor.body_frames[TENNIS_RACKET_BODY]
        for actor in actors
        if TENNIS_RACKET_BODY in actor.body_frames
    )
    if not racket_frames:
        return None

    for frame in racket_frames:
        frame.visible = bool(initial_visible)

    with server.gui.add_folder("Tennis", order=50.0):
        checkbox = server.gui.add_checkbox(
            "Show tennis racket",
            initial_value=initial_visible,
        )

    @checkbox.on_update
    def _(_evt) -> None:
        for frame in racket_frames:
            frame.visible = bool(checkbox.value)

    return checkbox


@dataclass(frozen=True)
class GroundPlaneBounds:
    """Dataset-derived horizontal bounds for the Viser ground grid."""

    center_xy: np.ndarray
    width: float
    height: float


@dataclass(frozen=True)
class InitialCameraView:
    """World-space camera pose selected from the displayed floating bases."""

    position: np.ndarray
    look_at: np.ndarray


def compute_initial_camera_view(
    base_positions_m: Sequence[np.ndarray],
    base_orientations_wxyz: Sequence[np.ndarray],
    *,
    minimum_distance_m: float = 3.0,
) -> InitialCameraView:
    """Frame all floating bases from the direction that they face.

    The local +x axis is treated as forward. When several actors are shown, the
    view targets their centroid, uses their mean heading, and moves farther away
    as their floating bases become more widely separated.
    """

    positions = np.asarray(base_positions_m, dtype=float)
    orientations = np.asarray(base_orientations_wxyz, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or positions.shape[0] == 0:
        raise ValueError("base_positions_m must contain at least one xyz position")
    if orientations.shape != (positions.shape[0], 4):
        raise ValueError("base_orientations_wxyz must contain one wxyz quaternion per position")
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise ValueError("Floating-base poses must contain only finite values")
    if minimum_distance_m <= 0.0:
        raise ValueError("minimum_distance_m must be positive")

    quaternion_norms = np.linalg.norm(orientations, axis=1)
    if np.any(quaternion_norms <= 1e-12):
        raise ValueError("Floating-base quaternions must have non-zero norm")
    quaternions = orientations / quaternion_norms[:, None]

    # First column of each quaternion rotation matrix: local +x in world space.
    w, x, y, z = quaternions.T
    forwards_xy = np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
        )
    )
    horizontal_norms = np.linalg.norm(forwards_xy, axis=1)
    usable = horizontal_norms > 1e-8
    if np.any(usable):
        headings = forwards_xy[usable] / horizontal_norms[usable, None]
        view_direction_xy = headings.mean(axis=0)
        if np.linalg.norm(view_direction_xy) <= 1e-8:
            # Opposing headings have no unique mean; keep the first actor's view.
            view_direction_xy = headings[0]
        else:
            view_direction_xy /= np.linalg.norm(view_direction_xy)
    else:
        view_direction_xy = np.array([1.0, 0.0])

    look_at = positions.mean(axis=0)
    horizontal_radius = float(np.max(np.linalg.norm(positions[:, :2] - look_at[:2], axis=1)))
    distance = max(float(minimum_distance_m), 2.0 * horizontal_radius + 2.0)
    position = look_at + np.array(
        [
            distance * view_direction_xy[0],
            distance * view_direction_xy[1],
            max(1.0, 0.35 * distance),
        ]
    )
    return InitialCameraView(position=position, look_at=look_at)


def compute_ground_plane_bounds(
    *,
    qpos: np.ndarray | None = None,
    xsens_positions_m: np.ndarray | None = None,
    robot_dof: int = 0,
    contains_object_in_qpos: bool = False,
    padding_m: float = 1.0,
    minimum_width_m: float | None = None,
    minimum_height_m: float | None = None,
) -> GroundPlaneBounds:
    """Compute a centered grid that covers every displayed motion position."""

    if padding_m < 0.0:
        raise ValueError("padding_m must be non-negative")
    if minimum_width_m is not None and minimum_width_m <= 0.0:
        raise ValueError("minimum_width_m must be positive when provided")
    if minimum_height_m is not None and minimum_height_m <= 0.0:
        raise ValueError("minimum_height_m must be positive when provided")

    point_sets: list[np.ndarray] = []
    if qpos is not None:
        robot_motion = np.asarray(qpos, dtype=float)
        if robot_motion.ndim != 2 or robot_motion.shape[0] == 0 or robot_motion.shape[1] < 3:
            raise ValueError("qpos must have shape [frames, values] and contain base xyz")
        point_sets.append(robot_motion[:, 0:2])
        if contains_object_in_qpos and robot_motion.shape[1] >= 7 + robot_dof + 7:
            point_sets.append(robot_motion[:, -7:-5])

    if xsens_positions_m is not None:
        xsens_positions = np.asarray(xsens_positions_m, dtype=float)
        if xsens_positions.ndim != 3 or xsens_positions.shape[-1] != 3 or xsens_positions.shape[0] == 0:
            raise ValueError("xsens_positions_m must have shape [frames, segments, 3]")
        point_sets.append(xsens_positions[..., 0:2].reshape(-1, 2))

    if not point_sets:
        raise ValueError("At least one motion dataset is required to calibrate the ground plane")
    points = np.concatenate(point_sets, axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise ValueError("Motion datasets contain no finite horizontal positions")

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    span = maximum - minimum
    width = float(span[0] + 2.0 * padding_m)
    height = float(span[1] + 2.0 * padding_m)
    if minimum_width_m is not None:
        width = max(width, float(minimum_width_m))
    if minimum_height_m is not None:
        height = max(height, float(minimum_height_m))
    return GroundPlaneBounds(center_xy=center, width=width, height=height)


def make_player(
    config: ViserConfig,
    qpos: np.ndarray | None = None,
    fps: int | None = None,
    xsens_motion: XsensHdf5Motion | None = None,
) -> viser.ViserServer:
    """Create a Viser player containing any configured combination of actors."""

    xsens_config = config if isinstance(config, XsensViserConfig) else None
    actor_modes = resolve_actor_modes(xsens_config.actor_modes if xsens_config is not None else ("robot",))
    show_robot = "robot" in actor_modes
    show_xsens = "xsens" in actor_modes or "g1_xsens" in actor_modes
    actor_offsets = resolve_actor_offsets(
        actor_modes,
        xsens_config.actor_spacing_m if xsens_config is not None else 0.0,
    )
    xsens_actor_offsets = {
        mode: offset for mode, offset in actor_offsets.items() if mode in {"xsens", "g1_xsens"}
    }
    record_output_path = resolve_record_output_path(config)
    if show_robot and qpos is None:
        raise ValueError(f"actor_modes={actor_modes} requires qpos motion for the robot actor")
    if show_xsens and (
        xsens_config is None or xsens_motion is None or xsens_config.xsens_hdf5 is None
    ):
        raise ValueError(f"actor_modes={actor_modes} requires an XsensViserConfig with --xsens-hdf5")

    server = viser.ViserServer()

    vr: ViserUrdf | None = None
    vo: ViserUrdf | None = None
    robot_root = None
    object_root = None
    robot_dof = 0
    robot_applier: QposViserApplier | None = None
    actual_robot_fps = float(fps if fps is not None else config.fps)

    if show_robot:
        robot_root = server.scene.add_frame("/robot", show_axes=False)
        robot_urdf_path = resolve_package_path(config.robot_urdf)
        robot_urdf = yourdfpy.URDF.load(
            str(robot_urdf_path),
            mesh_dir=str(robot_urdf_path.parent),
            load_meshes=True,
            build_scene_graph=True,
        )
        vr = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot")
        vr.show_visual = config.show_meshes
        robot_dof = len(vr.get_actuated_joint_limits())

        if config.object_urdf:
            object_root = server.scene.add_frame("/object", show_axes=False)
            object_urdf_path = resolve_package_path(config.object_urdf)
            object_urdf = yourdfpy.URDF.load(
                str(object_urdf_path),
                mesh_dir=str(object_urdf_path.parent),
                load_meshes=True,
                build_scene_graph=True,
            )
            vo = ViserUrdf(server, urdf_or_path=object_urdf, root_node_name="/object")
            vo.show_visual = config.show_meshes

        assert qpos is not None and robot_root is not None
        robot_applier = QposViserApplier(
            viser_robot=vr,
            robot_base_frame=robot_root,
            robot_dof=robot_dof,
            viser_object=vo if config.assume_object_in_qpos else None,
            object_base_frame=object_root if config.assume_object_in_qpos else None,
            contains_object_in_qpos=config.assume_object_in_qpos,
        )

    xsens_actors: dict[ActorMode, XsensUsdActor] = {}
    xsens_sampler: XsensMotionSampler | None = None
    g1_xsens_adapter: KinematicMorphologyAdapter | None = None
    g1_xsens_motion: KinematicMotion | None = None
    subject_reference_model = None
    if show_xsens:
        assert xsens_config is not None and xsens_motion is not None and xsens_config.xsens_hdf5 is not None
        xsens_sampler = XsensMotionSampler(xsens_motion)
        if "xsens" in actor_modes:
            subject_model_path = resolve_subject_xsens_usd(xsens_config.xsens_hdf5, xsens_config.xsens_usd)
            subject_model = load_xsens_usd_model(subject_model_path)
            validate_subject_xsens_usd(subject_model, xsens_config.xsens_hdf5)
            subject_reference_model = subject_model
            xsens_actors["xsens"] = XsensUsdActor(
                server,
                subject_model_path,
                model=subject_model,
                root_node_name="/xsens",
                show_meshes=xsens_config.show_xsens_meshes,
                show_landmarks=xsens_config.show_xsens_landmarks,
            )
        if "g1_xsens" in actor_modes:
            g1_model_path = resolve_g1_xsens_usd(xsens_config.g1_xsens_usd)
            g1_model = load_xsens_usd_model(g1_model_path)
            validate_g1_xsens_usd(g1_model)
            if subject_reference_model is None:
                subject_reference_model = build_subject_xsens_reference_model(
                    resolve_package_path(xsens_config.xsens_hdf5)
                )
            xsens_actors["g1_xsens"] = XsensUsdActor(
                server,
                g1_model_path,
                model=g1_model,
                root_node_name="/g1_xsens",
                show_meshes=xsens_config.show_xsens_meshes,
                show_landmarks=xsens_config.show_xsens_landmarks,
            )
            g1_xsens_adapter = build_xsens_morphology_adapter(
                g1_model,
                xsens_sampler.segment_names,
                grounding="none",
            )
            source_motion = KinematicMotion(
                xsens_sampler.segment_names,
                np.asarray(xsens_motion.positions_m, dtype=float),
                np.asarray(xsens_motion.quaternions_wijk, dtype=float),
                xsens_sampler.times_s.copy(),
            )
            g1_xsens_motion, root_motion_report = apply_xsens_root_motion(
                source_motion,
                g1_xsens_adapter.adapt_motion(source_motion),
                source_model=subject_reference_model,
                target_model=g1_model,
                grounding="match_lowest_soles",
                config=xsens_config.g1_xsens_root_motion,
            )
        for mode, actor in xsens_actors.items():
            actor.root.position = xsens_actor_offsets[mode]

    contains_object = bool(qpos is not None and robot_applier is not None and robot_applier.has_object_input(qpos))
    if qpos is not None and "robot" in actor_offsets:
        qpos = offset_qpos_positions(
            qpos,
            actor_offsets["robot"],
            has_object_input=contains_object,
        )
    xsens_ground_positions = None
    if show_xsens and xsens_motion is not None:
        xsens_ground_positions = np.concatenate(
            [
                xsens_motion.positions_m + xsens_actor_offsets[mode][None, None, :]
                for mode in xsens_actors
            ],
            axis=1,
        )
    ground = compute_ground_plane_bounds(
        qpos=qpos if show_robot else None,
        xsens_positions_m=xsens_ground_positions,
        robot_dof=robot_dof,
        contains_object_in_qpos=contains_object,
        padding_m=config.grid_padding,
        minimum_width_m=config.grid_width,
        minimum_height_m=config.grid_height,
    )
    server.scene.add_grid(
        "/grid",
        width=ground.width,
        height=ground.height,
        position=(float(ground.center_xy[0]), float(ground.center_xy[1]), 0.0),
    )
    print(
        f"[viser_player] ground center=({ground.center_xy[0]:.3f}, {ground.center_xy[1]:.3f}) | "
        f"extent=({ground.width:.3f}, {ground.height:.3f}) m"
    )
    for mode, offset in actor_offsets.items():
        print(f"[viser_player] {mode} root offset=({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}) m")
    if "g1_xsens" in xsens_actors:
        print(
            "[viser_player] g1_xsens "
            f"root_motion={root_motion_report.mode}, grounding=match_lowest_soles, "
            f"scale={root_motion_report.scale:.5f}, ground={root_motion_report.ground_height_m:.4f} m"
        )

    initial_base_positions: list[np.ndarray] = []
    initial_base_orientations: list[np.ndarray] = []
    if show_robot:
        assert qpos is not None
        initial_base_positions.append(np.asarray(qpos[0, 0:3], dtype=float))
        initial_base_orientations.append(np.asarray(qpos[0, 3:7], dtype=float))
    if xsens_actors:
        assert xsens_sampler is not None
        initial_xsens_pose = xsens_sampler.sample(float(xsens_sampler.times_s[0]))
        initial_g1_positions = None
        if "g1_xsens" in xsens_actors:
            assert g1_xsens_motion is not None
            initial_g1_positions = g1_xsens_motion.positions_m[0]
        for mode in xsens_actors:
            base_positions = (
                initial_g1_positions
                if mode == "g1_xsens"
                else initial_xsens_pose.positions_m
            )
            assert base_positions is not None
            initial_base_positions.append(
                np.asarray(base_positions[0], dtype=float) + xsens_actor_offsets[mode]
            )
            initial_base_orientations.append(
                np.asarray(initial_xsens_pose.quaternions_wxyz[0], dtype=float)
            )
    initial_camera = compute_initial_camera_view(initial_base_positions, initial_base_orientations)

    @server.on_client_connect
    def _set_initial_camera(client: viser.ClientHandle) -> None:
        client.camera.position = initial_camera.position
        client.camera.look_at = initial_camera.look_at
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    for connected_client in server.get_clients().values():
        _set_initial_camera(connected_client)

    with server.gui.add_folder("Camera", order=20.0):
        camera_follow = CameraFollowController(server, initial_enabled=config.camera_follow)

    robot_meshes_cb = None
    if vr is not None:
        with server.gui.add_folder("Robot", order=30.0):
            robot_meshes_cb = server.gui.add_checkbox(
                "Show robot meshes",
                initial_value=config.show_meshes,
            )

    xsens_display_controls = {}
    if xsens_config is not None and xsens_actors:
        with server.gui.add_folder("Xsens", order=40.0):
            for mode, actor in xsens_actors.items():
                label = "Xsens" if mode == "xsens" else "G1-proportioned Xsens"
                meshes_cb = server.gui.add_checkbox(
                    f"Show {label} meshes", initial_value=xsens_config.show_xsens_meshes
                )
                landmarks_cb = server.gui.add_checkbox(
                    f"Show {label} landmarks", initial_value=xsens_config.show_xsens_landmarks
                )
                xsens_display_controls[mode] = (actor, meshes_cb, landmarks_cb)

        add_tennis_racket_control(
            server,
            tuple(xsens_actors.values()),
            initial_visible=xsens_config.show_tennis_racket,
        )

    if robot_meshes_cb is not None:

        @robot_meshes_cb.on_update
        def _(_evt) -> None:
            assert vr is not None
            vr.show_visual = bool(robot_meshes_cb.value)
            if vo is not None:
                vo.show_visual = bool(robot_meshes_cb.value)

    for actor, meshes_cb, landmarks_cb in xsens_display_controls.values():

        @meshes_cb.on_update
        def _(_evt, actor=actor, checkbox=meshes_cb) -> None:
            actor.set_mesh_visibility(bool(checkbox.value))

        @landmarks_cb.on_update
        def _(_evt, actor=actor, checkbox=landmarks_cb) -> None:
            actor.set_landmark_visibility(bool(checkbox.value))

    if not show_xsens:
        assert qpos is not None and vr is not None and robot_root is not None and robot_applier is not None
        print(f"[viser_player] mode=robot | {qpos.shape[0]} frames | robot_dof={robot_dof}")

        def _apply_robot_frame(frame_idx: int) -> None:
            robot_applier.apply_frame(qpos, frame_idx)
            camera_follow.update_target(qpos[int(np.clip(frame_idx, 0, qpos.shape[0] - 1)), 0:3])

        if config.record_video:
            frame_indices = build_record_frame_indices(
                n_frames=int(qpos.shape[0]),
                start_frame=config.record_start_frame,
                end_frame=config.record_end_frame,
                stride=config.record_stride,
            )
            record_viser_sequence(
                server=server,
                apply_frame=_apply_robot_frame,
                frame_indices=frame_indices,
                output_path=record_output_path,
                width=config.record_width,
                height=config.record_height,
                fps=float(config.record_fps if config.record_fps is not None else actual_robot_fps),
                connect_timeout=config.record_connect_timeout,
                start_delay=config.record_start_delay,
                settle_time=config.record_settle_time,
                warmup_renders=config.record_warmup_renders,
                transport_format=config.record_transport_format,
            )
            if config.record_exit_after:
                raise SystemExit(0)

        create_motion_control_sliders(
            server=server,
            viser_robot=vr,
            robot_base_frame=robot_root,
            motion_sequence=qpos,
            robot_dof=robot_dof,
            viser_object=vo if config.assume_object_in_qpos else None,
            object_base_frame=object_root if config.assume_object_in_qpos else None,
            contains_object_in_qpos=config.assume_object_in_qpos,
            initial_fps=round(actual_robot_fps),
            initial_interp_mult=config.visual_fps_multiplier,
            initial_playback_speed=config.playback_speed,
            loop=config.loop,
            frame_times_s=np.arange(qpos.shape[0], dtype=float) / actual_robot_fps,
            on_pose_applied=lambda pose: camera_follow.update_target(pose[0:3]),
        )
        return server

    assert xsens_sampler is not None and xsens_actors
    master_times = xsens_sampler.times_s.copy()
    if show_robot:
        assert qpos is not None and robot_applier is not None
        robot_duration_s = max(0.0, (int(qpos.shape[0]) - 1) / actual_robot_fps)
        common_duration_s = min(xsens_sampler.duration_s, robot_duration_s)
        master_times = master_times[master_times <= common_duration_s + 1e-12]
        if master_times.size == 0:
            master_times = np.array([0.0])

    def _apply_time(time_s: float) -> None:
        xsens_pose = xsens_sampler.sample(time_s)
        g1_positions = None
        if "g1_xsens" in xsens_actors:
            assert g1_xsens_motion is not None
            lower, upper, weight = interpolation_window(g1_xsens_motion.times_s, time_s)
            g1_positions = (
                (1.0 - weight) * g1_xsens_motion.positions_m[lower]
                + weight * g1_xsens_motion.positions_m[upper]
            )

        avatar_positions: list[np.ndarray] = []
        for mode, actor in xsens_actors.items():
            actor_positions = g1_positions if mode == "g1_xsens" else xsens_pose.positions_m
            assert actor_positions is not None
            actor.apply_pose(
                xsens_sampler.segment_names,
                actor_positions,
                xsens_pose.quaternions_wxyz,
            )
            avatar_positions.append(actor_positions + xsens_actor_offsets[mode][None, :])
        robot_position = None
        if show_robot:
            assert qpos is not None and robot_applier is not None
            has_object = robot_applier.has_object_input(qpos)
            sampled_qpos = sample_qpos_at_time(
                qpos,
                time_s,
                fps=actual_robot_fps,
                robot_dof=robot_dof,
                has_object_input=has_object,
            )
            robot_applier.apply_qpos(sampled_qpos, has_object_input=has_object)
            robot_position = sampled_qpos[0:3]
        camera_follow.update_target(
            compute_camera_follow_target(
                robot_position_m=robot_position,
                avatar_positions_m=avatar_positions,
            )
        )

    playback_fps = _nominal_fps(master_times, config.fps)
    print(
        f"[viser_player] actors={','.join(actor_modes)} | {master_times.size} master frames | "
        f"duration={master_times[-1]:.3f}s | nominal_fps={playback_fps:.3f}"
    )
    if config.record_video:
        frame_indices = build_record_frame_indices(
            n_frames=int(master_times.size),
            start_frame=config.record_start_frame,
            end_frame=config.record_end_frame,
            stride=config.record_stride,
        )
        record_viser_sequence(
            server=server,
            apply_frame=lambda frame_idx: _apply_time(float(master_times[frame_idx])),
            frame_indices=frame_indices,
            output_path=record_output_path,
            width=config.record_width,
            height=config.record_height,
            fps=float(config.record_fps if config.record_fps is not None else playback_fps),
            connect_timeout=config.record_connect_timeout,
            start_delay=config.record_start_delay,
            settle_time=config.record_settle_time,
            warmup_renders=config.record_warmup_renders,
            transport_format=config.record_transport_format,
        )
        if config.record_exit_after:
            raise SystemExit(0)

    create_timed_motion_control_sliders(
        server,
        master_times,
        _apply_time,
        initial_fps=playback_fps,
        initial_interp_mult=config.visual_fps_multiplier,
        initial_playback_speed=config.playback_speed,
        loop=config.loop,
    )
    return server


def main(cfg: XsensViserConfig) -> None:
    actor_modes = resolve_actor_modes(cfg.actor_modes)
    qpos: np.ndarray | None = None
    fps: int | None = None
    xsens_motion: XsensHdf5Motion | None = None
    if "robot" in actor_modes:
        qpos, fps = load_npz(cfg.qpos_npz)
    if "xsens" in actor_modes or "g1_xsens" in actor_modes:
        if cfg.xsens_hdf5 is None:
            raise ValueError(f"actor_modes={actor_modes} requires --xsens-hdf5")
        xsens_motion = load_xsens_hdf5_motion(
            resolve_package_path(cfg.xsens_hdf5),
            target_fps=cfg.xsens_target_fps,
            frame_indices=cfg.xsens_frame_indices,
            include_tracked_props=True,
        )
    make_player(cfg, qpos=qpos, fps=fps, xsens_motion=xsens_motion)
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(XsensViserConfig))
