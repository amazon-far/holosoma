"""
Unified robot retargeting script for all task types:
- robot_only: Robot-only retargeting with ground interaction
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY, MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import (  # noqa: E402
    OrientationTrackingConfig,
    RetargeterConfig,
)
from holosoma_retargeting.config_types.retargeting import (  # noqa: E402
    RetargetingConfig,
    XsensMorphologyConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.data_utils.xsens_hdf5 import (  # noqa: E402
    XSENS_BODY_SEGMENT_NAMES,
    XsensHdf5Motion,
    load_xsens_hdf5_motion,
    resolve_xsens_hdf5_path,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,  # type: ignore[import-not-found]
)
from holosoma_retargeting.src.paths import DEMO_RESULTS_DIR  # noqa: E402
from holosoma_retargeting.src.utils import (  # noqa: E402
    augment_object_poses,
    calculate_scale_factor,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_urdf,
    create_scaled_multi_boxes_xml,
    estimate_human_orientation,
    extract_foot_sticking_sequence_velocity,
    extract_object_first_moving_frame,
    load_intermimic_data,
    load_object_data,
    preprocess_motion_data,
    transform_from_human_to_world,
    transform_y_up_to_z_up,
)
from holosoma_retargeting.xsens.morphology_adaptation import (  # noqa: E402
    adapt_xsens_motion_to_g1,
    adapt_xsens_tpose_to_g1,
)
from holosoma_retargeting.xsens.orientation_tracking import (  # noqa: E402
    XsensOrientationTargets,
    build_xsens_orientation_targets_from_calibration,
    describe_xsens_orientation_correspondences,
    load_xsens_orientation_targets,
)
from holosoma_retargeting.xsens.tennis_racket import (  # noqa: E402
    TennisRacketTargets,
    build_tennis_racket_targets,
    resolve_tennis_racket_attachment,
)
from holosoma_retargeting.xsens.tpose_calibration import (  # noqa: E402
    XsensTposeCalibrationConfig,
    solve_xsens_tpose_calibration,
    solve_xsens_tpose_calibration_from_data,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------- Constants -----------------------------

# Task-specific defaults
DEFAULT_DATA_FORMATS = {
    "robot_only": "smplh",
    "object_interaction": "smplh",
    "climbing": "mocap",
}

DATASET_OUTPUT_ALIASES = {
    # Keep the established output names for the bundled datasets while deriving
    # new dataset names from their input directories.
    "omomo_new": "omomo",
    "climb": "mocap_climb",
    "amass_smplx_processed": "amass_smplx",
}

RETARGETING_OUTPUT_FPS = 30.0


# Constants for numpy arrays (not in dataclass to avoid tyro parsing issues)
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_OBJECT_SCALE_NORMAL = np.array([1.0, 1.0, 1.0])
_AUGMENTATION_TRANSLATION = np.array([0.2, 0.0, 0.0])


# Type aliases
TaskType = Literal["robot_only", "object_interaction", "climbing"]
# DataFormat is imported from config_types.data_type


# ----------------------------- Helper Functions -----------------------------


def determine_save_dir(
    configured_save_dir: Path | None,
    *,
    results_root: Path,
    robot: str,
    task_type: str,
    data_path: Path,
    data_format: str,
) -> Path:
    """Resolve an explicit save directory or derive one from the input dataset.

    The inferred layout is ``<results_root>/<robot>/<task_type>/<dataset>``.
    ``dataset`` normally comes from the final component of ``data_path``; the
    data format is a fallback for paths without a usable final component.
    """
    if configured_save_dir is not None:
        return Path(configured_save_dir)

    dataset_dir = Path(data_path).name or data_format
    dataset_name = DATASET_OUTPUT_ALIASES.get(dataset_dir.casefold(), dataset_dir)
    return Path(results_root) / robot / task_type / dataset_name


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    task_config: TaskConfig,
    task_type: str,
) -> SimpleNamespace:
    """Create combined task constants from robot and motion data configs.

    Args:
        robot_config: Robot configuration
        motion_data_config: Motion data format configuration
        task_config: Task-specific configuration
        task_type: Type of task ("robot_only", "object_interaction", "climbing")

    Returns:
        SimpleNamespace with all task constants
    """
    task_constants = SimpleNamespace()

    # Copy all attributes from robot_config
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(task_constants, attr, getattr(robot_config, attr))

    # Copy legacy motion data constants (upper-case for compatibility)
    for attr, value in motion_data_config.legacy_constants().items():
        setattr(task_constants, attr, value)

    # Task-specific object setup
    if task_type == "robot_only":
        obj_name = task_config.object_name or "ground"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None
    elif task_type == "object_interaction":
        obj_name = task_config.object_name or "largebox"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = f"models/{obj_name}/{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = f"models/{obj_name}/{obj_name}.obj"
        task_constants.OBJECT_URDF_TEMPLATE = f"models/templates/{obj_name}.urdf.jinja"
    elif task_type == "climbing":
        obj_name = task_config.object_name or "multi_boxes"
        task_constants.OBJECT_NAME = obj_name
        object_dir = task_config.object_dir
        task_constants.OBJECT_DIR = str(object_dir) if object_dir else ""
        task_constants.OBJECT_URDF_FILE = str(object_dir / f"{obj_name}.urdf") if object_dir else f"{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = str(object_dir / f"{obj_name}.obj") if object_dir else f"{obj_name}.obj"
        task_constants.SCENE_XML_FILE = ""  # Will be set later

    return task_constants


def validate_config(cfg: RetargetingConfig) -> None:
    """Validate configuration consistency.

    Args:
        cfg: Configuration arguments

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate that data_format exists in registry (if provided)
    if cfg.data_format is not None and cfg.data_format not in DEMO_JOINTS_REGISTRY:
        available = ", ".join(sorted(DEMO_JOINTS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown data_format: '{cfg.data_format}'. "
            f"Available formats: {available}. "
            f"Add your format to DEMO_JOINTS_REGISTRY in config_types/data_type.py"
        )

    # Task-specific format requirements
    if cfg.task_type == "climbing" and cfg.data_format not in (None, "mocap"):
        raise ValueError("Climbing task requires 'mocap' data format")
    if cfg.task_type == "object_interaction" and cfg.data_format not in (None, "smplh"):
        raise ValueError("Object interaction requires 'smplh' data format")
    # robot_only accepts any format in the registry (already validated above)


def validate_xsens_morphology_selection(
    *,
    task_type: str,
    data_format: str,
    robot: str,
    config: XsensMorphologyConfig,
) -> None:
    """Fail early when G1 morphology adaptation is selected outside its contract."""

    if data_format != "xsens":
        return
    if config.mode == "direct":
        if config.root_motion.mode != "preserve_world":
            raise ValueError(
                "Non-default Xsens root-motion modes require xsens_morphology.mode='g1_proportioned'"
            )
        return
    if task_type != "robot_only" or robot != "g1":
        raise ValueError(
            "G1-proportioned Xsens morphology requires task_type='robot_only', data_format='xsens', and robot='g1'"
        )


def resolve_xsens_g1_model_path(
    robot_config: RobotConfig,
    morphology_config: XsensMorphologyConfig,
) -> Path:
    """Return the MuJoCo model used by all G1 morphology preparation paths."""

    if morphology_config.g1_model_path is not None:
        return morphology_config.g1_model_path
    return Path(robot_config.ROBOT_URDF_FILE).with_suffix(".xml")


def prepare_xsens_motion_for_retargeting(
    motion: XsensHdf5Motion,
    *,
    hdf5_path: Path,
    direct_scale: float,
    robot_config: RobotConfig,
    morphology_config: XsensMorphologyConfig,
) -> tuple[np.ndarray, float]:
    """Return optimizer positions and the remaining uniform preprocessing scale."""

    body_indices = [motion.segment_names.index(name) for name in XSENS_BODY_SEGMENT_NAMES]
    body_motion = XsensHdf5Motion(
        positions_m=np.asarray(motion.positions_m[:, body_indices], dtype=float),
        times_s=np.asarray(motion.times_s, dtype=float),
        stream_name=motion.stream_name,
        segment_names=list(XSENS_BODY_SEGMENT_NAMES),
        source_indices=[motion.source_indices[index] for index in body_indices],
        quaternions_wijk=np.asarray(motion.quaternions_wijk[:, body_indices], dtype=float),
        orientation_stream_name=motion.orientation_stream_name,
    )

    if morphology_config.mode == "direct":
        return np.asarray(body_motion.positions_m, dtype=float).copy(), direct_scale

    g1_model_path = resolve_xsens_g1_model_path(robot_config, morphology_config)
    adapted = adapt_xsens_motion_to_g1(
        body_motion,
        hdf5_path=hdf5_path,
        g1_model_path=g1_model_path,
        grounding=morphology_config.grounding,
        root_motion=morphology_config.root_motion,
        preserve_joint_offsets=morphology_config.preserve_joint_offsets,
    )
    logger.info(
        "Adapted Xsens motion to G1 proportions "
        "(root_motion=%s, grounding=%s, preserve_joint_offsets=%s)",
        morphology_config.root_motion.mode,
        morphology_config.grounding,
        morphology_config.preserve_joint_offsets,
    )
    return adapted.positions_m, 1.0


def create_ground_points(x_range: tuple[float, float], y_range: tuple[float, float], size: int) -> np.ndarray:
    """Create ground point meshgrid.

    Args:
        x_range: (min, max) x-coordinate range
        y_range: (min, max) y-coordinate range
        size: Number of points per dimension

    Returns:
        (N, 3) array of ground points
    """
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)


def load_motion_data(
    task_type: TaskType,
    data_format: str,
    data_path: Path,
    task_name: str,
    constants: SimpleNamespace,
    motion_data_config: MotionDataConfig,
) -> tuple[np.ndarray, np.ndarray, float, XsensHdf5Motion | None]:
    """Load motion data based on task type and format.

    Args:
        task_type: Type of task
        data_format: Data format ("lafan", "smplh", "mocap")
        data_path: Path to data directory
        task_name: Name of the task/sequence
        constants: Task constants
        motion_data_config: Motion data configuration

    Returns:
        Tuple of (human_joints, object_poses, smpl_scale, xsens_motion)
        - human_joints: (T, J, 3) array of joint positions
        - object_poses: (T, 7) array of object poses [qw, qx, qy, qz, x, y, z]
        - smpl_scale: Scaling factor for SMPL compatibility

    Raises:
        FileNotFoundError: If required data files are not found
    """
    logger.info("Loading motion data for task: %s, format: %s", task_name, data_format)

    xsens_motion: XsensHdf5Motion | None = None
    if task_type == "robot_only":
        if data_format == "lafan":
            npy_path = data_path / f"{task_name}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"LAFAN data file not found: {npy_path}")

            human_joints = np.load(str(npy_path))
            human_joints = transform_y_up_to_z_up(human_joints)
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            # LAFAN-specific spine adjustment
            human_joints[:, spine_joint_idx, -1] -= 0.06
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        elif data_format == "smplh":  # smplh
            pt_path = data_path / f"{task_name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")

            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
        elif data_format == "mocap":
            downsample = 4
            npy_file = data_path / f"{task_name}.npy"
            if not npy_file.exists():
                raise FileNotFoundError(f"MOCAP data file not found: {npy_file}")

            human_joints = np.load(str(npy_file))[::downsample]

            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height
        elif data_format == "smplx":
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height
        elif data_format == "xsens":
            hdf5_path = resolve_xsens_hdf5_path(data_path, task_name)
            target_fps = motion_data_config.target_fps if motion_data_config.target_fps is not None else None
            if target_fps is None:
                target_fps = RETARGETING_OUTPUT_FPS
            xsens_motion = load_xsens_hdf5_motion(
                hdf5_path,
                target_fps=target_fps,
                frame_start=motion_data_config.frame_start,
                max_frames=motion_data_config.max_frames,
                frame_indices=motion_data_config.frame_indices,
                include_tracked_props=True,
            )
            human_joints = xsens_motion.positions_m[:, : len(XSENS_BODY_SEGMENT_NAMES)]

            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height
        else:
            # For other custom data format, if it uses consistent .npz file like SMPLX,
            # you can use the same logic as SMPLX.
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height

        # Create dummy object poses for robot_only
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))

    elif task_type == "object_interaction":
        pt_path = data_path / f"{task_name}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")

        human_joints, object_poses = load_intermimic_data(str(pt_path))
        smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)

    elif task_type == "climbing":
        task_dir = data_path / task_name
        npy_files = list(task_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {task_dir}")

        npy_file = npy_files[0]
        # MOCAP-specific downsample factor
        downsample = 4
        human_joints = np.load(str(npy_file))[::downsample]
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))
        default_human_height = motion_data_config.default_human_height or 1.78
        smpl_scale = constants.ROBOT_HEIGHT / default_human_height

    logger.debug(
        "Loaded %d frames, scale factor: %.4f",
        human_joints.shape[0],
        smpl_scale,
    )
    return human_joints, object_poses, smpl_scale, xsens_motion


def setup_object_data(
    task_type: TaskType,
    constants: SimpleNamespace,
    object_dir: Path | None,
    smpl_scale: float,
    task_config: TaskConfig,
    augmentation: bool,
    object_scale_augmented: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Setup object-specific data (ground, object mesh, climbing terrain).
    Args:
        task_type: Type of task
        constants: Task constants
        object_dir: Object directory path (for climbing)
        smpl_scale: SMPL scaling factor
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        object_scale_augmented: Scale factor for augmented objects (default: [1.0, 1.0, 1.2])
    Returns:
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path)
    """
    object_scale_normal = np.array([1.0, 1.0, 1.0])
    if object_scale_augmented is None:
        object_scale_augmented = np.array([1.0, 1.0, 1.2])  # For climbing task augmentation
    logger.info("Setting up object data for task: %s", task_type)

    if task_type == "robot_only":
        # Create ground points meshgrid
        ground_pts = create_ground_points(task_config.ground_range, task_config.ground_range, task_config.ground_size)
        return ground_pts, ground_pts, None

    if task_type == "object_interaction":
        # Load object data
        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")

        object_local_pts, object_local_pts_demo = load_object_data(
            constants.OBJECT_MESH_FILE, smpl_scale=smpl_scale, sample_count=100
        )
        return object_local_pts, object_local_pts_demo, constants.OBJECT_URDF_FILE

    if task_type == "climbing":
        if object_dir is None:
            raise ValueError("object_dir must be provided for climbing task")

        # Setup climbing-specific object
        box_asset_xml = object_dir / "box_assets.xml"
        scene_xml_name = Path(constants.ROBOT_URDF_FILE).name.replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml")
        scene_xml_file = object_dir / scene_xml_name
        # Set SCENE_XML_FILE in constants BEFORE creating retargeter (needed for temp_retargeter)
        constants.SCENE_XML_FILE = str(scene_xml_file)

        np.random.seed(0)
        print("object mesh file: ", constants.OBJECT_MESH_FILE)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: (
                task_config.surface_weight_high
                if p[2] > task_config.surface_weight_threshold
                else task_config.surface_weight_low
            ),
            sample_count=100,
        )

        if augmentation:
            ground_pts = create_ground_points(
                task_config.climbing_ground_range, task_config.climbing_ground_range, task_config.climbing_ground_size
            )
            object_local_pts_demo = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
            object_scale = object_scale_augmented
            object_local_pts = object_scale * object_local_pts_demo
        else:
            object_scale = object_scale_normal
            object_local_pts_demo = object_local_pts_demo_original
            object_local_pts = object_local_pts_demo

        # Create scaled URDF and XML files
        scale_factors = tuple(float(value) for value in (object_scale * smpl_scale))
        object_urdf_file = create_scaled_multi_boxes_urdf(constants.OBJECT_URDF_FILE, scale_factors)
        object_asset_xml_path = create_scaled_multi_boxes_xml(str(box_asset_xml), scale_factors)
        new_scene_xml_path = create_new_scene_xml_file(str(scene_xml_file), scale_factors, object_asset_xml_path)
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file

    raise ValueError(f"Unknown task type: {task_type}")


def _compute_q_init_base(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter | None = None,
) -> np.ndarray:
    """Compute base robot pose initialization (q_init_base).
    This is a shared helper function used by both single and parallel processing.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses in format [qw, qx, qy, qz, x, y, z]
        constants: Task constants
        retargeter: Optional retargeter instance (needed for climbing)
    Returns:
        q_init_base in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints
    """
    if task_type == "robot_only":
        if data_format in {"lafan", "xsens"}:
            root_joint_name = "Spine1" if data_format == "lafan" else "Pelvis"
            root_joint_idx = constants.DEMO_JOINTS.index(root_joint_name)
            human_quat_init = estimate_human_orientation(human_joints, constants.DEMO_JOINTS)
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate(
                [human_joints[0, root_joint_idx, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)]
            )
        else:  # smplh
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "object_interaction":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "climbing":
        if retargeter is None:
            raise ValueError("retargeter is required for climbing task")
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        spine_joint_idx = retargeter.demo_joints.index("Spine1")
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate(
            [
                human_joints[0, spine_joint_idx],
                human_quat_init,
                np.zeros(constants.ROBOT_DOF),
            ]
        )
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return q_init_base


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert object poses from [qw, qx, qy, qz, x, y, z] to MuJoCo order [x, y, z, qw, qx, qy, qz].
    Args:
        object_poses: Object poses array of shape (T, 7) in format [qw, qx, qy, qz, x, y, z]
    Returns:
        Object poses array in MuJoCo order [x, y, z, qw, qx, qy, qz]
    """
    return object_poses[:, [4, 5, 6, 0, 1, 2, 3]]


def build_retargeter_kwargs_from_config(
    retargeter_config: RetargeterConfig,
    constants: SimpleNamespace,
    object_urdf_path: str | None,
    task_type: str,
) -> dict:
    """Build kwargs for InteractionMeshRetargeter from a RetargeterConfig.
    This is a convenience function that allows building kwargs directly from
    a RetargeterConfig without needing a full RetargetingConfig.
    Args:
        retargeter_config: Retargeter configuration
        constants: Task constants
        object_urdf_path: Path to object URDF file
        task_type: Type of task
    Returns:
        Dictionary of kwargs for InteractionMeshRetargeter
    """
    kwargs = {
        "task_constants": constants,
        "object_urdf_path": object_urdf_path,
        "q_a_init_idx": retargeter_config.q_a_init_idx,
        "activate_joint_limits": retargeter_config.activate_joint_limits,
        "activate_obj_non_penetration": retargeter_config.activate_obj_non_penetration,
        "activate_foot_sticking": retargeter_config.activate_foot_sticking,
        "foot_lock": retargeter_config.foot_lock,
        "penetration_tolerance": retargeter_config.penetration_tolerance,
        "foot_sticking_tolerance": retargeter_config.foot_sticking_tolerance,
        "self_collision": retargeter_config.self_collision,
        "orientation": retargeter_config.orientation,
        "step_size": retargeter_config.step_size,
        "initial_iterations": retargeter_config.initial_iterations,
        "iterations_per_frame": retargeter_config.iterations_per_frame,
        "visualize": retargeter_config.visualize,
        "debug": retargeter_config.debug,
        "w_nominal_tracking_init": retargeter_config.w_nominal_tracking_init,
    }
    if task_type == "climbing":
        kwargs["nominal_tracking_tau"] = retargeter_config.nominal_tracking_tau
    return kwargs


def load_orientation_targets_for_retargeting(
    *,
    orientation_config: OrientationTrackingConfig,
    robot_config: RobotConfig,
    robot: str,
    data_format: str,
    task_type: str,
    xsens_motion: XsensHdf5Motion | None,
    hdf5_path: Path | None,
    morphology_config: XsensMorphologyConfig,
) -> XsensOrientationTargets | None:
    """Load optional Xsens orientation/axis targets for retargeting."""
    if not orientation_config.enable:
        return None
    if data_format != "xsens" or task_type != "robot_only":
        raise ValueError("Orientation-aware retargeting currently supports only robot_only Xsens data")
    if xsens_motion is None:
        raise ValueError("Loaded Xsens motion is required for orientation-aware retargeting")
    if orientation_config.calibration_path is not None:
        return load_xsens_orientation_targets(
            calibration_path=orientation_config.calibration_path,
            motion_quaternions_wijk=xsens_motion.quaternions_wijk,
            segment_names=xsens_motion.segment_names,
        )
    if hdf5_path is None:
        raise ValueError("The source Xsens HDF5 path is required for automatic orientation calibration")

    calibration_config = XsensTposeCalibrationConfig(
        robot_type=robot,
        robot_urdf_file=robot_config.ROBOT_URDF_FILE,
        verbose=0,
    )
    if morphology_config.mode == "g1_proportioned":
        g1_model_path = resolve_xsens_g1_model_path(robot_config, morphology_config)
        logger.info(
            "Calibrating Xsens-to-G1 segment-frame orientation offsets from the G1-proportioned "
            "recording T-pose (position scale=1.0, grounding=%s, preserve_joint_offsets=%s)",
            morphology_config.grounding,
            morphology_config.preserve_joint_offsets,
        )
        calibration_tpose = adapt_xsens_tpose_to_g1(
            hdf5_path=hdf5_path,
            g1_model_path=g1_model_path,
            grounding=morphology_config.grounding,
            preserve_joint_offsets=morphology_config.preserve_joint_offsets,
        )
        calibration = solve_xsens_tpose_calibration_from_data(
            calibration_tpose,
            config=calibration_config,
            position_scale_factor=1.0,
        )
    else:
        logger.info(
            "Calibrating Xsens-to-G1 segment-frame orientation offsets from the direct human-sized recording T-pose"
        )
        calibration = solve_xsens_tpose_calibration(
            hdf5_path,
            config=calibration_config,
        )
    return build_xsens_orientation_targets_from_calibration(
        calibration,
        motion_quaternions_wijk=xsens_motion.quaternions_wijk,
        segment_names=xsens_motion.segment_names,
    )


def resolve_orientation_tracking_config(
    *,
    retargeter_config: RetargeterConfig,
    morphology_config: XsensMorphologyConfig,
    data_format: str,
    task_type: str,
    robot: str,
) -> RetargeterConfig:
    """Enable calibrated orientations for the default G1-proportioned Xsens path."""
    racket_requested = retargeter_config.orientation.tennis_racket.mode != "hand"
    if racket_requested and (task_type != "robot_only" or data_format != "xsens" or robot != "g1"):
        raise ValueError("Tennis-racket orientation modes support only robot_only G1 Xsens retargeting")
    auto_enable = (
        task_type == "robot_only"
        and data_format == "xsens"
        and robot == "g1"
        and morphology_config.mode == "g1_proportioned"
        and morphology_config.track_orientations
    ) or racket_requested
    if not auto_enable or retargeter_config.orientation.enable:
        return retargeter_config
    return replace(
        retargeter_config,
        orientation=replace(retargeter_config.orientation, enable=True),
    )


def describe_retargeting_setup(
    *,
    retargeter: InteractionMeshRetargeter,
    orientation_targets: XsensOrientationTargets | None,
    q_nominal_list: np.ndarray | None,
) -> tuple[str, ...]:
    """Describe the active optimizer objectives, constraints, and rotation mappings."""

    positional_pairs = ", ".join(f"{source}->{target}" for source, target in retargeter.laplacian_match_links.items())
    regularized_dofs = int(np.count_nonzero(np.asarray(retargeter.Q_diag, dtype=float)))
    nominal_active = (
        q_nominal_list is not None
        and retargeter.w_nominal_tracking_init > 0.0
        and len(retargeter.track_nominal_indices) > 0
    )
    nominal_status = "present" if q_nominal_list is not None else "absent"
    foot_sticking_active = retargeter.activate_foot_sticking and retargeter.q_a_init_idx < 12
    self_collision_active = bool(
        retargeter._self_collision_config is not None and retargeter._self_collision_config.enable
    )

    lines = [
        "Retargeting optimization setup:",
        "  Objectives:",
        (
            "    [active] interaction-mesh positional/relational tracking "
            f"(weight={retargeter.laplacian_weights}, mapped anchors={len(retargeter.laplacian_match_links)}): "
            f"{positional_pairs}"
        ),
        f"    [active] temporal smoothness (weight={retargeter.smooth_weight})",
        (
            f"    [{'active' if regularized_dofs else 'inactive'}] joint regularization "
            f"(weighted DoFs={regularized_dofs})"
        ),
        (
            f"    [{'active' if nominal_active else 'inactive'}] nominal-pose tracking "
            f"(weight={retargeter.w_nominal_tracking_init}, nominal trajectory={nominal_status})"
        ),
    ]

    if orientation_targets is None:
        lines.extend(
            [
                "    [inactive] full segment-orientation tracking",
                "    [inactive] segment-axis direction tracking",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "    [active] full segment-orientation tracking "
                    f"(weight={retargeter.orientation_config.orientation_weight}, "
                    f"mappings={len(orientation_targets.orientation_names)})"
                ),
                (
                    "    [active] segment-axis direction tracking "
                    f"(weight={retargeter.orientation_config.axis_weight}, "
                    f"axes={len(orientation_targets.axis_names)}): " + ", ".join(orientation_targets.axis_names)
                ),
            ]
        )

    lines.extend(
        [
            "  Hard constraints:",
            f"    [{'active' if retargeter.activate_joint_limits else 'inactive'}] joint limits",
            (
                f"    [{'active' if retargeter.activate_obj_non_penetration else 'inactive'}] "
                "ground/object non-penetration"
            ),
            f"    [{'active' if foot_sticking_active else 'inactive'}] detected foot sticking",
            f"    [{'active' if retargeter.foot_lock.enable else 'inactive'}] explicit foot-lock windows",
            f"    [{'active' if self_collision_active else 'inactive'}] self-collision avoidance",
            f"    [active] SQP trust region (step_size={retargeter.step_size})",
            "  Rotational correspondence:",
        ]
    )

    if orientation_targets is None:
        lines.append("    none; robot link orientations are unconstrained by Xsens segment rotations")
        return tuple(lines)

    lines.extend(
        f"    {line}"
        for line in describe_xsens_orientation_correspondences(
            orientation_targets.orientation_names,
            orientation_targets.orientation_robot_link_names,
            orientation_targets.orientation_offsets_wijk,
        )
    )
    return tuple(lines)


def log_retargeting_setup(
    *,
    retargeter: InteractionMeshRetargeter,
    orientation_targets: XsensOrientationTargets | None,
    q_nominal_list: np.ndarray | None,
) -> None:
    """Log the active optimizer setup once before motion retargeting."""

    for line in describe_retargeting_setup(
        retargeter=retargeter,
        orientation_targets=orientation_targets,
        q_nominal_list=q_nominal_list,
    ):
        logger.info("%s", line)


def initialize_robot_pose(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter,
    task_config: TaskConfig,
    augmentation: bool,
    save_dir: Path,
    task_name: str,
    augmentation_translation: np.ndarray | None = None,
    augmentation_rotation: float | None = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize robot pose (q_init, q_nominal) based on task.
    Returns qpos in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints.
    Object poses are returned in MuJoCo order: [0:3] position, [3:7] quaternion.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses (assumed to be in format: [quat, pos] or [pos, quat])
        constants: Task constants
        retargeter: Retargeter instance
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        save_dir: Save directory path
        task_name: Task name
        augmentation_translation: Translation vector for augmentation (default: [0.2, 0.0, 0.0])
    Returns:
        Tuple of (q_init, q_nominal, object_poses_augmented, human_joints_modified, object_poses_modified)
        where qpos is in MuJoCo order and object_poses are in MuJoCo order
    """
    # Use default if not provided
    if augmentation_translation is None:
        augmentation_translation = _AUGMENTATION_TRANSLATION
    logger.info("Initializing robot pose")

    if task_type == "robot_only":
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    if task_type == "object_interaction":
        if augmentation:
            object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
            object_poses_augmented = augment_object_poses(
                object_poses,
                object_moving_frame_idx,
                human_joints[0, 0, :],
                augmentation_translation,
                augmentation_rotation,
            )
            # Convert object_poses to MuJoCo order
            object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
            object_poses = convert_object_poses_to_mujoco_order(object_poses)

            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            return q_nominal[0], q_nominal, object_poses_augmented, human_joints, object_poses
        object_poses_augmented = object_poses.copy()
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
        return q_init, None, object_poses_augmented, human_joints, object_poses

    if task_type == "climbing":
        if augmentation:
            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            # Convert object_poses to MuJoCo order
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            return q_nominal[0], q_nominal, object_poses, human_joints, object_poses
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, retargeter)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    raise ValueError(f"Unknown task type: {task_type}")


def determine_output_path(
    task_type: TaskType,
    save_dir: Path,
    task_name: str,
    augmentation: bool,
) -> str:
    """Determine output file path based on task and augmentation.
    Args:
        task_type: Type of task
        save_dir: Save directory path
        task_name: Task name
        augmentation: Whether this is an augmentation run
    Returns:
        Output file path
    """
    if task_type == "robot_only":
        return str(save_dir / f"{task_name}.npz")
    if task_type in ("object_interaction", "climbing"):
        suffix = "_augmented" if augmentation else "_original"
        return str(save_dir / f"{task_name}{suffix}.npz")
    raise ValueError(f"Unknown task type: {task_type}")


# ----------------------------- Main -----------------------------


def main(cfg: RetargetingConfig) -> None:
    """Main retargeting pipeline.
    Args:
        cfg: Configuration arguments
    """
    # Validate configuration
    validate_config(cfg)

    robot = cfg.robot
    task_name = cfg.task_name
    task_type = cfg.task_type

    # Set defaults based on task type
    data_format: str = cfg.data_format or DEFAULT_DATA_FORMATS[task_type]
    data_path = cfg.data_path
    save_dir = determine_save_dir(
        cfg.save_dir,
        results_root=DEMO_RESULTS_DIR,
        robot=robot,
        task_type=task_type,
        data_path=data_path,
        data_format=data_format,
    )
    validate_xsens_morphology_selection(
        task_type=task_type,
        data_format=data_format,
        robot=robot,
        config=cfg.xsens_morphology,
    )
    retargeter_config = resolve_orientation_tracking_config(
        retargeter_config=cfg.retargeter,
        morphology_config=cfg.xsens_morphology,
        data_format=data_format,
        task_type=task_type,
        robot=robot,
    )

    os.makedirs(save_dir, exist_ok=True)
    logger.info("Task: %s, Type: %s, Format: %s", task_name, task_type, data_format)
    logger.info("Data path: %s, Save dir: %s", data_path, save_dir)

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)

    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = replace(cfg.motion_data_config, data_format=data_format, robot_type=robot)

    # Task-specific object setup: set default object_dir for climbing if not provided
    if task_type == "climbing" and cfg.task_config.object_dir is None:
        cfg.task_config = replace(cfg.task_config, object_dir=data_path / task_name)

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=task_type,
    )

    # Load motion data
    human_joints, object_poses, smpl_scale, xsens_motion = load_motion_data(
        task_type, data_format, data_path, task_name, constants, cfg.motion_data_config
    )
    hdf5_path = None
    if xsens_motion is not None:
        hdf5_path = resolve_xsens_hdf5_path(data_path, task_name)
        human_joints, smpl_scale = prepare_xsens_motion_for_retargeting(
            xsens_motion,
            hdf5_path=hdf5_path,
            direct_scale=smpl_scale,
            robot_config=cfg.robot_config,
            morphology_config=cfg.xsens_morphology,
        )
    orientation_targets = load_orientation_targets_for_retargeting(
        orientation_config=retargeter_config.orientation,
        robot_config=cfg.robot_config,
        robot=robot,
        data_format=data_format,
        task_type=task_type,
        xsens_motion=xsens_motion,
        hdf5_path=hdf5_path,
        morphology_config=cfg.xsens_morphology,
    )
    tennis_racket_targets: TennisRacketTargets | None = None
    if (
        xsens_motion is not None
        and orientation_targets is not None
        and "RightHandSword" in xsens_motion.segment_names
    ):
        assert hdf5_path is not None
        racket_attachment = resolve_tennis_racket_attachment(
            retargeter_config.orientation.tennis_racket,
            motion=xsens_motion,
            hdf5_path=hdf5_path,
        )
        tennis_racket_targets = build_tennis_racket_targets(xsens_motion, racket_attachment)
        logger.info(
            "Tennis-racket tracking: mode=%s, attachment=%s",
            retargeter_config.orientation.tennis_racket.mode,
            racket_attachment.calibration_source,
        )
    if (
        orientation_targets is not None
        and orientation_targets.orientation_target_rotations.shape[0] != human_joints.shape[0]
    ):
        raise ValueError(
            "Orientation target frame count does not match loaded motion data: "
            f"{orientation_targets.orientation_target_rotations.shape[0]} vs {human_joints.shape[0]}"
        )

    # Get toe names from motion data config (depends only on data_format)
    toe_names = cfg.motion_data_config.toe_names

    # Setup object data
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        task_type,
        constants,
        cfg.task_config.object_dir,
        smpl_scale,
        cfg.task_config,
        cfg.augmentation,
        object_scale_augmented=_OBJECT_SCALE_AUGMENTED,
    )

    # Create retargeter
    retargeter_kwargs = build_retargeter_kwargs_from_config(
        retargeter_config,
        constants,
        object_urdf_path,
        task_type,
    )
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)
    logger.info("Retargeter created")

    # Preprocess motion data
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(human_joints, retargeter, toe_names, smpl_scale)
    elif task_type in {"object_interaction", "climbing"}:
        human_joints, object_poses, _object_moving_frame_idx = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            scale=smpl_scale,
            object_poses=object_poses,
        )

    # Initialize robot pose
    q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
        task_type,
        data_format,
        human_joints,
        object_poses,
        constants,
        retargeter,
        cfg.task_config,
        cfg.augmentation,
        save_dir,
        task_name,
        augmentation_translation=_AUGMENTATION_TRANSLATION,
    )

    # Extract foot sticking sequences
    foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
        human_joints,
        retargeter.demo_joints,
        toe_names,
        frame_times_s=xsens_motion.times_s if xsens_motion is not None else None,
    )

    # Task-specific foot sticking adjustments
    if task_type == "object_interaction":
        # Disable initial sticking
        foot_sticking_sequences[0][toe_names[0]] = False
        foot_sticking_sequences[0][toe_names[1]] = False

    # Determine output path
    dest_res_path = determine_output_path(task_type, save_dir, task_name, cfg.augmentation)

    # Retarget motion
    log_retargeting_setup(
        retargeter=retargeter,
        orientation_targets=orientation_targets,
        q_nominal_list=q_nominal,
    )
    logger.info("Starting retargeting...")
    retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_poses=object_poses,
        object_poses_augmented=object_poses_augmented,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=q_nominal,
        orientation_targets=orientation_targets,
        tennis_racket_targets=tennis_racket_targets,
        original=not cfg.augmentation,
        dest_res_path=dest_res_path,
    )
    logger.info("Retargeting complete. Results saved to: %s", dest_res_path)

    if cfg.retargeter.debug:
        input("Press Enter to exit ...")


if __name__ == "__main__":
    cfg = tyro.cli(RetargetingConfig)
    main(cfg)
