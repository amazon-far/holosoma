"""
Unified parallel processing script for retargeting all task types:
- robot_only: Robot-only retargeting with ground interaction (LAFAN)
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain (MOCAP)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

# Add src to path for direct execution
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import ParallelRetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402

# Import reusable functions from robot_retarget.py
from holosoma_retargeting.examples.robot_retarget import (  # type: ignore[import-not-found]  # noqa: E402
    DEFAULT_DATA_FORMATS,
    build_retargeter_kwargs_from_config,
    create_task_constants,
    initialize_robot_pose,
    load_motion_data,
    setup_object_data,
)

# Import after path modification
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,  # type: ignore[import-not-found]
)
from holosoma_retargeting.src.utils import (  # type: ignore[import-not-found]  # noqa: E402
    extract_foot_sticking_sequence_velocity,
    preprocess_motion_data,
)

# ----------------------------- Constants -----------------------------

# Override save directories for parallel processing (use demo_results_parallel instead of demo_results)
PARALLEL_SAVE_DIRS = {
    "robot_only": "demo_results_parallel/{robot}/robot_only/omomo",
    "object_interaction": "demo_results_parallel/{robot}/object_interaction/omomo",
    "climbing": "demo_results_parallel/{robot}/climbing/mocap_climb",
}


def find_files(data_dir: Path, data_format: str, object_name: str | None = None):
    """Find files based on data format.

    Args:
        data_dir: Directory to search for files
        data_format: Data format ("lafan", "smplh", "mocap")
        object_name: Optional object name to filter files (for smplh format)

    Returns:
        Sorted list of file paths
    """
    data_dir = Path(data_dir)

    if data_format == "lafan":
        # LAFAN: .npy files in root directory
        files = [str(p) for p in data_dir.glob("*.npy")]
        return sorted(files)
    if data_format == "smplh":
        # SMPLH/OMOMO: .pt files (optionally filtered by object_name)
        if object_name:
            files = [str(p) for p in data_dir.glob(f"*{object_name}*.pt")]
        else:
            files = [str(p) for p in data_dir.glob("*.pt")]
        return sorted(files)
    if data_format == "mocap":
        # MOCAP: .npy files in subdirectories
        files = [str(p) for p in data_dir.glob("*/*.npy")]
        return sorted(files)
    if data_format == "smplx":
        # SMPL-X: .npz files in root directory
        files = [str(p) for p in data_dir.glob("*.npz")]
        return sorted(files)
    # For other data format, default to be consistent with SMPL-X
    files = [str(p) for p in data_dir.glob("*.npz")]
    return sorted(files)


def generate_augmentation_configs(task_type: str, augmentation: bool = True):
    """Generate augmentation configurations based on task type."""
    if task_type == "robot_only":
        # No augmentation for robot_only
        return [{"name": "original"}]

    if task_type == "object_interaction":
        """Generate different augmentation configurations for object interaction."""
        augmentations = []
        augmentations.append({"name": "original", "translation": np.array([0.0, 0.0, 0.0]), "rotation": 0.0})

        if augmentation:
            # Translation augmentations
            translations = [
                [0.2, 0.0, 0.0],  # forward
                [0.0, 0.2, 0.0],  # left
                [0.0, -0.2, 0.0],  # right
            ]
            for i, trans in enumerate(translations):
                augmentations.append({"name": f"trans_{i}", "translation": np.array(trans), "rotation": 0.0})

            # Rotation augmentations
            rotations = [np.pi / 4, -np.pi / 4]
            for i, rot in enumerate(rotations):
                augmentations.append(
                    {
                        "name": f"rot_{i}",
                        "translation": np.array([0.0, 0.2 * (-1) ** i, 0.0]),
                        "rotation": rot,
                    }
                )

        return augmentations

    if task_type == "climbing":
        """Generate augmentation configurations for climbing (object scaling)."""
        configs = [{"name": "original", "scale": np.array([1, 1, 1])}]
        if augmentation:
            configs.extend(
                {"name": f"z_scale_{z_scale}", "scale": np.array([1, 1, z_scale])} for z_scale in [0.8, 0.9, 1.1, 1.2]
            )
        return configs

    raise ValueError(f"Invalid task type: {task_type}")


def extract_task_name(file_path):
    """Extract task name from file path."""
    return Path(file_path).stem


def process_single_task(args):
    """Process a single task with all augmentations.

    This function follows the same structure as main() in robot_retarget.py,
    but handles multiple augmentations in a loop for parallel processing.
    """
    (
        file_path,
        save_dir,
        task_type,
        data_format,
        robot_config,
        motion_data_config,
        task_config,
        retargeter,
        augmentation,
    ) = args

    os.makedirs(save_dir, exist_ok=True)
    if task_type == "climbing":
        file_path = "/".join(file_path.split("/")[:-1])
        task_name = extract_task_name(file_path)
    else:
        task_name = extract_task_name(file_path)
    print(f"Processing: {task_name}")

    # Task-specific object setup: set default object_dir for climbing if not provided
    if task_type == "climbing" and task_config.object_dir is None:
        from dataclasses import replace

        task_config = replace(task_config, object_dir=Path(file_path))

    constants = create_task_constants(robot_config, motion_data_config, task_config, task_type)

    # Load motion data
    human_joints, object_poses, smpl_scale = load_motion_data(
        task_type, data_format, Path(file_path).parent, task_name, constants, motion_data_config
    )

    # Preserve original data (preprocess_motion_data modifies them in place)
    human_joints_original = human_joints.copy()
    object_poses_original = object_poses.copy()

    # Get toe names from motion data config (depends only on data_format)
    toe_names = motion_data_config.toe_names

    # Process all augmentations
    augmentations = generate_augmentation_configs(task_type, augmentation)
    print("The number of augmentations: ", len(augmentations))

    for k, aug_config in enumerate(augmentations):
        # Use fresh copies for each iteration
        human_joints = human_joints_original.copy()
        object_poses = object_poses_original.copy()
        aug_name = aug_config["name"]
        file_name = f"{save_dir}/{task_name}_{aug_name}.npz"

        print(f"  Processing augmentation: {aug_name}")

        # Setup object data
        if task_type == "climbing":
            print("obejct_dir: ", task_config.object_dir)
            object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
                task_type,
                constants,
                task_config.object_dir,
                smpl_scale,
                task_config,
                augmentation=(k > 0),
                object_scale_augmented=aug_config["scale"],
            )
        else:
            object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
                task_type,
                constants,
                task_config.object_dir,
                smpl_scale,
                task_config,
                augmentation=(k > 0),
            )

        # Create retargeter
        retargeter_kwargs = build_retargeter_kwargs_from_config(retargeter, constants, object_urdf_path, task_type)
        retargeter = InteractionMeshRetargeter(**retargeter_kwargs)

        # Preprocess motion data
        if task_type == "robot_only":
            human_joints = preprocess_motion_data(human_joints, retargeter, toe_names, smpl_scale)
        elif task_type in {"object_interaction", "climbing"}:
            human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
                human_joints, retargeter, toe_names, scale=smpl_scale, object_poses=object_poses
            )

        # Extract foot sticking sequences
        foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
            human_joints, retargeter.demo_joints, toe_names
        )

        # Task-specific foot sticking adjustments
        if task_type == "object_interaction":
            # Disable initial sticking
            foot_sticking_sequences[0][toe_names[0]] = False
            foot_sticking_sequences[0][toe_names[1]] = False

        # Determine if this is an augmentation run (k > 0 means we're augmenting)
        is_augmentation_run = k > 0

        if task_type == "object_interaction":
            # Initialize robot pose
            q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
                task_type,
                data_format,
                human_joints,
                object_poses,
                constants,
                retargeter,
                task_config,
                is_augmentation_run,
                save_dir,
                task_name,
                augmentation_translation=aug_config["translation"],
                augmentation_rotation=aug_config["rotation"],
            )
        else:
            # Initialize robot pose
            q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
                task_type,
                data_format,
                human_joints,
                object_poses,
                constants,
                retargeter,
                task_config,
                is_augmentation_run,
                save_dir,
                task_name,
            )

        # Check if file exists and skip retargeting if it does (after setting up conditions)
        if Path.exists(Path(file_name)):
            continue

        # Retarget motion
        retargeted_motions, _, _, _ = retargeter.retarget_motion(
            human_joint_motions=human_joints,
            object_poses=object_poses,
            object_poses_augmented=object_poses_augmented,
            object_points_local_demo=object_local_pts_demo,
            object_points_local=object_local_pts,
            foot_sticking_sequences=foot_sticking_sequences,
            q_a_init=q_init,
            q_nominal_list=q_nominal,
            original=(k == 0),
            dest_res_path=file_name,
        )


def main(cfg: ParallelRetargetingConfig) -> None:
    """Main parallel retargeting pipeline.

    Args:
        cfg: Configuration arguments
    """
    robot = cfg.robot
    task_type = cfg.task_type

    # Set defaults based on task type
    data_format: str = cfg.data_format or DEFAULT_DATA_FORMATS[task_type]
    save_dir = cfg.save_dir if cfg.save_dir is not None else Path(PARALLEL_SAVE_DIRS[task_type].format(robot=robot))
    data_dir = cfg.data_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"Task type: {task_type}, Format: {data_format}")
    print(f"Data dir: {data_dir}, Save dir: {save_dir}")

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)

    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=robot)

    if task_type == "robot_only":
        files = find_files(data_dir, data_format)
    else:
        files = find_files(data_dir, data_format, cfg.task_config.object_name)
    print(f"Found {len(files)} files for task type: {task_type}")

    # Pass configs to worker processes
    process_args = [
        (
            file_path,
            save_dir,
            task_type,
            data_format,
            cfg.robot_config,
            cfg.motion_data_config,
            cfg.task_config,
            cfg.retargeter,
            cfg.augmentation,
        )
        for file_path in files
    ]

    # Set up parallel processing
    max_workers = cfg.max_workers or mp.cpu_count()
    print(f"Using {max_workers} parallel workers")

    start_time = time.time()
    successful = 0
    failed = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_task, arg): arg[0] for arg in process_args}

        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
                print(f"Completed: {file_path}")
                successful += 1
            except Exception as e:
                print(f"Failed {file_path}: {e}")
                import traceback

                traceback.print_exc()
                failed += 1

    end_time = time.time()

    print("\n=== Processing Summary ===")
    print(f"Task type: {task_type}")
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    if len(files) > 0:
        print(f"Average time per file: {(end_time - start_time) / len(files):.2f} seconds")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(ParallelRetargetingConfig)
    main(cfg)
