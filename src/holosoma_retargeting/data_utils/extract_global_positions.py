#!/usr/bin/env python3
"""
Simple script to extract global positions from LAFAN dataset BVH files.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from lafan1 import extract, utils  # type: ignore[import-not-found]


def extract_global_positions(bvh_file_path):
    """
    Extract global positions from a BVH file.

    Args:
        bvh_file_path (str): Path to the BVH file

    Returns:
        dict: Dictionary containing:
            - 'positions': numpy array of shape (frames, joints, 3) with global positions
            - 'joint_names': list of joint names
            - 'parents': list of parent indices
            - 'num_frames': number of frames
            - 'num_joints': number of joints
    """
    # Read BVH file
    anim = extract.read_bvh(bvh_file_path)

    # Compute global positions using Forward Kinematics
    global_quats, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    return {
        "positions": global_positions / 100,
        "joint_names": anim.bones,
        "parents": anim.parents,
        "num_frames": global_positions.shape[0],
        "num_joints": global_positions.shape[1],
    }


def save_global_positions_to_npy(global_positions, output_path):
    """
    Save global positions to a .npy file.

    Args:
        global_positions (numpy.ndarray): Global positions array
        output_path (str): Output file path
    """
    np.save(output_path, global_positions)
    print(f"Saved global positions to: {output_path}")


@dataclass
class Config:
    """Configuration for extracting global positions from BVH files."""

    input_dir: str = "./lafan1/lafan"
    output_dir: str = "../demo_data/lafan"


def main(cfg: Config):
    """
    Main function to extract global positions from BVH files.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {cfg.input_dir} not found!")
        print("Please run the evaluation script first to generate BVH files.")
        return

    # Get list of BVH files
    bvh_files = [f.name for f in input_dir.iterdir() if f.is_file() and f.suffix == ".bvh"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each BVH file
    for bvh_file in bvh_files:  # Process first 3 files to avoid memory issues
        print(f"\nProcessing: {bvh_file}")

        bvh_path = input_dir / bvh_file

        # Extract global positions
        result = extract_global_positions(str(bvh_path))

        print(f"  Frames: {result['num_frames']}")
        print(f"  Joints: {result['num_joints']}")
        print(f"  Joint names: {result['joint_names']}")

        # Save to .npy file
        output_npy = output_dir / f"{bvh_file[:-4]}.npy"
        np.save(str(output_npy), result["positions"])


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
