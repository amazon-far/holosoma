#!/usr/bin/env python3
"""Convert LAFAN1 Retargeting Dataset CSV motion data to qpos-format .npz.

Supports H1 (26 cols), G1 (36 cols), H1_2, T1, etc.

CSV column order: root_x, root_y, root_z, root_qx, root_qy, root_qz, root_qw, joint_0, ..., joint_N

MuJoCo qpos order:  [x, y, z, qw, qx, qy, qz, joint_0, ..., joint_N]

Usage:
  python convert_csv_to_qpos.py --input motion.csv --output motion.npz --fps 30
  python convert_csv_to_qpos.py --input-dir ./csvs/ --output-dir ./npzs/ --fps 30
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tyro
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    input: str = ""
    """Path to a single CSV file."""

    input_dir: str = ""
    """Directory containing CSV files (batch mode)."""

    output: str = ""
    """Output .npz path (single file mode)."""

    output_dir: str = ""
    """Output directory (batch mode)."""

    fps: int = 30
    """Frames per second (LAFAN1 is 30 fps)."""


def csv_to_qpos(csv_path: str, output_path: str, fps: int) -> None:
    """Convert a single CSV file to qpos-format .npz.

    Works for any robot: auto-detects DOF count from column count.
    CSV must have 7 root columns (x,y,z,qx,qy,qz,qw) + N joint columns.
    """
    print(f"Reading: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    num_frames, num_cols = data.shape
    num_joints = num_cols - 7

    robot_type = {19: "H1", 23: "T1", 29: "G1"}.get(num_joints, f"{num_joints}DOF")
    print(f"Detected: {num_frames} frames, {num_cols} cols → {num_joints} joints ({robot_type})")

    # Root: columns 0-2 (pos), 3-6 (quat qx,qy,qz,qw)
    root_pos = data[:, 0:3]
    root_quat_csv = data[:, 3:7]
    joint_angles = data[:, 7:]

    # Reorder quaternion: CSV (qx,qy,qz,qw) → MuJoCo (qw,qx,qy,qz)
    root_quat_mj = np.zeros_like(root_quat_csv)
    root_quat_mj[:, 0] = root_quat_csv[:, 3]  # qw
    root_quat_mj[:, 1] = root_quat_csv[:, 0]  # qx
    root_quat_mj[:, 2] = root_quat_csv[:, 1]  # qy
    root_quat_mj[:, 3] = root_quat_csv[:, 2]  # qz

    # MuJoCo qpos: [x, y, z, qw, qx, qy, qz, joints...]
    qpos = np.concatenate([root_pos, root_quat_mj, joint_angles], axis=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, qpos=qpos, fps=fps)
    print(f"Saved: {output_path}  |  frames={num_frames}, fps={fps}, "
          f"duration={num_frames/fps:.1f}s, joints={num_joints}")


def main(cfg: Config) -> None:
    if cfg.input and cfg.output:
        csv_to_qpos(cfg.input, cfg.output, cfg.fps)

    elif cfg.input_dir and cfg.output_dir:
        input_dir = Path(cfg.input_dir)
        output_dir = Path(cfg.output_dir)
        csv_files = sorted(input_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return

        print(f"Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            csv_to_qpos(str(csv_file), str(output_dir / (csv_file.stem + ".npz")), cfg.fps)
        print(f"Done! {len(csv_files)} files → {output_dir}")

    else:
        print("Usage:")
        print("  Single: python convert_csv_to_qpos.py --input x.csv --output x.npz")
        print("  Batch:  python convert_csv_to_qpos.py --input-dir ./csvs/ --output-dir ./npzs/")


if __name__ == "__main__":
    main(tyro.cli(Config))
