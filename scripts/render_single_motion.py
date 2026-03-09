"""Render a single .npz motion file with robot mesh using MuJoCo and save as mp4.

Usage:
    python scripts/render_single_motion.py \
        --npz g1_npz/humanml/000002_chunk_0000.npz \
        --output motion_render.mp4
"""

from __future__ import annotations

import argparse
import math
import os

import imageio
import mujoco
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Render single npz motion with MuJoCo")
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz motion file")
    parser.add_argument("--output", type=str, default="motion_render.mp4", help="Output mp4 path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cam-distance", type=float, default=3.5)
    parser.add_argument("--cam-elevation", type=float, default=-20.0, help="MuJoCo elevation (negative = above)")
    parser.add_argument("--cam-azimuth", type=float, default=135.0)
    parser.add_argument("--xml", type=str,
                        default="src/holosoma/holosoma/data/robots/g1/g1_29dof.xml",
                        help="MuJoCo XML model path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load motion
    print(f"Loading motion: {args.npz}")
    with np.load(args.npz) as data:
        fps = int(data["fps"])
        joint_pos_all = data["joint_pos"]  # (T, 36) = [xyz(3) + quat_wxyz(4) + dof(29)]
        body_pos_w = data["body_pos_w"]    # (T, B, 3)

    num_frames = joint_pos_all.shape[0]
    duration_s = num_frames / fps
    print(f"Motion: {num_frames} frames @ {fps}fps = {duration_s:.1f}s")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(args.xml)
    data_mj = mujoco.MjData(model)

    assert model.nq == joint_pos_all.shape[1], (
        f"Model nq={model.nq} != motion joint_pos cols={joint_pos_all.shape[1]}"
    )

    # Setup renderer
    renderer = mujoco.Renderer(model, args.height, args.width)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.cam_distance
    cam.elevation = args.cam_elevation
    cam.azimuth = args.cam_azimuth

    # Find pelvis body for camera tracking
    pelvis_idx = 1  # body index in npz (0=world, 1=pelvis)

    # Render frames
    print(f"Rendering {num_frames} frames...")
    frames = []

    for i in range(num_frames):
        # Set qpos directly from motion data
        # joint_pos_all[i] = [x, y, z, qw, qx, qy, qz, j0, j1, ..., j28]
        data_mj.qpos[:] = joint_pos_all[i]
        mujoco.mj_forward(model, data_mj)

        # Camera follows pelvis
        pelvis_pos = body_pos_w[i, pelvis_idx]
        cam.lookat[:] = [pelvis_pos[0], pelvis_pos[1], pelvis_pos[2] + 0.3]

        renderer.update_scene(data_mj, camera=cam, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels.copy())

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{num_frames}")

    # Save video
    print(f"Encoding to {args.output}...")
    writer = imageio.get_writer(args.output, fps=fps, codec="libx264",
                                output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"Saved: {args.output} ({num_frames} frames @ {fps}fps, {duration_s:.1f}s)")


if __name__ == "__main__":
    main()
