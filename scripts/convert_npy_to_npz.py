"""Convert .npy motion files (PhUMA format) to .npz format compatible with MotionLoader.

Each .npy file contains a dictionary with:
  - root_trans: (T, 3) root translation
  - root_ori: (T, 4) root orientation quaternion (xyzw)
  - dof_pos: (T, 29) joint positions
  - fps: int

Output .npz format (matching existing MotionLoader expectations):
  - fps: int
  - joint_pos: (T, 7+29) = (T, 36)  [root_pos(3), root_quat_wxyz(4), dof_pos(29)]
  - joint_vel: (T, 6+29) = (T, 35)  [root_lin_vel(3), root_ang_vel(3), dof_vel(29)]
  - body_pos_w: (T, B, 3)
  - body_quat_w: (T, B, 4) in wxyz
  - body_lin_vel_w: (T, B, 3)
  - body_ang_vel_w: (T, B, 3)
  - joint_names: list of str
  - body_names: list of str

Usage:
    python scripts/convert_npy_to_npz.py --input_dir g1/ --output_dir g1_npz/ --xml_path src/holosoma/holosoma/data/robots/g1/g1_29dof.xml
    python scripts/convert_npy_to_npz.py --input_dir g1/ --output_dir g1_npz/ --xml_path src/holosoma/holosoma/data/robots/g1/g1_29dof.xml --max_workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np


def world_body_velocities(model: mujoco.MjModel, data: mujoco.MjData):
    """Per-body COM velocities in the world frame."""
    v = np.zeros((model.nbody, 6))
    for b in range(model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, v[b], 0)
    lin_w = v[:, 3:6]
    ang_w = v[:, 0:3]
    return lin_w, ang_w


def convert_single_file(
    npy_path: str,
    npz_path: str,
    xml_path: str,
    target_fps: int,
    min_frames: int,
) -> tuple[bool, str]:
    """Convert a single .npy file to .npz. Returns (success, message)."""
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        root_trans = data["root_trans"]  # (T, 3)
        root_ori_xyzw = data["root_ori"]  # (T, 4) xyzw
        dof_pos = data["dof_pos"]  # (T, 29)
        fps = int(data["fps"])

        T = root_trans.shape[0]
        if T < min_frames:
            return False, f"Skipped (too short: {T} frames)"

        # Convert root_ori from xyzw to wxyz for MuJoCo
        root_ori_wxyz = root_ori_xyzw[:, [3, 0, 1, 2]]

        # Load MuJoCo model (each process gets its own copy)
        model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(model)

        # Get joint and body names
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        joint_names = []
        for i in range(model.njnt):
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

        # Prepare storage
        all_joint_pos = np.zeros((T, model.nq), dtype=np.float64)
        all_joint_vel = np.zeros((T, model.nv), dtype=np.float64)
        all_body_pos_w = np.zeros((T, model.nbody, 3), dtype=np.float64)
        all_body_quat_w = np.zeros((T, model.nbody, 4), dtype=np.float64)
        all_body_lin_vel_w = np.zeros((T, model.nbody, 3), dtype=np.float64)
        all_body_ang_vel_w = np.zeros((T, model.nbody, 3), dtype=np.float64)

        dt = 1.0 / fps

        for t in range(T):
            # Set qpos: [root_pos(3), root_quat_wxyz(4), dof_pos(29)]
            mj_data.qpos[:3] = root_trans[t]
            mj_data.qpos[3:7] = root_ori_wxyz[t]
            mj_data.qpos[7:36] = dof_pos[t]

            # Compute joint velocities via finite differences
            if t == 0:
                # Forward difference
                next_trans = root_trans[1] if T > 1 else root_trans[0]
                mj_data.qvel[:3] = (next_trans - root_trans[t]) / dt
                next_dof = dof_pos[1] if T > 1 else dof_pos[0]
                mj_data.qvel[6:35] = (next_dof - dof_pos[t]) / dt
            elif t == T - 1:
                # Backward difference
                mj_data.qvel[:3] = (root_trans[t] - root_trans[t - 1]) / dt
                mj_data.qvel[6:35] = (dof_pos[t] - dof_pos[t - 1]) / dt
            else:
                # Central difference
                mj_data.qvel[:3] = (root_trans[t + 1] - root_trans[t - 1]) / (2 * dt)
                mj_data.qvel[6:35] = (dof_pos[t + 1] - dof_pos[t - 1]) / (2 * dt)

            # Angular velocity via finite differences on quaternions
            if t == 0:
                q0 = root_ori_wxyz[0]
                q1 = root_ori_wxyz[min(1, T - 1)]
            elif t == T - 1:
                q0 = root_ori_wxyz[t - 1]
                q1 = root_ori_wxyz[t]
            else:
                q0 = root_ori_wxyz[t - 1]
                q1 = root_ori_wxyz[t + 1]

            # Compute angular velocity from quaternion difference
            # dq = q1 * conj(q0), omega = 2 * log(dq) / dt_diff
            dt_diff = dt if (t == 0 or t == T - 1) else 2 * dt
            q0_conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
            # Hamilton product q1 * q0_conj (wxyz convention)
            aw, ax, ay, az = q1
            bw, bx, by, bz = q0_conj
            dq = np.array([
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ])
            # Normalize
            dq = dq / (np.linalg.norm(dq) + 1e-12)
            # Ensure shortest path
            if dq[0] < 0:
                dq = -dq
            # Extract axis-angle
            w_comp = np.clip(dq[0], -1.0, 1.0)
            angle = 2.0 * np.arccos(w_comp)
            sin_half = np.sqrt(max(0.0, 1.0 - w_comp * w_comp))
            if sin_half > 1e-8:
                axis = dq[1:] / sin_half
            else:
                axis = np.zeros(3)
            mj_data.qvel[3:6] = axis * angle / dt_diff

            # Run forward kinematics
            mujoco.mj_forward(model, mj_data)

            # Record data
            all_joint_pos[t] = mj_data.qpos[:].copy()
            all_joint_vel[t] = mj_data.qvel[:].copy()
            all_body_pos_w[t] = mj_data.xpos[:].copy()
            all_body_quat_w[t] = mj_data.xquat[:].copy()  # wxyz

            lin_vel_w, ang_vel_w = world_body_velocities(model, mj_data)
            all_body_lin_vel_w[t] = lin_vel_w
            all_body_ang_vel_w[t] = ang_vel_w

        # Resample to target_fps if needed
        if fps != target_fps:
            # Simple nearest-neighbor resampling
            duration = (T - 1) / fps
            new_T = int(duration * target_fps) + 1
            old_times = np.linspace(0, duration, T)
            new_times = np.linspace(0, duration, new_T)
            indices = np.searchsorted(old_times, new_times, side="right") - 1
            indices = np.clip(indices, 0, T - 1)

            all_joint_pos = all_joint_pos[indices]
            all_joint_vel = all_joint_vel[indices]
            all_body_pos_w = all_body_pos_w[indices]
            all_body_quat_w = all_body_quat_w[indices]
            all_body_lin_vel_w = all_body_lin_vel_w[indices]
            all_body_ang_vel_w = all_body_ang_vel_w[indices]
            output_fps = target_fps
        else:
            output_fps = fps

        # Save
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(
            npz_path,
            fps=np.array([output_fps]),
            joint_pos=all_joint_pos.astype(np.float32),
            joint_vel=all_joint_vel.astype(np.float32),
            body_pos_w=all_body_pos_w.astype(np.float32),
            body_quat_w=all_body_quat_w.astype(np.float32),
            body_lin_vel_w=all_body_lin_vel_w.astype(np.float32),
            body_ang_vel_w=all_body_ang_vel_w.astype(np.float32),
            joint_names=np.array(joint_names),
            body_names=np.array(body_names),
        )
        return True, f"OK ({all_joint_pos.shape[0]} frames @ {output_fps}fps)"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Convert .npy motion files to .npz format")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npz files")
    parser.add_argument(
        "--xml_path",
        type=str,
        default="src/holosoma/holosoma/data/robots/g1/g1_29dof.xml",
        help="Path to MuJoCo XML model",
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target FPS for output (default: 50)")
    parser.add_argument("--min_frames", type=int, default=30, help="Minimum number of frames (default: 30)")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Collect all .npy files
    npy_files = sorted(input_dir.rglob("*.npy"))
    print(f"Found {len(npy_files)} .npy files in {input_dir}")

    if not npy_files:
        print("No .npy files found, exiting.")
        return

    # Build task list: (npy_path, npz_path)
    tasks = []
    for npy_path in npy_files:
        rel = npy_path.relative_to(input_dir)
        npz_path = output_dir / rel.with_suffix(".npz")
        if npz_path.exists():
            continue
        tasks.append((str(npy_path), str(npz_path)))

    print(f"{len(tasks)} files to convert ({len(npy_files) - len(tasks)} already exist)")

    if not tasks:
        print("Nothing to do.")
        return

    success_count = 0
    fail_count = 0
    skip_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                convert_single_file,
                npy_path,
                npz_path,
                args.xml_path,
                args.target_fps,
                args.min_frames,
            ): (npy_path, npz_path)
            for npy_path, npz_path in tasks
        }

        for i, future in enumerate(as_completed(futures)):
            npy_path, npz_path = futures[future]
            ok, msg = future.result()
            if ok:
                success_count += 1
            elif msg.startswith("Skipped"):
                skip_count += 1
            else:
                fail_count += 1
                print(f"  FAIL: {npy_path}: {msg}")

            if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(
                    f"  [{i+1}/{len(tasks)}] "
                    f"ok={success_count} skip={skip_count} fail={fail_count} "
                    f"({rate:.1f} files/s)"
                )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s: {success_count} converted, {skip_count} skipped, {fail_count} failed")


if __name__ == "__main__":
    main()
