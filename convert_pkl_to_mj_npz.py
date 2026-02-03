#!/usr/bin/env python3
"""
Convert pkl motion file to full npz format using MuJoCo FK.

This script:
1. Loads pkl file (with numpy compatibility fix)
2. Converts to qpos format with optional FPS interpolation
3. Uses MuJoCo forward kinematics to compute body positions/quaternions
4. Saves as complete npz file ready for training

Usage:
    python convert_pkl_to_mj_npz.py input.pkl output.npz --input-fps 30 --output-fps 50
"""

import argparse
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F


# Fix numpy._core compatibility
class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core':
            module = 'numpy.core'
        elif module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        return super().find_class(module, name)


def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        return NumpyCompatUnpickler(f).load()


def interpolate_quaternions_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-8):
    """
    SLERP interpolation for quaternions (xyzw format).
    
    Args:
        q0: (N, 4) quaternions
        q1: (N, 4) quaternions
        t: (N,) blend values in [0, 1]
    
    Returns:
        (N, 4) interpolated quaternions
    """
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    
    # Make sure t has trailing dim for broadcasting
    if t.ndim == q0.ndim - 1:
        t = t.unsqueeze(-1)
    
    # Antipodal fix (take shortest path)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    
    # If very close, fall back to lerp
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    close = sin_theta.abs() < eps
    
    s0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    s1 = torch.sin(t * theta) / (sin_theta + eps)
    out = s0 * q0 + s1 * q1
    
    # Linear fallback for tiny angles
    out = torch.where(close, (1.0 - t) * q0 + t * q1, out)
    return F.normalize(out, dim=-1)


def interpolate_motion(root_pos, root_rot, dof_pos, input_fps, output_fps, device='cpu'):
    """
    Interpolate motion data from input_fps to output_fps.
    
    Args:
        root_pos: (T, 3) numpy array
        root_rot: (T, 4) numpy array (xyzw)
        dof_pos: (T, N) numpy array
        input_fps: input frames per second
        output_fps: output frames per second
        device: torch device
    
    Returns:
        Tuple of (root_pos_interp, root_rot_interp, dof_pos_interp) as numpy arrays
    """
    # Convert to torch
    root_pos_t = torch.from_numpy(root_pos).to(torch.float32).to(device)
    root_rot_t = torch.from_numpy(root_rot).to(torch.float32).to(device)
    dof_pos_t = torch.from_numpy(dof_pos).to(torch.float32).to(device)
    
    input_frames = root_pos.shape[0]
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (input_frames - 1) * input_dt
    
    # Generate output timestamps
    times = torch.arange(0, duration, output_dt, device=device, dtype=torch.float32)
    output_frames = times.shape[0]
    
    print(f"Interpolating: {input_frames} frames @ {input_fps}fps → {output_frames} frames @ {output_fps}fps")
    print(f"Duration: {duration:.2f} seconds")
    
    # Compute frame blend
    phase = times / duration
    index_0 = (phase * (input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(input_frames - 1, device=device))
    blend = phase * (input_frames - 1) - index_0
    
    # Interpolate positions and dof (linear)
    root_pos_interp = (1 - blend.unsqueeze(1)) * root_pos_t[index_0] + blend.unsqueeze(1) * root_pos_t[index_1]
    dof_pos_interp = (1 - blend.unsqueeze(1)) * dof_pos_t[index_0] + blend.unsqueeze(1) * dof_pos_t[index_1]
    
    # Interpolate rotations (slerp)
    root_rot_interp = interpolate_quaternions_slerp(root_rot_t[index_0], root_rot_t[index_1], blend)
    
    # Convert back to numpy
    return (
        root_pos_interp.cpu().numpy(),
        root_rot_interp.cpu().numpy(),
        dof_pos_interp.cpu().numpy()
    )


def convert_pkl_to_mj_npz(
    pkl_path: str,
    output_path: str,
    robot_xml_path: str,
    input_fps: int = 30,
    output_fps: int = 30,
    device: str = 'cpu',
):
    """
    Convert pkl to full npz format using MuJoCo FK.
    
    Args:
        pkl_path: Path to input pkl file
        output_path: Path to output npz file
        robot_xml_path: Path to MuJoCo robot XML file
        input_fps: Input FPS from pkl file
        output_fps: Desired output FPS
        device: torch device for interpolation
    """
    # Load pkl file
    print(f"Loading pkl file: {pkl_path}")
    data = load_pkl(pkl_path)
    
    # Extract data
    root_pos = data['root_pos']  # (T, 3)
    root_rot_xyzw = data['root_rot']  # (T, 4) xyzw
    dof_pos = data['dof_pos']  # (T, 29)
    dof_names = data['dof_names']
    actual_input_fps = data.get('fps', input_fps)
    
    T = root_pos.shape[0]
    num_dof = dof_pos.shape[1]
    
    print(f"Input FPS: {actual_input_fps}")
    print(f"Number of frames: {T}")
    print(f"Duration: {T / actual_input_fps:.2f} seconds")
    print(f"Number of DOFs: {num_dof}")
    
    # Convert root_rot from xyzw to wxyz for MuJoCo
    root_rot_wxyz = np.zeros_like(root_rot_xyzw)
    root_rot_wxyz[:, 0] = root_rot_xyzw[:, 3]  # w
    root_rot_wxyz[:, 1] = root_rot_xyzw[:, 0]  # x
    root_rot_wxyz[:, 2] = root_rot_xyzw[:, 1]  # y
    root_rot_wxyz[:, 3] = root_rot_xyzw[:, 2]  # z
    
    # Interpolate if needed
    if output_fps != actual_input_fps:
        print(f"\nInterpolating from {actual_input_fps}fps to {output_fps}fps...")
        root_pos, root_rot_wxyz, dof_pos = interpolate_motion(
            root_pos, root_rot_wxyz, dof_pos, actual_input_fps, output_fps, device
        )
        T = root_pos.shape[0]
    
    # Load MuJoCo model
    print(f"\nLoading MuJoCo model: {robot_xml_path}")
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    data_mj = mujoco.MjData(model)
    
    # Get DOF mapping
    dof_name_list = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_name_list.append(dof_name)
    
    print(f"MuJoCo model has {len(dof_name_list)} DOFs")
    dof_index_list = [dof_names.index(dof_name) for dof_name in dof_name_list]
    
    # Get body names
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    joint_names_out = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    
    print(f"MuJoCo model has {model.nbody} bodies")
    
    # Prepare output arrays
    joint_pos = np.zeros((T, 7 + num_dof), dtype=np.float32)
    joint_vel = np.zeros((T, 6 + num_dof), dtype=np.float32)
    body_pos_w = np.zeros((T, model.nbody, 3), dtype=np.float64)
    body_quat_w = np.zeros((T, model.nbody, 4), dtype=np.float64)
    body_lin_vel_w = np.zeros((T, model.nbody, 3), dtype=np.float64)
    body_ang_vel_w = np.zeros((T, model.nbody, 3), dtype=np.float64)
    
    # Process each frame using MuJoCo FK
    print(f"\nComputing forward kinematics for {T} frames...")
    dt = 1.0 / output_fps
    
    for t in range(T):
        if t % 100 == 0:
            print(f"  Frame {t}/{T}")
        
        # Set qpos
        data_mj.qpos[:3] = root_pos[t]
        data_mj.qpos[3:7] = root_rot_wxyz[t]
        data_mj.qpos[7:] = dof_pos[t, dof_index_list]
        
        # Set qvel to zero for this frame (will compute velocities later)
        data_mj.qvel[:] = 0
        
        # Run forward kinematics
        mujoco.mj_forward(model, data_mj)
        
        # Store joint_pos
        joint_pos[t, :3] = root_pos[t]
        joint_pos[t, 3:7] = root_rot_wxyz[t]
        joint_pos[t, 7:] = dof_pos[t]
        
        # Store body poses
        body_pos_w[t] = data_mj.xpos
        body_quat_w[t] = data_mj.xquat  # wxyz format
        
    # Compute velocities using finite differences
    print("\nComputing velocities...")
    
    # Joint velocities
    joint_vel[:, :3] = np.gradient(root_pos, dt, axis=0)  # root linear velocity
    
    # Root angular velocity (approximation from quaternion)
    for t in range(T):
        if t == 0:
            joint_vel[t, 3:6] = (joint_vel[1, 3:6] if T > 1 else 0)
        elif t == T - 1:
            joint_vel[t, 3:6] = joint_vel[t-1, 3:6]
        else:
            # Simple finite difference
            joint_vel[t, 3:6] = 0  # Placeholder
    
    # DOF velocities
    joint_vel[:, 6:] = np.gradient(dof_pos, dt, axis=0)
    
    # Body velocities
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)
    
    # Body angular velocities (placeholder)
    body_ang_vel_w[:] = 0
    
    # Save to npz
    print(f"\nSaving npz file: {output_path}")
    np.savez(
        output_path,
        fps=np.array([output_fps], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_names=np.array(joint_names_out, dtype=str),
        body_names=np.array(body_names, dtype=str),
    )
    
    print("\n✓ Conversion complete!")
    print(f"Output shapes:")
    print(f"  joint_pos: {joint_pos.shape}")
    print(f"  joint_vel: {joint_vel.shape}")
    print(f"  body_pos_w: {body_pos_w.shape}")
    print(f"  body_quat_w: {body_quat_w.shape}")
    print(f"  body_lin_vel_w: {body_lin_vel_w.shape}")
    print(f"  body_ang_vel_w: {body_ang_vel_w.shape}")
    print(f"  FPS: {output_fps}")
    print(f"  Duration: {T / output_fps:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pkl motion file to full npz format using MuJoCo FK")
    parser.add_argument("pkl_path", type=str, help="Path to input pkl file")
    parser.add_argument("output_path", type=str, help="Path to output npz file")
    parser.add_argument("--robot-xml", type=str, default=None, help="Path to MuJoCo robot XML file")
    parser.add_argument("--input-fps", type=int, default=30, help="Input FPS (default: 30)")
    parser.add_argument("--output-fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for interpolation (default: cpu)")
    
    args = parser.parse_args()
    
    # Auto-detect robot XML if not provided
    if args.robot_xml is None:
        # Assume g1-29dof by default
        robot_xml = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
        print(f"Using default robot XML: {robot_xml}")
        args.robot_xml = robot_xml
    
    convert_pkl_to_mj_npz(
        args.pkl_path,
        args.output_path,
        args.robot_xml,
        args.input_fps,
        args.output_fps,
        args.device,
    )
