#!/usr/bin/env python3
"""
Visualize npz motion file using MuJoCo viewer.

This script loads a npz file and plays it back in MuJoCo viewer
to verify the motion is correct.
"""

import argparse
import time

import mujoco
import mujoco.viewer as mjv
import numpy as np


def visualize_npz(npz_path: str, robot_xml_path: str, loop: bool = True):
    """
    Visualize npz motion file in MuJoCo viewer.
    
    Args:
        npz_path: Path to npz file
        robot_xml_path: Path to MuJoCo robot XML file
        loop: Whether to loop the motion
    """
    # Load npz file
    print(f"Loading npz file: {npz_path}")
    data_npz = np.load(npz_path, allow_pickle=True)
    
    print(f"Keys: {list(data_npz.keys())}")
    
    fps = int(data_npz['fps'][0]) if 'fps' in data_npz else 30
    joint_pos = data_npz['joint_pos']  # (T, 36) = [xyz(3), wxyz(4), dof(29)]
    
    num_frames = joint_pos.shape[0]
    dt = 1.0 / fps
    duration = num_frames / fps
    
    print(f"FPS: {fps}")
    print(f"Number of frames: {num_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"joint_pos shape: {joint_pos.shape}")
    
    # Load MuJoCo model, patching floor texture to match main branch sim
    print(f"\nLoading MuJoCo model: {robot_xml_path}")
    import os, tempfile, shutil
    with open(robot_xml_path) as f:
        xml_text = f.read()
    # Replace the floor checker texture colors to match the main branch simulator
    xml_text = xml_text.replace(
        'rgb1=".2 .3 .4" rgb2=".1 .15 .2"',
        'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" mark="edge" markrgb="0.8 0.8 0.8"',
    )
    # Also update texrepeat for the floor material to match main branch (5 5)
    xml_text = xml_text.replace('texrepeat="1 1"', 'texrepeat="5 5"')
    # Write patched XML next to original so mesh paths resolve correctly
    xml_dir = os.path.dirname(os.path.abspath(robot_xml_path))
    patched_path = os.path.join(xml_dir, "_visualize_patched.xml")
    with open(patched_path, "w") as f:
        f.write(xml_text)
    try:
        model = mujoco.MjModel.from_xml_path(patched_path)
    finally:
        os.remove(patched_path)
    data_mj = mujoco.MjData(model)
    
    # Get DOF names from model
    dof_name_list = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_name_list.append(dof_name)
    
    print(f"MuJoCo model has {len(dof_name_list)} DOFs")
    print(f"DOF names: {dof_name_list[:5]}...")
    
    # Check if npz has joint_names
    if 'joint_names' in data_npz:
        joint_names_npz = data_npz['joint_names'].tolist()
        print(f"NPZ joint names: {joint_names_npz[:5]}...")
        
        # Create mapping
        dof_index_list = [joint_names_npz.index(dof_name) for dof_name in dof_name_list]
    else:
        print("Warning: No joint_names in npz, assuming 1:1 mapping")
        dof_index_list = list(range(len(dof_name_list)))
    
    # Mutable state for key callback
    playback_state = {"frame_idx": 0, "reset_requested": False}

    def key_callback(keycode):
        # Backspace = 259 in GLFW
        if keycode == 259:
            playback_state["reset_requested"] = True
            print("\n=== Reset requested (BACKSPACE) ===\n")

    # Launch viewer
    print("\nLaunching MuJoCo viewer...")
    print("Press BACKSPACE to reset, ESC to close")
    viewer = mjv.launch_passive(
        model, data_mj, show_left_ui=False, show_right_ui=False, key_callback=key_callback
    )
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0

    viewer.cam.distance = 3.0
    viewer.cam.elevation = -20.0
    viewer.cam.azimuth = 45.0

    # Playback loop
    frame_idx = 0
    print(f"\nPlaying motion... (Frame 0/{num_frames})")
    
    while viewer.is_running():
        start_time = time.perf_counter()

        # Handle reset (BACKSPACE)
        if playback_state["reset_requested"]:
            playback_state["reset_requested"] = False
            frame_idx = 0
            print(f"Playing motion... (Frame 0/{num_frames})")

        # Set robot state from motion data
        data_mj.qpos[:3] = joint_pos[frame_idx, :3]  # root position
        data_mj.qpos[3:7] = joint_pos[frame_idx, 3:7]  # root quaternion (wxyz)
        data_mj.qpos[7:] = joint_pos[frame_idx, 7:][dof_index_list]  # dof positions

        # Zero velocity (we're just playing back positions)
        data_mj.qvel[:] = 0

        # Forward kinematics
        mujoco.mj_forward(model, data_mj)
        viewer.sync()

        # Print progress every 30 frames
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}/{num_frames} ({frame_idx/num_frames*100:.1f}%)")

        # Advance frame
        frame_idx += 1
        if frame_idx >= num_frames:
            if loop:
                frame_idx = 0
                print("\n=== Looping motion ===\n")
            else:
                print("\n=== Motion playback complete ===")
                break

        # Sleep to maintain correct FPS
        end_time = time.perf_counter()
        sleep_time = max(0, dt - (end_time - start_time))
        time.sleep(sleep_time)
    
    viewer.close()
    print("\nViewer closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize npz motion file in MuJoCo")
    parser.add_argument("npz_path", type=str, help="Path to npz file")
    parser.add_argument("--robot-xml", type=str, default=None, help="Path to MuJoCo robot XML file")
    parser.add_argument("--no-loop", action="store_true", help="Play once and exit (default: loop)")
    
    args = parser.parse_args()
    
    # Auto-detect robot XML if not provided
    if args.robot_xml is None:
        robot_xml = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
        print(f"Using default robot XML: {robot_xml}")
        args.robot_xml = robot_xml
    
    visualize_npz(args.npz_path, args.robot_xml, loop=not args.no_loop)
