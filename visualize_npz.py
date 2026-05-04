#!/usr/bin/env python3
"""
Visualize npz motion file using MuJoCo viewer.

This script loads a npz file and plays it back in MuJoCo viewer
to verify the motion is correct.

Usage:
$PYTHON visualize_npz.py output.npz                    # Unitree G1                                                                                                                                                                                                                           
$PYTHON visualize_npz.py output.npz --robot-xml path/to/robot.xml  # Other Robot                                                                                                                                                                                                                          
$PYTHON visualize_npz.py output.npz --no-loop           # Single Loop
"""

import argparse
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer as mjv
import numpy as np


def _write_scene_stl(vertices, faces, stl_path):
    """Write a binary STL file from vertices and faces."""
    num_faces = faces.shape[0]
    with open(stl_path, 'wb') as f:
        f.write(b'\x00' * 80)  # header
        f.write(np.uint32(num_faces).tobytes())
        for i in range(num_faces):
            v0 = vertices[faces[i, 0]]
            v1 = vertices[faces[i, 1]]
            v2 = vertices[faces[i, 2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal /= norm_len
            else:
                normal = np.array([0, 0, 1], dtype=np.float32)
            f.write(normal.astype(np.float32).tobytes())
            f.write(v0.astype(np.float32).tobytes())
            f.write(v1.astype(np.float32).tobytes())
            f.write(v2.astype(np.float32).tobytes())
            f.write(np.uint16(0).tobytes())  # attribute byte count


def _build_model_with_scene(robot_xml_path, scene_verts, scene_faces, tmpdir):
    """Build a MuJoCo model that includes both the robot and scene mesh."""
    stl_path = str(Path(tmpdir) / "scene_mesh.stl")
    # MuJoCo STL face limit is 200000; decimate if needed
    MAX_FACES = 190000
    if len(scene_faces) > MAX_FACES:
        print(f"  Decimating scene mesh: {len(scene_faces)} -> {MAX_FACES} faces")
        idx = np.random.default_rng(42).choice(len(scene_faces), MAX_FACES, replace=False)
        scene_faces = scene_faces[idx]

    _write_scene_stl(scene_verts, scene_faces, stl_path)

    # Load robot XML content and get its directory for asset resolution
    robot_path = Path(robot_xml_path).resolve()
    robot_dir = str(robot_path.parent)

    # Create a new XML that includes the robot and adds scene mesh
    # Use MjSpec to load robot and add scene geom
    spec = mujoco.MjSpec.from_file(str(robot_path))

    # Add mesh asset
    mesh = spec.add_mesh()
    mesh.name = "scene_mesh"
    mesh.file = stl_path

    # Add scene body with mesh geom
    scene_body = spec.worldbody.add_body()
    scene_body.name = "scene"
    geom = scene_body.add_geom()
    geom.name = "scene_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = "scene_mesh"
    geom.rgba = [0.5, 0.6, 0.7, 0.7]
    geom.contype = 0
    geom.conaffinity = 0

    model = spec.compile()
    return model


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

    # Load scene mesh if available
    has_scene = 'mesh_vertices' in data_npz and 'mesh_faces' in data_npz
    if has_scene:
        scene_verts = data_npz['mesh_vertices'].astype(np.float32)
        scene_faces = data_npz['mesh_faces'].astype(np.int32)
        print(f"\nScene mesh: {scene_verts.shape[0]} vertices, {scene_faces.shape[0]} triangles")
        print(f"  vertex range: min={scene_verts.min(axis=0)}, max={scene_verts.max(axis=0)}")

    # Load MuJoCo model (with scene mesh if available)
    print(f"\nLoading MuJoCo model: {robot_xml_path}")
    if has_scene:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _build_model_with_scene(robot_xml_path, scene_verts, scene_faces, tmpdir)
            _run_viewer(model, joint_pos, fps, dt, num_frames, duration, loop)
    else:
        model = mujoco.MjModel.from_xml_path(robot_xml_path)
        _run_viewer(model, joint_pos, fps, dt, num_frames, duration, loop)


def _run_viewer(model, joint_pos, fps, dt, num_frames, duration, loop):
    """Run the MuJoCo viewer playback loop."""
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

    # Map joint names (npz joint order may differ from model)
    num_dof = joint_pos.shape[1] - 7
    dof_index_list = list(range(num_dof))

    # Launch viewer
    print("\nLaunching MuJoCo viewer...")
    print("Press ESC to close viewer")
    viewer = mjv.launch_passive(model, data_mj, show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0

    viewer.cam.distance = 3.0
    viewer.cam.elevation = -20.0
    viewer.cam.azimuth = 45.0
    viewer.cam.lookat[:] = joint_pos[0, :3]

    # Playback loop
    frame_idx = 0
    print(f"\nPlaying motion... (Frame 0/{num_frames})")

    while viewer.is_running():
        start_time = time.perf_counter()

        # Set robot state from motion data
        data_mj.qpos[:3] = joint_pos[frame_idx, :3]  # root position
        data_mj.qpos[3:7] = joint_pos[frame_idx, 3:7]  # root quaternion (wxyz)
        # Only set DOFs that exist in model (skip scene body which has no joints)
        model_ndof = model.nq - 7  # exclude free joint (3 pos + 4 quat)
        npz_ndof = joint_pos.shape[1] - 7
        ndof_to_set = min(model_ndof, npz_ndof)
        data_mj.qpos[7:7 + ndof_to_set] = joint_pos[frame_idx, 7:7 + ndof_to_set]

        # Zero velocity
        data_mj.qvel[:] = 0

        # Forward kinematics
        mujoco.mj_forward(model, data_mj)
        viewer.cam.lookat[:] = joint_pos[frame_idx, :3]
        viewer.sync()

        # Print progress every 50 frames
        if frame_idx % 50 == 0:
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
