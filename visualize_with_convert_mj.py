#!/usr/bin/env python3
"""
Wrapper to visualize npz files using convert_data_format_mj.py's visualization.

This script converts a final npz file back to qpos format and uses
convert_data_format_mj.py for visualization.
"""

import argparse
import numpy as np
import tempfile
import subprocess
import os


def detect_robot_from_npz(npz_path: str, data: dict) -> tuple[str, int]:
    """
    Detect robot type and DOF from npz file.
    
    Args:
        npz_path: Path to npz file
        data: Loaded npz data
        
    Returns:
        (robot_type, num_dof) tuple
    """
    # Try to detect from file path first
    npz_path_lower = npz_path.lower()
    if "h1_2" in npz_path_lower:
        return ("h1_2", 27)
    elif "statue" in npz_path_lower:
        return ("statue_v0_1", 29)
    elif "g1" in npz_path_lower:
        return ("g1", 29)
    
    # Detect from joint_pos shape
    if 'joint_pos' in data:
        joint_pos = data['joint_pos']
        if joint_pos.ndim == 2:
            num_dof = joint_pos.shape[1] - 7  # Subtract root (3 pos + 4 quat)
            if num_dof == 27:
                return ("h1_2", 27)
            elif num_dof == 29:
                return ("g1", 29)
    
    # Default to g1
    return ("g1", 29)


def npz_to_qpos_for_visualization(npz_path: str, output_path: str):
    """
    Extract qpos from final npz for visualization.
    
    Args:
        npz_path: Path to final npz file
        output_path: Path to output qpos npz file
        
    Returns:
        (robot_type, num_dof) tuple
    """
    print(f"Loading npz file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # Detect robot type
    robot_type, num_dof = detect_robot_from_npz(npz_path, data)
    print(f"Detected robot: {robot_type} ({num_dof} DOF)")
    
    # Extract qpos (already in correct format)
    joint_pos = data['joint_pos']  # (T, 7+DOF) = [xyz(3), wxyz(4), dof(N)]
    fps = int(data['fps'][0]) if 'fps' in data else 30
    joint_names = data.get('joint_names', None)

    print(f"FPS: {fps}")
    print(f"Frames: {joint_pos.shape[0]}")
    print(f"Duration: {joint_pos.shape[0] / fps:.2f} seconds")

    # Save as qpos npz (include joint_names if available)
    print(f"Saving qpos npz: {output_path}")
    save_dict = {
        'qpos': joint_pos,
        'fps': 1.0 / fps,  # dt format expected by convert_data_format_mj.py
    }
    if joint_names is not None:
        save_dict['joint_names'] = joint_names
        print(f"Including {len(joint_names)} joint names")

    np.savez(output_path, **save_dict)
    
    print("✓ Conversion complete!")
    return robot_type, num_dof


def visualize_with_mujoco(qpos_path: str, robot_type: str, num_dof: int):
    """
    Visualize using convert_data_format_mj.py.
    
    Args:
        qpos_path: Path to qpos npz file
        robot_type: Robot type (g1, h1_2, etc.)
        num_dof: Number of DOF
    """
    # Change to retargeting directory
    retargeting_dir = "src/holosoma_retargeting/holosoma_retargeting"
    
    # Use the actual robot type (h1_2 and statue_v0_1 are now in _ROBOT_JOINT_NAMES_DEFAULT)
    base_robot_type = robot_type
    
    cmd = [
        "python",
        "data_conversion/convert_data_format_mj.py",
        "--input-file", os.path.abspath(qpos_path),
        "--robot", base_robot_type,
        "--robot-config.robot-dof", str(num_dof),
        "--object-name", "ground",
        "--input-fps", "30",  # Will be read from file
        "--output-fps", "30",  # Same as input for visualization
        "--no-once",  # Loop mode for visualization
    ]
    
    # For h1_2 and statue_v0_1, also override robot URDF/XML path
    if robot_type == "h1_2":
        # Choose between h1_2 (51 DOF with hands) and h1_2_handless (27 DOF)
        if num_dof == 27:
            h1_2_urdf = os.path.abspath("src/holosoma/holosoma/data/robots/h1_2/h1_2_handless.urdf")
            h1_2_xml = os.path.abspath("src/holosoma/holosoma/data/robots/h1_2/h1_2_handless.xml")
        else:
            h1_2_urdf = os.path.abspath("src/holosoma/holosoma/data/robots/h1_2/h1_2.urdf")
            h1_2_xml = os.path.abspath("src/holosoma/holosoma/data/robots/h1_2/h1_2.xml")

        if os.path.exists(h1_2_urdf):
            cmd.extend(["--robot-config.robot-urdf-file", h1_2_urdf])
        else:
            # Use XML directly (convert_data_format_mj.py converts .xml to .urdf internally)
            cmd.extend(["--robot-config.robot-urdf-file", h1_2_xml])
    elif robot_type == "statue_v0_1":
        # Check if URDF exists, otherwise use XML path
        statue_urdf = os.path.abspath("src/holosoma/holosoma/data/robots/statue_v0_1/statue_v0_1.urdf")
        statue_xml = os.path.abspath("src/holosoma/holosoma/data/robots/statue_v0_1/statue_v0_1.xml")
        if os.path.exists(statue_urdf):
            cmd.extend(["--robot-config.robot-urdf-file", statue_urdf])
        else:
            # Use XML directly (convert_data_format_mj.py converts .xml to .urdf internally)
            cmd.extend(["--robot-config.robot-urdf-file", statue_xml])
    
    print(f"\nStarting MuJoCo visualization...")
    print(f"Working directory: {retargeting_dir}")
    print(f"Robot: {robot_type} ({num_dof} DOF)")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress ESC in the MuJoCo viewer window to exit")
    print("-" * 70)
    
    try:
        subprocess.run(cmd, cwd=retargeting_dir, check=True)
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\n\nError running visualization: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize npz motion file using convert_data_format_mj.py"
    )
    parser.add_argument("npz_path", type=str, help="Path to npz file to visualize")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary qpos file")
    
    args = parser.parse_args()
    
    # Create temporary qpos file and detect robot
    if args.keep_temp:
        temp_qpos = "temp_qpos_for_viz.npz"
        robot_type, num_dof = npz_to_qpos_for_visualization(args.npz_path, temp_qpos)
        print(f"\nTemporary file saved: {temp_qpos}")
        visualize_with_mujoco(temp_qpos, robot_type, num_dof)
    else:
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            temp_qpos = tmp.name
        
        try:
            robot_type, num_dof = npz_to_qpos_for_visualization(args.npz_path, temp_qpos)
            visualize_with_mujoco(temp_qpos, robot_type, num_dof)
        finally:
            # Clean up
            if os.path.exists(temp_qpos):
                os.remove(temp_qpos)
                print(f"\nTemporary file cleaned up: {temp_qpos}")
