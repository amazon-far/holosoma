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


def npz_to_qpos_for_visualization(npz_path: str, output_path: str):
    """
    Extract qpos from final npz for visualization.
    
    Args:
        npz_path: Path to final npz file
        output_path: Path to output qpos npz file
    """
    print(f"Loading npz file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract qpos (already in correct format)
    joint_pos = data['joint_pos']  # (T, 36) = [xyz(3), wxyz(4), dof(29)]
    fps = int(data['fps'][0]) if 'fps' in data else 30
    
    print(f"FPS: {fps}")
    print(f"Frames: {joint_pos.shape[0]}")
    print(f"Duration: {joint_pos.shape[0] / fps:.2f} seconds")
    
    # Save as qpos npz
    print(f"Saving qpos npz: {output_path}")
    np.savez(
        output_path,
        qpos=joint_pos,
        fps=1.0 / fps,  # dt format expected by convert_data_format_mj.py
    )
    
    print("✓ Conversion complete!")


def visualize_with_mujoco(qpos_path: str, robot_xml: str):
    """
    Visualize using convert_data_format_mj.py.
    
    Args:
        qpos_path: Path to qpos npz file
        robot_xml: Path to robot XML file
    """
    # Change to retargeting directory
    retargeting_dir = "src/holosoma_retargeting/holosoma_retargeting"
    
    cmd = [
        "python",
        "data_conversion/convert_data_format_mj.py",
        "--input-file", os.path.abspath(qpos_path),
        "--robot", "g1",
        "--robot-config.robot-dof", "29",
        "--object-name", "ground",
        "--input-fps", "30",  # Will be read from file
        "--output-fps", "30",  # Same as input for visualization
        "--no-once",  # Loop mode for visualization
    ]
    
    print(f"\nStarting MuJoCo visualization...")
    print(f"Working directory: {retargeting_dir}")
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
    parser.add_argument("--robot-xml", type=str, default=None, 
                       help="Path to robot XML (default: g1_29dof)")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary qpos file")
    
    args = parser.parse_args()
    
    # Auto-detect robot XML
    if args.robot_xml is None:
        args.robot_xml = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    
    # Create temporary qpos file
    if args.keep_temp:
        temp_qpos = "temp_qpos_for_viz.npz"
        npz_to_qpos_for_visualization(args.npz_path, temp_qpos)
        print(f"\nTemporary file saved: {temp_qpos}")
        visualize_with_mujoco(temp_qpos, args.robot_xml)
    else:
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            temp_qpos = tmp.name
        
        try:
            npz_to_qpos_for_visualization(args.npz_path, temp_qpos)
            visualize_with_mujoco(temp_qpos, args.robot_xml)
        finally:
            # Clean up
            if os.path.exists(temp_qpos):
                os.remove(temp_qpos)
                print(f"\nTemporary file cleaned up: {temp_qpos}")
