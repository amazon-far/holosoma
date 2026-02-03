#!/usr/bin/env python3
"""
Add initial pose padding to the beginning of a motion npz file.

This script adds N frames of the initial pose at the beginning of the motion,
useful for allowing the robot to settle before starting the motion.
"""

import argparse
import numpy as np
from pathlib import Path


def add_init_padding(input_path: str, output_path: str, padding_frames: int = 100):
    """
    Add initial pose padding to motion data.
    
    Args:
        input_path: Path to input npz file
        output_path: Path to output npz file
        padding_frames: Number of frames to pad with initial pose
    """
    print(f"Loading: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    
    # Get fps
    fps = int(data['fps'][0]) if data['fps'].ndim > 0 else int(data['fps'])
    
    # Get first frame for all arrays
    joint_pos_0 = data['joint_pos'][0:1]  # (1, 36)
    joint_vel_0 = np.zeros_like(data['joint_vel'][0:1])  # Zero velocity for padding
    body_pos_w_0 = data['body_pos_w'][0:1]  # (1, N, 3)
    body_quat_w_0 = data['body_quat_w'][0:1]  # (1, N, 4)
    body_lin_vel_w_0 = np.zeros_like(data['body_lin_vel_w'][0:1])  # Zero velocity
    body_ang_vel_w_0 = np.zeros_like(data['body_ang_vel_w'][0:1])  # Zero velocity
    
    # Create padding by repeating first frame
    joint_pos_padding = np.repeat(joint_pos_0, padding_frames, axis=0)
    joint_vel_padding = np.repeat(joint_vel_0, padding_frames, axis=0)
    body_pos_w_padding = np.repeat(body_pos_w_0, padding_frames, axis=0)
    body_quat_w_padding = np.repeat(body_quat_w_0, padding_frames, axis=0)
    body_lin_vel_w_padding = np.repeat(body_lin_vel_w_0, padding_frames, axis=0)
    body_ang_vel_w_padding = np.repeat(body_ang_vel_w_0, padding_frames, axis=0)
    
    # Concatenate padding with original data
    joint_pos_new = np.concatenate([joint_pos_padding, data['joint_pos']], axis=0)
    joint_vel_new = np.concatenate([joint_vel_padding, data['joint_vel']], axis=0)
    body_pos_w_new = np.concatenate([body_pos_w_padding, data['body_pos_w']], axis=0)
    body_quat_w_new = np.concatenate([body_quat_w_padding, data['body_quat_w']], axis=0)
    body_lin_vel_w_new = np.concatenate([body_lin_vel_w_padding, data['body_lin_vel_w']], axis=0)
    body_ang_vel_w_new = np.concatenate([body_ang_vel_w_padding, data['body_ang_vel_w']], axis=0)
    
    # Handle object data if present
    if 'object_pos_w' in data:
        object_pos_w_0 = data['object_pos_w'][0:1]
        object_quat_w_0 = data['object_quat_w'][0:1]
        object_lin_vel_w_0 = np.zeros_like(data['object_lin_vel_w'][0:1])
        
        object_pos_w_padding = np.repeat(object_pos_w_0, padding_frames, axis=0)
        object_quat_w_padding = np.repeat(object_quat_w_0, padding_frames, axis=0)
        object_lin_vel_w_padding = np.repeat(object_lin_vel_w_0, padding_frames, axis=0)
        
        object_pos_w_new = np.concatenate([object_pos_w_padding, data['object_pos_w']], axis=0)
        object_quat_w_new = np.concatenate([object_quat_w_padding, data['object_quat_w']], axis=0)
        object_lin_vel_w_new = np.concatenate([object_lin_vel_w_padding, data['object_lin_vel_w']], axis=0)
    
    # Original stats
    orig_frames = data['joint_pos'].shape[0]
    orig_duration = orig_frames / fps
    
    # New stats
    new_frames = joint_pos_new.shape[0]
    new_duration = new_frames / fps
    padding_duration = padding_frames / fps
    
    print()
    print("=" * 70)
    print("Original:")
    print(f"  Frames: {orig_frames}")
    print(f"  Duration: {orig_duration:.2f}s @ {fps}fps")
    print()
    print("Padding:")
    print(f"  Frames: {padding_frames}")
    print(f"  Duration: {padding_duration:.2f}s (static initial pose)")
    print()
    print("Result:")
    print(f"  Frames: {new_frames} ({padding_frames} padding + {orig_frames} original)")
    print(f"  Duration: {new_duration:.2f}s @ {fps}fps")
    print("=" * 70)
    
    # Save new npz
    print(f"\nSaving: {output_path}")
    save_dict = {
        'fps': data['fps'],
        'joint_pos': joint_pos_new.astype(data['joint_pos'].dtype),
        'joint_vel': joint_vel_new.astype(data['joint_vel'].dtype),
        'body_pos_w': body_pos_w_new.astype(data['body_pos_w'].dtype),
        'body_quat_w': body_quat_w_new.astype(data['body_quat_w'].dtype),
        'body_lin_vel_w': body_lin_vel_w_new.astype(data['body_lin_vel_w'].dtype),
        'body_ang_vel_w': body_ang_vel_w_new.astype(data['body_ang_vel_w'].dtype),
        'joint_names': data['joint_names'],
        'body_names': data['body_names'],
    }
    
    # Add object data if present
    if 'object_pos_w' in data:
        save_dict.update({
            'object_pos_w': object_pos_w_new.astype(data['object_pos_w'].dtype),
            'object_quat_w': object_quat_w_new.astype(data['object_quat_w'].dtype),
            'object_lin_vel_w': object_lin_vel_w_new.astype(data['object_lin_vel_w'].dtype),
        })
    
    np.savez(output_path, **save_dict)
    
    print()
    print("✓ Padding complete!")
    print()
    print("Verification:")
    print(f"  Original first frame position: {data['joint_pos'][0, :3]}")
    print(f"  Padded first frame position:   {joint_pos_new[0, :3]}")
    print(f"  Frame {padding_frames} position (should match): {joint_pos_new[padding_frames, :3]}")
    print(f"  Frame {padding_frames+1} position (original frame 1): {joint_pos_new[padding_frames+1, :3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add initial pose padding to motion npz file"
    )
    parser.add_argument("input_npz", type=str, help="Path to input npz file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Path to output npz file (default: input_padded.npz)")
    parser.add_argument("--padding-frames", "-p", type=int, default=100,
                       help="Number of padding frames (default: 100)")
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        input_path = Path(args.input_npz)
        args.output = str(input_path.parent / f"{input_path.stem}_padded{input_path.suffix}")
    
    add_init_padding(args.input_npz, args.output, args.padding_frames)
