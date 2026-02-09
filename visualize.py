import os
import argparse
import numpy as np
import mujoco
import imageio
import joblib
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize motion data with optional reference motion')
    parser.add_argument('--input', type=str, required=True,
                        help='Input motion file path (.npy or .pkl)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output video file path (.mp4)')
    parser.add_argument('--ref_motion', type=str, default=None,
                        help='Reference motion file path (.npy or .pkl) for ghost rendering (optional)')
    parser.add_argument('--scene', type=str, 
                        default='src/holosoma/holosoma/data/robots/g1/g1_29dof.xml',
                        help='Scene XML file path')
    parser.add_argument('--ghost_alpha', type=float, default=0.45,
                        help='Alpha transparency for ghost/reference motion (0.0-1.0)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video')
    parser.add_argument('--image_shape', type=int, nargs=2, default=[480*2, 640*2],
                        help='Render image shape [height, width]')
    parser.add_argument('--input_fps', type=int, default=None,
                        help='Input motion fps (Hz). If not specified, will try to detect from data or use default')
    parser.add_argument('--ref_fps', type=int, default=None,
                        help='Reference motion fps (Hz). If not specified, will try to detect from data or use default')
    return parser.parse_args()


def set_qpos(root_pos, root_ori, dof_pos):
    """
    Convert root position, orientation, and DOF positions to MuJoCo qpos format.
    
    Args:
        root_pos: [3] - root position (x, y, z)
        root_ori: [4] - root orientation quaternion in xyzw order (ProtoMotions format)
        dof_pos: [num_dofs] - joint positions
    
    Returns:
        qpos: [num_dofs+7] - MuJoCo qpos format [pos(3), quat_wxyz(4), joints(...)]
    """
    qpos = np.zeros(len(dof_pos)+7)
    qpos[0: 3] = root_pos
    # Convert xyzw (ProtoMotions) -> wxyz (MuJoCo) using index reordering [3,0,1,2]
    qpos[3: 7] = root_ori[..., [3, 0, 1, 2]]
    qpos[7:  ] = dof_pos

    return qpos


def load_motion_data(file_path, default_fps=30):
    """
    Load motion data from .npy or .pkl file.
    
    Args:
        file_path: Path to motion file (.npy or .pkl)
        default_fps: Default fps to use if not found in data
    
    Returns:
        dict with keys: 'root_trans', 'root_ori', 'dof_pos', 'fps'
    """
    if file_path.endswith('.pkl'):
        # Load pkl file (processed_phuma format)
        data = joblib.load(file_path)
        # Get first value from dictionary (pkl format: {filename: {data}})
        motion_data = list(data.values())[0]
        
        return {
            'root_trans': motion_data['root_trans_offset'],
            'root_ori': motion_data['root_rot'],
            'dof_pos': motion_data['dof'],
            'fps': motion_data.get('fps', default_fps)
        }
    else:
        # Load npy file (from play_g1.py, typically 50fps for IsaacGym)
        motion_data = np.load(file_path, allow_pickle=True).item()
        return {
            'root_trans': motion_data['root_trans'],
            'root_ori': motion_data['root_ori'],
            'dof_pos': motion_data['dof_pos'],
            'fps': motion_data.get('fps', default_fps)
        }


def quat_slerp(q1, q2, t):
    """
    Spherical linear interpolation (SLERP) for quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z] or [x, y, z, w]
        q2: Second quaternion [w, x, y, z] or [x, y, z, w]
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated quaternion
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, negate one quaternion for shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate angle
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    
    # SLERP formula
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    result = w1 * q1 + w2 * q2
    
    return result / np.linalg.norm(result)


def resample_motion(motion_data, target_fps, original_fps):
    """
    Resample motion data to match target fps using linear interpolation for positions/angles
    and SLERP for quaternions.
    
    Args:
        motion_data: dict with 'root_trans', 'root_ori', 'dof_pos'
        target_fps: Target frames per second
        original_fps: Original frames per second
    
    Returns:
        Resampled motion data dict
    """
    if original_fps == target_fps:
        return motion_data
    
    num_original_frames = len(motion_data['dof_pos'])
    original_time = np.arange(num_original_frames) / original_fps
    duration = original_time[-1]
    num_target_frames = int(duration * target_fps)
    target_time = np.linspace(0, duration, num_target_frames)
    
    resampled = {}
    
    # Resample root_trans (linear interpolation)
    root_trans = motion_data['root_trans']
    resampled_root_trans = np.zeros((num_target_frames, root_trans.shape[1]), dtype=root_trans.dtype)
    for dim in range(root_trans.shape[1]):
        resampled_root_trans[:, dim] = np.interp(target_time, original_time, root_trans[:, dim])
    resampled['root_trans'] = resampled_root_trans
    
    # Resample root_ori (SLERP for quaternions)
    root_ori = motion_data['root_ori']  # [x, y, z, w] format
    resampled_root_ori = np.zeros((num_target_frames, 4), dtype=root_ori.dtype)
    for i, t in enumerate(target_time):
        # Find surrounding frames
        idx = np.searchsorted(original_time, t)
        if idx == 0:
            resampled_root_ori[i] = root_ori[0]
        elif idx >= num_original_frames:
            resampled_root_ori[i] = root_ori[-1]
        else:
            # Interpolate between idx-1 and idx
            t_local = (t - original_time[idx-1]) / (original_time[idx] - original_time[idx-1])
            # Convert to [w, x, y, z] for slerp (assuming input is [x, y, z, w])
            q1 = root_ori[idx-1][[3, 0, 1, 2]]  # [w, x, y, z]
            q2 = root_ori[idx][[3, 0, 1, 2]]    # [w, x, y, z]
            q_interp = quat_slerp(q1, q2, t_local)
            resampled_root_ori[i] = q_interp[[1, 2, 3, 0]]  # Back to [x, y, z, w]
    resampled['root_ori'] = resampled_root_ori
    
    # Resample dof_pos (linear interpolation)
    dof_pos = motion_data['dof_pos']
    resampled_dof_pos = np.zeros((num_target_frames, dof_pos.shape[1]), dtype=dof_pos.dtype)
    for dim in range(dof_pos.shape[1]):
        resampled_dof_pos[:, dim] = np.interp(target_time, original_time, dof_pos[:, dim])
    resampled['dof_pos'] = resampled_dof_pos
    
    resampled['fps'] = target_fps
    return resampled


def main():
    args = parse_args()
    
    # Determine default fps based on file type
    # play_g1.py output (npy) is typically 50fps (IsaacGym default)
    # pkl files are typically 30fps
    input_default_fps = 50 if args.input.endswith('.npy') else 30
    ref_default_fps = 50 if args.ref_motion and args.ref_motion.endswith('.npy') else 30
    
    # Load motion data
    print(f"Loading motion data from {args.input}...")
    input_fps_override = args.input_fps if args.input_fps is not None else input_default_fps
    motion_data = load_motion_data(args.input, default_fps=input_fps_override)
    render_image_shape = tuple(args.image_shape)
    
    input_fps = motion_data['fps']
    print(f"Input motion: {len(motion_data['dof_pos'])} frames @ {input_fps} fps")
    
    # Load reference motion if provided
    has_ref_motion = args.ref_motion is not None
    ref_fps = None
    if has_ref_motion:
        print(f"Loading reference motion from {args.ref_motion}...")
        ref_fps_override = args.ref_fps if args.ref_fps is not None else ref_default_fps
        ref_motion_data = load_motion_data(args.ref_motion, default_fps=ref_fps_override)
        ref_fps = ref_motion_data['fps']
        print(f"Reference motion: {len(ref_motion_data['dof_pos'])} frames @ {ref_fps} fps")
        
        # Resample motions to match fps (use the higher fps as target)
        target_fps = max(input_fps, ref_fps)
        if input_fps != target_fps:
            print(f"Resampling input motion from {input_fps} fps to {target_fps} fps...")
            motion_data = resample_motion(motion_data, target_fps, input_fps)
        if ref_fps != target_fps:
            print(f"Resampling reference motion from {ref_fps} fps to {target_fps} fps...")
            ref_motion_data = resample_motion(ref_motion_data, target_fps, ref_fps)
    
    dof_pos = motion_data['dof_pos'].squeeze()
    root_pos = motion_data['root_trans'].squeeze()
    initial_root_pos = root_pos[0, :2].copy()
    root_pos[:, :2] -= initial_root_pos
    root_ori = motion_data['root_ori'].squeeze()
    num_frames = len(dof_pos)
    print(f"Using {num_frames} frames for rendering")
    
    if has_ref_motion:
        ref_dof_pos = ref_motion_data['dof_pos'].squeeze()
        ref_root_pos = ref_motion_data['root_trans'].squeeze()
        ref_root_pos[:, :2] -= initial_root_pos  # Use same initial offset
        ref_root_ori = ref_motion_data['root_ori'].squeeze()
        num_ref_frames = len(ref_dof_pos)
        num_frames = min(num_frames, num_ref_frames)
        print(f"Aligned to {num_frames} frames")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(args.scene)
    model.vis.quality.offsamples = 16  # High quality anti-aliasing
    data = mujoco.MjData(model)
    data.qpos = set_qpos(root_pos=root_pos[0], root_ori=root_ori[0], dof_pos=dof_pos[0])
    mujoco.mj_resetData(model, data)
    
    # Setup ghost model if reference motion is provided
    ghost_model = None
    ghost_data = None
    if has_ref_motion:
        ghost_model = mujoco.MjModel.from_xml_path(args.scene)
        ghost_model.vis.quality.offsamples = 16  # High quality anti-aliasing
        ghost_data = mujoco.MjData(ghost_model)
        ghost_data.qpos = set_qpos(root_pos=ref_root_pos[0], root_ori=ref_root_ori[0], dof_pos=ref_dof_pos[0])
        mujoco.mj_resetData(ghost_model, ghost_data)
        
        # Set ghost transparency
        for i in range(ghost_model.ngeom):
            ghost_model.geom_rgba[i, 3] = args.ghost_alpha
    
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 0
    camera.distance = 2.2  # Distance between camera and tracking target
    camera.elevation = -20.0  # Camera elevation angle
    camera.azimuth = -140.0  # Camera azimuth angle
    
    frames = []
    
    if has_ref_motion:
        # Render with ghost/reference motion
        print(f"Rendering with reference motion (ghost alpha={args.ghost_alpha})...")
        with mujoco.Renderer(model, render_image_shape[0], render_image_shape[1]) as renderer:
            with mujoco.Renderer(ghost_model, render_image_shape[0], render_image_shape[1]) as ghost_renderer:
                for i in tqdm(range(num_frames), desc="Rendering frames"):
                    # Update main model
                    data.qpos = set_qpos(root_pos=root_pos[i], root_ori=root_ori[i], dof_pos=dof_pos[i])
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=scene_option)
                    pixels = renderer.render()
                    
                    # Update ghost model
                    ghost_data.qpos = set_qpos(root_pos=ref_root_pos[i], root_ori=ref_root_ori[i], dof_pos=ref_dof_pos[i])
                    mujoco.mj_forward(ghost_model, ghost_data)
                    ghost_renderer.update_scene(ghost_data, camera=camera, scene_option=scene_option)
                    ghost_pixels = ghost_renderer.render()
                    
                    # Combine pixels
                    combined_pixels = (1 - args.ghost_alpha) * pixels.astype(np.float32) + args.ghost_alpha * ghost_pixels.astype(np.float32)
                    combined_pixels = combined_pixels.astype(np.uint8)
                    frames.append(combined_pixels)
    else:
        # Render without ghost
        print("Rendering without reference motion...")
        with mujoco.Renderer(model, render_image_shape[0], render_image_shape[1]) as renderer:
            for i in tqdm(range(num_frames), desc="Rendering frames"):
                data.qpos = set_qpos(root_pos=root_pos[i], root_ori=root_ori[i], dof_pos=dof_pos[i])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()
                frames.append(pixels)
    
    # Save video
    print(f"Saving video to {args.output}...")
    writer = imageio.get_writer(args.output, format='FFMPEG', fps=args.fps, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved successfully!")


if __name__ == "__main__":
    main()