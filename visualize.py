import os
import tempfile
import argparse
import numpy as np
import mujoco
import imageio
import joblib
import trimesh
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
    parser.add_argument('--ghost_color', type=str, default='0.2 0.4 1.0',
                        help='RGB triple for the ghost robot (space-separated floats 0-1). '
                             'Set to "" to keep the original geom colors.')
    parser.add_argument('--ghost_hide_terrain', action='store_true', default=True,
                        help='Hide the terrain mesh in the ghost render (default: True). '
                             'Mask-based composition is used so the main terrain stays unchanged.')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video')
    parser.add_argument('--image_shape', type=int, nargs=2, default=[480*2, 640*2],
                        help='Render image shape [height, width]')
    parser.add_argument('--input_fps', type=int, default=None,
                        help='Input motion fps (Hz). If not specified, will try to detect from data or use default')
    parser.add_argument('--ref_fps', type=int, default=None,
                        help='Reference motion fps (Hz). If not specified, will try to detect from data or use default')
    parser.add_argument('--terrain_npz', type=str, default=None,
                        help='Terrain npz file path containing mesh_vertices and mesh_faces for scene rendering')
    return parser.parse_args()


def create_scene_with_terrain(base_xml_path, terrain_npz_path, tmpdir):
    """
    Create a modified MuJoCo XML that includes terrain mesh from an npz file.

    The terrain mesh is exported as an OBJ file and added to the XML scene.
    Returns the path to the modified XML file.
    """
    data = np.load(terrain_npz_path, allow_pickle=True)
    vertices = data['mesh_vertices']
    faces = data['mesh_faces']

    # Export terrain mesh as OBJ
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    obj_path = os.path.join(tmpdir, 'terrain.obj')
    mesh.export(obj_path)

    # Read the base XML and inject the terrain mesh
    base_xml_dir = os.path.dirname(os.path.abspath(base_xml_path))
    with open(base_xml_path, 'r') as f:
        xml_content = f.read()

    # Make meshdir absolute so the XML works from any location
    xml_content = xml_content.replace('meshdir="./', f'meshdir="{base_xml_dir}/')
    xml_content = xml_content.replace("meshdir='./", f"meshdir='{base_xml_dir}/")

    # Add terrain mesh asset (before closing </asset> of the second asset block, the scene one)
    terrain_asset = f'        <mesh name="terrain_mesh" file="{obj_path}" scale="1 1 1"/>\n'
    terrain_geom = f'        <geom name="terrain" type="mesh" mesh="terrain_mesh" rgba="0.5 0.55 0.6 1.0" contype="15" conaffinity="15"/>\n'

    # Insert terrain mesh in the second <asset> block (scene assets)
    # Find the second </asset> tag
    first_asset_end = xml_content.find('</asset>')
    second_asset_end = xml_content.find('</asset>', first_asset_end + 1)
    xml_content = xml_content[:second_asset_end] + terrain_asset + '    ' + xml_content[second_asset_end:]

    # Hide the default MuJoCo floor (keep it for contact references but make invisible)
    import re
    xml_content = re.sub(
        r'<geom([^>]*name="floor"[^>]*)/>',
        r'<geom\1 rgba="0 0 0 0" group="4"/>',
        xml_content,
    )
    # Insert terrain geom before the (now hidden) floor geom
    floor_geom_pos = xml_content.find('name="floor"')
    geom_start = xml_content.rfind('<geom', 0, floor_geom_pos)
    xml_content = xml_content[:geom_start] + terrain_geom + '        ' + xml_content[geom_start:]

    modified_xml_path = os.path.join(tmpdir, 'scene_with_terrain.xml')
    with open(modified_xml_path, 'w') as f:
        f.write(xml_content)

    return modified_xml_path


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
    if args.terrain_npz is None:
        root_pos[:, :2] -= initial_root_pos
    root_ori = motion_data['root_ori'].squeeze()
    num_frames = len(dof_pos)
    print(f"Using {num_frames} frames for rendering")
    
    if has_ref_motion:
        ref_dof_pos = ref_motion_data['dof_pos'].squeeze()
        ref_root_pos = ref_motion_data['root_trans'].squeeze()
        if args.terrain_npz is None:
            ref_root_pos[:, :2] -= initial_root_pos  # Use same initial offset
        ref_root_ori = ref_motion_data['root_ori'].squeeze()
        num_ref_frames = len(ref_dof_pos)
        num_frames = min(num_frames, num_ref_frames)
        print(f"Aligned to {num_frames} frames")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create scene with terrain if terrain_npz is provided
    _tmpdir = None
    scene_path = args.scene
    if args.terrain_npz is not None:
        _tmpdir = tempfile.mkdtemp()
        scene_path = create_scene_with_terrain(args.scene, args.terrain_npz, _tmpdir)
        print(f"Created scene with terrain: {scene_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    model.vis.quality.offsamples = 16  # High quality anti-aliasing
    data = mujoco.MjData(model)
    data.qpos = set_qpos(root_pos=root_pos[0], root_ori=root_ori[0], dof_pos=dof_pos[0])
    mujoco.mj_resetData(model, data)
    
    # Setup ghost model if reference motion is provided
    ghost_model = None
    ghost_data = None
    ghost_terrain_geom_id = -1
    ghost_color_rgb = None
    if has_ref_motion:
        ghost_model = mujoco.MjModel.from_xml_path(scene_path)
        # Ghost is rendered as a solid-color silhouette and is composited via
        # depth-mask, so MSAA on the ghost model only adds cost (depth resolve
        # in MSAA mode is especially expensive). Drop it to 1x.
        ghost_model.vis.quality.offsamples = 1
        ghost_data = mujoco.MjData(ghost_model)
        ghost_data.qpos = set_qpos(root_pos=ref_root_pos[0], root_ori=ref_root_ori[0], dof_pos=ref_dof_pos[0])
        mujoco.mj_resetData(ghost_model, ghost_data)

        if args.ghost_color.strip():
            ghost_color_rgb = np.array(
                [float(c) for c in args.ghost_color.split()], dtype=np.float32
            )
            assert ghost_color_rgb.shape == (3,), \
                f'--ghost_color must be 3 floats; got {args.ghost_color!r}'

        if args.ghost_hide_terrain and args.terrain_npz is not None:
            ghost_terrain_geom_id = mujoco.mj_name2id(
                ghost_model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain'
            )

        # Per-geom render color/alpha. Geom alpha=1.0 here keeps the ghost robot
        # fully opaque in its OWN render; the screen-space ghost_alpha controls
        # how strongly the ghost overlays the main render in the mask region.
        for i in range(ghost_model.ngeom):
            if i == ghost_terrain_geom_id:
                ghost_model.geom_rgba[i] = [0.0, 0.0, 0.0, 0.0]
                continue
            if ghost_color_rgb is not None:
                ghost_model.geom_rgba[i, :3] = ghost_color_rgb
            ghost_model.geom_rgba[i, 3] = 1.0
    
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
        # Render with ghost/reference motion using a depth mask: only pixels
        # where the ghost model rendered actual geometry (i.e. the ghost robot,
        # since terrain is hidden) are blended onto the main render. This keeps
        # the main terrain/sky pixels untouched.
        print(f"Rendering with reference motion (ghost alpha={args.ghost_alpha})...")
        with mujoco.Renderer(model, render_image_shape[0], render_image_shape[1]) as renderer:
            with mujoco.Renderer(ghost_model, render_image_shape[0], render_image_shape[1]) as ghost_renderer:
                for i in tqdm(range(num_frames), desc="Rendering frames"):
                    # Update main model
                    data.qpos = set_qpos(root_pos=root_pos[i], root_ori=root_ori[i], dof_pos=dof_pos[i])
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=scene_option)
                    pixels = renderer.render()

                    # Update ghost model + render color and depth.
                    ghost_data.qpos = set_qpos(root_pos=ref_root_pos[i], root_ori=ref_root_ori[i], dof_pos=ref_dof_pos[i])
                    mujoco.mj_forward(ghost_model, ghost_data)
                    ghost_renderer.update_scene(ghost_data, camera=camera, scene_option=scene_option)
                    ghost_pixels = ghost_renderer.render()

                    ghost_renderer.enable_depth_rendering()
                    ghost_renderer.update_scene(ghost_data, camera=camera, scene_option=scene_option)
                    ghost_depth = ghost_renderer.render()
                    ghost_renderer.disable_depth_rendering()

                    # Background pixels have depth at the far plane (very large value).
                    # Geometry pixels have finite, small depth. Use a generous threshold.
                    finite_depth = np.isfinite(ghost_depth) & (ghost_depth > 0)
                    if finite_depth.any():
                        far_threshold = np.percentile(ghost_depth[finite_depth], 99) * 10.0
                    else:
                        far_threshold = np.inf
                    ghost_mask = finite_depth & (ghost_depth < far_threshold)

                    combined_pixels = pixels.astype(np.float32).copy()
                    if ghost_mask.any():
                        blend = (
                            (1.0 - args.ghost_alpha) * pixels.astype(np.float32)
                            + args.ghost_alpha * ghost_pixels.astype(np.float32)
                        )
                        combined_pixels[ghost_mask] = blend[ghost_mask]
                    frames.append(combined_pixels.astype(np.uint8))
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

    # Cleanup temporary directory
    if _tmpdir is not None:
        import shutil
        shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()