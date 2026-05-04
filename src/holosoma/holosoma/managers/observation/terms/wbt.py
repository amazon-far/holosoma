"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand, MultiMotionCommand
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand | MultiMotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, (MotionCommand, MultiMotionCommand)), (
        f"Expected MotionCommand or MultiMotionCommand, got {type(motion_command)}"
    )
    return motion_command


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_ref_pos_xy_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """VideoMimic-style narrowed command: only the xy part of motion_ref_pos_b.

    Drops the z component so that step-function discontinuities in the target
    pelvis height (e.g. synthetic stair clips, or any clip with abrupt vertical
    teleports) do not enter the actor obs. Mirrors VideoMimic's
    `_obs_torso_xy_rel` (`robot_deepmimic.py:669`: `rel_xy = obs[:, :2]`).
    """
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos[:, :2].reshape(env.num_envs, -1)


def motion_ref_yaw_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """VideoMimic-style narrowed command: yaw error only as (cos, sin).

    `motion_ref_ori_b` exposes the full body-frame relative orientation as a
    6D rotation rep (first two columns of the matrix), so roll/pitch leak
    into the actor. This term collapses that to a single yaw angle and
    returns ``(cos, sin)`` so the policy never sees the angle wrap-around at
    ±π. Conceptually matches VideoMimic's scalar `_obs_torso_yaw_rel`
    (`robot_deepmimic.py:718-735`).
    """
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    # ori is (num_envs, 4) xyzw quaternion of the target frame in the robot's
    # reference frame. Yaw = atan2(2(wz + xy), 1 - 2(yy + zz)). Same formula
    # the codebase uses in `yaw_quat` (`utils/rotations.py:34`).
    qx, qy, qz, qw = ori[:, 0], ori[:, 1], ori[:, 2], ori[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    unit_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    vel_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w.clone(),
        motion_command.robot_ref_quat_w.clone(),
        motion_command.simulator_object_lin_vel_w,
        unit_quat,
    )
    return vel_b.view(env.num_envs, -1)


#########################################################################################################
## Future Motion Targets (for motion encoder)
#########################################################################################################

def future_motion_targets(
    env: WholeBodyTrackingManager,
    future_num_steps: int = 20,
    future_max_steps: int = 95,
    include_key_body_pos: bool = True,
) -> torch.Tensor:
    """Future motion targets for motion encoder.
    
    Returns concatenated future motion features:
    - root_height: [future_num_steps, 1]
    - roll_pitch: [future_num_steps, 2]
    - base_lin_vel: [future_num_steps, 3]
    - base_yaw_vel: [future_num_steps, 1]
    - dof_pos: [future_num_steps, num_dofs]
    - local_key_body_pos: [future_num_steps, num_tracked_bodies * 3]  (optional)
    
    Total per step (with key body): 1 + 2 + 3 + 1 + num_dofs + num_tracked_bodies * 3
    Total per step (without key body): 1 + 2 + 3 + 1 + num_dofs
    
    Args:
        env: WholeBodyTrackingManager environment
        future_num_steps: Number of future steps to sample
        future_max_steps: Maximum future horizon in steps
        include_key_body_pos: If True, include local key body positions; if False, omit them.
    
    Returns:
        Tensor of shape [num_envs, future_num_steps * feature_dim]
    """
    from holosoma.utils.rotations import get_euler_xyz, quat_rotate_inverse
    
    motion_command = _get_motion_command_and_assert_type(env)
    is_multi = hasattr(motion_command, 'motion_library')

    # Generate future time step indices (linearly spaced)
    tar_obs_steps = torch.linspace(
        start=1,
        end=future_max_steps,
        steps=future_num_steps,
        device=env.device,
        dtype=torch.long,
    )

    # Current time steps + future offsets, clamped to valid range
    current_time_steps = motion_command.time_steps  # [num_envs]
    future_time_steps = current_time_steps[:, None] + tar_obs_steps[None, :]  # [num_envs, future_num_steps]

    if is_multi:
        lib = motion_command.motion_library
        per_env_totals = lib.time_step_totals[motion_command.motion_indices]
        future_time_steps = torch.min(future_time_steps, (per_env_totals - 1)[:, None])
        future_time_steps = torch.clamp(future_time_steps, min=0)
    else:
        motion = motion_command.motion
        future_time_steps = torch.clamp(future_time_steps, 0, motion.time_step_total - 1)

    num_envs = env.num_envs
    num_steps = future_num_steps

    if is_multi:
        mi = motion_command.motion_indices  # [num_envs]
        mi_exp = mi[:, None].expand(-1, num_steps)  # [num_envs, num_steps]
        root_body_raw_idx = lib._body_indexes[0].item()

        # 3D indexing: lib._body_pos_w[mi, t, body]
        root_pos = lib._body_pos_w[mi_exp, future_time_steps, root_body_raw_idx]
        root_quat = lib._body_quat_w[mi_exp, future_time_steps, root_body_raw_idx]
        root_height = root_pos[..., 2:3]
        roll, pitch, yaw = get_euler_xyz(root_quat.reshape(-1, 4), w_last=True)
        roll_pitch = torch.stack([roll, pitch], dim=-1).reshape(num_envs, num_steps, 2)
        base_lin_vel = lib._body_lin_vel_w[mi_exp, future_time_steps, root_body_raw_idx]
        base_ang_vel = lib._body_ang_vel_w[mi_exp, future_time_steps, root_body_raw_idx]
        base_yaw_vel = base_ang_vel[..., 2:3]
        dof_pos = lib._joint_pos[mi_exp, future_time_steps][:, :, lib._joint_indexes]
        default_dof_pos = env.default_dof_pos_base[:, None, :]
        dof_pos_rel = dof_pos - default_dof_pos

        if include_key_body_pos:
            body_pos_robot_order = lib._body_pos_w[mi_exp, future_time_steps][:, :, lib._body_indexes]
            tracked_body_indexes = motion_command.tracked_body_indexes
            tracked_body_pos = body_pos_robot_order[:, :, tracked_body_indexes]
            root_pos_expanded = root_pos[:, :, None, :]
            root_quat_expanded = root_quat[:, :, None, :]
            local_body_pos = tracked_body_pos - root_pos_expanded
            local_body_pos_flat = local_body_pos.reshape(-1, 3)
            root_quat_flat = root_quat_expanded.expand(-1, -1, local_body_pos.shape[2], -1).reshape(-1, 4)
            local_body_pos_b = quat_rotate_inverse(root_quat_flat, local_body_pos_flat, w_last=True)
            local_key_body_pos = local_body_pos_b.reshape(num_envs, num_steps, -1)
    else:
        # Original single-motion path
        env_indices = torch.arange(num_envs, device=env.device)[:, None].expand(-1, num_steps)
        root_body_raw_idx = motion._body_indexes[0].item()

        root_pos = motion._body_pos_w[future_time_steps, root_body_raw_idx]
        root_quat = motion._body_quat_w[future_time_steps, root_body_raw_idx]
        root_height = root_pos[..., 2:3]
        roll, pitch, yaw = get_euler_xyz(root_quat.reshape(-1, 4), w_last=True)
        roll_pitch = torch.stack([roll, pitch], dim=-1).reshape(num_envs, num_steps, 2)
        base_lin_vel = motion._body_lin_vel_w[future_time_steps, root_body_raw_idx]
        base_ang_vel = motion._body_ang_vel_w[future_time_steps, root_body_raw_idx]
        base_yaw_vel = base_ang_vel[..., 2:3]
        dof_pos = motion._joint_pos[future_time_steps][:, :, motion._joint_indexes]
        default_dof_pos = env.default_dof_pos_base[:, None, :]
        dof_pos_rel = dof_pos - default_dof_pos

        if include_key_body_pos:
            body_pos_robot_order = motion._body_pos_w[future_time_steps][:, :, motion._body_indexes]
            tracked_body_indexes = motion_command.tracked_body_indexes
            tracked_body_pos = body_pos_robot_order[:, :, tracked_body_indexes]
            root_pos_expanded = root_pos[:, :, None, :]
            root_quat_expanded = root_quat[:, :, None, :]
            local_body_pos = tracked_body_pos - root_pos_expanded
            local_body_pos_flat = local_body_pos.reshape(-1, 3)
            root_quat_flat = root_quat_expanded.expand(-1, -1, local_body_pos.shape[2], -1).reshape(-1, 4)
            local_body_pos_b = quat_rotate_inverse(root_quat_flat, local_body_pos_flat, w_last=True)
            local_key_body_pos = local_body_pos_b.reshape(num_envs, num_steps, -1)

    # Build future_obs (common to both paths)
    if include_key_body_pos:
        future_obs = torch.cat([
            root_height, roll_pitch, base_lin_vel, base_yaw_vel, dof_pos_rel, local_key_body_pos,
        ], dim=-1)
    else:
        future_obs = torch.cat([
            root_height, roll_pitch, base_lin_vel, base_yaw_vel, dof_pos_rel,
        ], dim=-1)
    
    # Flatten: [num_envs, num_steps * feature_dim]
    result = future_obs.reshape(num_envs, -1)
    
    # Debug logging (only when save_debug=True, env 0, first 200 timesteps, and not already logged)
    if hasattr(env, '_debug_future_motion_log') and env._debug_future_motion_log is not None:
        current_timestep = motion_command.time_steps[0].item()
        # Initialize logged timesteps set if not exists
        if not hasattr(env, '_debug_logged_timesteps'):
            env._debug_logged_timesteps = set()
        # Only log if we haven't logged this timestep yet
        if current_timestep < 200 and current_timestep not in env._debug_logged_timesteps:
            log_data = {
                'timestep': current_timestep,
                'future_timesteps': future_time_steps[0].cpu().numpy().tolist(),
                'root_height': root_height[0].cpu().numpy().tolist(),
                'roll_pitch': roll_pitch[0].cpu().numpy().tolist(),
                'base_lin_vel': base_lin_vel[0].cpu().numpy().tolist(),
                'base_yaw_vel': base_yaw_vel[0].cpu().numpy().tolist(),
                'dof_pos_rel_first': dof_pos_rel[0, 0].cpu().numpy().tolist(),
                'result_shape': result.shape,
                'result_mean': result[0].mean().item(),
                'result_std': result[0].std().item(),
                # Add motion_command and ref_quat logging
                'motion_command_first_10': motion_command.joint_pos[0, :10].cpu().numpy().tolist() if hasattr(motion_command, 'joint_pos') else None,
                'motion_command_vel_first_10': motion_command.joint_vel[0, :10].cpu().numpy().tolist() if hasattr(motion_command, 'joint_vel') else None,
                'ref_quat_xyzw': motion_command.ref_quat_w[0].cpu().numpy().tolist() if hasattr(motion_command, 'ref_quat_w') else None,
            }
            if include_key_body_pos:
                log_data['local_key_body_pos_first_body'] = local_key_body_pos[0, 0, :3].cpu().numpy().tolist()
            env._debug_future_motion_log.append(log_data)
            env._debug_logged_timesteps.add(current_timestep)
        elif current_timestep == 200 and not getattr(env, '_debug_log_saved', False):
            # Motion reached timestep 200 - save immediately
            import json
            from pathlib import Path
            if hasattr(env, '_debug_log_path') and env._debug_log_path:
                debug_log_path = Path(env._debug_log_path)
                with open(debug_log_path, 'w') as f:
                    json.dump(env._debug_future_motion_log, f, indent=2)
                env._debug_log_saved = True
                from loguru import logger
                logger.info(f"Debug log auto-saved at timestep 200 to: {debug_log_path} ({len(env._debug_future_motion_log)} entries)")
    
    return result


#########################################################################################################
## Height Map Observation (for height map encoder)
#########################################################################################################

def height_map_obs(
    env: WholeBodyTrackingManager,
    height_scan_offset: float = 0.5,
    min_height: float = -5.0,
) -> torch.Tensor:
    """Height map observation for terrain perception via CNN + cross-attention.

    Reads from the IsaacSim ``height_map_scanner`` RayCaster sensor, matching
    KraftonLab's MapScanWrapper implementation. The sensor updates every 5
    policy steps (configured via ``update_period`` in the simulator).

    Returns (x_local, y_local, height) per grid point, where
    height = sensor_z - ray_hit_z - offset (positive = above ground).

    Args:
        env: WholeBodyTrackingManager environment.
        height_scan_offset: Height offset subtracted from scan (default 0.5m).
        min_height: Value to use for missed rays (default -5.0).

    Returns:
        Tensor of shape [num_envs, num_rays * 3].
    """
    scanner = env.simulator.scene.sensors["height_map_scanner"]

    # Build grid coordinates once (matching sensor's GridPatternCfg ordering)
    if not hasattr(env, "_height_map_grid_xy"):
        cfg = scanner.cfg.pattern_cfg
        x = torch.arange(
            start=-cfg.size[0] / 2, end=cfg.size[0] / 2 + 1.0e-9,
            step=cfg.resolution, device=env.device,
        )
        y = torch.arange(
            start=-cfg.size[1] / 2, end=cfg.size[1] / 2 + 1.0e-9,
            step=cfg.resolution, device=env.device,
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        env._height_map_grid_x = grid_x.flatten()  # (num_rays,)
        env._height_map_grid_y = grid_y.flatten()  # (num_rays,)

    num_envs = env.num_envs
    num_rays = env._height_map_grid_x.shape[0]

    # Height = sensor_pos_z - ray_hit_z - offset  (KraftonLab convention)
    height_map = (
        scanner.data.pos_w[:, 2].unsqueeze(1)
        - scanner.data.ray_hits_w[..., 2]
        - height_scan_offset
    )

    # Handle invalid hits
    height_map = torch.where(torch.isnan(height_map), torch.full_like(height_map, min_height), height_map)
    height_map = torch.where(torch.isinf(height_map), torch.full_like(height_map, min_height), height_map)

    # Build output: (num_envs, num_rays, 3) with (x_local, y_local, height)
    map_scan = torch.zeros(num_envs, num_rays, 3, device=env.device)
    map_scan[:, :, 0] = env._height_map_grid_x
    map_scan[:, :, 1] = env._height_map_grid_y
    map_scan[:, :, 2] = height_map

    return map_scan.reshape(num_envs, -1)


def videomimic_height_map_obs(
    env: WholeBodyTrackingManager,
    height_scan_offset: float = 0.5,
    min_height: float = -5.0,
) -> torch.Tensor:
    """VideoMimic-style heightmap obs.

    Reads from the ``videomimic_height_map_scanner`` RayCaster sensor. Grid
    shape / resolution / channel count come from
    ``SimulatorInitConfig.videomimic_height_map``:

    - ``num_channels == 1``: returns height only, shape ``(num_envs, H*W)``.
    - ``num_channels == 3``: returns ``(x_local, y_local, height)`` per grid
      point, shape ``(num_envs, H*W*3)``.

    height = sensor_z - ray_hit_z - offset (positive = above ground).
    """
    scanner = env.simulator.scene.sensors["videomimic_height_map_scanner"]

    height_map = (
        scanner.data.pos_w[:, 2].unsqueeze(1)
        - scanner.data.ray_hits_w[..., 2]
        - height_scan_offset
    )

    height_map = torch.where(torch.isnan(height_map), torch.full_like(height_map, min_height), height_map)
    height_map = torch.where(torch.isinf(height_map), torch.full_like(height_map, min_height), height_map)

    vm_cfg = env.simulator.simulator_config.videomimic_height_map
    if vm_cfg.num_channels == 1:
        return height_map

    if vm_cfg.num_channels != 3:
        raise ValueError(
            f"videomimic_height_map.num_channels must be 1 or 3, got {vm_cfg.num_channels}"
        )

    # xyz variant — build local (x, y) grid once, same convention as height_map_obs.
    if not hasattr(env, "_videomimic_height_map_grid_xy"):
        cfg = scanner.cfg.pattern_cfg
        x = torch.arange(
            start=-cfg.size[0] / 2, end=cfg.size[0] / 2 + 1.0e-9,
            step=cfg.resolution, device=env.device,
        )
        y = torch.arange(
            start=-cfg.size[1] / 2, end=cfg.size[1] / 2 + 1.0e-9,
            step=cfg.resolution, device=env.device,
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        env._videomimic_height_map_grid_x = grid_x.flatten()
        env._videomimic_height_map_grid_y = grid_y.flatten()
        env._videomimic_height_map_grid_xy = True

    num_envs = env.num_envs
    num_rays = env._videomimic_height_map_grid_x.shape[0]
    map_scan = torch.zeros(num_envs, num_rays, 3, device=env.device)
    map_scan[:, :, 0] = env._videomimic_height_map_grid_x
    map_scan[:, :, 1] = env._videomimic_height_map_grid_y
    map_scan[:, :, 2] = height_map
    return map_scan.reshape(num_envs, -1)


#########################################################################################################
## Foot contact observations (intended for critic-only / privileged input)
#########################################################################################################


def _foot_contact_body_indexes(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Cache and return body indexes for the two foot contact points."""
    if not hasattr(env, "_foot_contact_body_indexes"):
        body_names = env.simulator.body_names
        env._foot_contact_body_indexes = torch.tensor(
            [body_names.index("left_foot_contact_point"), body_names.index("right_foot_contact_point")],
            dtype=torch.long,
            device=env.device,
        )
    return env._foot_contact_body_indexes


def actual_foot_contact(env: WholeBodyTrackingManager, threshold: float = 1.0) -> torch.Tensor:
    """Actual robot foot contact as binary (left, right), shape (num_envs, 2).

    Uses ``contact_forces_history`` max-over-history (same pattern as
    :class:`UndesiredContacts`) so brief contact losses between physics substeps
    are still captured.
    """
    indexes = _foot_contact_body_indexes(env)
    # (num_envs, history_length, num_bodies, 3) → (num_envs, 2) float in {0, 1}
    net_forces = env.simulator.contact_forces_history[:, :, indexes, :]
    force_norm = torch.norm(net_forces, dim=-1)
    contact = torch.max(force_norm, dim=1)[0] > threshold
    return contact.float()


def target_foot_contact(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Target foot contact from motion data, shape (num_envs, 2) in {0.0, 1.0}."""
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.motion_foot_contact
