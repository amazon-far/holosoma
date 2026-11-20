from __future__ import annotations

import torch

from holosoma.utils.rotation_6d import (
    heading_quat_from_root_quat,
    rot6d_to_rotmat,
    quat_to_rotmat,
    world_to_reference_frame,
)
from holosoma.utils.rotations import quat_apply, quat_rotate_inverse


def _get_arm_tracking_action_term(env):
    """Resolve the active arm-tracking action term via capability-first lookup.

    Iterates registered action manager terms in declaration order and returns
    the first term that exposes a ``get_base_arm_actions`` method.

    Raises:
        KeyError: If no registered action term exposes ``get_base_arm_actions``.
    """
    for _, term in env.action_manager.iter_terms():
        if hasattr(term, "get_base_arm_actions"):
            return term

    registered = [name for name, _ in env.action_manager.iter_terms()]
    raise KeyError(
        f"No action term exposing get_base_arm_actions() found. "
        f"Registered terms: {registered}"
    )


def _get_arm_command_state(env):
    arm_command = env.command_manager.get_state("arm_command")
    if arm_command is None:
        raise AttributeError("arm_command is not registered with the command manager.")
    return arm_command


def _get_current_ee_pose_in_reference_frame(env, arm_command) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current EE position/quaternion in the same frame as target command."""
    current_ee_pos_world = env.simulator._rigid_body_pos[:, arm_command.ee_body_indices, :]
    current_ee_quat_world = env.simulator._rigid_body_rot[:, arm_command.ee_body_indices, :]

    if arm_command.reference_frame == "torso":
        torso_body_index = getattr(arm_command, "torso_body_index", None)
        if torso_body_index is None:
            torso_body_index = env.simulator.find_rigid_body_indice("torso_link")
        ref_pos = env.simulator._rigid_body_pos[:, torso_body_index]
        ref_quat = env.simulator._rigid_body_rot[:, torso_body_index]
    elif arm_command.reference_frame == "base":
        ref_pos = env.simulator.robot_root_states[:, :3]
        ref_quat = env.simulator.robot_root_states[:, 3:7]
    elif arm_command.reference_frame == "heading":
        root_quat = env.simulator.robot_root_states[:, 3:7]
        ref_pos = env.simulator.robot_root_states[:, :3]
        ref_quat = heading_quat_from_root_quat(root_quat)
    elif arm_command.reference_frame == "ground_heading":
        root_pos = env.simulator.robot_root_states[:, :3]
        root_quat = env.simulator.robot_root_states[:, 3:7]
        ref_pos = root_pos.clone()
        ref_pos[:, 2] = 0.0
        ref_quat = heading_quat_from_root_quat(root_quat)
    elif arm_command.reference_frame == "world":
        return current_ee_pos_world, current_ee_quat_world
    else:
        raise ValueError(
            f"Unsupported reference_frame '{arm_command.reference_frame}'. "
            f"Expected 'base', 'torso', 'world', 'heading', or 'ground_heading'."
        )

    current_ee_pos = torch.zeros_like(current_ee_pos_world)
    current_ee_quat = torch.zeros_like(current_ee_quat_world)
    for i in range(2):
        current_ee_pos[:, i], current_ee_quat[:, i] = world_to_reference_frame(
            current_ee_pos_world[:, i], current_ee_quat_world[:, i], ref_pos, ref_quat
        )

    return current_ee_pos, current_ee_quat


def _tracking_errors(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (position squared error, rotational Frobenius/trace error), both [num_envs]."""
    arm_command = _get_arm_command_state(env)
    current_ee_pos, current_ee_quat = _get_current_ee_pose_in_reference_frame(env, arm_command)

    # Position error: mean over left/right arm of squared L2 distance.
    pos_error = torch.sum(torch.square(current_ee_pos - arm_command.target_ee_pos), dim=-1).mean(dim=-1)

    # Rotational error: tr(I - R_current^T R_target) = 3 - tr(R_current^T R_target)
    current_ee_rotmat = quat_to_rotmat(current_ee_quat.reshape(-1, 4)).reshape(current_ee_quat.shape[0], 2, 3, 3)
    target_ee_rotmat = rot6d_to_rotmat(arm_command.target_ee_rot6d.reshape(-1, 6)).reshape(
        arm_command.target_ee_rot6d.shape[0], 2, 3, 3
    )
    rel = torch.matmul(current_ee_rotmat.transpose(-2, -1), target_ee_rotmat)
    trace_rel = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rot_error = (3.0 - trace_rel).mean(dim=-1)

    return pos_error, rot_error


def tracking_ee_pos(env, tracking_sigma: float = 0.02, **kwargs) -> torch.Tensor:
    """Reward end-effector position tracking using exponential error shaping."""
    pos_error, _ = _tracking_errors(env)
    return torch.exp(-pos_error / tracking_sigma)


def tracking_ee_rot(env, tracking_sigma: float = 0.25, **kwargs) -> torch.Tensor:
    """Reward end-effector orientation tracking using Frobenius/trace rotational error."""
    _, rot_error = _tracking_errors(env)
    return torch.exp(-rot_error / tracking_sigma)


def alive(env, **kwargs) -> torch.Tensor:
    """Small per-step alive reward."""
    return torch.ones(env.num_envs, device=env.device)


def penalty_termination(env, **kwargs) -> torch.Tensor:
    """One-time penalty for non-timeout termination."""
    return (env.reset_buf * ~env.time_out_buf).float()


def tracking_orientation(env, tracking_sigma: float = 0.10, **kwargs) -> torch.Tensor:
    """Reward base (pelvis) uprightness using exponential kernel.

    Computes exp(-error / sigma) where error = ||g_xy||^2 is the squared
    magnitude of the projected gravity vector's x/y components in the base
    frame. When perfectly upright error=0 → reward=1. The sigma parameter
    controls the tolerance: at error=sigma the reward is ~0.37.

    For reference, ||g_xy||^2 ≈ sin²(tilt_angle), so:
      - sigma=0.25 → reward ~0.37 at ~30° tilt
      - sigma=0.10 → reward ~0.37 at ~18° tilt
      - sigma=0.50 → reward ~0.37 at ~45° tilt
    """
    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(env.simulator.base_quat, gravity, w_last=True)
    error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    return torch.exp(-error / tracking_sigma)


def tracking_torso_orientation(env, tracking_sigma: float = 0.10, **kwargs) -> torch.Tensor:
    """Reward torso (last waist link) uprightness using exponential kernel.

    Same as tracking_orientation but measured at the torso_link frame (the
    body after waist_pitch_joint) rather than the pelvis. This captures the
    combined effect of all three waist joints (yaw/roll/pitch) and penalises
    torso lean even when the pelvis is upright.

    Sigma semantics are identical to tracking_orientation.
    """
    torso_body_index = env.simulator.find_rigid_body_indice("torso_link")
    torso_quat = env.simulator._rigid_body_rot[:, torso_body_index]

    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(torso_quat, gravity, w_last=True)
    error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    return torch.exp(-error / tracking_sigma)


def penalty_base_ang_vel(env, **kwargs) -> torch.Tensor:
    """Penalize all base angular velocity (body frame x/y/z components).

    For arm-tracking the robot should remain standing still, so yaw rotation
    is equally undesirable as roll and pitch.
    """
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    ang_vel_body = quat_rotate_inverse(env.simulator.base_quat, ang_vel_world, w_last=True)
    return torch.sum(torch.square(ang_vel_body), dim=1)


def penalty_base_lin_vel_xy(env, **kwargs) -> torch.Tensor:
    """Penalize base linear drift in x/y (body frame)."""
    lin_vel_world = env.simulator.robot_root_states[:, 7:10]
    lin_vel_body = quat_rotate_inverse(env.simulator.base_quat, lin_vel_world, w_last=True)
    return torch.sum(torch.square(lin_vel_body[:, :2]), dim=1)


def tracking_base_height(env, base_height_target: float = 0.73, tracking_sigma: float = 3.0, **kwargs) -> torch.Tensor:
    """Reward base height tracking using exponential error shaping.

    Uses exp(-error_cm² / sigma) so the reward is 1.0 at target and decays
    with distance. The sigma parameter controls the tolerance in cm.

    Args:
        base_height_target: Target height in meters (default 0.73 for G1).
        tracking_sigma: Shaping width in cm. At error = sqrt(sigma) cm,
            reward ≈ 0.37. Default 3.0 means ~1.7 cm gives half reward.

    Example with sigma=3.0:
        0 cm error → reward = 1.0
        1 cm error → reward ≈ 0.72
        2 cm error → reward ≈ 0.26
        3 cm error → reward ≈ 0.05
    """
    arm_command = _get_arm_command_state(env)
    if hasattr(arm_command, "target_base_height"):
        target = arm_command.target_base_height
    else:
        target = base_height_target
    error_cm = (env.simulator.robot_root_states[:, 2] - target) * 100.0
    return torch.exp(-torch.square(error_cm) / tracking_sigma)


def penalty_leg_action_rate(env, **kwargs) -> torch.Tensor:
    """Penalize action delta of leg joints between consecutive policy steps."""
    n_leg = len(env.leg_dof_indices)
    prev_leg = env.action_manager.prev_action[:, :n_leg]
    curr_leg = env.action_manager.action[:, :n_leg]
    return torch.sum(torch.square(prev_leg - curr_leg), dim=1)


def penalty_arm_action_rate(env, **kwargs) -> torch.Tensor:
    """Penalize action delta of arm joints between consecutive policy steps."""
    n_leg = len(env.leg_dof_indices)
    prev_arm = env.action_manager.prev_action[:, n_leg:]
    curr_arm = env.action_manager.action[:, n_leg:]
    return torch.sum(torch.square(prev_arm - curr_arm), dim=1)


def penalty_arm_action_accel(env, **kwargs) -> torch.Tensor:
    """Penalize second-order action differences (action acceleration) for arm joints."""
    n_leg = len(env.leg_dof_indices)
    curr = env.action_manager.action[:, n_leg:]
    prev = env.action_manager.prev_action[:, n_leg:]
    prev_prev = env.action_manager.prev_prev_action[:, n_leg:]
    accel = curr - 2 * prev + prev_prev
    return torch.sum(torch.square(accel), dim=1)


def penalty_leg_action_accel(env, **kwargs) -> torch.Tensor:
    """Penalize second-order action differences (action acceleration) for leg joints."""
    n_leg = len(env.leg_dof_indices)
    curr = env.action_manager.action[:, :n_leg]
    prev = env.action_manager.prev_action[:, :n_leg]
    prev_prev = env.action_manager.prev_prev_action[:, :n_leg]
    accel = curr - 2 * prev + prev_prev
    return torch.sum(torch.square(accel), dim=1)


def penalty_arm_residual_l2(env, **kwargs) -> torch.Tensor:
    """Penalize arm residual magnitude."""
    try:
        action_term = _get_arm_tracking_action_term(env)
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)

    arm_residuals = getattr(action_term, "arm_residuals", None)
    if arm_residuals is None:
        # Split-arm controller mode has no arm residual policy branch.
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(arm_residuals), dim=-1)


def penalty_dof_pos_limits(
    env, soft_dof_pos_limit: float = 0.95, use_arm_dofs_only: bool = False, **kwargs
) -> torch.Tensor:
    """Penalize joint positions outside soft limits."""
    limits_source = getattr(env.simulator, "hard_dof_pos_limits", env.simulator.dof_pos_limits)
    m = (limits_source[:, 0] + limits_source[:, 1]) / 2.0
    r = limits_source[:, 1] - limits_source[:, 0]
    lower_soft = m - 0.5 * r * soft_dof_pos_limit
    upper_soft = m + 0.5 * r * soft_dof_pos_limit

    q = env.simulator.dof_pos
    if use_arm_dofs_only:
        idx = env.arm_dof_indices
        q = q[:, idx]
        lower_soft = lower_soft[idx]
        upper_soft = upper_soft[idx]

    out_of_limits = -(q - lower_soft).clip(max=0.0)
    out_of_limits += (q - upper_soft).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def penalty_arm_dof_vel(env, **kwargs) -> torch.Tensor:
    """Penalize high arm joint velocities."""
    arm_vel = env.simulator.dof_vel[:, env.arm_dof_indices]
    return torch.sum(torch.square(arm_vel), dim=1)


def penalty_leg_dof_vel(env, **kwargs) -> torch.Tensor:
    """Penalize high leg joint velocities."""
    leg_vel = env.simulator.dof_vel[:, env.leg_dof_indices]
    return torch.sum(torch.square(leg_vel), dim=1)


def penalty_pose_deviation(
    env,
    include_arm_dofs: bool = False,
    pose_weights: list[float] | None = None,
    **kwargs,
) -> torch.Tensor:
    """Penalize deviation from default pose (locomotion-pose style with arm masking).

    Args:
        include_arm_dofs: If False, arm-joint weights are zeroed.
        pose_weights: Optional per-DOF weights in full DOF order.
    """
    qpos = env.simulator.dof_pos
    qdef = env.default_dof_pos
    pose_error = torch.square(qpos - qdef)

    weights = torch.ones((pose_error.shape[1],), device=env.device, dtype=pose_error.dtype)
    if pose_weights is not None:
        weights = torch.tensor(pose_weights, device=env.device, dtype=pose_error.dtype)
        if weights.numel() != pose_error.shape[1]:
            raise ValueError(
                f"pose_weights length ({weights.numel()}) must match num_dof ({pose_error.shape[1]})."
            )

    if not include_arm_dofs:
        weights = weights.clone()
        weights[env.arm_dof_indices] = 0.0

    return torch.sum(pose_error * weights.unsqueeze(0), dim=1)


def penalty_close_feet_xy(env, close_feet_threshold: float = 0.05, **kwargs) -> torch.Tensor:
    """Penalize when feet are too close together in the xy plane.

    Mirrors the locomotion penalty to discourage crossed or colliding legs
    during standing arm-tracking tasks.
    """
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    # Base forward direction for yaw-aligned lateral distance
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    base_forward = quat_apply(env.base_quat, forward, w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )
    return (feet_distance < close_feet_threshold).float()


def penalty_feet_ori(env, **kwargs) -> torch.Tensor:
    """Penalize feet orientation deviation from flat.

    Mirrors the locomotion penalty to keep feet level during arm tracking.
    """
    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)
    return (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
    )


def _get_mean_action_term(env):
    """Find an action term with mean action buffers by checking for _mean_action_current.

    Returns the first action term that has a ``_mean_action_current`` attribute,
    or None if no such term is registered.
    """
    for _, term in env.action_manager.iter_terms():
        if hasattr(term, "_mean_action_current"):
            return term
    return None


def penalty_mean_action_accel_arm(env, **kwargs) -> torch.Tensor:
    """Penalize second-order differences of mean action for arm joints.

    Computes sum((mu_k - 2·mu_{k-1} + mu_{k-2})² for arm joints) per environment,
    operating on the deterministic policy output (mean action) rather than the
    sampled action.

    Returns zero if mean action buffers are not available.
    """
    term = _get_mean_action_term(env)
    if term is None:
        return torch.zeros(env.num_envs, device=env.device)
    n_leg = len(env.leg_dof_indices)
    curr = term._mean_action_current[:, n_leg:]
    prev = term._mean_action_prev[:, n_leg:]
    prev_prev = term._mean_action_prev_prev[:, n_leg:]
    accel = curr - 2 * prev + prev_prev
    return torch.sum(torch.square(accel), dim=1)


def penalty_mean_action_accel_leg(env, **kwargs) -> torch.Tensor:
    """Penalize second-order differences of mean action for leg joints.

    Computes sum((mu_k - 2·mu_{k-1} + mu_{k-2})² for leg joints) per environment,
    operating on the deterministic policy output (mean action) rather than the
    sampled action.

    Returns zero if mean action buffers are not available.
    """
    term = _get_mean_action_term(env)
    if term is None:
        return torch.zeros(env.num_envs, device=env.device)
    n_leg = len(env.leg_dof_indices)
    curr = term._mean_action_current[:, :n_leg]
    prev = term._mean_action_prev[:, :n_leg]
    prev_prev = term._mean_action_prev_prev[:, :n_leg]
    accel = curr - 2 * prev + prev_prev
    return torch.sum(torch.square(accel), dim=1)

