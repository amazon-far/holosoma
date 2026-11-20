from __future__ import annotations

import torch

from holosoma.utils.rotation_6d import (
    heading_quat_from_root_quat,
    quat_to_rot6d,
    quat_to_rotmat,
    rot6d_to_rotmat,
    world_to_reference_frame,
)
from holosoma.utils.rotations import quat_rotate_inverse


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


def arm_command(env, **kwargs) -> torch.Tensor:
    """SE(3) targets from arm command manager state.

    Shape: [num_envs, 18]
    """
    arm_command_state = env.command_manager.get_state("arm_command")
    if arm_command_state is None:
        raise AttributeError("arm_command is not registered with the command manager.")
    return arm_command_state.command


def ee_error_pose(env, **kwargs) -> torch.Tensor:
    """End-effector pose error for both hands in command order.

    Order: [left_pos_error(3), left_rot_error_rot6d(6), right_pos_error(3), right_rot_error_rot6d(6)]

    Shape: [num_envs, 18]
    """
    pos_error, rot_error = _get_ee_error_in_command_frame(env)
    rot_error_rot6d = _rotmat_to_rot6d(rot_error)
    return torch.cat(
        [
            pos_error[:, 0],
            rot_error_rot6d[:, 0],
            pos_error[:, 1],
            rot_error_rot6d[:, 1],
        ],
        dim=-1,
    )


def ee_error_pose_tanh(env, temperature: float = 0.01, **kwargs) -> torch.Tensor:
    """High-gain saturated EE position error for mm-level precision.

    Applies ``tanh(pos_error / temperature)`` per xyz component so that
    small errors produce O(1) observation signals while large errors are
    clamped.  With the default temperature of 0.01 m the linear regime
    covers roughly ±5 mm — exactly the range that matters for fine tracking.

    Shape: [num_envs, 6]  (left_xyz, right_xyz)
    """
    pos_error, _ = _get_ee_error_in_command_frame(env)
    # pos_error: [num_envs, 2, 3]
    saturated = torch.tanh(pos_error / temperature)
    return saturated.reshape(env.num_envs, -1)


def base_pos(env, **kwargs) -> torch.Tensor:
    """Root position in world frame.

    Shape: [num_envs, 3]
    """
    return env.simulator.robot_root_states[:, :3]


def base_rot6d(env, **kwargs) -> torch.Tensor:
    """Root orientation as 6D rotation representation.

    Shape: [num_envs, 6]
    """
    base_quat = env.simulator.robot_root_states[:, 3:7]  # xyzw
    return quat_to_rot6d(base_quat)


def projected_gravity(env, **kwargs) -> torch.Tensor:
    """Gravity vector projected into the base frame.

    Shape: [num_envs, 3]
    """
    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    return quat_rotate_inverse(env.simulator.base_quat, gravity, w_last=True)


def base_ang_vel(env, **kwargs) -> torch.Tensor:
    """Root angular velocity in body frame.

    Shape: [num_envs, 3]
    """
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(env.simulator.base_quat, ang_vel_world, w_last=True)


def dof_pos_relative(env, **kwargs) -> torch.Tensor:
    """Joint positions relative to arm-tracking default joint pose.

    Shape: [num_envs, num_dof]
    """
    return env.simulator.dof_pos - env.default_dof_pos[0]


def dof_vel(env, **kwargs) -> torch.Tensor:
    """Joint velocities.

    Shape: [num_envs, num_dof]
    """
    return env.simulator.dof_vel


def last_action(env, **kwargs) -> torch.Tensor:
    """Last processed action used by joint control term.

    Shape: [num_envs, num_dof]
    """
    action_term = _get_arm_tracking_action_term(env)
    processed_actions = getattr(action_term, "_processed_actions", None)
    if processed_actions is not None:
        return processed_actions
    # Fallback: if only policy-space actions are available (e.g. split control),
    # map them into full-DOF vector using known policy_dof_indices.
    policy_actions = env.action_manager.action
    policy_dof_indices = getattr(action_term, "policy_dof_indices", None)
    if policy_dof_indices is None:
        return policy_actions

    full_actions = torch.zeros((env.num_envs, env.num_dof), device=env.device, dtype=policy_actions.dtype)
    full_actions[:, policy_dof_indices] = policy_actions
    return full_actions


def last_residual_policy_action(env, **kwargs) -> torch.Tensor:
    """Last residual-policy action with arm residuals after residual clip/scale.

    Legs/torso remain in raw policy coordinates. The arm slice is replaced with
    ``action_term.arm_residuals`` so the observation matches the residual input
    actually applied before controller fusion.

    Shape: [num_envs, dim_actions]
    """
    action_term = _get_arm_tracking_action_term(env)
    leg_actions = getattr(action_term, "leg_actions", None)
    arm_residuals = getattr(action_term, "arm_residuals", None)
    if leg_actions is not None and arm_residuals is not None:
        if leg_actions.shape[0] == env.num_envs and arm_residuals.shape[0] == env.num_envs:
            return torch.cat([leg_actions, arm_residuals], dim=-1)
    return env.action_manager.action


def controller_arm_joint_command(env, **kwargs) -> torch.Tensor:
    """Controller-produced arm joint command used as residual-policy context.

    Shape: [num_envs, num_arm_dof]
    """
    action_term = _get_arm_tracking_action_term(env)
    get_base_arm_actions = getattr(action_term, "get_base_arm_actions", None)
    if get_base_arm_actions is None:
        return torch.zeros((env.num_envs, len(env.arm_dof_indices)), device=env.device)
    return get_base_arm_actions(force_recompute=False)


def _get_arm_command_state(env):
    arm_command_state = env.command_manager.get_state("arm_command")
    if arm_command_state is None:
        raise AttributeError("arm_command is not registered with the command manager.")
    return arm_command_state


def _get_current_ee_pose_in_command_frame(env) -> tuple[torch.Tensor, torch.Tensor]:
    arm_command_state = _get_arm_command_state(env)
    current_ee_pos_world = env.simulator._rigid_body_pos[:, arm_command_state.ee_body_indices, :]
    current_ee_quat_world = env.simulator._rigid_body_rot[:, arm_command_state.ee_body_indices, :]

    if arm_command_state.reference_frame == "torso":
        torso_body_index = getattr(arm_command_state, "torso_body_index", None)
        if torso_body_index is None:
            torso_body_index = env.simulator.find_rigid_body_indice("torso_link")
        ref_pos = env.simulator._rigid_body_pos[:, torso_body_index]
        ref_quat = env.simulator._rigid_body_rot[:, torso_body_index]
    elif arm_command_state.reference_frame == "base":
        ref_pos = env.simulator.robot_root_states[:, :3]
        ref_quat = env.simulator.robot_root_states[:, 3:7]
    elif arm_command_state.reference_frame == "heading":
        root_quat = env.simulator.robot_root_states[:, 3:7]
        ref_pos = env.simulator.robot_root_states[:, :3]
        ref_quat = heading_quat_from_root_quat(root_quat)
    elif arm_command_state.reference_frame == "ground_heading":
        root_pos = env.simulator.robot_root_states[:, :3]
        root_quat = env.simulator.robot_root_states[:, 3:7]
        ref_pos = root_pos.clone()
        ref_pos[:, 2] = 0.0
        ref_quat = heading_quat_from_root_quat(root_quat)
    elif arm_command_state.reference_frame == "world":
        return current_ee_pos_world, current_ee_quat_world
    else:
        raise ValueError(
            f"Unsupported reference_frame '{arm_command_state.reference_frame}'. "
            f"Expected 'base', 'torso', 'world', 'heading', or 'ground_heading'."
        )

    current_ee_pos = torch.zeros_like(current_ee_pos_world)
    current_ee_quat = torch.zeros_like(current_ee_quat_world)
    for arm_idx in range(current_ee_pos_world.shape[1]):
        current_ee_pos[:, arm_idx], current_ee_quat[:, arm_idx] = world_to_reference_frame(
            current_ee_pos_world[:, arm_idx],
            current_ee_quat_world[:, arm_idx],
            ref_pos,
            ref_quat,
        )
    return current_ee_pos, current_ee_quat


def _get_ee_error_in_command_frame(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return translational and rotational configuration errors for both hands.

    Returns:
        pos_error: [num_envs, 2, 3], translation of g_de = g_d^{-1} g
        rot_error: [num_envs, 2, 3, 3], rotation of g_de = g_d^{-1} g
    """
    arm_command_state = _get_arm_command_state(env)
    current_ee_pos, current_ee_quat = _get_current_ee_pose_in_command_frame(env)

    current_rotmat = quat_to_rotmat(current_ee_quat.reshape(-1, 4)).reshape(env.num_envs, 2, 3, 3)
    target_rotmat = rot6d_to_rotmat(arm_command_state.target_ee_rot6d.reshape(-1, 6)).reshape(env.num_envs, 2, 3, 3)

    target_rotmat_t = target_rotmat.transpose(-2, -1)
    pos_delta = current_ee_pos - arm_command_state.target_ee_pos
    pos_error = torch.matmul(target_rotmat_t, pos_delta.unsqueeze(-1)).squeeze(-1)
    rot_error = torch.matmul(target_rotmat_t, current_rotmat)

    return pos_error, rot_error


def _rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the 6D representation via first two columns."""
    first_two_columns = rotmat[..., :, :2]
    return first_two_columns.transpose(-2, -1).reshape(*rotmat.shape[:-2], 6)

