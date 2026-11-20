"""Observation terms for the combined loco-arm-tracking environment.

Each function takes ``env`` as its first argument and returns a ``torch.Tensor``
with shape ``[num_envs, ...]``.  Functions are self-contained (no imports from
the sibling arm_tracking or locomotion term modules) so that the
loco_arm_tracking environment remains fully independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from holosoma.utils.rotation_6d import quat_to_rotmat, rot6d_to_rotmat
from holosoma.utils.rotations import quat_rotate_inverse

if TYPE_CHECKING:
    from holosoma.managers.command.terms.locomotion import LocomotionGait


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
    """Retrieve the arm command state from the command manager."""
    arm_command_state = env.command_manager.get_state("arm_command")
    if arm_command_state is None:
        raise AttributeError(
            "arm_command is not registered with the command manager."
        )
    return arm_command_state


def _get_current_ee_pose_in_command_frame(env):
    """Compute current EE poses in the arm command reference frame.

    Returns:
        current_ee_pos: [num_envs, 2, 3]
        current_ee_quat: [num_envs, 2, 4] (xyzw)
    """
    from holosoma.utils.rotation_6d import heading_quat_from_root_quat, world_to_reference_frame

    arm_command_state = _get_arm_command_state(env)
    current_ee_pos_world = env.simulator._rigid_body_pos[
        :, arm_command_state.ee_body_indices, :
    ]
    current_ee_quat_world = env.simulator._rigid_body_rot[
        :, arm_command_state.ee_body_indices, :
    ]

    if arm_command_state.reference_frame == "torso":
        torso_body_index = getattr(
            arm_command_state, "torso_body_index", None
        )
        if torso_body_index is None:
            torso_body_index = env.simulator.find_rigid_body_indice(
                "torso_link"
            )
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
            f"Unsupported reference_frame "
            f"'{arm_command_state.reference_frame}'. "
            f"Expected 'base', 'torso', 'world', 'heading', or 'ground_heading'."
        )

    current_ee_pos = torch.zeros_like(current_ee_pos_world)
    current_ee_quat = torch.zeros_like(current_ee_quat_world)
    for arm_idx in range(current_ee_pos_world.shape[1]):
        current_ee_pos[:, arm_idx], current_ee_quat[:, arm_idx] = (
            world_to_reference_frame(
                current_ee_pos_world[:, arm_idx],
                current_ee_quat_world[:, arm_idx],
                ref_pos,
                ref_quat,
            )
        )
    return current_ee_pos, current_ee_quat


def _get_ee_error_in_command_frame(env):
    """Return translational and rotational configuration errors for both hands.

    Returns:
        pos_error: [num_envs, 2, 3], translation of g_de = g_d^{-1} g
        rot_error: [num_envs, 2, 3, 3], rotation of g_de = g_d^{-1} g
    """
    arm_command_state = _get_arm_command_state(env)
    current_ee_pos, current_ee_quat = _get_current_ee_pose_in_command_frame(
        env
    )

    current_rotmat = quat_to_rotmat(
        current_ee_quat.reshape(-1, 4)
    ).reshape(env.num_envs, 2, 3, 3)
    target_rotmat = rot6d_to_rotmat(
        arm_command_state.target_ee_rot6d.reshape(-1, 6)
    ).reshape(env.num_envs, 2, 3, 3)

    target_rotmat_t = target_rotmat.transpose(-2, -1)
    pos_delta = current_ee_pos - arm_command_state.target_ee_pos
    pos_error = torch.matmul(
        target_rotmat_t, pos_delta.unsqueeze(-1)
    ).squeeze(-1)
    rot_error = torch.matmul(target_rotmat_t, current_rotmat)

    return pos_error, rot_error


def _rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the 6D representation."""
    first_two_columns = rotmat[..., :, :2]
    return first_two_columns.transpose(-2, -1).reshape(
        *rotmat.shape[:-2], 6
    )


def arm_command(env, **kwargs) -> torch.Tensor:
    """SE(3) targets from arm command manager state.

    Shape: [num_envs, 18]
    """
    arm_command_state = _get_arm_command_state(env)
    return arm_command_state.command


def ee_error_pose(env, **kwargs) -> torch.Tensor:
    """End-effector pose error for both hands in command order.

    Order: [left_pos_error(3), left_rot_error_rot6d(6),
            right_pos_error(3), right_rot_error_rot6d(6)]

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


def ee_error_pose_tanh(
    env, temperature: float = 0.01, **kwargs
) -> torch.Tensor:
    """High-gain saturated EE position error for mm-level precision.

    Applies ``tanh(pos_error / temperature)`` per xyz component so that
    small errors produce O(1) observation signals while large errors are
    clamped.

    Shape: [num_envs, 6]  (left_xyz, right_xyz)
    """
    pos_error, _ = _get_ee_error_in_command_frame(env)
    # pos_error: [num_envs, 2, 3]
    saturated = torch.tanh(pos_error / temperature)
    return saturated.reshape(env.num_envs, -1)


def command_lin_vel(env, **kwargs) -> torch.Tensor:
    """Commanded linear velocity (x, y).

    Shape: [num_envs, 2]
    """
    return env.command_manager.commands[:, :2]


def command_ang_vel(env, **kwargs) -> torch.Tensor:
    """Commanded angular velocity (yaw).

    Shape: [num_envs, 1]
    """
    return env.command_manager.commands[:, 2:3]


def sin_phase(env, **kwargs) -> torch.Tensor:
    """Sine of the gait phase.

    Shape: [num_envs, 2]
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError(
            "locomotion_gait is not registered with the command manager."
        )
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.sin(phase)


def cos_phase(env, **kwargs) -> torch.Tensor:
    """Cosine of the gait phase.

    Shape: [num_envs, 2]
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError(
            "locomotion_gait is not registered with the command manager."
        )
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.cos(phase)


def projected_gravity(env, **kwargs) -> torch.Tensor:
    """Gravity vector projected into the base frame.

    Shape: [num_envs, 3]
    """
    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    return quat_rotate_inverse(
        env.simulator.base_quat, gravity, w_last=True
    )


def torso_projected_gravity(env, **kwargs) -> torch.Tensor:
    """Gravity vector projected into the torso (last waist link) frame.

    Provides the critic with direct observability of torso tilt, which is
    the quantity shaped by the ``tracking_torso_orientation`` reward.

    Shape: [num_envs, 3]
    """
    torso_body_index = env.simulator.find_rigid_body_indice("torso_link")
    torso_quat = env.simulator._rigid_body_rot[:, torso_body_index]
    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    return quat_rotate_inverse(torso_quat, gravity, w_last=True)


def base_ang_vel(env, **kwargs) -> torch.Tensor:
    """Root angular velocity in body frame.

    Shape: [num_envs, 3]
    """
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(
        env.simulator.base_quat, ang_vel_world, w_last=True
    )


def base_lin_vel(env, **kwargs) -> torch.Tensor:
    """Root linear velocity in body frame (critic only).

    Shape: [num_envs, 3]
    """
    lin_vel_world = env.simulator.robot_root_states[:, 7:10]
    return quat_rotate_inverse(
        env.simulator.base_quat, lin_vel_world, w_last=True
    )


def dof_pos_relative(env, **kwargs) -> torch.Tensor:
    """Joint positions relative to default joint pose.

    Uses the full [num_envs, num_dof] default_dof_pos tensor since it may
    vary per environment.

    Shape: [num_envs, num_dof]
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env, **kwargs) -> torch.Tensor:
    """Joint velocities.

    Shape: [num_envs, num_dof]
    """
    return env.simulator.dof_vel


def last_residual_policy_action(env, **kwargs) -> torch.Tensor:
    """Last residual-policy action with arm residuals after clip/scale.

    Legs/torso remain in raw policy coordinates. The arm slice is replaced
    with ``action_term.arm_residuals`` so the observation matches the
    residual input actually applied before controller fusion.

    Shape: [num_envs, dim_actions]
    """
    action_term = _get_arm_tracking_action_term(env)
    leg_actions = getattr(action_term, "leg_actions", None)
    arm_residuals = getattr(action_term, "arm_residuals", None)
    if leg_actions is not None and arm_residuals is not None:
        if (
            leg_actions.shape[0] == env.num_envs
            and arm_residuals.shape[0] == env.num_envs
        ):
            return torch.cat([leg_actions, arm_residuals], dim=-1)
    return env.action_manager.action


def controller_arm_joint_command(env, **kwargs) -> torch.Tensor:
    """Controller-produced arm joint command used as residual-policy context.

    Shape: [num_envs, num_arm_dof]
    """
    action_term = _get_arm_tracking_action_term(env)
    get_base_arm_actions = getattr(
        action_term, "get_base_arm_actions", None
    )
    if get_base_arm_actions is None:
        return torch.zeros(
            (env.num_envs, len(env.arm_dof_indices)), device=env.device
        )
    return get_base_arm_actions(force_recompute=False)


def ee_body_twist(env, **kwargs) -> torch.Tensor:
    """End-effector body-frame twist (linear + angular velocity) for both arms.

    Reads the pre-computed ``ee_body_twist`` from the active arm controller.
    This is V_RE_E computed with q_dot[0:3]=0 (no pelvis linear velocity),
    which matches what is available at inference time.

    Order: [left_lin_vel(3), left_ang_vel(3),
            right_lin_vel(3), right_ang_vel(3)]

    Shape: [num_envs, 12]
    """
    action_term = _get_arm_tracking_action_term(env)
    controller = getattr(action_term, "base_controller", None)
    twist = getattr(controller, "ee_body_twist", None)
    if twist is None:
        return torch.zeros((env.num_envs, 12), device=env.device)
    return torch.cat([twist[0], twist[1]], dim=-1)


def ee_body_twist_full(env, **kwargs) -> torch.Tensor:
    """EE body-frame twist with true pelvis linear velocity (critic/reward).

    Reads ``ee_body_twist_true`` from the controller, which is computed as
    J_RE_E @ q_dot_true (pelvis linear velocity not zeroed).

    Shape: [num_envs, 12]
    """
    action_term = _get_arm_tracking_action_term(env)
    controller = getattr(action_term, "base_controller", None)
    twist_true = getattr(controller, "ee_body_twist_true", None)
    if twist_true is None:
        return torch.zeros((env.num_envs, 12), device=env.device)
    return torch.cat([twist_true[0], twist_true[1]], dim=-1)


def prev_prev_action(env, **kwargs) -> torch.Tensor:
    """Action from two steps ago (second-step previous action signal).

    Provides the policy with two-step action history needed for
    acceleration awareness.

    TIMING: read at observation time, after this step's action a_{T-1} was
    pushed, so this holds a_{T-3} relative to the action a_T about to be chosen
    (a_{T-2} is not observed). Kept as is for the policies trained with it
    (v19); v19_fixed uses ``last_raw_action`` / ``prev_raw_action``.

    Shape: [num_envs, dim_actions]
    """
    return env.action_manager.prev_prev_action


def last_raw_action(env, **kwargs) -> torch.Tensor:
    """Raw policy action a_{T-1}, the one applied over the step that just ended.

    Raw means the policy output before any clip, residual scale or controller
    merge (for FastSAC, ``tanh(mu) * action_boundaries``). Paired with
    ``prev_raw_action`` in the same units and adjacent in time; together they
    are what the rate (a_T - a_{T-1}) and acceleration penalties act on.

    Shape: [num_envs, dim_actions]
    """
    return env.action_manager.action


def prev_raw_action(env, **kwargs) -> torch.Tensor:
    """Raw policy action a_{T-2}, one step before ``last_raw_action``.

    Shape: [num_envs, dim_actions]
    """
    return env.action_manager.prev_action


def base_height_command(env, **kwargs) -> torch.Tensor:
    """Return current height target [num_envs, 1]."""
    arm_command = env.command_manager.get_state("arm_command")
    return arm_command.target_base_height.unsqueeze(-1)


def base_height_error(env, **kwargs) -> torch.Tensor:
    """Return signed height error (current - target) [num_envs, 1]."""
    arm_command = env.command_manager.get_state("arm_command")
    current = env.simulator.robot_root_states[:, 2]
    return (current - arm_command.target_base_height).unsqueeze(-1)


def gait_avg_velocity_state(env, **kwargs) -> torch.Tensor:
    """Critic-only obs exposing the gait-averaged velocity accumulator state.

    The gait-averaged leakage penalties read a per-stride running mean that is
    held internal state (not visible in the actor observation). Exposing it to
    the critic restores Markovian-ness of the per-step reward so the value
    function can assign credit correctly. The actor never needs this term
    (it learns the behavior; only the critic needs the bookkeeping).

    Returns [num_envs, 5]:
        [wz_running_mean, v_perp_running_mean,   # current penalized values
         wz_cycle_mean, v_perp_cycle_mean,       # last completed full-cycle mean
         sample_count_norm]                      # cycle progress proxy (count/50)
    """
    state = env.command_manager.get_state("gait_avg_velocity")
    if state is None or state.wz_running_mean is None:
        return torch.zeros(env.num_envs, 5, device=env.device)

    return torch.stack(
        [
            state.wz_running_mean,
            state.v_perp_running_mean,
            state.wz_cycle_mean,
            state.v_perp_cycle_mean,
            state.sample_count / 50.0,
        ],
        dim=-1,
    )


def _commanded_trunk_gravity(env, attr: str) -> torch.Tensor:
    """Commanded gravity direction [num_envs, 3]; upright when not commanded."""
    state = env.command_manager.get_state("arm_command")
    target = getattr(state, attr, None) if state is not None else None
    if target is not None:
        return target
    upright = torch.zeros(env.num_envs, 3, device=env.device)
    upright[:, 2] = -1.0
    return upright


def pelvis_gravity_command(env, **kwargs) -> torch.Tensor:
    """Commanded pelvis gravity direction [num_envs, 3]."""
    return _commanded_trunk_gravity(env, "target_pelvis_gravity").clone()


def torso_gravity_command(env, **kwargs) -> torch.Tensor:
    """Commanded torso gravity direction [num_envs, 3]."""
    return _commanded_trunk_gravity(env, "target_torso_gravity").clone()


def pelvis_gravity_error(env, **kwargs) -> torch.Tensor:
    """Measured minus commanded pelvis gravity direction [num_envs, 3]."""
    return projected_gravity(env) - _commanded_trunk_gravity(env, "target_pelvis_gravity")


def torso_gravity_error(env, **kwargs) -> torch.Tensor:
    """Measured minus commanded torso gravity direction [num_envs, 3]."""
    return torso_projected_gravity(env) - _commanded_trunk_gravity(env, "target_torso_gravity")

