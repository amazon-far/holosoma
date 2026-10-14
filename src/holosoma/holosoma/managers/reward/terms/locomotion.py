"""Reward terms for locomotion tasks.

These terms are migrated from LeggedRobotBase._reward_* methods to be
compatible with the reward manager system.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from holosoma.managers.observation.terms.locomotion import (
    base_forward_vector,
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)
from holosoma.simulator.shared.dof_limits import soft_limits
from holosoma.utils.rotations import (
    quat_apply,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
    from holosoma.managers.command.terms.locomotion import LocomotionGait


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile."""

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)
    stance = cubic_bezier_interpolation(torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x)
    swing = cubic_bezier_interpolation(torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1)
    return torch.where(x <= 0.5, stance, swing)


# ================================================================================================
# Termination Rewards
# ================================================================================================


def termination(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Terminal reward/penalty for early termination (excluding timeouts).

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return (env.reset_buf * ~env.time_out_buf).float()


# ================================================================================================
# Penalty Rewards
# ================================================================================================


def penalty_action_rate(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def penalty_orientation(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize non-flat base orientation.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    projected = get_projected_gravity(env)
    return torch.sum(torch.square(projected[:, :2]), dim=1)


def _is_axis_index(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def penalty_feet_ori(
    env: LeggedRobotLocomotionManager, feet: Sequence[str] = ("#feet",), sole_axes: Sequence[int] = (0, 1)
) -> torch.Tensor:
    """Penalize feet orientation deviation from flat.

    Args:
        env: The environment instance
        feet: Foot bodies (exact names or ``#tag`` references); the per-foot penalty is summed over them
        sole_axes: The two foot-frame axes spanning the sole plane; gravity projected onto them is the tilt.
            ``(0, 1)`` for a z-up foot frame, ``(0, 2)`` for a foot link whose x-z plane is the sole.

    Returns:
        Reward tensor [num_envs]
    """
    axes = [int(a) for a in sole_axes] if all(_is_axis_index(a) for a in sole_axes) else []
    if len(axes) != 2 or len(set(axes)) != 2 or not all(0 <= a <= 2 for a in axes):
        raise ValueError(
            f"penalty_feet_ori: sole_axes must be two distinct integer axes in 0..2, got {list(sole_axes)}"
        )
    idx = env.robot_names.resolve_body_indices(feet)
    quats = env.simulator._rigid_body_rot[:, idx]  # [N, F, 4]
    num_envs, num_feet = quats.shape[:2]
    gravity = gravity_vector(env).unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 3)
    foot_gravity = quat_rotate_inverse(quats.reshape(-1, 4), gravity, w_last=True).reshape(num_envs, num_feet, 3)
    tilt = foot_gravity[..., axes]
    return torch.sum(torch.sum(torch.square(tilt), dim=-1) ** 0.5, dim=1)


# ================================================================================================
# Limit Rewards
# ================================================================================================


def limits_dof_pos(env: LeggedRobotLocomotionManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    lower_soft_limit, upper_soft_limit = soft_limits(env.simulator.dof_pos_limits, soft_dof_pos_limit).unbind(1)

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ================================================================================================
# Tracking and Task Rewards
# ================================================================================================


def tracking_lin_vel(env: LeggedRobotLocomotionManager, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - get_base_lin_vel(env)[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def tracking_ang_vel(env: LeggedRobotLocomotionManager, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    ang_vel = get_base_ang_vel(env)
    ang_vel_error = torch.square(commands[:, 2] - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def penalty_ang_vel_xy(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize xy axes base angular velocity.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    ang_vel = get_base_ang_vel(env)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def penalty_close_feet_xy(
    env: LeggedRobotLocomotionManager, close_feet_threshold: float = 0.05, feet: Sequence[str] = ("#feet",)
) -> torch.Tensor:
    """Penalize when feet are too close together in xy plane.

    Args:
        env: The environment instance
        close_feet_threshold: Minimum distance threshold between feet
        feet: The left/right foot pair (exact names or ``#tag`` references); must resolve to exactly 2 bodies

    Returns:
        Reward tensor [num_envs]
    """
    idx = env.robot_names.resolve_body_indices(feet)
    if idx.numel() != 2:
        raise ValueError(
            f"penalty_close_feet_xy is a lateral left/right pair term: feet={list(feet)!r} resolved to "
            f"{idx.numel()} bodies ({env.robot_names.resolve_body_names(feet)}), expected exactly 2."
        )
    feet_xy = env.simulator._rigid_body_pos[:, idx, :2]  # [N, 2, 2]
    left_foot_xy, right_foot_xy = feet_xy[:, 0], feet_xy[:, 1]

    # Get base orientation
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    # Calculate perpendicular distance in base-local coordinates
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )

    # Return penalty when feet are too close
    return (feet_distance < close_feet_threshold).float()


def base_height(
    env: LeggedRobotLocomotionManager,
    desired_base_height: float = 0.89,
    zero_vel_penalty_scale: float = 1.0,
    stance_penalty_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize base height away from target.

    Args:
        env: The environment instance
        desired_base_height: Target base height
        zero_vel_penalty_scale: Multiplier for base height penalty when robot has zero velocity commands
        stance_penalty_scale: Multiplier for base height penalty when robot is in stance mode

    Returns:
        Reward tensor [num_envs]
    """
    base_height_penalty = torch.square(
        env.terrain_manager.get_state("locomotion_terrain").base_heights - desired_base_height
    )

    # Apply stronger penalty for zero velocity commands if configured
    if zero_vel_penalty_scale != 1.0:
        commands = env.command_manager.commands
        zero_vel_mask = torch.norm(commands[:, :2], dim=1) < 0.1
        base_height_penalty = torch.where(
            zero_vel_mask, base_height_penalty * zero_vel_penalty_scale, base_height_penalty
        )

    # Apply stronger penalty for stance mode if configured (used in decoupled locomotion)
    if stance_penalty_scale != 1.0 and hasattr(env, "stance_mask"):
        base_height_penalty = torch.where(
            env.stance_mask, base_height_penalty * stance_penalty_scale, base_height_penalty
        )

    return base_height_penalty


def feet_phase(
    env: LeggedRobotLocomotionManager, swing_height: float = 0.08, tracking_sigma: float = 0.25
) -> torch.Tensor:
    """Reward for tracking desired foot height based on gait phase.

    Based on MuJoCo Playground's implementation.

    Args:
        env: The environment instance
        swing_height: Maximum height during swing phase
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    # Foot heights relative to the terrain, one column per foot: [N, F]
    feet_heights = env.terrain_manager.get_state("locomotion_terrain").feet_heights
    gait_state = cast("LocomotionGait", env.command_manager.get_state("locomotion_gait"))
    phase = cast("torch.Tensor", gait_state.phase)  # [N, F]
    if feet_heights.ndim != 2 or feet_heights.shape[1] != phase.shape[1]:
        raise ValueError(
            f"feet_phase needs one terrain foot height per gait phase column: feet_heights has shape "
            f"{tuple(feet_heights.shape)}, phase has {phase.shape[1]} columns. Declare the robot's 'feet_height' "
            "body tag (or set TerrainTermCfg.feet_height_bodies) so the terrain term raycasts one height per foot."
        )

    # Expected foot height from the phase, per foot; squared error summed over feet
    expected_heights = _expected_foot_height(phase, swing_height)
    total_error = torch.sum(torch.square(feet_heights - expected_heights), dim=1)

    return torch.exp(-total_error / tracking_sigma)


def pose(
    env: LeggedRobotLocomotionManager,
    pose_weights: list[float],
) -> torch.Tensor:
    """Reward for maintaining default pose.

    Penalizes deviation from default joint positions with weighted importance.

    Args:
        env: The environment instance
        pose_weights: List of weights for each DOF (must match num_dof)

    Returns:
        Reward tensor [num_envs]
    """
    # Get current joint positions
    qpos = env.simulator.dof_pos

    # Convert pose_weights to tensor
    weights = torch.tensor(pose_weights, device=env.device, dtype=torch.float32)

    # Calculate squared deviation from default pose
    # Use env.default_dof_pos which is already set up from robot config
    pose_error = torch.square(qpos - env.default_dof_pos)

    # Weight and sum the errors
    weighted_error = pose_error * weights.unsqueeze(0)

    return torch.sum(weighted_error, dim=1)


def penalty_stumble(env: LeggedRobotLocomotionManager, feet: Sequence[str] = ("#feet",)) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces.

    Args:
        env: The environment instance
        feet: Foot bodies (exact names or ``#tag`` references)

    Returns:
        Reward tensor [num_envs]
    """
    idx = env.robot_names.resolve_body_indices(feet)
    return torch.any(
        torch.norm(env.simulator.contact_forces[:, idx, :2], dim=2)
        > 4 * torch.abs(env.simulator.contact_forces[:, idx, 2]),
        dim=1,
    )


def alive(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Reward for staying alive.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)
