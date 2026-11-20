from __future__ import annotations

import torch

from holosoma.managers.observation.terms.arm_tracking import projected_gravity


def base_height_termination(env, min_height: float = 0.3, **kwargs) -> torch.Tensor:
    """Terminate if robot falls."""
    current_height = env.simulator.robot_root_states[:, 2]
    terminated = current_height < min_height
    return terminated


def base_orientation_termination(env, threshold_x: float = 0.7, threshold_y: float = 0.7, **kwargs) -> torch.Tensor:
    """Terminate if projected gravity indicates excessive roll/pitch tilt."""
    grav = projected_gravity(env)
    tilt_x = torch.abs(grav[:, 0]) > threshold_x
    tilt_y = torch.abs(grav[:, 1]) > threshold_y
    return tilt_x | tilt_y


def dof_position_limit_termination(
    env,
    soft_dof_pos_limit: float = 0.95,
    use_arm_dofs_only: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Terminate when joint positions exceed soft joint limits."""
    limits_source = getattr(env.simulator, "hard_dof_pos_limits", env.simulator.dof_pos_limits)
    midpoint = (limits_source[:, 0] + limits_source[:, 1]) / 2.0
    span = limits_source[:, 1] - limits_source[:, 0]
    lower_soft = midpoint - 0.5 * span * soft_dof_pos_limit
    upper_soft = midpoint + 0.5 * span * soft_dof_pos_limit

    q = env.simulator.dof_pos
    if use_arm_dofs_only:
        idx = env.arm_dof_indices
        q = q[:, idx]
        lower_soft = lower_soft[idx]
        upper_soft = upper_soft[idx]

    lower_violation = (q < lower_soft.unsqueeze(0)).any(dim=1)
    upper_violation = (q > upper_soft.unsqueeze(0)).any(dim=1)
    return lower_violation | upper_violation
