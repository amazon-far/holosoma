"""Shared DOF-limit tensor construction.

The soft position limits are a symmetric contraction of the hard limits toward their
midpoint by ``soft_dof_pos_limit``; several backends build the same four limit tensors from
``robot_config``. This module hosts that pure config->tensor math so MuJoCo and IsaacSim share
one implementation, and IsaacGym (whose hard limits come from the URDF props array, not the
config) shares just the contraction formula.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig


def soft_limit_range(lower: float, upper: float, soft_factor: float) -> tuple[float, float]:
    """Contract a ``[lower, upper]`` hard limit toward its midpoint by ``soft_factor``.

    Returns the ``(soft_lower, soft_upper)`` used as the working DOF position limits: with
    ``soft_factor == 1`` this is the hard range, and smaller factors shrink it symmetrically.
    """
    m = (lower + upper) / 2
    r = upper - lower
    return m - 0.5 * r * soft_factor, m + 0.5 * r * soft_factor


def build_dof_limits_from_config(
    robot_config: RobotConfig, num_dof: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the DOF-limit tensors from ``robot_config`` (the config-sourced backends' path).

    Parameters
    ----------
    robot_config : RobotConfig
        Source of the per-DOF position/velocity/effort limit lists and ``soft_dof_pos_limit``.
    num_dof : int
        Number of robot DOFs.
    device : str
        Device for the returned tensors.

    Returns
    -------
    (hard_dof_pos_limits, dof_pos_limits, dof_vel_limits, torque_limits)
        ``hard_dof_pos_limits`` / ``dof_pos_limits`` are ``[num_dof, 2]`` (lower, upper); the
        latter are soft-contracted. ``dof_vel_limits`` / ``torque_limits`` are ``[num_dof]``.
    """
    hard_dof_pos_limits = torch.zeros(num_dof, 2, dtype=torch.float, device=device, requires_grad=False)
    dof_pos_limits = torch.zeros(num_dof, 2, dtype=torch.float, device=device, requires_grad=False)
    dof_vel_limits = torch.zeros(num_dof, dtype=torch.float, device=device, requires_grad=False)
    torque_limits = torch.zeros(num_dof, dtype=torch.float, device=device, requires_grad=False)

    for i in range(num_dof):
        lower = robot_config.dof_pos_lower_limit_list[i]
        upper = robot_config.dof_pos_upper_limit_list[i]
        hard_dof_pos_limits[i, 0] = lower
        hard_dof_pos_limits[i, 1] = upper
        dof_vel_limits[i] = robot_config.dof_vel_limit_list[i]
        torque_limits[i] = robot_config.dof_effort_limit_list[i]
        # Soft limits: symmetric contraction toward the midpoint.
        soft_lower, soft_upper = soft_limit_range(lower, upper, robot_config.soft_dof_pos_limit)
        dof_pos_limits[i, 0] = soft_lower
        dof_pos_limits[i, 1] = soft_upper

    return hard_dof_pos_limits, dof_pos_limits, dof_vel_limits, torque_limits
