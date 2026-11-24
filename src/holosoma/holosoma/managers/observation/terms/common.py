"""Observation terms for any policy task: base velocities, projected gravity, joint state and actions."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from holosoma.utils.rotations import quat_rotate_inverse
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.policy_task import PolicyTask


@functools.cache
def _broadcast_axis(axis: tuple[float, ...], device: str | torch.device, num_envs: int) -> torch.Tensor:
    """Read-only ``[num_envs, len(axis)]`` view of a constant axis vector."""
    return to_torch(list(axis), device=device).unsqueeze(0).expand(num_envs, -1)


def gravity_vector(env: PolicyTask, up_axis_idx: int = 2) -> torch.Tensor:
    return _broadcast_axis(tuple(get_axis_params(-1.0, up_axis_idx)), env.device, env.num_envs)


def base_forward_vector(env: PolicyTask) -> torch.Tensor:
    return _broadcast_axis((1.0, 0.0, 0.0), env.device, env.num_envs)


def get_base_lin_vel(env: PolicyTask) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(env.simulator.base_quat, lin_vel_world, w_last=True)


def get_base_ang_vel(env: PolicyTask) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(env.simulator.base_quat, ang_vel_world, w_last=True)


def get_projected_gravity(env: PolicyTask) -> torch.Tensor:
    return quat_rotate_inverse(env.simulator.base_quat, gravity_vector(env), w_last=True)


def base_lin_vel(env: PolicyTask) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: PolicyTask) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: PolicyTask) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: PolicyTask) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: PolicyTask) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: PolicyTask) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action
