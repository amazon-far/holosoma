"""Basic locomotion observation terms.

These functions compute individual observation components for legged locomotion tasks.
Each function mirrors the manager-based observation pipeline that replaced the legacy direct `_get_obs_*()` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

# Re-exported so configs naming these terms under this module keep resolving.
from holosoma.managers.observation.terms.common import (  # noqa: F401
    actions,
    base_ang_vel,
    base_forward_vector,
    base_lin_vel,
    dof_pos,
    dof_vel,
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
    projected_gravity,
)

if TYPE_CHECKING:
    from holosoma.envs.policy_task import PolicyTask
    from holosoma.managers.command.terms.locomotion import LocomotionGait


def command_lin_vel(env: PolicyTask) -> torch.Tensor:
    """Commanded linear velocity (x, y).

    Returns:
        Tensor of shape [num_envs, 2]

    Equivalent to:
        env.command_manager.commands[:, :2]
    """
    return env.command_manager.commands[:, :2]


def command_ang_vel(env: PolicyTask) -> torch.Tensor:
    """Commanded angular velocity (yaw).

    Returns:
        Tensor of shape [num_envs, 1]

    Equivalent to:
        env.command_manager.commands[:, 2:3]
    """
    return env.command_manager.commands[:, 2:3]


def sin_phase(env: PolicyTask) -> torch.Tensor:
    """Sine of the gait phase.

    Returns:
        Tensor of shape [num_envs, 1]
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.sin(phase)


def cos_phase(env: PolicyTask) -> torch.Tensor:
    """Cosine of the gait phase.

    Returns:
        Tensor of shape [num_envs, 1]
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.cos(phase)
