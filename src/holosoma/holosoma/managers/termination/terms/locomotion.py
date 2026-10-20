"""Locomotion-specific termination terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from holosoma.managers.observation.terms.locomotion import get_projected_gravity
from holosoma.utils.safe_torch_import import torch


def _apply_probability(mask: torch.Tensor, probability: float, device: torch.device) -> torch.Tensor:
    """Optionally apply probabilistic gating to a mask."""
    if probability >= 1.0:
        return mask
    if probability <= 0.0:
        return torch.zeros_like(mask, dtype=torch.bool)
    sample = torch.rand(1, device=device)
    return mask & (sample < probability)


def contact_forces_exceeded(env: Any, bodies: Sequence[str], force_threshold: float = 1.0) -> torch.Tensor:
    """Terminate when any of ``bodies`` (exact names or ``#tag`` references) has a contact force above threshold.

    Note: If you want to disable contact termination, simply don't add this term to your
    termination config instead of using a flag.
    """
    indices = env.robot_names.resolve_body_indices(bodies)
    contact_forces = env.simulator.contact_forces[:, indices, :]
    return torch.any(torch.norm(contact_forces, dim=-1) > force_threshold, dim=1)


def gravity_tilt_exceeded(env: Any, threshold_x: float, threshold_y: float) -> torch.Tensor:
    """Terminate if projected gravity exceeds roll/pitch thresholds."""
    if not getattr(env.config.termination, "terminate_by_gravity", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    grav = get_projected_gravity(env)
    tilt_x = torch.abs(grav[:, 0]) > threshold_x
    tilt_y = torch.abs(grav[:, 1]) > threshold_y
    return tilt_x | tilt_y


def base_height_below_threshold(env: Any, min_height: float) -> torch.Tensor:
    """Terminate if base height drops below threshold."""
    if not getattr(env.config.termination, "terminate_by_low_height", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    base_height = env.simulator.robot_root_states[:, 2]
    return base_height < min_height


def dof_velocity_limit_exceeded(env: Any, probability: float = 1.0) -> torch.Tensor:
    """Terminate when DOF velocity limits are exceeded."""
    if not getattr(env.config.termination, "terminate_when_close_to_dof_vel_limit", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    delta = (
        torch.abs(env.simulator.dof_vel)
        - env.dof_vel_limits * env.config.termination_scales.termination_close_to_dof_vel_limit
    ).clip(min=0.0, max=1.0)
    violation = torch.sum(delta, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)


def torque_limit_exceeded(env: Any, probability: float = 1.0) -> torch.Tensor:
    """Terminate when actuator torques exceed limits."""
    if not getattr(env.config.termination, "terminate_when_close_to_torque_limit", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    torques = env.action_manager.get_term("joint_control").torques
    delta = (
        torch.abs(torques) - env.torque_limits * env.config.termination_scales.termination_close_to_torque_limit
    ).clip(min=0.0, max=1.0)
    violation = torch.sum(delta, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)
