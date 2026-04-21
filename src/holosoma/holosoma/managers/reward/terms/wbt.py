"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def penalty_dof_acc(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize joint accelerations.
    
    Computes the sum of squared joint accelerations:
        sum((dof_vel - last_dof_vel) / dt)^2
    
    For environments that just reset, returns 0 to avoid spurious penalties.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    # Initialize last_dof_vel buffer if it doesn't exist
    if not hasattr(env, '_last_dof_vel'):
        env._last_dof_vel = torch.zeros_like(env.simulator.dof_vel)
        env._last_dof_vel[:] = env.simulator.dof_vel
        # Also track if this is the first call per environment
        env._dof_acc_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Verify buffer shapes match
    assert env._last_dof_vel.shape == env.simulator.dof_vel.shape, \
        f"Shape mismatch: _last_dof_vel {env._last_dof_vel.shape} vs dof_vel {env.simulator.dof_vel.shape}"
    
    # Check which environments need initialization (first call or after reset)
    # reset_buf is 1 when environment just reset
    needs_init = env.reset_buf.bool() | ~env._dof_acc_initialized
    
    # For environments that just reset, update last_dof_vel to current velocity
    # This prevents computing acceleration from pre-reset state
    if needs_init.any():
        env._last_dof_vel[needs_init] = env.simulator.dof_vel[needs_init]
    
    # Compute acceleration: (dof_vel - last_dof_vel) / dt
    dof_acc = (env.simulator.dof_vel - env._last_dof_vel) / env.dt
    
    # Compute squared acceleration penalty
    result = torch.sum(torch.square(dof_acc), dim=1)
    
    # Zero out penalty for environments that just reset (should already be ~0, but ensure it)
    result[needs_init] = 0.0
    
    # Update last_dof_vel for next step (after computing acceleration)
    env._last_dof_vel[:] = env.simulator.dof_vel
    env._dof_acc_initialized[:] = True
    
    return result


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class FootContactMatch(RewardTermBase):
    """VideoMimic-style foot contact matching reward.

    Returns the number of feet (left, right) whose actual contact state matches
    the target contact state from the motion data: ``Σ (actual == target)``.
    Range per env: ``{0, 1, 2}`` before scaling by ``weight``.

    - Actual contact = ``contact_forces_history[ankle_roll_link].norm() > threshold``
      (max over history dim, matching :class:`UndesiredContacts`).
    - Target contact = ``motion_command.motion_foot_contact`` (from the npz
      ``foot_contact`` field, binary 0.0/1.0).
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        body_names = env.simulator.body_names  # type: ignore[attr-defined]
        self.foot_body_indexes = torch.tensor(
            [body_names.index("left_foot_contact_point"), body_names.index("right_foot_contact_point")],
            dtype=torch.long,
            device=env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # Actual foot contact: (num_envs, 2) bool
        net_forces = env.simulator.contact_forces_history[:, :, self.foot_body_indexes, :]  # type: ignore[attr-defined]
        actual_contact = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > self.threshold

        # Target foot contact: (num_envs, 2) float in {0.0, 1.0}
        motion_command = _get_motion_command_and_assert_type(env)
        target_contact = motion_command.motion_foot_contact > 0.5  # to bool for equality

        return torch.sum((actual_contact == target_contact).float(), dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
