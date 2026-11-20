from __future__ import annotations

import importlib

import torch

from holosoma.managers.action.terms.joint_control import JointPositionActionTerm


def _build_controller(cfg, env):
    """Instantiate arm controller from config.

    Returns a ``NullController`` when no controller class is specified,
    ensuring ``base_controller`` is always non-None and exposes
    ``pose_error`` / ``integral_error`` for observation and reward access.
    """
    controller_class = cfg.params.get("controller_class", None)
    if controller_class is None:
        from holosoma.envs.arm_tracking.null_controller import NullController
        return NullController(env)

    module_path, class_name = controller_class.split(":", 1)
    module = importlib.import_module(module_path)
    ControllerClass = getattr(module, class_name)
    controller_kwargs = cfg.params.get("controller_kwargs", {})
    return ControllerClass(env, **controller_kwargs)


class ResidualArmTrackingAction(JointPositionActionTerm):
    """Residual RL action on top of base controller.
    
    Inherits from JointPositionActionTerm to leverage PD control for position→torque conversion.
    
    Architecture:
    - Legs: Full RL control (learns standing balance from scratch)
    - Arms: Base controller + RL residuals (end-effector control with learned corrections)
    """

    def __init__(self, cfg, env):
        # Initialize parent class (sets up PD gains, buffers, etc.)
        super().__init__(cfg, env)
        
        # Arm-specific parameters
        # Residual scale is applied before JointPositionActionTerm action_scale.
        # Effective residual scale = arm_residual_scale * robot.control.action_scale.
        self.arm_residual_scale = cfg.params.get("arm_residual_scale", cfg.params.get("arm_action_scale", 1.0))
        self.arm_clip_actions = cfg.params.get("arm_clip_actions", 1.0)

        # Per-DOF action scale override for arm joints.
        # The global action_scale (e.g. 0.25) is tuned for leg PD stability.
        # Arms benefit from a larger scale (default 1.0) so that
        # arm_residual_scale can be kept smaller while still covering the
        # full joint range, improving fine-control resolution.
        arm_action_scale = cfg.params.get("arm_action_scale_override", None)
        if arm_action_scale is not None:
            self.action_scales[env.arm_dof_indices] = arm_action_scale
        
        # Base controller (Jacobian-based IK, GAC, or NullController for baseline)
        self.base_controller = _build_controller(cfg, env)
        # FastSAC can use this to align tanh action boundaries with policy-controlled DOFs.
        self.policy_dof_indices = torch.arange(self.env.num_dof, dtype=torch.long, device=self.env.device)

        # Cache validity for controller arm targets. This allows observation
        # computation and action processing to share a single controller solve.
        self._base_arm_actions_valid = False

    def setup(self):
        """Initialize buffers after all managers are created."""
        super().setup()  # Initialize parent buffers (torques, action queues, etc.)
        
        # Buffers for monitoring
        self.leg_actions = torch.zeros((self.env.num_envs, len(self.env.leg_dof_indices)), device=self.env.device)
        self.base_arm_actions = torch.zeros((self.env.num_envs, len(self.env.arm_dof_indices)), device=self.env.device)
        self.arm_residuals = torch.zeros_like(self.base_arm_actions)
        self._base_arm_actions_valid = False
        
        # Initialize base controller (always non-None: real controller or NullController)
        self.base_controller.setup()
        # Propagate FK noise settings from env config to controller
        if hasattr(self.env, "fk_noise_enabled"):
            self.base_controller.enable_fk_noise = self.env.fk_noise_enabled
            self.base_controller.fk_noise_sigma = self.env.fk_noise_sigma

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process actions: split legs (direct) and arms (base + residual).
        
        Args:
            actions: RL policy output [num_envs, num_dof]
                     First len(leg_dof_indices) are leg actions (direct)
                     Remaining are arm residuals
        """
        leg_indices = self.env.leg_dof_indices
        arm_indices = self.env.arm_dof_indices
        
        # Split actions
        leg_actions = actions[:, :len(leg_indices)]
        arm_residuals = actions[:, len(leg_indices):]
        
        # Clip residuals, then apply controller-mode residual scaling.
        arm_residuals = (
            torch.clip(arm_residuals, -self.arm_clip_actions, self.arm_clip_actions) * self.arm_residual_scale
        )
        
        # Use cached base arm actions when available (e.g., precomputed for observations).
        base_arm_actions = self.get_base_arm_actions()
        
        # Combine: legs (direct) + arms (base + residual)
        combined_actions = torch.zeros((self.env.num_envs, self.env.num_dof), device=self.env.device)
        combined_actions[:, leg_indices] = leg_actions
        combined_actions[:, arm_indices] = base_arm_actions + arm_residuals
        # if not torch.isfinite(combined_actions).all():
        #     combined_actions = torch.nan_to_num(combined_actions, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store for monitoring
        self.leg_actions[:] = leg_actions
        self.base_arm_actions[:] = base_arm_actions
        self.arm_residuals[:] = arm_residuals

        # Consume current-step cache. Next step should recompute.
        self._base_arm_actions_valid = False
        
        # Pass to parent class for PD control processing (clipping, scaling, delay, torque computation)
        super().process_actions(combined_actions)

    def _fallback_base_arm_actions(self, arm_indices: torch.Tensor) -> torch.Tensor:
        """Fallback base arm actions if controller is not initialized.

        Returns zeros so that _compute_torques reconstructs default_dof_pos:
            target = 0 * action_scale + default_dof_pos = default_dof_pos
        """
        return torch.zeros((self.env.num_envs, len(arm_indices)), device=self.env.device)

    def get_base_arm_actions(self, force_recompute: bool = False) -> torch.Tensor:
        """Return controller arm joint targets for this control step.

        The result is cached so observation terms and action processing can reuse
        the same controller output without an extra solve.
        """
        arm_indices = self.env.arm_dof_indices
        if force_recompute or not self._base_arm_actions_valid:
            self.base_arm_actions[:] = self.base_controller.compute_arm_actions()
            self._base_arm_actions_valid = True
        return self.base_arm_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action buffers."""
        super().reset(env_ids)  # Reset parent buffers (torques, action queues, etc.)
        
        # Reset monitoring buffers
        if env_ids is None:
            self.leg_actions.zero_()
            self.base_arm_actions.zero_()
            self.arm_residuals.zero_()
        else:
            self.leg_actions[env_ids] = 0.0
            self.base_arm_actions[env_ids] = 0.0
            self.arm_residuals[env_ids] = 0.0
        self._base_arm_actions_valid = False

        # Reset base controller internal states (integral error, pose error, etc.)
        self.base_controller.reset(env_ids)

