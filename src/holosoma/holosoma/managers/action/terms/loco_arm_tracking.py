"""Action terms for the combined loco-arm-tracking environment.

Provides these action term classes:

- ``LocoArmResidualAction``: Leg joint position targets (direct) + arm
  residual corrections on top of the GAC controller output.
- ``LocoArmMeanActionResidualAction``: ``LocoArmResidualAction`` plus
  deterministic mean-action history tracking, for the mean-action
  acceleration penalties.

All inherit from ``JointPositionActionTerm`` for PD torque computation.
"""

from __future__ import annotations

import importlib

import torch

from holosoma.managers.action.terms.joint_control import JointPositionActionTerm


def _build_controller(cfg, env):
    """Instantiate arm controller from config.

    Returns a ``NullController`` when no controller class is specified.
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


class LocoArmResidualAction(JointPositionActionTerm):
    """Leg joint targets + arm residual on top of the GAC controller.

    The policy outputs an action vector of dimension ``num_dof``:
    - First ``len(leg_dof_indices)`` elements: direct leg joint commands.
    - Remaining ``len(arm_dof_indices)`` elements: arm residual corrections.

    Arm joint targets are computed as:
        arm_target = gac_output + clip(residual) * arm_residual_scale

    Config params:
        arm_residual_scale (float): Scale for arm residuals (default: 3.0).
        arm_clip_actions (float): Clip bound for arm residuals (default: 1.0).
        controller_class (str): Dotted path to controller class.
        controller_kwargs (dict): Extra kwargs for controller constructor.
        arm_action_scale_override (float|None): Override action_scale for arm DOFs.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.arm_residual_scale = cfg.params.get("arm_residual_scale", 3.0)
        self.arm_clip_actions = cfg.params.get("arm_clip_actions", 1.0)

        # Optional per-DOF action scale override for arm joints.
        arm_action_scale = cfg.params.get("arm_action_scale_override", None)
        if arm_action_scale is not None:
            self.action_scales[env.arm_dof_indices] = arm_action_scale

        # Base controller (GAC or NullController).
        self.base_controller = _build_controller(cfg, env)

        # For observation term / FastSAC compatibility.
        self.policy_dof_indices = torch.arange(
            env.num_dof, dtype=torch.long, device=env.device
        )

        self._base_arm_actions_valid = False

    def setup(self):
        """Initialize buffers after all managers are created."""
        super().setup()

        num_envs = self.env.num_envs
        device = self.env.device

        self.leg_actions = torch.zeros(
            (num_envs, len(self.env.leg_dof_indices)), device=device
        )
        self.base_arm_actions = torch.zeros(
            (num_envs, len(self.env.arm_dof_indices)), device=device
        )
        self.arm_residuals = torch.zeros_like(self.base_arm_actions)
        self._processed_actions = torch.zeros(
            (num_envs, self.env.num_dof), device=device
        )
        self._base_arm_actions_valid = False

        # Initialize base controller.
        self.base_controller.setup()
        if hasattr(self.env, "fk_noise_enabled"):
            self.base_controller.enable_fk_noise = self.env.fk_noise_enabled
            self.base_controller.fk_noise_sigma = self.env.fk_noise_sigma

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process actions: legs direct, arms = GAC + residual."""
        leg_indices = self.env.leg_dof_indices
        arm_indices = self.env.arm_dof_indices

        leg_actions = actions[:, : len(leg_indices)]
        arm_residuals = actions[:, len(leg_indices):]

        # Clip and scale arm residuals.
        arm_residuals = (
            torch.clip(
                arm_residuals, -self.arm_clip_actions, self.arm_clip_actions
            )
            * self.arm_residual_scale
        )

        # Get GAC controller output (cached per step).
        base_arm_actions = self.get_base_arm_actions()

        # Compose full-DOF command.
        combined = torch.zeros(
            (self.env.num_envs, self.env.num_dof), device=self.env.device
        )
        combined[:, leg_indices] = leg_actions
        combined[:, arm_indices] = base_arm_actions + arm_residuals

        # Store for monitoring / observation terms.
        self.leg_actions[:] = leg_actions
        self.base_arm_actions[:] = base_arm_actions
        self.arm_residuals[:] = arm_residuals
        self._processed_actions[:] = combined

        # Invalidate cache for next step.
        self._base_arm_actions_valid = False

        # Pass combined command to parent for PD control.
        super().process_actions(combined)

    def get_base_arm_actions(
        self, force_recompute: bool = False
    ) -> torch.Tensor:
        """Return GAC controller arm joint targets (cached per step)."""
        if force_recompute or not self._base_arm_actions_valid:
            self.base_arm_actions[:] = self.base_controller.compute_arm_actions()
            self._base_arm_actions_valid = True
        return self.base_arm_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action buffers and controller state."""
        super().reset(env_ids)

        if env_ids is None:
            self.leg_actions.zero_()
            self.base_arm_actions.zero_()
            self.arm_residuals.zero_()
            self._processed_actions.zero_()
        else:
            self.leg_actions[env_ids] = 0.0
            self.base_arm_actions[env_ids] = 0.0
            self.arm_residuals[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
        self._base_arm_actions_valid = False

        self.base_controller.reset(env_ids)


class LocoArmMeanActionResidualAction(LocoArmResidualAction):
    """Loco-arm residual action with true policy mean action buffer tracking.

    Extends LocoArmResidualAction to track the deterministic policy mean
    history (mu_k, mu_{k-1}, mu_{k-2}) for use by the mean-action
    acceleration penalty.

    When ``use_true_action_mean`` is True (default), the true mean is
    provided by the algorithm (FastSAC) via the _incoming_action_mean
    attribute set by the ActionManager. Falls back to the sampled action
    if not available.

    Config params (in addition to LocoArmResidualAction params):
        use_true_action_mean (bool): Whether to use the true policy mean
            (default: True).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.use_true_action_mean = cfg.params.get("use_true_action_mean", True)

    def setup(self):
        """Initialize mean action buffers after all managers are created."""
        super().setup()

        self._warned_no_action_mean = False

        # Mean action history buffers (all shape [num_envs, num_dof])
        self._mean_action_current = torch.zeros(
            (self.env.num_envs, self.env.num_dof), device=self.env.device
        )
        self._mean_action_prev = torch.zeros(
            (self.env.num_envs, self.env.num_dof), device=self.env.device
        )
        self._mean_action_prev_prev = torch.zeros(
            (self.env.num_envs, self.env.num_dof), device=self.env.device
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Track mean action history, then delegate to parent pipeline."""
        # Shift mean action history buffers
        self._mean_action_prev_prev[:] = self._mean_action_prev
        self._mean_action_prev[:] = self._mean_action_current

        # Use the true policy mean if configured and available
        action_mean = getattr(self, "_incoming_action_mean", None)
        if self.use_true_action_mean and action_mean is not None:
            self._mean_action_current[:] = action_mean
        else:
            if self.use_true_action_mean and not self._warned_no_action_mean:
                import warnings
                warnings.warn(
                    "LocoArmMeanActionResidualAction: use_true_action_mean=True "
                    "but no action_mean provided by the algorithm. Falling back "
                    "to sampled action.",
                    stacklevel=2,
                )
                self._warned_no_action_mean = True
            self._mean_action_current[:] = actions

        # Delegate to parent for normal action pipeline
        super().process_actions(actions)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset mean action buffers for specified environments."""
        super().reset(env_ids)

        if env_ids is None:
            self._mean_action_current.zero_()
            self._mean_action_prev.zero_()
            self._mean_action_prev_prev.zero_()
        else:
            self._mean_action_current[env_ids] = 0.0
            self._mean_action_prev[env_ids] = 0.0
            self._mean_action_prev_prev[env_ids] = 0.0

