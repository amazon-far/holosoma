"""Action terms for joint-level control."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from holosoma.managers.action.base import ActionTermBase

if TYPE_CHECKING:
    from holosoma.config_types.action import ActionTermCfg


class JointPositionActionTerm(ActionTermBase):
    """Action term for joint position control with PD controller.

    This term processes raw actions as joint position targets and computes
    torques using a PD controller. Supports:
    - Action scaling
    - Action clipping
    - Action delay (if configured)
    - Torque randomization (if configured)
    - Torque clipping
    """

    def __init__(self, cfg: ActionTermCfg, env: Any):
        """Initialize joint position action term.

        Args:
            cfg: Configuration for this action term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        super().__init__(cfg, env)
        if cfg.controller is None:
            raise ValueError("JointPositionActionTerm requires ActionTermCfg.controller.")
        self.controller = cfg.controller

        # Get action dimension from environment
        self._action_dim: int = env.num_dof

        # Initialize action buffers
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._actions_after_delay = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # Initialize torque buffer
        self.torques = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # Sub-step torque history: [num_envs, decimation, num_dof], allocated in setup().
        self._substep_idx: int = 0

        # Cache previous DOF velocities for derivative control
        self._prev_dof_vel = torch.zeros(env.num_envs, env.num_dof, device=env.device)

        # Default actuator scaling (may be overridden by randomization terms)
        self._kp_scale = torch.ones(env.num_envs, self._action_dim, device=env.device)
        self._kd_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim: float = 0.0
        self._randomize_torque_rfi: bool = False

        # PD gains and action scales
        self.p_gains = torch.zeros(self._action_dim, dtype=torch.float, device=env.device)
        self.d_gains = torch.zeros_like(self.p_gains)
        self.i_gains = torch.zeros_like(self.p_gains)
        self.action_scales = torch.zeros_like(self.p_gains)

        self._configure_pd_gains(env)
        self._configure_action_scales(env)

        # Expose references on the environment for backward compatibility
        env.p_gains = self.p_gains
        env.d_gains = self.d_gains
        env.i_gains = self.i_gains
        env.action_scales = self.action_scales

        # Action delay queue will be initialized in setup() after randomization manager is ready
        self.action_queue: torch.Tensor | None = None

    def setup(self) -> None:
        """Setup action term after all managers are initialized.

        Initialize action delay queue if control delay randomization is enabled.
        This must be called after the randomization manager is set up.
        """
        super().setup()

        if getattr(self.env, "_randomize_ctrl_delay", False):
            max_delay = self.env._ctrl_delay_step_range[1]
            self.action_queue = torch.zeros(self.env.num_envs, max_delay + 1, self._action_dim, device=self.env.device)

        # Allocate sub-step torque history buffer
        decimation = self.env.simulator.simulator_config.sim.control_decimation_steps
        self.torques_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)
        self.dof_pos_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)
        self.dof_vel_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)

        # IsaacGym creates randomization buffers before the action manager exists.
        # Once we reach setup(), try attaching any pre-created actuator scales.
        self._attach_actuator_randomizer_scales()

        enabled, rfi_lim = self.env._pending_torque_rfi
        self.configure_torque_rfi(enabled=enabled, rfi_lim=rfi_lim)
        self.env._pending_torque_rfi = (False, 0.0)

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self._action_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions: clip and apply delay if configured.

        Args:
            actions: Raw action tensor [num_envs, action_dim]
        """
        self._substep_idx = 0
        # Store raw actions
        assert self._raw_actions is not None
        self._raw_actions[:] = actions

        # Clip actions
        if self.cfg.clip is not None:
            lower, upper = self.cfg.clip
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError(f"ActionTermCfg.clip must be finite and ordered, got {self.cfg.clip}.")
            assert self._processed_actions is not None
            self._processed_actions[:] = torch.clip(actions, lower, upper)
            # Log clipping fraction
            self.env.log_dict["action_clip_frac"] = (
                (self._processed_actions == lower) | (self._processed_actions == upper)
            ).sum() / self._processed_actions.numel()
        else:
            assert self._processed_actions is not None
            self._processed_actions[:] = actions
            self.env.log_dict["action_clip_frac"] = torch.tensor(0.0)

        # Apply action delay if configured
        if getattr(self.env, "_randomize_ctrl_delay", False):
            self._apply_action_delay()
        else:
            assert self._processed_actions is not None
            self._actions_after_delay[:] = self._processed_actions

    def _apply_action_delay(self) -> None:
        """Apply action delay based on domain randomization settings."""
        assert self.action_queue is not None, "action_queue must be initialized in setup()"
        assert self._processed_actions is not None

        # Update action queue
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = self._processed_actions.clone()

        # Apply uniform delay
        self._actions_after_delay[:] = self.action_queue[
            torch.arange(self.env.num_envs, device=self.env.device), self.env.action_delay_idx
        ].clone()

    def apply_actions(self) -> None:
        """Apply processed actions by computing and applying torques."""
        # Compute torques using PD controller
        self.torques[:] = self._compute_torques(self._actions_after_delay)
        # Record sub-step torques/dof_pos/dof_vel
        self.torques_substep[:, self._substep_idx] = self.torques
        self.dof_pos_substep[:, self._substep_idx] = self.env.simulator.dof_pos
        self.dof_vel_substep[:, self._substep_idx] = self.env.simulator.dof_vel
        self._substep_idx += 1
        # Apply torques to simulator
        self.env.simulator.apply_torques_at_dof(self.torques)
        # Cache velocities for next derivative computation
        self._prev_dof_vel.copy_(self.env.simulator.dof_vel)

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute torques from actions using PD controller.

        Args:
            actions: Action tensor [num_envs, action_dim]

        Returns:
            Torque tensor [num_envs, action_dim]
        """
        # Scale actions
        actions_scaled = actions * self.action_scales

        # Compute torques based on control type
        control_type = self.controller.control_type

        if control_type == "P":
            # Position control
            torques = (
                self._kp_scale * self.p_gains * (actions_scaled + self.env.default_dof_pos - self.env.simulator.dof_pos)
                - self._kd_scale * self.d_gains * self.env.simulator.dof_vel
            )
        elif control_type == "V":
            # Velocity control
            torques = (
                self._kp_scale * self.p_gains * (actions_scaled - self.env.simulator.dof_vel)
                - self._kd_scale * self.d_gains * (self.env.simulator.dof_vel - self._prev_dof_vel) / self.env.sim_dt
            )
        elif control_type == "T":
            # Torque control
            torques = actions_scaled
        else:
            raise ValueError(f"Unknown controller type: {control_type}")

        # Apply torque randomization if configured
        if self._randomize_torque_rfi:
            torques = (
                torques
                + (torch.rand_like(torques) * 2.0 - 1.0) * self._rfi_lim * self._rfi_lim_scale * self.env.torque_limits
            )

        # Clip torques if configured
        if self.controller.clip_torques:
            torques = torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)

        return torques

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action term state.

        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        super().reset(env_ids)

        # Reset action delay queue if applicable
        if self.action_queue is not None:
            if env_ids is None:
                self.action_queue.zero_()
            else:
                self.action_queue[env_ids] = 0.0

        # Reset torques
        if env_ids is None:
            self.torques.zero_()
        else:
            self.torques[env_ids] = 0.0

        # Reset cached velocities
        if env_ids is None:
            self._prev_dof_vel.zero_()
        else:
            self._prev_dof_vel[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Hooks for randomization manager

    def attach_actuator_scales(
        self, kp_scale: torch.Tensor, kd_scale: torch.Tensor, rfi_lim_scale: torch.Tensor
    ) -> None:
        """Attach shared actuator scaling tensors provided by the randomization manager."""
        self._kp_scale = kp_scale
        self._kd_scale = kd_scale
        self._rfi_lim_scale = rfi_lim_scale

    def update_pd_scales(self, env_ids: torch.Tensor, kp_values: torch.Tensor, kd_values: torch.Tensor) -> None:
        """Fallback PD-scale update when no shared buffers are registered."""
        self._kp_scale[env_ids] = kp_values
        self._kd_scale[env_ids] = kd_values

    def update_rfi_scales(self, env_ids: torch.Tensor, rfi_values: torch.Tensor) -> None:
        """Fallback RFI-scale update when no shared buffers are registered."""
        self._rfi_lim_scale[env_ids] = rfi_values

    def configure_torque_rfi(self, *, enabled: bool, rfi_lim: float | None = None) -> None:
        """Configure residual force injection behaviour."""
        if rfi_lim is not None:
            self._rfi_lim = float(rfi_lim)
        if enabled:
            self._require_finite_torque_limits("torque RFI")
        self._randomize_torque_rfi = enabled

    def get_pd_scale_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return references to the PD gain scale buffers."""
        return self._kp_scale, self._kd_scale

    def get_rfi_scale_tensor(self) -> torch.Tensor:
        """Return reference to the RFI limit scale buffer."""
        return self._rfi_lim_scale

    def get_prev_dof_vel(self) -> torch.Tensor:
        """Return cached previous DOF velocities."""
        return self._prev_dof_vel

    # ------------------------------------------------------------------
    # Internal helpers

    def _attach_actuator_randomizer_scales(self) -> None:
        """Attach shared actuator randomizer buffers if they exist."""
        rand_manager = getattr(self.env, "randomization_manager", None)
        if rand_manager is None:
            return

        get_state = getattr(rand_manager, "get_state", None)
        if not callable(get_state):
            return

        state = get_state("actuator_randomizer_state")
        if state is None:
            return

        self.attach_actuator_scales(state.kp_scale_tensor, state.kd_scale_tensor, state.rfi_lim_scale_tensor)

    def _configure_pd_gains(self, env: Any) -> None:
        robot_config = env.robot_config
        # Per-DOF gains from the exact-name / #joint_tag keys; coverage is validated at config construction.
        stiffness, damping = self.controller.resolve_gains(robot_config)
        p_by_name = dict(zip(robot_config.dof_names, stiffness))
        d_by_name = dict(zip(robot_config.dof_names, damping))

        for i, name in enumerate(env.dof_names):
            if name not in p_by_name:
                raise ValueError(f"Simulator DOF '{name}' is not in robot_config.dof_names.")
            self.p_gains[i] = p_by_name[name]
            self.d_gains[i] = d_by_name[name]
            self.i_gains[i] = 0.0

    def _configure_action_scales(self, env: Any) -> None:
        configured = self.cfg.scale
        if isinstance(configured, tuple):
            if len(configured) != self._action_dim:
                raise ValueError(f"ActionTermCfg.scale has {len(configured)} entries, expected {self._action_dim}.")
            base_scales = torch.tensor(configured, dtype=torch.float, device=env.device)
        else:
            if not math.isfinite(configured):
                raise ValueError(f"ActionTermCfg.scale must be finite, got {configured!r}.")
            base_scales = torch.full((self._action_dim,), configured, dtype=torch.float, device=env.device)
        if not bool(torch.isfinite(base_scales).all()):
            raise ValueError(f"ActionTermCfg.scale entries must be finite, got {configured!r}.")

        if self.controller.action_scales_by_effort_limit_over_p_gain:
            invalid = [
                env.dof_names[i]
                for i, effort in enumerate(env.torque_limits.tolist())
                if self.p_gains[i] != 0.0 and not math.isfinite(effort)
            ]
            if invalid:
                raise ValueError(
                    "action_scales_by_effort_limit_over_p_gain requires finite simulator effort limits for "
                    f"DOFs with nonzero P gain; unbounded DOFs: {invalid}. Set dof_effort_limits or disable "
                    "effort-derived action scales."
                )
            for i, effort in enumerate(env.torque_limits.tolist()):
                stiffness = self.p_gains[i]
                if stiffness == 0.0:
                    self.action_scales[i] = 0.0
                else:
                    self.action_scales[i] = base_scales[i] * effort / stiffness
        else:
            self.action_scales[:] = base_scales

    def _require_finite_torque_limits(self, feature: str) -> None:
        invalid = [
            self.env.dof_names[i]
            for i, finite in enumerate(torch.isfinite(self.env.torque_limits).tolist())
            if not finite
        ]
        if invalid:
            raise ValueError(
                f"{feature} requires finite simulator effort limits; unbounded DOFs: {invalid}. "
                "Set dof_effort_limits or disable this feature."
            )
