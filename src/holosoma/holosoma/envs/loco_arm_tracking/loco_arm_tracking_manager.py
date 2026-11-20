"""Combined locomotion + arm tracking environment.

LocoArmTrackingManager simultaneously tracks root-frame velocity commands
(walking) and end-effector pose targets (arm tracking). It inherits from
BaseTask and is independent of ArmTrackingManager and
LeggedRobotLocomotionManager.
"""

from __future__ import annotations

import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.torch_utils import torch_rand_float


class LocoArmTrackingManager(BaseTask):
    """Combined locomotion + arm tracking with dual command generation."""

    def __init__(self, tyro_config, *, device):
        # Store FK noise config before super().__init__ triggers manager setup
        self.fk_noise_enabled: bool = getattr(tyro_config, "fk_noise_enabled", False)
        self.fk_noise_sigma: float = getattr(tyro_config, "fk_noise_sigma", 0.0)
        super().__init__(tyro_config, device=device)

    def _get_task_name(self) -> str:
        return "loco_arm_tracking"

    def _load_assets(self):
        """Load robot assets and compute DOF index groups.

        Arm/leg/torso DOF indices are computed eagerly here because action
        term constructors may access them during ActionManager initialization,
        which runs before _init_buffers().
        """
        super()._load_assets()
        self.arm_dof_indices = self._get_arm_dof_indices()
        self.torso_dof_indices = self._get_torso_dof_indices()
        self.leg_dof_indices = self._get_leg_dof_indices()

    def _setup_robot_body_indices(self):
        """Set up body indices needed by the simulator and reward terms."""
        foot_body_names = [
            s for s in self.body_names
            if self.robot_config.foot_body_name in s
        ]
        self.feet_indices = torch.zeros(
            len(foot_body_names), dtype=torch.long,
            device=self.device, requires_grad=False,
        )
        for i, name in enumerate(foot_body_names):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(name)

        foot_height_names = [
            s for s in self.body_names
            if self.robot_config.foot_height_name in s
        ]
        self.feet_height_indices = torch.zeros(
            len(foot_height_names), dtype=torch.long,
            device=self.device, requires_grad=False,
        )
        for i, name in enumerate(foot_height_names):
            self.feet_height_indices[i] = self.simulator.find_rigid_body_indice(name)

        termination_contact_names = []
        for name in self.robot_config.terminate_after_contacts_on:
            termination_contact_names.extend(
                [s for s in self.body_names if name in s]
            )
        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long,
            device=self.device, requires_grad=False,
        )
        for i, name in enumerate(termination_contact_names):
            self.termination_contact_indices[i] = (
                self.simulator.find_rigid_body_indice(name)
            )

        if self.robot_config.has_torso:
            self.torso_name = self.robot_config.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

    def reset_all(self):
        """Reset all robots and return initial observations."""
        self._init_buffers()
        return super().reset_all()

    def _init_buffers(self):
        super()._init_buffers()
        self.base_quat = self.simulator.base_quat
        self.need_to_refresh_envs = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._configure_default_dof_pos()

        # Recompute DOF indices if not already set
        if self.arm_dof_indices is None:
            self.arm_dof_indices = self._get_arm_dof_indices()
        if self.torso_dof_indices is None:
            self.torso_dof_indices = self._get_torso_dof_indices()
        if self.leg_dof_indices is None:
            self.leg_dof_indices = self._get_leg_dof_indices()

        # Control delay randomization (required by action terms)
        self._randomize_ctrl_delay = False

        # Push randomization buffers
        self._push_robots_enabled = False
        self._max_push_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )
        self.record_push_robot_vel_buf = torch.zeros_like(self.push_robot_vel_buf)

        # Step counter for curriculum / logging
        self.common_step_counter = 0

        # Foot touchdown shock reward: track previous contact state
        self._prev_foot_contact = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )

    def _configure_default_dof_pos(self):
        """Set default standing pose from robot config."""
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device
        )
        for i, name in enumerate(self.dof_names):
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(
                    f"Missing default joint angle for DOF '{name}' "
                    f"in robot configuration."
                )
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)
        self.default_dof_pos = self.default_dof_pos_base.repeat(
            self.num_envs, 1
        ).clone()

    # ------------------------------------------------------------------
    # DOF index computation
    # ------------------------------------------------------------------

    def _get_arm_dof_indices(self):
        """Get arm DOF indices for G1 29-DOF."""
        arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        indices = [
            self.dof_names.index(name)
            for name in arm_joint_names
            if name in self.dof_names
        ]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _get_torso_dof_indices(self):
        """Get torso/waist DOF indices from robot config."""
        torso_names = getattr(self.robot_config, "waist_dof_names", None) or []
        indices = [
            self.dof_names.index(n) for n in torso_names if n in self.dof_names
        ]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _get_leg_dof_indices(self):
        """Get leg DOF indices (all DOFs except arms)."""
        all_indices = set(range(self.num_dof))
        arm_indices = set(self.arm_dof_indices.cpu().tolist())
        leg_indices = sorted(all_indices - arm_indices)
        return torch.tensor(leg_indices, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Reset callbacks
    # ------------------------------------------------------------------

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        """Reset to standing pose with small randomization."""
        if len(env_ids) == 0:
            return

        if target_states is not None:
            dof_pos = target_states["dof_pos"]
            dof_vel = target_states["dof_vel"]
            root_states = target_states["root_states"]
        else:
            # Standing pose (same as arm_tracking env)
            dof_pos = self.default_dof_pos[env_ids] * torch_rand_float(
                0.9, 1.1, (len(env_ids), self.num_dof), device=str(self.device)
            )
            dof_vel = torch.zeros(
                (len(env_ids), self.num_dof), device=self.device
            )

            # Reset root state to standing
            root_states = self.base_init_state.unsqueeze(0).repeat(len(env_ids), 1)
            root_states[:, :3] += (
                self.terrain_manager.get_state("locomotion_terrain")
                .env_origins[env_ids]
            )

        # Write to simulator tensors
        self.simulator.dof_pos[env_ids] = dof_pos
        self.simulator.dof_vel[env_ids] = dof_vel
        self.simulator.robot_root_states[env_ids] = root_states

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        """Reset episode counters and flags."""
        self.need_to_refresh_envs[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self._pending_episode_update_mask[env_ids] = True
        self._prev_foot_contact[env_ids] = False

    # ------------------------------------------------------------------
    # Step callbacks
    # ------------------------------------------------------------------

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(
            env_ids, self.simulator.all_root_states
        )
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False

        # Refresh tensors so controller sees valid state
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _pre_compute_observations_callback(self):
        """Refresh base_quat and terrain heights before observation compute."""
        self.base_quat[:] = self.simulator.base_quat[:]
        self.terrain_manager.update_heights()

    def _update_counters_each_step(self):
        self.common_step_counter += 1

    def _update_tasks_callback(self):
        """Update commands, curriculum, and randomization each step."""
        super()._update_tasks_callback()

        # Assign commands to simulator for headless recording
        if (
            hasattr(self.simulator, "headless_recording")
            and self.simulator.headless_recording
        ):
            if hasattr(self.command_manager, "commands"):
                self.simulator.commands = self.command_manager.commands

    def _setup_simulator_control(self):
        """Push command state to simulator for viewer display."""
        self.simulator.commands = self.command_manager.commands

    # ------------------------------------------------------------------
    # Push robots (domain randomization)
    # ------------------------------------------------------------------

    def _push_robots(self, env_ids):
        """Random pushes the robots via randomized base velocity."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 2:
            raise ValueError(
                "Push velocity vector must have exactly 2 components."
            )

        rand = torch.rand(len(env_ids), 2, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = (
            self.push_robot_vel_buf[env_ids].clone()
        )
        self.simulator.robot_root_states[env_ids, 7:9] = (
            self.push_robot_vel_buf[env_ids]
        )
        self.simulator.set_actor_root_state_tensor_robots(
            env_ids, self.simulator.robot_root_states
        )
        self._max_push_vel = max_vel_tensor.clone()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _update_log_dict(self):
        avg = self._get_average_episode_tracker().get_average()
        self.log_dict["average_episode_length"] = avg.detach().cpu()

        # Add arm tracking metrics
        arm_command = self.command_manager.get_state("arm_command")
        if arm_command and hasattr(arm_command, "metrics"):
            self.log_dict.update(arm_command.metrics)

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError(
                "AverageEpisodeLengthTracker not registered."
            )
        return tracker

    @property
    def average_episode_length(self) -> float:
        avg = self._get_average_episode_tracker().get_average()
        return float(avg.detach().cpu().item())

    # ------------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------------

    def set_is_evaluating(self, command=None):
        """Set evaluation mode with optional fixed velocity command."""
        super().set_is_evaluating()
        commands = self.command_manager.commands
        commands.zero_()
        if command is not None:
            command_tensor = torch.as_tensor(
                command, device=self.device, dtype=commands.dtype
            )
            commands[:] = command_tensor.view(1, -1).expand_as(commands)
        gait_state = self.command_manager.get_state("locomotion_gait")
        if gait_state is not None:
            gait_state.set_eval_mode(True)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def get_checkpoint_state(self) -> dict[str, torch.Tensor | float]:
        state: dict[str, torch.Tensor | float] = {}
        state["average_episode_tracker"] = (
            self._get_average_episode_tracker().state_dict()
        )
        return state

    def load_checkpoint_state(
        self, state: dict[str, torch.Tensor | float] | None
    ) -> None:
        if not state:
            return
        tracker_state = state.get("average_episode_tracker")
        if tracker_state is not None:
            tracker = self._get_average_episode_tracker()
            tracker.load_state_dict(tracker_state)
            tracker.suppress_next_update()

    def synchronize_curriculum_state(
        self, *, device: str, world_size: int
    ) -> None:
        if world_size <= 1:
            return
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        tracker = self._get_average_episode_tracker()
        avg_tensor = tracker.get_average().clone().detach().to(device)
        torch.distributed.broadcast(avg_tensor, src=0)
        tracker.set_average(avg_tensor.to(self.device), suppress_update=False)
