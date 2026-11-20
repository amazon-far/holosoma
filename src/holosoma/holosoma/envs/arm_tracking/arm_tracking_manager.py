from __future__ import annotations

import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.torch_utils import torch_rand_float


class ArmTrackingManager(BaseTask):
    """Arm tracking with standing balance using residual RL."""

    def __init__(self, tyro_config, *, device):
        # Store FK noise config before super().__init__ triggers manager setup
        self.fk_noise_enabled: bool = getattr(tyro_config, "fk_noise_enabled", False)
        self.fk_noise_sigma: float = getattr(tyro_config, "fk_noise_sigma", 0.0)
        super().__init__(tyro_config, device=device)

    def _get_task_name(self) -> str:
        return "arm_tracking"

    def _load_assets(self):
        """Load robot assets and eagerly compute arm/leg DOF indices.

        Manager term constructors (e.g. ``ResidualArmTrackingAction.__init__``)
        access ``env.arm_dof_indices`` during ``ActionManager`` initialization,
        which runs before ``_init_buffers()``.  Computing the indices here —
        right after ``dof_names`` becomes available — ensures they exist in
        time.
        """
        super()._load_assets()
        self.arm_dof_indices = self._get_arm_dof_indices()
        self.torso_dof_indices = self._get_torso_dof_indices()
        self.leg_dof_indices = self._get_leg_dof_indices()
    
    def reset_all(self):
        """Reset all robots and return initial observations."""
        self._init_buffers()
        return super().reset_all()

    def _init_buffers(self):
        super()._init_buffers()
        self.base_quat = self.simulator.base_quat
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._configure_default_dof_pos()

        # Arm tracking specific buffers (reuse indices computed in _load_assets
        # if already available, otherwise compute now for safety)
        if self.arm_dof_indices is None:
            self.arm_dof_indices = self._get_arm_dof_indices()
        if self.torso_dof_indices is None:
            self.torso_dof_indices = self._get_torso_dof_indices()
        if self.leg_dof_indices is None:
            self.leg_dof_indices = self._get_leg_dof_indices()

        # Foot body indices (needed by foot penalty reward terms)
        foot_body_names = [s for s in self.body_names if self.robot_config.foot_body_name in s]
        self.feet_indices = torch.zeros(len(foot_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(foot_body_names):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(name)
        
        # Control delay randomization (required by action terms)
        self._randomize_ctrl_delay = False

        # Push randomization buffers (used by dynamics randomization)
        self._push_robots_enabled = False
        self._max_push_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.push_robot_vel_buf = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self.record_push_robot_vel_buf = torch.zeros_like(self.push_robot_vel_buf)

    def _configure_default_dof_pos(self):
        """Set default standing pose."""
        self.default_dof_pos_base = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for i, name in enumerate(self.dof_names):
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()

    def _get_arm_dof_indices(self):
        """Get predefined arm DOF indices for G1."""
        # Explicit arm joint names for G1 29-DOF
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
        indices = [self.dof_names.index(name) for name in arm_joint_names if name in self.dof_names]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _get_torso_dof_indices(self):
        """Get torso/waist DOF indices from robot config."""
        torso_names = getattr(self.robot_config, "waist_dof_names", None) or []
        indices = [self.dof_names.index(n) for n in torso_names if n in self.dof_names]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _get_leg_dof_indices(self):
        """Get leg DOF indices (all except arms)."""
        all_indices = set(range(self.num_dof))
        arm_indices = set(self.arm_dof_indices.cpu().tolist())
        leg_indices = sorted(all_indices - arm_indices)
        return torch.tensor(leg_indices, dtype=torch.long, device=self.device)
    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 2:
            raise ValueError("Arm tracking push velocity vector must have exactly 2 components.")

        rand = torch.rand(len(env_ids), 2, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        """Reset to standing pose."""
        # Skip if no envs to reset
        if len(env_ids) == 0:
            return
            
        if target_states is not None:
            dof_pos = target_states["dof_pos"]
            dof_vel = target_states["dof_vel"]
            root_states = target_states["root_states"]
        else:
            if self.is_evaluating and self.num_envs == 1:
                print("Using Fixed initial dof pos")
                dof_pos = torch.tensor([[-0.3096,  0.0000,  0.0000,  0.6860, -0.3939,  0.0000, -0.3199,  0.0000,
                                        0.0000,  0.6941, -0.3851,  0.0000,  0.0000,  0.0000,  0.0000,  0.2051,
                                        0.1873,  0.0000,  0.5973,  0.0000,  0.0000,  0.0000,  0.2167, -0.1841,
                                        0.0000,  0.6078,  0.0000,  0.0000,  0.0000]], device='cuda:0')
                print("dof pos", dof_pos)
            # Standing pose with small randomization
            else:
                dof_pos = self.default_dof_pos[env_ids].clone()

            dof_vel = torch.zeros((len(env_ids), self.num_dof), device=self.device)

            # Reset root state to standing
            root_states = self.base_init_state.unsqueeze(0).repeat(len(env_ids), 1)
            root_states[:, :3] += self.terrain_manager.get_state("locomotion_terrain").env_origins[env_ids]
        
        # Write to simulator tensors
        self.simulator.dof_pos[env_ids] = dof_pos
        self.simulator.dof_vel[env_ids] = dof_vel
        self.simulator.robot_root_states[env_ids] = root_states

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        self.need_to_refresh_envs[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self._pending_episode_update_mask[env_ids] = True

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        
        # CRITICAL: Refresh tensors immediately so controller sees valid state
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _pre_compute_observations_callback(self):
        self.base_quat[:] = self.simulator.base_quat[:]

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
            raise RuntimeError("AverageEpisodeLengthTracker not registered.")
        return tracker
