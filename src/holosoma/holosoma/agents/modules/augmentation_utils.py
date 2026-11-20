from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch


class SymmetryUtils:
    """Symmetry utilities for humanoid robots using configuration-driven approach.

    This class provides symmetry transformations for data augmentation and mirroring
    operations. All robot-specific information is loaded from the environment's
    robot configuration.
    """

    def __init__(self, env: Any) -> None:
        """Initializes the symmetry utilities.

        Parameters
        ----------
        env : Any
            Environment object containing robot configuration and observation specifications.
        """
        self.env = env
        self.robot_config = env.robot_config

        # Initialize attributes that will be set during initialization
        self.observation_dims: Dict[str, int] = {}
        self.observation_dims_single_frame: Dict[str, int] = {}  # Dimension without history
        self.history_lengths: Dict[str, int] = {}
        self.sub_observation_keys: Dict[str, List[str]] = {}
        self.sub_observation_indices: Dict[str, Dict[str, torch.Tensor]] = {}
        self.sub_observation_indices_single_frame: Dict[
            str, Dict[str, torch.Tensor]
        ] = {}  # Indices within single frame
        self.sub_observation_dims: Dict[str, int] = {}
        self.joint_index_map: torch.Tensor = torch.empty(0)
        self.sign_flip_mask: torch.Tensor = torch.empty(0)

        self._init_observation_config()
        self._init_robot_config()

    def _init_observation_config(self) -> None:
        """Initializes observation configuration from environment.

        This method sets up observation dimensions, indices, and keys required for parsing
        and transforming observation tensors. This is generic across all humanoid robots.
        """
        self.history_lengths = self.env.history_length

        # Check if using observation manager or direct system
        if getattr(self.env, "observation_manager", None):
            # Observation manager system - get info from observation manager
            self._init_observation_config_from_manager()
        else:
            # Direct observation system
            self._init_observation_config_direct()

    def _init_observation_config_direct(self) -> None:
        """Initialize observation config from direct system."""
        # sub_observation_dims contains the dimension of each observation component
        # e.g.) sub_observation_dims: {base_ang_vel: 3, projected_gravity: 3, ...}
        # observation_dims contains the total dimension per observation type (including history)
        # e.g.) observation_dims: {actor_obs: (3 + 3 + ...) * history_length}
        self.sub_observation_dims = self.env.config.obs.obs_dims
        self.observation_dims = self.env.dim_obs

        self.sub_observation_indices = {}
        self.sub_observation_indices_single_frame = {}
        self.sub_observation_keys = {}
        self.observation_dims_single_frame = {}

        for obs_key, obs_config in self.env.config.obs.obs_dict.items():
            # e.g.) obs_key: actor_obs, obs_config: [base_ang_vel, projected_gravity, ...]
            sub_obs_keys = sorted(obs_config)
            self.sub_observation_keys[obs_key] = sub_obs_keys
            self.sub_observation_indices[obs_key] = {}
            self.sub_observation_indices_single_frame[obs_key] = {}
            idx = 0
            for key in sub_obs_keys:
                # Indices for single frame (used during mirroring after reshape)
                self.sub_observation_indices_single_frame[obs_key][key] = torch.arange(
                    idx, idx + self.sub_observation_dims[key], device=self.env.device
                )
                # For direct config, observations may not have explicit history handling in indices
                # These indices work with the full flattened observation
                self.sub_observation_indices[obs_key][key] = torch.arange(
                    idx, idx + self.sub_observation_dims[key], device=self.env.device
                )
                idx += self.sub_observation_dims[key]

            # Store single-frame dimension (sum of all sub-observation dimensions)
            self.observation_dims_single_frame[obs_key] = idx

    def _init_observation_config_from_manager(self) -> None:
        """Initialize observation config from observation manager."""
        import torch

        obs_manager = self.env.observation_manager

        # Compute sub_observation_dims: dimension of each observation term
        self.sub_observation_dims = {}
        self.observation_dims = {}
        self.observation_dims_single_frame = {}
        self.sub_observation_keys = {}
        self.sub_observation_indices = {}
        self.sub_observation_indices_single_frame = {}

        for group_name, group_cfg in obs_manager.cfg.groups.items():
            # Compute dimension for each term in this group
            # Only include enabled terms (disabled terms are excluded from the
            # observation vector by the ObservationManager)
            term_keys = sorted(
                name for name, cfg in group_cfg.terms.items() if cfg.enabled
            )
            self.sub_observation_keys[group_name] = term_keys
            self.sub_observation_indices[group_name] = {}
            self.sub_observation_indices_single_frame[group_name] = {}

            idx = 0
            idx_single_frame = 0
            for term_name in term_keys:
                # Compute the term once to get its dimension
                term_cfg = group_cfg.terms[term_name]
                obs = obs_manager._compute_term(group_name, term_name, term_cfg)
                term_dim = obs.shape[1]

                # Store the base dimension (without history)
                self.sub_observation_dims[term_name] = term_dim

                # Create indices for this term in the concatenated observation
                # Account for history at group level
                if group_cfg.history_length > 1:
                    term_dim_with_history = term_dim * group_cfg.history_length
                else:
                    term_dim_with_history = term_dim

                # Indices for full observation (with history flattened)
                self.sub_observation_indices[group_name][term_name] = torch.arange(
                    idx, idx + term_dim_with_history, device=self.env.device
                )
                # Indices for single frame (used during mirroring after reshape)
                self.sub_observation_indices_single_frame[group_name][term_name] = torch.arange(
                    idx_single_frame, idx_single_frame + term_dim, device=self.env.device
                )
                idx += term_dim_with_history
                idx_single_frame += term_dim

            # Total dimension for this group (with history)
            self.observation_dims[group_name] = idx
            # Total dimension for this group (single frame, without history)
            self.observation_dims_single_frame[group_name] = idx_single_frame

    def _init_robot_config(self) -> None:
        """Initializes robot configuration from config files.

        This method loads joint mappings and sign flip configurations directly from
        the robot config.
        """
        # Validate required config attributes
        if not self.robot_config.dof_names:
            raise ValueError("Robot config must contain 'dof_names' list")
        if not self.robot_config.symmetry_joint_names:
            raise ValueError("Robot config must contain 'symmetry_joint_names' mapping")
        if not self.robot_config.flip_sign_joint_names:
            raise ValueError("Robot config must contain 'flip_sign_joint_names' list")

        # Get joint configuration from robot config
        dof_names = self.robot_config.dof_names
        joint_name_mapping = self.robot_config.symmetry_joint_names
        sign_flip_joints = self.robot_config.flip_sign_joint_names

        # Create name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(dof_names)}

        # Build joint index mapping
        joint_index_mapping = {}
        for joint1, joint2 in joint_name_mapping.items():
            if joint1 in name_to_idx and joint2 in name_to_idx:
                joint_index_mapping[name_to_idx[joint1]] = name_to_idx[joint2]

        # Create joint mapping tensor
        self.joint_index_map = torch.tensor(
            [joint_index_mapping.get(i, i) for i in range(len(dof_names))], device=self.env.device, dtype=torch.long
        )

        # Create sign flip mask
        sign_flip_indices = {name_to_idx[name] for name in sign_flip_joints if name in name_to_idx}
        self.sign_flip_mask = torch.tensor(
            [-1.0 if i in sign_flip_indices else 1.0 for i in range(len(dof_names))],
            device=self.env.device,
            dtype=torch.float,
        )


    def augment_observations(self, obs: torch.Tensor, env: Any, obs_list: Sequence[str]) -> torch.Tensor:
        """Applies x-z plane symmetry transformation for observation data augmentation.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor with shape [batch_size, obs_dim].
            Contains robot state observations to be augmented.
        env : object
            Environment object (passed for compatibility).
        obs_list : Sequence[str]
            List of observation component names to process.
            Example: ["actor_obs"] or ["actor_state_obs", "perception_obs"].

        Returns
        -------
        torch.Tensor
            Augmented observation tensor with doubled batch size.
            Original data is concatenated with mirrored data along batch dimension.
        """
        mirrored_obs = self.mirror_xz_plane(obs, env, obs_list)
        return torch.cat((obs, mirrored_obs), dim=0)

    def augment_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Applies x-z plane symmetry transformation for action data augmentation.

        Parameters
        ----------
        actions : torch.Tensor
            Action tensor with shape [batch_size, action_dim].
            Contains robot actions to be augmented.

        Returns
        -------
        torch.Tensor
            Augmented action tensor with doubled batch size.
            Original data is concatenated with mirrored data along batch dimension.
        """
        mirrored_actions = self.mirror_action_xz_plane(actions)
        return torch.cat((actions, mirrored_actions), dim=0)

    def mirror_xz_plane(self, observation: torch.Tensor, env: Any, obs_list: Sequence[str]) -> torch.Tensor:
        """Performs x-z plane symmetry transformation on observation tensor.

        This function parses the observation tensor and applies appropriate mirroring
        to each component based on its physical meaning and coordinate system.

        Parameters
        ----------
        observation : torch.Tensor
            Input observation tensor with shape [batch_size, obs_dim].
            Contains concatenated observation components as specified in obs_list.
        env : object
            Environment object (passed for compatibility).
        obs_list : List[str]
            List of observation component names that define the structure of the observation tensor.

        Returns
        -------
        torch.Tensor
            Mirrored observation tensor with same shape as input.
            Y-axis components are typically negated, joint configurations are remapped.
        """
        # Create a copy of the observation
        mirrored_obs_all = observation.clone()
        B, _ = mirrored_obs_all.shape

        # Parse observation components and apply mirroring
        idx = 0

        for obs_key in obs_list:
            cur_obs_length = self.observation_dims[obs_key]
            history_length = self.history_lengths[obs_key]

            # LAYOUT: the ObservationManager stacks history PER TERM, so a group
            # is term-major -- term A's H frames, then term B's H frames -- not
            # step-major. Each term is therefore mirrored on its own block,
            # viewed as [B, H, term_dim]; the mirror functions all act on the
            # last dimension, so every frame is mirrored in one call.
            #
            # At H == 1 (every config here) term-major and step-major coincide.
            offset = idx
            for sub_obs_key in self.sub_observation_keys[obs_key]:
                term_dim = self.sub_observation_dims[sub_obs_key]
                span = term_dim * history_length
                frames = mirrored_obs_all[..., offset : offset + span].reshape(
                    B, history_length, term_dim
                )
                mirrored = getattr(self, f"mirror_obs_{sub_obs_key}")(frames)
                mirrored_obs_all[..., offset : offset + span] = mirrored.reshape(
                    B, span
                )
                offset += span

            if offset != idx + cur_obs_length:
                raise RuntimeError(
                    f"mirror_xz_plane: group '{obs_key}' terms span "
                    f"{offset - idx} dims but the group is {cur_obs_length}. "
                    "The term list and the observation layout disagree."
                )
            idx += cur_obs_length

        return mirrored_obs_all

    def mirror_action_xz_plane(self, action: torch.Tensor) -> torch.Tensor:
        """Performs x-z plane symmetry transformation on action tensor.

        Parameters
        ----------
        action : torch.Tensor
            Input action tensor with shape [batch_size, n_dofs].
            Contains joint position commands or torques with original action order [a0, a1, a2, ..., aN].

        Returns
        -------
        torch.Tensor
            Mirrored action tensor with same shape as input.
            Joint mappings are applied and signs are flipped as appropriate.
            Returns [a_mapped0 * sign0, a_mapped1 * sign1, ..., a_mappedN * signN] (mirrored and sign-flipped).
        """
        return action[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_base_lin_vel(self, base_lin_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the base linear velocity in robot's base frame.

        Parameters
        ----------
        base_lin_vel : torch.Tensor
            Base linear velocity with layout [v_x, v_y, v_z] in base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored velocity with y-component negated: [v_x, -v_y, v_z].
        """
        base_lin_vel[..., 1] = -base_lin_vel[..., 1]  # Flip y component
        return base_lin_vel

    def mirror_obs_base_ang_vel(self, base_ang_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the base angular velocity in robot's base frame.

        Parameters
        ----------
        base_ang_vel : torch.Tensor
            Base angular velocity with layout [ω_x, ω_y, ω_z] in base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored angular velocity with x and z components negated: [-ω_x, ω_y, -ω_z].
        """
        base_ang_vel[..., 0] = -base_ang_vel[..., 0]  # Flip x component
        base_ang_vel[..., 2] = -base_ang_vel[..., 2]  # Flip z component
        return base_ang_vel

    def mirror_obs_base_orientation(self, base_orientation: torch.Tensor) -> torch.Tensor:
        """Mirrors the base orientation representation.

        Parameters
        ----------
        base_orientation : torch.Tensor
            Base orientation with layout [θ_x, θ_y, θ_z] (Euler angles or similar representation).

        Returns
        -------
        torch.Tensor
            Mirrored orientation with y-component negated: [θ_x, -θ_y, θ_z].
        """
        base_orientation[..., 1] = -base_orientation[..., 1]  # Flip y component
        return base_orientation

    def mirror_obs_projected_gravity(self, projected_gravity: torch.Tensor) -> torch.Tensor:
        """Mirrors the projected gravity vector in robot's base frame.

        Parameters
        ----------
        projected_gravity : torch.Tensor
            Gravity vector projected into base frame with layout [g_x, g_y, g_z].

        Returns
        -------
        torch.Tensor
            Mirrored gravity vector with y-component negated: [g_x, -g_y, g_z].
        """
        projected_gravity[..., 1] = -projected_gravity[..., 1]  # Flip y component
        return projected_gravity

    def mirror_obs_command_lin_vel(self, command_lin_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded linear velocity.

        Parameters
        ----------
        command_lin_vel : torch.Tensor
            Commanded linear velocity with layout [v_x_cmd, v_y_cmd, v_z_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored velocity command with y-component negated: [v_x_cmd, -v_y_cmd, v_z_cmd].
        """
        command_lin_vel[..., 1] = -command_lin_vel[..., 1]  # Flip y component
        return command_lin_vel

    def mirror_obs_command_ang_vel(self, command_ang_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded angular velocity (yaw only).

        Parameters
        ----------
        command_ang_vel : torch.Tensor
            Commanded yaw angular velocity (1D scalar) with layout [ω_yaw_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored yaw angular velocity command with sign negated: [-ω_yaw_cmd].
        """
        command_ang_vel[..., 0] = -command_ang_vel[..., 0]
        return command_ang_vel

    def mirror_obs_command_stand(self, command_stand: torch.Tensor) -> torch.Tensor:
        """Mirrors the stand command (no transformation needed).

        Parameters
        ----------
        command_stand : torch.Tensor
            Stand command signal (scalar or binary).
        Returns
        -------
        torch.Tensor
            Unchanged stand command.
        """
        return command_stand

    def mirror_obs_command_waist_dofs(self, command_waist_dofs: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded waist joint positions.

        Parameters
        ----------
        command_waist_dofs : torch.Tensor
            Waist joint commands with layout [yaw_cmd, roll_cmd, pitch_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored waist commands with yaw and roll negated: [-yaw_cmd, -roll_cmd, pitch_cmd].
        """
        # flip yaw
        command_waist_dofs[..., 0] = -command_waist_dofs[..., 0]
        # flip roll
        command_waist_dofs[..., 1] = -command_waist_dofs[..., 1]
        return command_waist_dofs

    def mirror_obs_command_base_height(self, command_base_height: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded base height (no transformation needed).

        Parameters
        ----------
        command_base_height : torch.Tensor
            Commanded base height (scalar).
            Inputs: [height_cmd].

        Returns
        -------
        torch.Tensor
            Unchanged height command.
            Outputs: [height_cmd].
        """
        return command_base_height

    def mirror_obs_sin_phase(self, sin_phase: torch.Tensor) -> torch.Tensor:
        """Mirrors the sine phase for gait timing by swapping left/right channels.

        Parameters
        ----------
        sin_phase : torch.Tensor
            Sine of gait phase with layout [sin(φ_left), sin(φ_right)].

        Returns
        -------
        torch.Tensor
            Mirrored phase with left/right swapped: [sin(φ_right), sin(φ_left)].
        """
        return sin_phase[..., [1, 0]]

    def mirror_obs_cos_phase(self, cos_phase: torch.Tensor) -> torch.Tensor:
        """Mirrors the cosine phase for gait timing by swapping left/right channels.

        Parameters
        ----------
        cos_phase : torch.Tensor
            Cosine of gait phase with layout [cos(φ_left), cos(φ_right)].

        Returns
        -------
        torch.Tensor
            Mirrored phase with left/right swapped: [cos(φ_right), cos(φ_left)].
        """
        return cos_phase[..., [1, 0]]

    def mirror_obs_torso_projected_gravity(self, torso_projected_gravity: torch.Tensor) -> torch.Tensor:
        """Torso gravity direction; same reflection as the pelvis one (negate y)."""
        return self.mirror_obs_projected_gravity(torso_projected_gravity)

    def mirror_obs_pelvis_gravity_command(self, pelvis_gravity_command: torch.Tensor) -> torch.Tensor:
        """Commanded pelvis gravity direction: a direction, so negate y."""
        return self.mirror_obs_projected_gravity(pelvis_gravity_command)

    def mirror_obs_torso_gravity_command(self, torso_gravity_command: torch.Tensor) -> torch.Tensor:
        """Commanded torso gravity direction: a direction, so negate y."""
        return self.mirror_obs_projected_gravity(torso_gravity_command)

    def mirror_obs_pelvis_gravity_error(self, pelvis_gravity_error: torch.Tensor) -> torch.Tensor:
        """Difference of two mirrored directions, so it mirrors the same way."""
        return self.mirror_obs_projected_gravity(pelvis_gravity_error)

    def mirror_obs_torso_gravity_error(self, torso_gravity_error: torch.Tensor) -> torch.Tensor:
        """Difference of two mirrored directions, so it mirrors the same way."""
        return self.mirror_obs_projected_gravity(torso_gravity_error)


    def mirror_obs_dof_pos(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """Mirrors the joint positions using joint mapping and sign flipping.

        Parameters
        ----------
        dof_pos : torch.Tensor
            Joint positions with layout following the robot's DOF order.
            Typically [hip_joints..., knee_joints..., ankle_joints..., waist_joints..., arm_joints...].
            Inputs: [q0, q1, q2, ..., qN] (original joint order).

        Returns
        -------
        torch.Tensor
            Mirrored joint positions with left-right mapping applied and signs flipped for appropriate joints.
            Outputs: [q_mapped0 * sign0, q_mapped1 * sign1, ..., q_mappedN * signN] (mirrored and sign-flipped).
        """
        return dof_pos[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_dof_vel(self, dof_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the joint velocities (same mapping as joint positions).

        Parameters
        ----------
        dof_vel : torch.Tensor
            Joint velocities with same layout as dof_pos.
            Inputs: [qd0, qd1, qd2, ..., qdN] (original joint velocities).

        Returns
        -------
        torch.Tensor
            Mirrored joint velocities with same transformation as joint positions.
            Outputs: [qd_mapped0 * sign0, qd_mapped1 * sign1, ..., qd_mappedN * signN] (mirrored and sign-flipped).
        """
        return dof_vel[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Mirrors the previous actions (same mapping as joint positions).

        Parameters
        ----------
        actions : torch.Tensor
            Previous action values with same layout as dof_pos.
            Inputs: [a0, a1, a2, ..., aN] (original actions).

        Returns
        -------
        torch.Tensor
            Mirrored actions with same transformation as joint positions.
            Outputs: [a_mapped0 * sign0, a_mapped1 * sign1, ..., a_mappedN * signN] (mirrored and sign-flipped).
        """
        return actions[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_dof_pos_relative(self, dof_pos_relative: torch.Tensor) -> torch.Tensor:
        """Same shape and ordering as dof_pos; mirroring is identical."""
        return self.mirror_obs_dof_pos(dof_pos_relative)

    def mirror_obs_prev_prev_action(self, prev_prev_action: torch.Tensor) -> torch.Tensor:
        """Two-step-old action; same layout as actions, mirroring is identical."""
        return self.mirror_obs_actions(prev_prev_action)

    def mirror_obs_base_height_command(self, base_height_command: torch.Tensor) -> torch.Tensor:
        """Scalar target z (shape [..., 1]). Invariant under M = diag(1,-1,1)."""
        return base_height_command

    def mirror_obs_base_height_error(self, base_height_error: torch.Tensor) -> torch.Tensor:
        """Scalar z error (shape [..., 1]). Invariant under M = diag(1,-1,1)."""
        return base_height_error

    def mirror_obs_last_action(self, last_action: torch.Tensor) -> torch.Tensor:
        """Same shape and ordering as actions; mirroring is identical."""
        return self.mirror_obs_actions(last_action)

    def mirror_obs_last_raw_action(self, last_raw_action: torch.Tensor) -> torch.Tensor:
        """Raw policy action a_{T-1}; same layout as actions, mirroring is identical."""
        return self.mirror_obs_actions(last_raw_action)

    def mirror_obs_prev_raw_action(self, prev_raw_action: torch.Tensor) -> torch.Tensor:
        """Raw policy action a_{T-2}; same layout as actions, mirroring is identical."""
        return self.mirror_obs_actions(prev_raw_action)


    def mirror_obs_controller_arm_joint_command(
        self, controller_arm_joint_command: torch.Tensor
    ) -> torch.Tensor:
        """Mirrors GAC's arm joint command (shape [..., 14], arm_dof_names order).

        Layout: [L_shoulder_p, L_shoulder_r, L_shoulder_y, L_elbow,
                 L_wrist_r, L_wrist_p, L_wrist_y,
                 R_shoulder_p, R_shoulder_r, R_shoulder_y, R_elbow,
                 R_wrist_r, R_wrist_p, R_wrist_y]

        Per-arm sign flip mask follows flip_sign_joint_names: roll/yaw axes flip,
        pitch/elbow axes do not -> [+, -, -, +, -, +, -]. Then swap L<->R 7-blocks.
        """
        sign_7 = torch.tensor(
            [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            device=controller_arm_joint_command.device,
            dtype=controller_arm_joint_command.dtype,
        )
        left = controller_arm_joint_command[..., :7].clone() * sign_7
        right = controller_arm_joint_command[..., 7:].clone() * sign_7
        return torch.cat([right, left], dim=-1)


    def mirror_obs_ee_apply_force(self, ee_apply_force: torch.Tensor) -> torch.Tensor:
        """Mirrors the end-effector applied forces in base frame.

        Parameters
        ----------
        ee_apply_force : torch.Tensor
            Applied forces with layout [left_fx, left_fy, left_fz, right_fx, right_fy, right_fz].
            Forces are already transformed to base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored forces with left-right swapped and y-components negated:
            [right_fx, -right_fy, right_fz, left_fx, -left_fy, left_fz].
        """
        # Note: this force is already transformed to the base frame
        left_ee_apply_force = ee_apply_force[..., :3].clone()
        left_ee_apply_force[..., 1] = -left_ee_apply_force[..., 1]
        right_ee_apply_force = ee_apply_force[..., 3:].clone()
        right_ee_apply_force[..., 1] = -right_ee_apply_force[..., 1]
        return torch.cat([right_ee_apply_force, left_ee_apply_force], dim=-1)

    def mirror_obs_arm_command(self, arm_command: torch.Tensor) -> torch.Tensor:
        """Mirrors the arm command (per-hand position + rot6d) under the sagittal plane.

        Each hand contributes a 9-block ``[pos(3), rot6d(6)]`` where ``rot6d`` stores
        the first two columns of the rotation matrix R as
        ``[c0_x, c0_y, c0_z, c1_x, c1_y, c1_z]``. Under the reflection
        ``M = diag(1, -1, 1)``, R transforms as ``R' = M R M``, which yields:

          - ``col0 = R[:, 0]`` -> ``M @ col0`` (negate y)
          - ``col1 = R[:, 1]`` -> ``-M @ col1`` (negate x and z)

        Position is a vector and transforms as ``M @ pos`` (negate y). The mirrored
        left arm sees what the right arm saw, and vice versa, so the L/R 9-blocks
        are also swapped.

        Parameters
        ----------
        arm_command : torch.Tensor
            Arm command with shape ``[..., 18]`` and layout
            ``[left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6)]``.

        Returns
        -------
        torch.Tensor
            Mirrored arm command (new tensor, input is not mutated). The L/R
            9-blocks are swapped and each block has the per-hand sign mask
            ``[+, -, +,  +, -, +,  -, +, -]`` applied.
        """
        sign_9 = torch.tensor(
            [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            device=arm_command.device,
            dtype=arm_command.dtype,
        )
        left_block = arm_command[..., :9].clone() * sign_9
        right_block = arm_command[..., 9:].clone() * sign_9
        return torch.cat([right_block, left_block], dim=-1)

    def mirror_obs_ee_error_pose(self, ee_error_pose: torch.Tensor) -> torch.Tensor:
        """Mirrors the per-hand end-effector pose error (position + rot6d).

        Mathematically identical to :meth:`mirror_obs_arm_command`. The rotation
        error is ``R_err = R_target^T @ R_curr``; under conjugation by
        ``M = diag(1, -1, 1)``, both ``R_target`` and ``R_curr`` map to ``M(.)M``,
        so ``R_err' = M @ R_err @ M``. The position error
        ``R_target^T (p_curr - p_target)`` transforms identically by the same
        conjugation argument, so the per-hand 9-block sign mask is the same:
        ``[+, -, +,  +, -, +,  -, +, -]``. L/R 9-blocks are swapped.

        Parameters
        ----------
        ee_error_pose : torch.Tensor
            EE pose error with shape ``[..., 18]`` and layout
            ``[left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6)]``.

        Returns
        -------
        torch.Tensor
            Mirrored pose error (new tensor, input is not mutated).
        """
        sign_9 = torch.tensor(
            [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            device=ee_error_pose.device,
            dtype=ee_error_pose.dtype,
        )
        left_block = ee_error_pose[..., :9].clone() * sign_9
        right_block = ee_error_pose[..., 9:].clone() * sign_9
        return torch.cat([right_block, left_block], dim=-1)

    def mirror_obs_ee_error_pose_tanh(self, ee_error_pose_tanh: torch.Tensor) -> torch.Tensor:
        """Mirrors the per-hand tanh-squashed EE position error.

        The squashed error is a 3-vector per hand and transforms as a vector
        under ``M = diag(1, -1, 1)``: only the y-component is negated. L/R
        3-blocks are swapped.

        Parameters
        ----------
        ee_error_pose_tanh : torch.Tensor
            Tanh-squashed EE pos error with shape ``[..., 6]`` and layout
            ``[left_xyz(3), right_xyz(3)]``.

        Returns
        -------
        torch.Tensor
            Mirrored tanh-squashed pos error (new tensor, input is not mutated).
            Per-hand 3-block sign mask: ``[+, -, +]``; L/R 3-blocks swapped.
        """
        sign_3 = torch.tensor(
            [1.0, -1.0, 1.0],
            device=ee_error_pose_tanh.device,
            dtype=ee_error_pose_tanh.dtype,
        )
        left_block = ee_error_pose_tanh[..., :3].clone() * sign_3
        right_block = ee_error_pose_tanh[..., 3:].clone() * sign_3
        return torch.cat([right_block, left_block], dim=-1)

    def mirror_obs_ee_body_twist(self, ee_body_twist: torch.Tensor) -> torch.Tensor:
        """Mirrors the per-hand body twist (linear + angular velocity).

        Under ``M = diag(1, -1, 1)``:

          - linear velocity is a vector: ``v -> M @ v`` (negate y)
          - angular velocity is a pseudovector:
            ``omega -> det(M) * (M @ omega) = -(M @ omega)`` (negate x and z)

        L/R 6-blocks are swapped.

        Parameters
        ----------
        ee_body_twist : torch.Tensor
            EE body twist with shape ``[..., 12]`` and layout
            ``[left_lin(3), left_ang(3), right_lin(3), right_ang(3)]``.

        Returns
        -------
        torch.Tensor
            Mirrored body twist (new tensor, input is not mutated). Per-hand
            6-block sign mask: ``[+, -, +,  -, +, -]``; L/R 6-blocks swapped.
        """
        sign_6 = torch.tensor(
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            device=ee_body_twist.device,
            dtype=ee_body_twist.dtype,
        )
        left_block = ee_body_twist[..., :6].clone() * sign_6
        right_block = ee_body_twist[..., 6:].clone() * sign_6
        return torch.cat([right_block, left_block], dim=-1)

    def mirror_obs_ee_body_twist_full(self, ee_body_twist_full: torch.Tensor) -> torch.Tensor:
        """Mirrors the critic-only full per-hand body twist.

        Identical in shape, layout, and operation to
        :meth:`mirror_obs_ee_body_twist`; this variant exists for the critic
        observation namespace. Implementation simply delegates.

        Parameters
        ----------
        ee_body_twist_full : torch.Tensor
            EE body twist with shape ``[..., 12]`` and layout
            ``[left_lin(3), left_ang(3), right_lin(3), right_ang(3)]``.

        Returns
        -------
        torch.Tensor
            Mirrored body twist (new tensor, input is not mutated).
        """
        return self.mirror_obs_ee_body_twist(ee_body_twist_full)


    def mirror_obs_gait_avg_velocity_state(
        self, gait_avg_state: torch.Tensor
    ) -> torch.Tensor:
        """Mirror gait-averaged velocity accumulator state under sagittal-plane reflection.

        Layout: ``[wz_running_mean, v_perp_running_mean, wz_cycle_mean,
                   v_perp_cycle_mean, sample_count_norm]`` — shape ``[..., 5]``.

        Under x-z plane (sagittal) reflection:
        - wz (yaw rate) flips sign (rotation about z reverses).
        - v_perp (signed lateral velocity) flips sign (y-direction reverses).
        - sample_count_norm is a scalar counter — invariant.

        Parameters
        ----------
        gait_avg_state : torch.Tensor
            Gait-averaged velocity accumulator observation with shape ``[..., 5]``.

        Returns
        -------
        torch.Tensor
            Mirrored accumulator state with wz and v_perp components negated.
        """
        mirrored = gait_avg_state.clone()
        mirrored[..., 0] = -mirrored[..., 0]  # wz_running_mean
        mirrored[..., 1] = -mirrored[..., 1]  # v_perp_running_mean
        mirrored[..., 2] = -mirrored[..., 2]  # wz_cycle_mean
        mirrored[..., 3] = -mirrored[..., 3]  # v_perp_cycle_mean
        # mirrored[..., 4] unchanged (sample_count_norm)
        return mirrored
