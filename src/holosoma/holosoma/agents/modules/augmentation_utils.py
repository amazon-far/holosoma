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
            term_keys = sorted(group_cfg.terms.keys())
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

        # Build a NAME-BASED foot/leg channel mirror map for gait-phase observations.
        # Gait phase obs (sin_phase/cos_phase) have one channel per foot, ordered to
        # match env.feet_indices == [body for body in body_names if foot_body_name in body].
        # Under a left-right mirror, each foot's phase channel must be SWAPPED with its
        # mirror foot's channel. The previous implementation hard-coded "negate channel 0"
        # which only happens to be (partially) correct when channel 0 is the LEFT foot
        # (i.e. left-first body_names). This derives the permutation by name so it is
        # correct for any foot ordering and any number of feet (bipeds and quadrupeds).
        self.phase_channel_map = self._build_phase_channel_map()

    def _mirror_body_name(self, name: str) -> str:
        """Return the left<->right mirror of a body name. Side tokens are matched only
        at WORD BOUNDARIES (name start, or surrounded by '_') so mid-word letters are
        never mistaken for a side prefix -- e.g. the 'l' in 'ankle_roll' must NOT be
        read as an 'l_' side marker. Returns the name unchanged if no side token found.
        """
        import re

        # whole-word 'left'/'right'
        if re.search(r"(?<![a-z])left(?![a-z])", name) and "right" not in name:
            return re.sub(r"(?<![a-z])left(?![a-z])", "right", name, count=1)
        if re.search(r"(?<![a-z])right(?![a-z])", name) and "left" not in name:
            return re.sub(r"(?<![a-z])right(?![a-z])", "left", name, count=1)
        # prefix l_/r_ (start) or _l_/_r_ (after an underscore) ONLY
        if re.match(r"^l_", name):
            return "r_" + name[2:]
        if re.match(r"^r_", name):
            return "l_" + name[2:]
        if "_l_" in name:
            return name.replace("_l_", "_r_", 1)
        if "_r_" in name:
            return name.replace("_r_", "_l_", 1)
        return name

    def _build_phase_channel_map(self) -> torch.Tensor:
        """Permutation tensor p such that mirrored_phase[..., i] = phase[..., p[i]].

        p[i] = index of the foot channel that is the left-right mirror of channel i.
        Falls back to identity for any channel whose mirror can't be resolved by name
        (so behavior degrades to "unchanged" rather than wrong)."""
        body_names = list(getattr(self.env, "body_names", []) or [])
        foot_pattern = self.robot_config.foot_body_name
        foot_names = [b for b in body_names if foot_pattern in b] if foot_pattern else []
        name_to_chan = {n: i for i, n in enumerate(foot_names)}
        perm = []
        for i, n in enumerate(foot_names):
            mirror = self._mirror_body_name(n)
            perm.append(name_to_chan.get(mirror, i))  # identity fallback
        return torch.tensor(perm, device=self.env.device, dtype=torch.long)

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
            mirrored_obs = mirrored_obs_all[..., idx : idx + cur_obs_length]
            # Reshape to [batch, history_length, single_frame_obs_dim]
            mirrored_obs = mirrored_obs.reshape(
                B, self.history_lengths[obs_key], self.observation_dims_single_frame[obs_key]
            )
            # Apply mirroring to each sub-observation component using single-frame indices
            for sub_obs_key in self.sub_observation_keys[obs_key]:
                mirrored_obs[..., self.sub_observation_indices_single_frame[obs_key][sub_obs_key]] = getattr(
                    self, f"mirror_obs_{sub_obs_key}"
                )(mirrored_obs[..., self.sub_observation_indices_single_frame[obs_key][sub_obs_key]])
            mirrored_obs_all[..., idx : idx + cur_obs_length] = mirrored_obs.reshape(B, cur_obs_length)
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
        """Mirrors the sine of the gait phase under a left-right reflection.

        Layout: one channel per foot, ordered to match env.feet_indices.
        A left-right mirror SWAPS each foot's phase channel with its mirror foot's
        channel (mirrored[..., i] = phase[..., phase_channel_map[i]]). This is correct
        for any foot ordering / count, unlike the previous hard "negate channel 0".

        NOTE (changed 2026-06-30): this differs from the old behavior for left-first
        bipeds too — the old code negated channel 0 and left channel 1 untouched, which
        is not a consistent leg swap.
        """
        return sin_phase[..., self.phase_channel_map]

    def mirror_obs_cos_phase(self, cos_phase: torch.Tensor) -> torch.Tensor:
        """Mirrors the cosine of the gait phase under a left-right reflection.

        See mirror_obs_sin_phase: channels are swapped by the name-derived
        phase_channel_map so the transform is correct for any foot ordering.
        """
        return cos_phase[..., self.phase_channel_map]

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
