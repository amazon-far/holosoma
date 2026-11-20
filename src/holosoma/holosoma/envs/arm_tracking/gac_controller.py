"""
GAC (Geometric Admittance Control) controller for end-effector tracking.

Implements geometric impedance control for dual-arm manipulation.
"""

from __future__ import annotations

import torch
from holosoma.envs.arm_tracking.controller_base import ArmControllerBase


class GAC_controller(ArmControllerBase):
    """
    Geometric Admittance Controller for end-effector tracking.
    
    Inherits from ArmControllerBase which provides:
    - get_ee_jacobian(): Compute end-effector Jacobians in root frame
    - get_ee_pose(): Get current end-effector poses in root frame
    - get_torso_pose(): Get current torso pose in root frame
    - get_arm_state(): Get current joint positions and velocities
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

        # Controller gains / constants
        # self.K_p_gain = 25000.0
        # self.K_R_gain = 25000.0

        self.K_p_gain = 5000.0
        self.K_R_gain = 5000.0
        self.damping_ratio = kwargs.get("damping_ratio", 1.0)
        self.jjt_reg_eps = 1e-3

        # Precompute constant batched tensors reused every step.
        eye3 = torch.eye(3, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        eye6 = torch.eye(6, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        self._eye3 = eye3
        self._eye6 = eye6

        self.K_p = eye3 * self.K_p_gain
        self.K_R = eye3 * self.K_R_gain

        # Mass matrix: diag([1, 1, 1, 1.5, 1.5, 1.5]) * 1.5
        M_diag = torch.tensor([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], device=self.device) * 1.5
        M = torch.diag(M_diag).unsqueeze(0).expand(self.num_envs, -1, -1)
        self.M_inv = torch.inverse(M)

        # Damping: K_d = sqrt(block_diag(K_p, K_R)) * 2 * damping_ratio
        K_pR = torch.zeros((self.num_envs, 6, 6), device=self.device)
        K_pR[:, :3, :3] = self.K_p
        K_pR[:, 3:, 3:] = self.K_R
        self.K_d = torch.sqrt(K_pR) * 2 * self.damping_ratio

        self.Ad_CeCr_template = torch.zeros((self.num_envs, 6, 6), device=self.device)
        self.Ad_CeCr_template[:, :3, :3] = eye3
        self.Ad_CeCr_template[:, 3:, 3:] = eye3
        self.JJT_reg = eye6 * self.jjt_reg_eps

        self.num_dof_per_arm = self.num_arm_dof // 2
        self.left_arm_indices = self.arm_dof_indices[:self.num_dof_per_arm]
        self.right_arm_indices = self.arm_dof_indices[self.num_dof_per_arm:]

        self._Ad_CeCr = torch.zeros((self.num_envs, 6, 6), device=self.device)
        self._Ad_CeE = torch.zeros((self.num_envs, 6, 6), device=self.device)
        self._qdot_next_total = torch.zeros((self.num_envs, self.num_total_dof), device=self.device)

        self.use_nullspace_rest_posture = True
        self.k_null = 50.0
        self.nullspace_damping = 0.0025

        # self.use_nullspace_rest_posture = True
        # self.k_null = 10.0
        # self.nullspace_damping = 0.0025


    def compute_arm_actions(self, dt: float | None = None) -> torch.Tensor:
        """
        Compute arm joint targets for end-effector control.

        Args:
            dt: Integration timestep. If None, uses self.env.dt.

        Returns:
            torch.Tensor: [num_envs, num_arm_dof] arm joint targets
        """
        # Safety check: return default pose if robot state is not initialized
        q, q_dot = self.get_robot_state()
        arm_q = q[:, self.arm_dof_indices]
        # When use_angular_base_jacobian is True, q_dot has 6 prepended entries
        # (free-flyer velocity). Offset arm indices accordingly for NaN check.
        arm_qdot_idx = self.arm_dof_indices + 6 if self.use_angular_base_jacobian else self.arm_dof_indices
        arm_qdot = q_dot[:, arm_qdot_idx]
        if torch.isnan(arm_q).any() or torch.isnan(arm_qdot).any() or not hasattr(self, 'arm_actions'):
            # Robot not initialized yet, return default arm pose.
            # Return zeros so that _compute_torques reconstructs:
            #   target = 0 * action_scale + default_dof_pos = default_dof_pos
            return torch.zeros((self.num_envs, self.num_arm_dof), device=self.device)

        if dt is None:
            dt = self.env.dt

        # Get arm targets from command
        arm_command = self.env.command_manager.get_state("arm_command")
        if arm_command is not None:
            # Get target end-effector poses (in root/torso frame)
            target_ee_pos = arm_command.target_ee_pos  # [num_envs, 2, 3]
            target_ee_rot6d = arm_command.target_ee_rot6d  # [num_envs, 2, 6]

            ee_pos, ee_quat, ee_jacobian_world = self.get_ee_pose_and_jacobian()
            ref_pos, ref_quat, ref_jacobian_world = self.get_reference_pose_and_jacobian()

            # Notation: W=world frame, R=reference frame, E=end-effector frame
            p_WR = ref_pos
            R_WR_mat = self._quat_to_rotm(ref_quat)
            R_RW = R_WR_mat.transpose(-2, -1)
            J_WR_Cr = ref_jacobian_world

            ee_rotm = self._quat_to_rotm(ee_quat.reshape(-1, 4)).reshape(self.num_envs, 2, 3, 3)
            target_ee_rotm = self._rot6d_to_rotm(target_ee_rot6d.reshape(-1, 6)).reshape(self.num_envs, 2, 3, 3)

            qdot_next_total = self._qdot_next_total
            qdot_next_total.zero_()

            for arm_idx in range(2):
                J_WE_Ce = ee_jacobian_world[:, arm_idx]
                p_WE = ee_pos[:, arm_idx]
                R_WE = ee_rotm[:, arm_idx]

                # Position offset from EE to reference frame
                p_CeCr = -p_WE + p_WR

                # Adjoint for position offset: Ad_CeCr = [I, hat([p_CeCr]); 0, I]
                Ad_CeCr = self._Ad_CeCr
                Ad_CeCr.copy_(self.Ad_CeCr_template)
                Ad_CeCr[:, :3, 3:] = self._skew(p_CeCr)
                
                # Relative Jacobian: J_RE_Ce = J_WE_Ce - Ad_CeCr @ J_WR_Cr
                J_RE_Ce = J_WE_Ce - torch.bmm(Ad_CeCr, J_WR_Cr)
                
                # Transform to EE frame: Ad_CeE = [R_WE^T, 0; 0, R_WE^T]
                R_EW = R_WE.transpose(-2, -1)
                Ad_CeE = self._Ad_CeE
                Ad_CeE.zero_()
                Ad_CeE[:, :3, :3] = R_EW
                Ad_CeE[:, 3:, 3:] = R_EW
                J_RE_E = torch.bmm(Ad_CeE, J_RE_Ce)

                # Pose of EE in reference frame
                p_RE = torch.bmm(R_RW, (p_WE - p_WR).unsqueeze(-1)).squeeze(-1)
                R_RE = torch.bmm(R_RW, R_WE)

                # Reference pose (in configured reference frame)
                ref_p = target_ee_pos[:, arm_idx]  # [N, 3]
                ref_R = target_ee_rotm[:, arm_idx]  # [N, 3, 3]

                # Store body-frame pose error and update integral for obs/reward
                self._compute_body_frame_pose_error(arm_idx, p_RE, R_RE, ref_p, ref_R)
                self._update_integral(arm_idx, dt)

                # Position wrench: f_p = R_RE^T @ ref_R @ K_p @ ref_R^T @ (p_RE - ref_p)
                pos_error = p_RE - ref_p  # [N, 3]
                f_p = torch.bmm(
                    torch.bmm(torch.bmm(R_RE.transpose(-2, -1), ref_R), self.K_p),
                    torch.bmm(ref_R.transpose(-2, -1), pos_error.unsqueeze(-1))
                ).squeeze(-1)  # [N, 3]

                # Rotation wrench: f_R = vee(K_R @ ref_R^T @ R_RE - R_RE^T @ ref_R @ K_R)
                term1 = torch.bmm(torch.bmm(self.K_R, ref_R.transpose(-2, -1)), R_RE)
                term2 = torch.bmm(torch.bmm(R_RE.transpose(-2, -1), ref_R), self.K_R)
                f_R = self._vee(term1 - term2)  # [N, 3]

                # Combined wrench
                f_g = torch.cat([f_p, f_R], dim=-1)  # [N, 6]
                # Current body twist
                V_RE_E = self._compute_ee_body_twist(arm_idx, J_RE_E, q_dot)  # [N, 6]

                # Admittance update: V_next = V + dt * M^-1 @ (-f_g - K_d @ V)
                f_d = torch.bmm(self.K_d, V_RE_E.unsqueeze(-1)).squeeze(-1)  # [N, 6]
                V_next = V_RE_E + dt * torch.bmm(self.M_inv, (-f_g - f_d).unsqueeze(-1)).squeeze(-1)  # [N, 6]

                arm_slice = self.left_arm_indices if arm_idx == 0 else self.right_arm_indices

                # When using full Jacobian (with free-flyer cols), arm DOF
                # columns are offset by 6 in the Jacobian matrix.
                if self.use_angular_base_jacobian:
                    jac_arm_cols = arm_slice + 6
                else:
                    jac_arm_cols = arm_slice
                J_ctrl = J_RE_E[:, :, jac_arm_cols]

                # Task-space to joint-space mapping with null-space rest posture
                J = J_ctrl  # [N, 6, n_arm]

                # Damped least-squares matrix
                JJT = torch.bmm(J, J.transpose(-2, -1))  # [N, 6, 6]
                A = JJT + self.nullspace_damping * self._eye6

                # Primary task term: qdot_task = J^T (J J^T + λI)^-1 V_next
                x_task = torch.linalg.solve(A, V_next.unsqueeze(-1))  # [N, 6, 1]
                qdot_task = torch.bmm(J.transpose(-2, -1), x_task).squeeze(-1)  # [N, n_arm]

                if self.use_nullspace_rest_posture:
                    q_arm = q[:, arm_slice]  # [N, n_arm]
                    q_rest = self.env.default_dof_pos[:, arm_slice]  # [N, n_arm]

                    # Secondary objective in joint space
                    qdot_rest = -self.k_null * (q_arm - q_rest)  # [N, n_arm]

                    # Compute N qdot_rest = qdot_rest - J^T (J J^T + λI)^-1 (J qdot_rest)
                    J_qdot_rest = torch.bmm(J, qdot_rest.unsqueeze(-1))  # [N, 6, 1]
                    x_null = torch.linalg.solve(A, J_qdot_rest)  # [N, 6, 1]
                    qdot_null = qdot_rest - torch.bmm(J.transpose(-2, -1), x_null).squeeze(-1)  # [N, n_arm]

                    qdot_next = qdot_task + qdot_null
                else:
                    qdot_next = qdot_task

                qdot_next_total[:, arm_slice] = qdot_next
                
            # Integrate to get joint positions: q_cmd = q + dt * qdot_next
            q_cmd = q + dt * qdot_next_total

            # Convert absolute joint targets into the normalised action space
            # expected by JointPositionActionTerm._compute_torques:
            #   target = action * action_scale + default_dof_pos
            # So action = (q_cmd - default_dof_pos) / action_scale
            arm_q_cmd = q_cmd[:, self.arm_dof_indices]
            arm_default = self.env.default_dof_pos[:, self.arm_dof_indices]
            arm_action_scale = self.env.action_scales[self.arm_dof_indices]
            self.arm_actions[:] = (arm_q_cmd - arm_default) / arm_action_scale.clamp(min=1e-8)
        else:
            # Fallback to default arm pose: zeros reconstruct default_dof_pos
            self.arm_actions[:] = 0.0
            self.pose_error[:] = 0.0

        return self.arm_actions
