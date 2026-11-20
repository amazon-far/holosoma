"""Null controller for baseline (no-controller) mode.

Inherits from ArmControllerBase to provide pose_error, integral_error,
and ee_body_twist tracking without producing any joint commands.  This
ensures the observation and reward systems always have a valid
``base_controller`` to read from, eliminating the need for fallback
chains or shim objects.
"""

from __future__ import annotations

import torch
from holosoma.envs.arm_tracking.controller_base import ArmControllerBase


class NullController(ArmControllerBase):
    """Controller that tracks error state but outputs zero arm actions.

    Used when ``controller_class`` is ``None`` in the experiment config.
    The RL policy controls all joints directly; this controller only
    maintains ``pose_error``, ``integral_error``, and ``ee_body_twist``
    for observation and reward access.
    """

    def setup(self):
        """Initialize controller buffers and pre-allocate Adjoint scratch tensors."""
        super().setup()
        eye3 = torch.eye(3, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        self._Ad_CeCt_template = torch.zeros((self.num_envs, 6, 6), device=self.device)
        self._Ad_CeCt_template[:, :3, :3] = eye3
        self._Ad_CeCt_template[:, 3:, 3:] = eye3
        self._Ad_CeCt = torch.zeros((self.num_envs, 6, 6), device=self.device)
        self._Ad_CeE = torch.zeros((self.num_envs, 6, 6), device=self.device)

    def compute_arm_actions(self, dt: float | None = None) -> torch.Tensor:
        """Compute arm actions (always zeros) while updating error and twist state."""
        if not hasattr(self, "arm_actions"):
            return torch.zeros(
                (self.num_envs, self.num_arm_dof), device=self.device
            )

        if dt is None:
            dt = self.env.dt

        arm_command = self.env.command_manager.get_state("arm_command")
        if arm_command is not None:
            target_ee_pos = arm_command.target_ee_pos
            target_ee_rot6d = arm_command.target_ee_rot6d

            ee_pos, ee_quat, ee_jac_w = self.get_ee_pose_and_jacobian()
            ref_pos, ref_quat, ref_jac_w = self.get_reference_pose_and_jacobian()
            _, q_dot = self.get_robot_state()

            R_WT = self._quat_to_rotm(ref_quat)
            R_TW = R_WT.transpose(-2, -1)

            ee_rotm = self._quat_to_rotm(
                ee_quat.reshape(-1, 4)
            ).reshape(self.num_envs, 2, 3, 3)
            target_ee_rotm = self._rot6d_to_rotm(
                target_ee_rot6d.reshape(-1, 6)
            ).reshape(self.num_envs, 2, 3, 3)

            for arm_idx in range(2):
                J_WE_Ce = ee_jac_w[:, arm_idx]
                p_WE = ee_pos[:, arm_idx]
                R_WE = ee_rotm[:, arm_idx]

                # Adjoint for position offset
                p_CeCt = -p_WE + ref_pos
                Ad_CeCt = self._Ad_CeCt
                Ad_CeCt.copy_(self._Ad_CeCt_template)
                Ad_CeCt[:, :3, 3:] = self._skew(p_CeCt)

                # Relative Jacobian in world frame
                J_TE_Ce = J_WE_Ce - torch.bmm(Ad_CeCt, ref_jac_w)

                # Transform to EE body frame
                R_EW = R_WE.transpose(-2, -1)
                Ad_CeE = self._Ad_CeE
                Ad_CeE.zero_()
                Ad_CeE[:, :3, :3] = R_EW
                Ad_CeE[:, 3:, 3:] = R_EW
                J_TE_E = torch.bmm(Ad_CeE, J_TE_Ce)

                # EE pose in reference frame
                p_TE = torch.bmm(
                    R_TW, (p_WE - ref_pos).unsqueeze(-1)
                ).squeeze(-1)
                R_TE = torch.bmm(R_TW, R_WE)

                ref_p = target_ee_pos[:, arm_idx]
                ref_R = target_ee_rotm[:, arm_idx]

                self._compute_body_frame_pose_error(
                    arm_idx, p_TE, R_TE, ref_p, ref_R
                )
                self._update_integral(arm_idx, dt)
                self._compute_ee_body_twist(arm_idx, J_TE_E, q_dot)
        else:
            self.pose_error[:] = 0.0
            self.ee_body_twist[:] = 0.0

        # Null controller: return zeros (default_dof_pos reconstructed by PD)
        self.arm_actions[:] = 0.0
        return self.arm_actions
