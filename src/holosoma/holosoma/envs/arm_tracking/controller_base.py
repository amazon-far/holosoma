"""
Base controller class for arm tracking with Jacobian utilities.

Provides common functionality for all arm controllers:
- Jacobian computation for end-effectors
- Forward kinematics
- Utility methods for control algorithms
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod

from holosoma.utils.rotation_6d import heading_quat_from_root_quat, quat_to_rotmat

MUJOCO_TO_ISAAC = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]

ISAAC_TO_MUJOCO = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
]

class ArmControllerBase(ABC):
    """
    Base class for arm controllers with Jacobian computation.
    
    Subclasses should implement:
    - compute_arm_actions(): Main control logic
    """

    def __init__(
        self,
        env,
        leaky_sigma: float = 0.1,
        max_integral: float = 2.0,
        integral_gate_width_pos: float = 0.015,
        integral_gate_width_rot: float = 0.05,
        use_angular_base_jacobian: bool = False,
        **kwargs,
    ):
        if max_integral <= 0:
            raise ValueError(
                f"max_integral must be positive, got {max_integral}"
            )

        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # When True, the heading/ground_heading Jacobian and ee_body_twist
        # include the floating-base angular velocity contribution (free-flyer
        # columns). This makes the relative Jacobian and twist sensitive to
        # body yaw-rate. Requires retraining models trained without this flag.
        self.use_angular_base_jacobian = use_angular_base_jacobian

        # Leaky-integral parameters
        self.leaky_sigma = leaky_sigma
        self.max_integral = max_integral
        # Norm-wise gate widths for translation / rotation channels
        self.integral_gate_width_pos = integral_gate_width_pos
        self.integral_gate_width_rot = integral_gate_width_rot

        # FK noise injection config (Tier 3: first-order SE(3) approximation)
        self.enable_fk_noise: bool = False
        self.fk_noise_sigma: float = 0.0

        # Get arm DOF indices
        self.arm_dof_indices = self._get_arm_dof_indices()
        self.num_arm_dof = len(self.arm_dof_indices)

        # Get total number of the joints
        self.num_total_dof = env.simulator.num_dof

        # Get end-effector and torso body indices
        self.ee_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.torso_body_name = "torso_link"
        self.base_body_name = "pelvis"
        self.ee_body_indices = self._get_ee_body_indices()
        self.torso_body_index = self.env.simulator.find_rigid_body_indice(self.torso_body_name)

        # Pre-allocated error tensors: [2, num_envs, 6] (arm_idx, env, [e_p; e_R])
        # Updated in-place via [:] to avoid per-step allocation overhead.
        self.pose_error = torch.zeros(
            (2, self.num_envs, 6), device=self.device
        )
        self.integral_error = torch.zeros(
            (2, self.num_envs, 6), device=self.device
        )

        # Pre-allocated EE body-frame twist: [2, num_envs, 6] (arm_idx, env, [v_lin; v_ang])
        # Updated in-place via [:] to avoid per-step allocation overhead.
        self.ee_body_twist = torch.zeros(
            (2, self.num_envs, 6), device=self.device
        )

        # Cached relative Jacobian in EE body frame: [2, num_envs, 6, nv]
        # Only meaningful when use_angular_base_jacobian=True.
        # Stores J_RE_E per arm for critic/reward ground-truth twist computation.
        nv = 6 + self.num_total_dof if self.use_angular_base_jacobian else self.num_total_dof
        self._J_RE_E_cached = torch.zeros(
            (2, self.num_envs, 6, nv), device=self.device
        )# Ground-truth EE body twist with true pelvis linear velocity.
        # Only differs from ee_body_twist when use_angular_base_jacobian=True
        # and reference_frame=ground_heading.
        self.ee_body_twist_true = torch.zeros(
            (2, self.num_envs, 6), device=self.device
        )

        def _get_physx_link_paths(robot):
            link_paths = robot.root_physx_view.link_paths
            # link_paths is typically [num_envs][num_links]
            return list(link_paths[0])

        def _find_unique_link_idx(link_paths, link_name: str) -> int:
            def _leaf_name(path: str) -> str:
                name = str(path)
                if "/" in name:
                    name = name.rsplit("/", 1)[-1]
                if ":" in name:
                    name = name.rsplit(":", 1)[-1]
                return name

            # Strict exact match on normalized leaf name to avoid
            # accidental substring matches (e.g., "pelvis" vs "pelvis_contour_link").
            hits = [i for i, p in enumerate(link_paths) if _leaf_name(p) == link_name]
            if len(hits) != 1:
                # print candidates to debug
                cand = [p for p in link_paths if link_name in str(p)][-10:]
                leaf_names = sorted({str(p).split("/")[-1].split(":")[-1] for p in link_paths})
                print(f"[GAC][link-options] requested='{link_name}', available_leaf_names={leaf_names}")
                raise RuntimeError(f"Link '{link_name}' matches {len(hits)} entries. Candidates (last 10): {cand}")
            return hits[0]

        # Problem: Jacobian function is indexed by link index - different from joint index.
        robot = self.env.simulator._robot
        self._robot_is_fixed_base = bool(getattr(robot, "is_fixed_base", False))
        J = robot.root_physx_view.get_jacobians()
        link_paths = _get_physx_link_paths(robot)

        eeL_link_idx = _find_unique_link_idx(link_paths, self.ee_body_names[0])
        eeR_link_idx = _find_unique_link_idx(link_paths, self.ee_body_names[1])
        torso_link_idx = _find_unique_link_idx(link_paths, self.torso_body_name)
        base_link_idx = _find_unique_link_idx(link_paths, self.base_body_name)

        # IMPORTANT: fixed-base offset rule (Isaac Lab tutorial)
        # fixed base: jacobian rows exclude root body => body/link index is shifted by -1
        if self._robot_is_fixed_base and len(link_paths) == J.shape[1] + 1:
            eeL_jac_row = eeL_link_idx - 1
            eeR_jac_row = eeR_link_idx - 1
            torso_jac_row = torso_link_idx - 1
        else:
            eeL_jac_row = eeL_link_idx
            eeR_jac_row = eeR_link_idx
            torso_jac_row = torso_link_idx

        if self._robot_is_fixed_base:
            # Fixed-base articulations have no valid base/root Jacobian row.
            base_jac_row = None
        else:
            base_jac_row = base_link_idx

        self._jac_row_left = eeL_jac_row
        self._jac_row_right = eeR_jac_row
        self._jac_row_torso = torso_jac_row
        self._jac_row_base = base_jac_row

    def _get_arm_dof_indices(self):
        """Get arm DOF indices from environment."""
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
        dof_names = self.env.simulator.dof_names
        indices = [dof_names.index(name) for name in arm_joint_names if name in dof_names]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _get_ee_body_indices(self):
        """Get body indices for end-effectors."""
        indices = []
        for name in self.ee_body_names:
            idx = self.env.simulator.find_rigid_body_indice(name)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def setup(self):
        """Initialize controller buffers."""
        # Buffer for computed arm actions
        self.arm_actions = torch.zeros(
            (self.num_envs, self.num_arm_dof), device=self.device
        )

    def _vee(self, S: torch.Tensor) -> torch.Tensor:
        """Extract vector from skew-symmetric matrix. [N, 3, 3] -> [N, 3]"""
        return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=-1)
    
    def _skew(self, v: torch.Tensor) -> torch.Tensor:
        """Compute skew-symmetric matrix from vector. [N, 3] -> [N, 3, 3]"""
        batch_size = v.shape[0]
        skew = torch.zeros((batch_size, 3, 3), device=v.device, dtype=v.dtype)
        skew[:, 0, 1] = -v[:, 2]
        skew[:, 0, 2] = v[:, 1]
        skew[:, 1, 0] = v[:, 2]
        skew[:, 1, 2] = -v[:, 0]
        skew[:, 2, 0] = -v[:, 1]
        skew[:, 2, 1] = v[:, 0]
        return skew

    def _compute_body_frame_pose_error(
        self,
        arm_idx: int,
        p_RE: torch.Tensor,
        R_RE: torch.Tensor,
        ref_p: torch.Tensor,
        ref_R: torch.Tensor,
    ) -> torch.Tensor:
        """Compute body-frame pose error [e_p; e_R] and store in self.pose_error.

        Parameters
        ----------
        arm_idx : int
            Arm index (0=left, 1=right).
        p_RE : Tensor [N, 3]
            End-effector position in reference frame.
        R_RE : Tensor [N, 3, 3]
            End-effector rotation in reference frame.
        ref_p : Tensor [N, 3]
            Target position in reference frame.
        ref_R : Tensor [N, 3, 3]
            Target rotation in reference frame.

        Returns
        -------
        Tensor [N, 6]
            Body-frame pose error ``[e_p; e_R]``.
        """
        pos_error = p_RE - ref_p  # [N, 3]
        e_p = torch.bmm(
            R_RE.transpose(-2, -1), pos_error.unsqueeze(-1)
        ).squeeze(-1)  # [N, 3]
        e_R = self._vee(
            torch.bmm(ref_R.transpose(-2, -1), R_RE)
            - torch.bmm(R_RE.transpose(-2, -1), ref_R)
        )  # [N, 3]
        e = torch.cat([e_p, e_R], dim=-1)  # [N, 6]
        self.pose_error[arm_idx][:] = e
        return e

    def _update_integral(self, arm_idx: int, dt: float) -> None:
        """Apply exponential-gated leaky-integral update and clamp.

        Uses norm-wise gating: the translation (dims 0-2) and rotation
        (dims 3-5) channels each compute a scalar gate from their
        respective error norm::

            gate_pos = exp(-(||e_p|| / gate_width_pos) ** 2)
            gate_rot = exp(-(||e_R|| / gate_width_rot) ** 2)
            I_new = I_old * (1 - leaky_sigma * dt)
                    + pose_error * dt * gate

        The Gaussian gate suppresses integral accumulation when the
        error norm is large, preventing windup during transients while
        allowing full integration near zero error.

        Then clamps element-wise to ``[-max_integral, max_integral]``.

        Parameters
        ----------
        arm_idx : int
            Arm index (0=left, 1=right).
        dt : float
            Integration timestep.
        """
        err = self.pose_error[arm_idx]       # [N, 6]
        prev = self.integral_error[arm_idx]  # [N, 6]

        # Norm-wise Gaussian gate (one scalar per channel per env)
        pos_norm = err[:, :3].norm(dim=-1, keepdim=True)  # [N, 1]
        rot_norm = err[:, 3:].norm(dim=-1, keepdim=True)  # [N, 1]

        gate_pos = torch.exp(
            -(pos_norm / self.integral_gate_width_pos) ** 2
        )  # [N, 1]
        gate_rot = torch.exp(
            -(rot_norm / self.integral_gate_width_rot) ** 2
        )  # [N, 1]

        gate = torch.cat([
            gate_pos.expand_as(err[:, :3]),
            gate_rot.expand_as(err[:, 3:]),
        ], dim=-1)  # [N, 6]

        updated = (
            prev * (1.0 - self.leaky_sigma * dt)
            + err * dt * gate
        )
        self.integral_error[arm_idx][:] = torch.clamp(
            updated, -self.max_integral, self.max_integral
        )

    def _compute_ee_body_twist(
        self,
        arm_idx: int,
        J_RE_E: torch.Tensor,
        q_dot: torch.Tensor,
    ) -> torch.Tensor:
        """Compute and store body-frame EE twist for the given arm.

        Parameters
        ----------
        arm_idx : int
            Arm index (0=left, 1=right).
        J_RE_E : Tensor [N, 6, num_dof]
            Relative Jacobian in EE body frame.
        q_dot : Tensor [N, num_dof]
            Joint velocities (or full qdot with free-flyer when
            use_angular_base_jacobian is True).

        Returns
        -------
        Tensor [N, 6]
            Body-frame twist ``V_RE_E = J_RE_E @ q_dot``.
        """
        twist = torch.bmm(J_RE_E, q_dot.unsqueeze(-1)).squeeze(-1)
        self.ee_body_twist[arm_idx][:] = twist

        # Compute ground-truth twist with true pelvis linear velocity
        if self.use_angular_base_jacobian:
            root_states = self.env.simulator.robot_root_states
            v_lin = root_states[:, 7:10]
            q_dot_true = q_dot.clone()
            q_dot_true[:, 0:3] = v_lin
            twist_true = torch.bmm(J_RE_E, q_dot_true.unsqueeze(-1)).squeeze(-1)
            self.ee_body_twist_true[arm_idx][:] = twist_true
        else:
            self.ee_body_twist_true[arm_idx][:] = twist

        return twist

    def _rot6d_to_rotm(self, rot6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D rotation to rotation matrix. [N, 6] -> [N, 3, 3]"""
        x = rot6d[:, :3]
        y = rot6d[:, 3:6]
        x = x / torch.norm(x, dim=-1, keepdim=True)
        z = torch.cross(x, y, dim=-1)
        z = z / torch.norm(z, dim=-1, keepdim=True)
        y = torch.cross(z, x, dim=-1)
        return torch.stack([x, y, z], dim=-1)

    def _quat_to_rotm(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (xyzw) to rotation matrix."""
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=-1)
        ], dim=-2)

    def _rotm_to_quat(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion (xyzw)."""
        # Batched, branch-stable conversion that avoids division by very small w.
        m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

        trace = m00 + m11 + m22
        eps = 1e-8

        w = torch.empty_like(trace)
        x = torch.empty_like(trace)
        y = torch.empty_like(trace)
        z = torch.empty_like(trace)

        mask_trace = trace > 0.0
        if mask_trace.any():
            s = torch.sqrt(torch.clamp(trace[mask_trace] + 1.0, min=eps)) * 2.0
            w[mask_trace] = 0.25 * s
            x[mask_trace] = (m21[mask_trace] - m12[mask_trace]) / s
            y[mask_trace] = (m02[mask_trace] - m20[mask_trace]) / s
            z[mask_trace] = (m10[mask_trace] - m01[mask_trace]) / s

        mask_x = (~mask_trace) & (m00 >= m11) & (m00 >= m22)
        if mask_x.any():
            s = torch.sqrt(torch.clamp(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x], min=eps)) * 2.0
            w[mask_x] = (m21[mask_x] - m12[mask_x]) / s
            x[mask_x] = 0.25 * s
            y[mask_x] = (m01[mask_x] + m10[mask_x]) / s
            z[mask_x] = (m02[mask_x] + m20[mask_x]) / s

        mask_y = (~mask_trace) & (~mask_x) & (m11 >= m22)
        if mask_y.any():
            s = torch.sqrt(torch.clamp(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y], min=eps)) * 2.0
            w[mask_y] = (m02[mask_y] - m20[mask_y]) / s
            x[mask_y] = (m01[mask_y] + m10[mask_y]) / s
            y[mask_y] = 0.25 * s
            z[mask_y] = (m12[mask_y] + m21[mask_y]) / s

        mask_z = (~mask_trace) & (~mask_x) & (~mask_y)
        if mask_z.any():
            s = torch.sqrt(torch.clamp(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z], min=eps)) * 2.0
            w[mask_z] = (m10[mask_z] - m01[mask_z]) / s
            x[mask_z] = (m02[mask_z] + m20[mask_z]) / s
            y[mask_z] = (m12[mask_z] + m21[mask_z]) / s
            z[mask_z] = 0.25 * s

        quat = torch.stack([x, y, z, w], dim=-1)
        quat = quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=eps)
        return quat

    def get_base_pose(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get root pose (position and rotation matrix), using xyzw quaternion convention."""
        """This is the "base frame" - or pelvis frame"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        root_states = self.env.simulator.robot_root_states[env_ids]
        root_pos_w = root_states[:, :3]
        root_quat_xyzw = root_states[:, 3:7]
        return root_pos_w, root_quat_xyzw

    def get_torso_pose(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current torso pose in world frame."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        torso_pos_w = self.env.simulator._rigid_body_pos[env_ids, self.torso_body_index]
        torso_quat_w = self.env.simulator._rigid_body_rot[env_ids, self.torso_body_index]

        return torso_pos_w, torso_quat_w

    def _get_joint_jacobian_in_dof_order(self) -> torch.Tensor:
        """Get PhysX Jacobian with columns aligned to simulator DOF order."""
        jacobian_w = self.env.simulator._robot.root_physx_view.get_jacobians()

        # Normalize Jacobian columns: fixed-base [num_dof], free-base [6 + num_dof]
        raw_cols = jacobian_w.shape[-1]
        if raw_cols == self.num_total_dof:
            jacobian_joint = jacobian_w
        elif raw_cols == self.num_total_dof + 6:
            jacobian_joint = jacobian_w[..., 6:]
        else:
            raise RuntimeError(
                f"Unexpected Jacobian column count {raw_cols}, expected "
                f"{self.num_total_dof} or {self.num_total_dof + 6}."
            )

        dof_ids = getattr(self.env.simulator, "dof_ids", None)
        if dof_ids is not None:
            dof_ids_t = torch.as_tensor(dof_ids, dtype=torch.long, device=jacobian_joint.device)
            jacobian_joint = jacobian_joint.index_select(-1, dof_ids_t)

        return jacobian_joint

    def _get_full_jacobian_in_dof_order(self) -> torch.Tensor:
        """Get PhysX Jacobian including free-flyer columns [6 + num_dof].

        Returns a Jacobian with shape [..., 6 + num_dof] where columns 0:6
        correspond to the floating-base (linear xyz, angular xyz) and columns
        6: correspond to actuated joints in simulator DOF order.
        """
        jacobian_w = self.env.simulator._robot.root_physx_view.get_jacobians()
        raw_cols = jacobian_w.shape[-1]

        if raw_cols == self.num_total_dof + 6:
            jacobian_full = jacobian_w
        elif raw_cols == self.num_total_dof:
            # Fixed-base: prepend zero free-flyer columns
            zeros = torch.zeros(
                *jacobian_w.shape[:-1], 6,
                device=jacobian_w.device, dtype=jacobian_w.dtype,
            )
            jacobian_full = torch.cat([zeros, jacobian_w], dim=-1)
        else:
            raise RuntimeError(
                f"Unexpected Jacobian column count {raw_cols}, expected "
                f"{self.num_total_dof} or {self.num_total_dof + 6}."
            )

        # Reorder joint columns (after the 6 free-flyer cols) to match DOF order
        dof_ids = getattr(self.env.simulator, "dof_ids", None)
        if dof_ids is not None:
            dof_ids_t = torch.as_tensor(dof_ids, dtype=torch.long, device=jacobian_full.device)
            joint_cols = jacobian_full[..., 6:].index_select(-1, dof_ids_t)
            jacobian_full = torch.cat([jacobian_full[..., :6], joint_cols], dim=-1)

        return jacobian_full

    def get_ee_jacobian(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute end-effector and torso Jacobians in world frame using PhysX."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.use_angular_base_jacobian:
            jacobian_w_env = self._get_full_jacobian_in_dof_order()[env_ids]
        else:
            jacobian_w_env = self._get_joint_jacobian_in_dof_order()[env_ids]

        # End-effector Jacobians in the world frame
        ee_jac_indices = [self._jac_row_left, self._jac_row_right]

        ee_jacobian_w = jacobian_w_env[:, ee_jac_indices, :, :]

        torso_jac_w = jacobian_w_env[:, self._jac_row_torso, :, :]

        return ee_jacobian_w, torso_jac_w

    def get_base_jacobian(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get base/root Jacobian in world frame.

        For fixed-root articulations this returns zeros because the root is kinematically fixed.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self._jac_row_base is None:
            ncols = (self.num_total_dof + 6) if self.use_angular_base_jacobian else self.num_total_dof
            return torch.zeros(
                (env_ids.numel(), 6, ncols),
                device=self.device,
                dtype=self.env.simulator.dof_pos.dtype,
            )

        if self.use_angular_base_jacobian:
            jacobian_w_env = self._get_full_jacobian_in_dof_order()[env_ids]
        else:
            jacobian_w_env = self._get_joint_jacobian_in_dof_order()[env_ids]
        return jacobian_w_env[:, self._jac_row_base, :, :]

    def get_reference_pose_and_jacobian(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get pose and Jacobian for the active arm-command reference frame."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        arm_command = self.env.command_manager.get_state("arm_command")
        reference_frame = getattr(arm_command, "reference_frame", "base")

        if reference_frame == "torso":
            ref_pos, ref_quat = self.get_torso_pose(env_ids)
            _, ref_jacobian = self.get_ee_jacobian(env_ids)
            return ref_pos, ref_quat, ref_jacobian

        if reference_frame == "base":
            ref_pos, ref_quat = self.get_base_pose(env_ids)
            ref_jacobian = self.get_base_jacobian(env_ids)
            return ref_pos, ref_quat, ref_jacobian

        if reference_frame == "heading":
            return self._compute_heading_pose_and_jacobian(env_ids, ground_attached=False)

        if reference_frame == "ground_heading":
            return self._compute_heading_pose_and_jacobian(env_ids, ground_attached=True)

        if reference_frame == "world":
            ref_pos = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=self.env.simulator.dof_pos.dtype)
            ref_quat = torch.zeros((env_ids.numel(), 4), device=self.device, dtype=self.env.simulator.dof_pos.dtype)
            ref_quat[:, 3] = 1.0
            ncols = (self.num_total_dof + 6) if self.use_angular_base_jacobian else self.num_total_dof
            ref_jacobian = torch.zeros(
                (env_ids.numel(), 6, ncols),
                device=self.device,
                dtype=self.env.simulator.dof_pos.dtype,
            )
            return ref_pos, ref_quat, ref_jacobian

        raise ValueError(
            f"Unsupported reference_frame '{reference_frame}'. "
            f"Expected 'base', 'torso', 'world', 'heading', or 'ground_heading'."
        )

    def _compute_heading_pose_and_jacobian(
        self, env_ids: torch.Tensor, ground_attached: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute heading (H) or ground-heading (H₀) frame pose and Jacobian.

        Notation:
            W = world, P = pelvis, H = heading (or H₀ = ground-heading)
            C_X = world-aligned Cartesian frame at X's origin
            J_WX_Cx = Jacobian of frame X w.r.t. world, expressed in C_X

        Heading frame H:
            p_WH = p_WP,  R_WH = R_z(ψ)
            J_WH_Ch = [Jv_WP_Cp; e_z · J_ψ]

        Ground-heading frame H₀:
            p_WH0 = [p_WP_x, p_WP_y, 0],  R_WH0 = R_z(ψ)
            J_WH0_Ch0 = [P_xy · Jv_WP_Cp; e_z · J_ψ]

        Args:
            env_ids: Environment indices.
            ground_attached: If True, compute H₀ (ground-heading).

        Returns:
            p_WH: [N, 3] reference frame origin in world.
            q_WH: [N, 4] reference frame orientation (xyzw).
            J_WH_Ch: [N, 6, nv] reference frame Jacobian.
        """
        p_WP, q_WP = self.get_base_pose(env_ids)
        q_WH = heading_quat_from_root_quat(q_WP)

        # Frame origin
        if ground_attached:
            p_WH = p_WP.clone()
            p_WH[:, 2] = 0.0
        else:
            p_WH = p_WP

        # Pelvis Jacobian: J_WP_Cp = [Jv_WP_Cp; Jw_WP_Cp]
        J_WP_Cp = self.get_base_jacobian(env_ids)  # [N, 6, nv]
        Jv_WP_Cp = J_WP_Cp[:, :3, :]  # [N, 3, nv]
        Jw_WP_Cp = J_WP_Cp[:, 3:, :]  # [N, 3, nv]

        # Exact yaw-rate Jacobian: ψ̇ = aᵀ · ω_WP
        # where a = [-fz·fx/d, -fz·fy/d, 1], f = R_WP · e_x, d = fx²+fy²
        R_WP = quat_to_rotmat(q_WP)  # [N, 3, 3]
        f = R_WP[:, :, 0]  # forward axis in world [N, 3]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        d = (fx * fx + fy * fy).clamp(min=1e-8)  # [N]

        a = torch.stack([
            -fz * fx / d,
            -fz * fy / d,
            torch.ones_like(fz),
        ], dim=-1)  # [N, 3]

        # J_psi = aᵀ @ Jw_WP_Cp: [N, 1, 3] @ [N, 3, nv] -> [N, 1, nv]
        J_psi = torch.bmm(a.unsqueeze(1), Jw_WP_Cp)  # [N, 1, nv]

        # Angular part: Jw_WH_Ch = e_z · J_psi (yaw-only)
        Jw_WH_Ch = torch.zeros_like(Jw_WP_Cp)
        Jw_WH_Ch[:, 2, :] = J_psi.squeeze(1)

        # Linear part
        if ground_attached:
            # P_xy · Jv_WP_Cp: zero z-row to match H₀ definition (z=0).
            # When ground_attached, The output is J_WH0_Ch0
            # This creates a non-zero free-flyer linear-z column in the
            # relative Jacobian J_RE_E, enabling the actor/critic asymmetry.
            Jv_WH_Ch = Jv_WP_Cp.clone()
            Jv_WH_Ch[:, 2, :] = 0.0
        else:
            Jv_WH_Ch = Jv_WP_Cp

        # Assemble J_WH_Ch = [Jv_WH_Ch; Jw_WH_Ch]
        J_WH_Ch = torch.cat([Jv_WH_Ch, Jw_WH_Ch], dim=1)  # [N, 6, nv]

        return p_WH, q_WH, J_WH_Ch

    def _apply_fk_noise(
        self,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        jacobian_w: torch.Tensor,
        noise_sigma: float,
        arm_dof_indices: list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply first-order SE(3) FK noise via Jacobian-based perturbation.

        Samples joint noise δq ~ N(0, σ²I) for arm DOFs, maps to a 6D twist
        via the Jacobian, and applies the perturbation to the EE pose using
        the Rodrigues formula (exponential map) for the rotation part.

        Parameters
        ----------
        ee_pos_w : Tensor [num_envs, num_ee, 3]
        ee_quat_w : Tensor [num_envs, num_ee, 4] (xyzw convention)
        jacobian_w : Tensor [num_envs, num_ee, 6, num_dof]
        noise_sigma : float
        arm_dof_indices : indices of arm DOFs in the Jacobian columns

        Returns
        -------
        (noisy_pos, noisy_quat) with same shapes as inputs.
        """
        num_envs, num_ee = ee_pos_w.shape[0], ee_pos_w.shape[1]
        device = ee_pos_w.device
        dtype = ee_pos_w.dtype

        if isinstance(arm_dof_indices, list):
            arm_dof_indices = torch.tensor(arm_dof_indices, device=device, dtype=torch.long)

        num_arm_dof = arm_dof_indices.shape[0]

        # Sample joint noise: [num_envs, num_arm_dof]
        dq = torch.randn(num_envs, num_arm_dof, device=device, dtype=dtype) * noise_sigma

        # Extract arm columns from Jacobian: [num_envs, num_ee, 6, num_arm_dof]
        # When use_angular_base_jacobian=True, Jacobian has 6 extra free-flyer
        # columns at the front; offset arm indices accordingly.
        if self.use_angular_base_jacobian:
            J_arm = jacobian_w[..., arm_dof_indices + 6]
        else:
            J_arm = jacobian_w[..., arm_dof_indices]

        # Map to task-space twist: [num_envs, num_ee, 6]
        dx = torch.einsum("neij,nj->nei", J_arm, dq)

        dp = dx[..., :3]  # [num_envs, num_ee, 3]
        dw = dx[..., 3:]  # [num_envs, num_ee, 3]

        # Position perturbation
        noisy_pos = ee_pos_w + dp

        # --- Rodrigues formula: R_delta = exp([dw]×) ---
        dw_flat = dw.reshape(-1, 3)
        theta = torch.norm(dw_flat, dim=-1, keepdim=True)  # [N, 1]
        theta_sq = theta * theta

        K = self._skew(dw_flat)          # [N, 3, 3]
        K_sq = torch.bmm(K, K)           # [N, 3, 3]

        # Taylor expansion for small angles to avoid division by zero
        small = (theta.squeeze(-1) < 1e-6)
        sinc = torch.where(small, 1.0 - theta_sq.squeeze(-1) / 6.0,
                           torch.sin(theta.squeeze(-1)) / theta.squeeze(-1).clamp(min=1e-8))
        cosc = torch.where(small, 0.5 - theta_sq.squeeze(-1) / 24.0,
                           (1.0 - torch.cos(theta.squeeze(-1))) / theta_sq.squeeze(-1).clamp(min=1e-8))

        eye3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        R_delta = eye3 + sinc[:, None, None] * K + cosc[:, None, None] * K_sq

        # Compose: R_noisy = R_nom @ R_delta
        R_nom = self._quat_to_rotm(ee_quat_w.reshape(-1, 4))  # [N, 3, 3]
        R_noisy = torch.bmm(R_nom, R_delta)

        # Convert back to quaternion
        noisy_quat = self._rotm_to_quat(R_noisy).reshape(num_envs, num_ee, 4)

        return noisy_pos, noisy_quat

    def get_ee_pose(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current end-effector poses in world frame (p_WE, q_WE)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        ee_pos_w = self.env.simulator._rigid_body_pos[env_ids][:, self.ee_body_indices]
        ee_quat_w = self.env.simulator._rigid_body_rot[env_ids][:, self.ee_body_indices]

        # Apply FK noise if enabled (Tier 3: first-order SE(3) approximation)
        if self.enable_fk_noise and self.fk_noise_sigma > 0.0:
            ee_jacobian_w, _ = self.get_ee_jacobian(env_ids)
            ee_pos_w, ee_quat_w = self._apply_fk_noise(
                ee_pos_w, ee_quat_w, ee_jacobian_w,
                self.fk_noise_sigma, self.arm_dof_indices,
            )

        return ee_pos_w, ee_quat_w

    def get_ee_pose_and_jacobian(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get end-effector poses and Jacobians in world frame."""
        ee_pos_w, ee_quat_w = self.get_ee_pose(env_ids)
        ee_jacobian_w, _ = self.get_ee_jacobian(env_ids)
        return ee_pos_w, ee_quat_w, ee_jacobian_w

    def get_arm_state(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current arm joint positions and velocities."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        arm_pos = self.env.simulator.dof_pos[env_ids][:, self.arm_dof_indices]
        arm_vel = self.env.simulator.dof_vel[env_ids][:, self.arm_dof_indices]
        
        return arm_pos, arm_vel
    
    def get_robot_state(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full robot state: return full q and qdot.

        When use_angular_base_jacobian is True, qdot is prepended with
        [0, 0, 0, ang_vel_w] (6 free-flyer velocity entries: zero linear,
        measured angular) to match the full Jacobian column count.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        joint_pos = self.env.simulator.dof_pos[env_ids]
        joint_vel = self.env.simulator.dof_vel[env_ids]

        if self.use_angular_base_jacobian:
            # Build qdot_full = [0, 0, 0, ang_vel_w(3), joint_vel(N)]
            # Angular velocity from root states (indices 10:13 = ang_vel in world)
            root_states = self.env.simulator.robot_root_states[env_ids]
            ang_vel_w = root_states[:, 10:13]  # world-frame angular velocity
            lin_vel_zero = torch.zeros_like(ang_vel_w)
            joint_vel = torch.cat([lin_vel_zero, ang_vel_w, joint_vel], dim=-1)

        return joint_pos, joint_vel

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset controller internal state for specified environments."""
        if not hasattr(self, "arm_actions"):
            return

        if env_ids is None:
            self.arm_actions.zero_()
            self.pose_error.zero_()
            self.integral_error.zero_()
            self.ee_body_twist.zero_()
        else:
            self.arm_actions[env_ids] = 0.0
            self.pose_error[:, env_ids, :] = 0.0
            self.integral_error[:, env_ids, :] = 0.0
            self.ee_body_twist[:, env_ids, :] = 0.0

    def compute_arm_actions(self, dt: float | None = None) -> torch.Tensor:
        """
        Compute arm joint targets for end-effector control.

        Args:
            dt: Integration timestep. If None, uses self.env.dt.

        Returns:
            torch.Tensor: [num_envs, num_arm_dof] arm joint targets
        """
        raise NotImplementedError("Subclasses must implement compute_arm_actions()")
