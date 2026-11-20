from __future__ import annotations

import math
import os

import torch

from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.rotation_6d import (
    heading_quat_from_root_quat,
    quat_to_rot6d,
    quat_to_rotmat,
    rot6d_to_rotmat,
    world_to_reference_frame,
)


def _rotmat_to_rot6d_cmd(rotmat: torch.Tensor) -> torch.Tensor:
    """Rotation matrices -> 6D representation (first two columns, row-major)."""
    return rotmat[..., :, :2].transpose(-2, -1).reshape(*rotmat.shape[:-2], 6)


class ArmTrackingCommand(CommandTermBase):
    """Generate arm tracking targets in SE(3) space."""

    def setup(self):
        self.num_envs = self.env.num_envs
        self.device = self.env.device

        # Config parameters
        self.reference_frame = str(self.cfg.params.get("reference_frame", "base")).lower()
        if self.reference_frame not in {"base", "torso", "world", "heading", "ground_heading"}:
            raise ValueError(
                f"Unsupported reference_frame '{self.reference_frame}'. "
                f"Expected 'base', 'torso', 'world', 'heading', or 'ground_heading'."
            )
        self.use_fixed_sampling = bool(self.cfg.params.get("use_fixed_sampling", False))
        self.torso_body_index = self.env.simulator.find_rigid_body_indice("torso_link")

        # Target arm joint positions (for base controller)
        self.target_arm_pos = torch.zeros(
            (self.num_envs, len(self.env.arm_dof_indices)), device=self.device
        )

        # Target end-effector poses in SE(3): pos(3) + rot6d(6) for left and right
        # Initialize to reasonable default (arms at sides)
        self.target_ee_pos = torch.zeros((self.num_envs, 2, 3), device=self.device)
        self.target_ee_pos[:, 0, 0] = 0.15  # Left arm: +X (left side)
        self.target_ee_pos[:, 0, 1] = 0.20  # Left arm: +Y (left side)
        self.target_ee_pos[:, 1, 0] = 0.15  # Right arm: +X (right side)
        self.target_ee_pos[:, 1, 1] = -0.20  # Right arm: -Y (right side)
        self.target_ee_pos[:, :, 2] = 0.05  # Both arms: Z height
        
        # Initialize rotations to identity (first two columns of identity matrix)
        self.target_ee_rot6d = torch.zeros((self.num_envs, 2, 6), device=self.device)
        self.target_ee_rot6d[:, :, 0] = 1.0  # x-axis: [1, 0, 0]
        self.target_ee_rot6d[:, :, 4] = 1.0  # y-axis: [0, 1, 0]

        self.target_ee_pos_default = self.target_ee_pos.clone()
        self.target_ee_rot6d_default = self.target_ee_rot6d.clone()

        self.target_ee_rotmat_default = rot6d_to_rotmat(self.target_ee_rot6d_default.reshape(-1, 6)).reshape(
            self.target_ee_rot6d_default.shape[0], 2, 3, 3
        )

        # Get end-effector body names (customize for your robot)
        self.ee_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.ee_body_indices = self._get_ee_body_indices()

        # Rejection sampling parameters
        self.max_rejection_attempts = 100
        self.min_manipulability = 0.10  # Threshold to avoid singularities

        self.wrist_joint_bounds_weight = float(self.cfg.params.get("wrist_joint_bounds_weight", 0.3))
        self.other_joint_bounds_weight = float(self.cfg.params.get("other_joint_bounds_weight", 0.7))

        self.update_joint_limits()

        # Initialize defaults from environment default arm pose to avoid
        # hard-coded near-singular SE(3) targets.
        # env_ids_all = torch.arange(self.num_envs, device=self.device)
        # default_arm_pos = self.env.default_dof_pos[:, self.env.arm_dof_indices]
        # ee_pos_world_default, ee_quat_world_default = self._compute_fk(env_ids_all, default_arm_pos)
        # ee_pos_ref_default, ee_rot6d_ref_default = self._to_reference_frame(
        #     env_ids_all, ee_pos_world_default, ee_quat_world_default
        # )
        # self.target_arm_pos[:] = default_arm_pos
        # self.target_ee_pos_default = ee_pos_ref_default
        # self.target_ee_rot6d_default = ee_rot6d_ref_default
        # self.target_ee_pos[:] = self.target_ee_pos_default
        # self.target_ee_rot6d[:] = self.target_ee_rot6d_default

        # Tracking metrics
        self.metrics = {}

        # --- Trajectory mode config (Task 1.1) ---
        self.trajectory_mode = self.cfg.params.get("trajectory_mode", None)
        if self.trajectory_mode not in {None, "non_smooth", "smooth", "precomputed"}:
            raise ValueError(
                f"Invalid trajectory_mode '{self.trajectory_mode}'. "
                f"Expected None, 'non_smooth', 'smooth', or 'precomputed'."
            )

        self.sampling_interval = int(self.cfg.params.get("sampling_interval", 500))
        if self.sampling_interval <= 0:
            raise ValueError(f"sampling_interval must be > 0, got {self.sampling_interval}")

        self.interpolation_method = str(self.cfg.params.get("interpolation_method", "linear"))

        # Max per-joint step between consecutive waypoints (non-smooth mode only)
        # Default: 30 degrees (~0.5236 rad)
        self.max_joint_step = float(self.cfg.params.get("max_joint_step", math.radians(30)))

        self.total_steps = int(self.env.max_episode_length)
        self.num_waypoints = self.total_steps // self.sampling_interval

        if self.trajectory_mode not in {None, "precomputed"} and self.num_waypoints == 0:
            raise ValueError(
                f"num_waypoints is 0 (total_steps={self.total_steps} < "
                f"sampling_interval={self.sampling_interval}). "
                f"At least 1 waypoint is required when trajectory_mode is enabled."
            )

        # --- Trajectory state tensors (Task 1.2) ---
        if self.trajectory_mode == "precomputed":
            trajectory_file = self.cfg.params.get("trajectory_file", None)
            self._load_trajectory_dataset(trajectory_file)
            self.trajectory_pos = torch.zeros(
                (self.num_envs, self.total_steps, 2, 3), dtype=torch.float32, device=self.device
            )
            self.trajectory_rot6d = torch.zeros(
                (self.num_envs, self.total_steps, 2, 6), dtype=torch.float32, device=self.device
            )
            self.trajectory_index = torch.zeros(
                (self.num_envs,), dtype=torch.int64, device=self.device
            )
        elif self.trajectory_mode is not None:
            self.trajectory_pos = torch.zeros(
                (self.num_envs, self.total_steps, 2, 3), dtype=torch.float32, device=self.device
            )
            self.trajectory_rot6d = torch.zeros(
                (self.num_envs, self.total_steps, 2, 6), dtype=torch.float32, device=self.device
            )
            self.trajectory_index = torch.zeros(
                (self.num_envs,), dtype=torch.int64, device=self.device
            )

        # --- Height trajectory config ---
        self.height_range_min = float(self.cfg.params.get("height_range_min", 0.68))
        self.height_range_max = float(self.cfg.params.get("height_range_max", 0.78))
        self.height_smooth_prob = float(self.cfg.params.get("height_smooth_prob", 0.70))
        self.height_max_step = float(self.cfg.params.get("height_max_step", 0.10))
        self.height_nominal = float(self.cfg.params.get("height_nominal", 0.73))

        if self.height_range_min > self.height_range_max:
            raise ValueError(
                f"height_range_min ({self.height_range_min}) must be <= "
                f"height_range_max ({self.height_range_max})."
            )
        if self.height_max_step <= 0:
            raise ValueError(
                f"height_max_step must be > 0, got {self.height_max_step}"
            )

        # Height target buffer (always allocated, initialized to nominal so reward
        # functions can read it even when trajectory_mode is None)
        self.target_base_height = torch.full(
            (self.num_envs,), self.height_nominal, dtype=torch.float32, device=self.device
        )

        # --- Posture (trunk orientation) command ---
        # Commanded orientation is the target gravity direction in each body's
        # frame: upright is (0, 0, -1), a forward pitch of theta is
        # (sin theta, 0, -cos theta). This is exactly the roll/pitch degrees of
        # freedom with no yaw, and it mirrors under the sagittal plane by
        # negating y. Ranges are in DEGREES and default to zero, so a config
        # that does not set them commands upright and behaves as before.
        self.pelvis_pitch_range_deg = tuple(
            self.cfg.params.get("pelvis_pitch_range_deg", (0.0, 0.0))
        )
        self.torso_pitch_range_deg = tuple(
            self.cfg.params.get("torso_pitch_range_deg", (0.0, 0.0))
        )
        for name, rng in (
            ("pelvis_pitch_range_deg", self.pelvis_pitch_range_deg),
            ("torso_pitch_range_deg", self.torso_pitch_range_deg),
        ):
            if len(rng) != 2 or rng[0] > rng[1]:
                raise ValueError(f"{name} must be (min, max) with min <= max, got {rng}")

        # Optional waist range. When set, the torso pitch is sampled as the
        # pelvis pitch plus this waist bend, so the torso can never end up
        # behind the pelvis (a backward waist bend). When None, the torso is
        # sampled independently from torso_pitch_range_deg as before.
        waist_rng = self.cfg.params.get("waist_pitch_range_deg", None)
        self.waist_pitch_range_deg = None if waist_rng is None else tuple(waist_rng)
        if self.waist_pitch_range_deg is not None and (
            len(self.waist_pitch_range_deg) != 2
            or self.waist_pitch_range_deg[0] > self.waist_pitch_range_deg[1]
        ):
            raise ValueError(
                f"waist_pitch_range_deg must be (min, max) with min <= max, "
                f"got {self.waist_pitch_range_deg}"
            )
        # Curriculum scale on the sampled pitch, in [0, 1]. The pipeline's
        # scheduler ramps it during Phase DE so the lean widens gradually.
        self.posture_range_scale = float(self.cfg.params.get("posture_range_scale", 1.0))
        # Separate, smaller lean while walking (non-zero velocity command). When
        # these are set, each episode samples a standing and a walking pitch
        # and the current velocity command selects between them. When unset,
        # the one sampled pitch applies regardless of the velocity command.
        # The command slews toward its goal at posture_rate_deg_s, so
        # stand/walk switches and resets never step the command.
        walk_pelvis = self.cfg.params.get("walking_pelvis_pitch_range_deg", None)
        walk_waist = self.cfg.params.get("walking_waist_pitch_range_deg", None)
        self.walking_pelvis_pitch_range_deg = None if walk_pelvis is None else tuple(walk_pelvis)
        self.walking_waist_pitch_range_deg = None if walk_waist is None else tuple(walk_waist)
        if (self.walking_pelvis_pitch_range_deg is None) != (self.walking_waist_pitch_range_deg is None):
            raise ValueError(
                "walking_pelvis_pitch_range_deg and walking_waist_pitch_range_deg "
                "must be set together"
            )
        self.posture_rate_deg_s = float(self.cfg.params.get("posture_rate_deg_s", 30.0))

        # Sampled per-episode pitch (rad), [pelvis, torso], for standing and
        # walking, and the slewed pitch actually commanded.
        self._posture_goal_pitch = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self._posture_walk_goal_pitch = torch.zeros_like(self._posture_goal_pitch)
        self._posture_pitch = torch.zeros_like(self._posture_goal_pitch)

        self.target_pelvis_gravity = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.target_pelvis_gravity[:, 2] = -1.0
        self.target_torso_gravity = self.target_pelvis_gravity.clone()

        # Full-episode height trajectory buffer (only when trajectory_mode is active)
        if self.trajectory_mode is not None:
            self.height_trajectory = torch.full(
                (self.num_envs, self.total_steps),
                self.height_nominal,
                dtype=torch.float32,
                device=self.device,
            )

    def _get_ee_body_indices(self):
        """Get body indices for end-effectors."""
        indices = []
        for name in self.ee_body_names:
            idx = self.env.simulator.find_rigid_body_indice(name)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """Sample feasible SE(3) targets using joint sampling + FK + rejection."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        self._sample_posture_command(env_ids)

        if self.use_fixed_sampling:
            self.target_ee_pos[env_ids] = self.target_ee_pos_default[env_ids]
            self.target_ee_rot6d[env_ids] = self.target_ee_rot6d_default[env_ids]
            # Reproject z for ground_heading (defaults are in base frame)
            if self.reference_frame == "ground_heading":
                self.target_ee_pos[env_ids, :, 2] += self.target_base_height[env_ids].unsqueeze(-1)
            return

        if self.trajectory_mode == "precomputed":
            # Randomly select one trajectory per environment from the loaded dataset
            indices = torch.randint(0, self._num_dataset_trajectories, (len(env_ids),))
            # Copy selected CPU slices to GPU buffers for reset env_ids only
            self.trajectory_pos[env_ids] = self._cpu_trajectory_pos[indices].to(self.device)
            self.trajectory_rot6d[env_ids] = self._cpu_trajectory_rot6d[indices].to(self.device)

            # Reset trajectory index and set initial targets
            self.trajectory_index[env_ids] = 0
            self.target_ee_pos[env_ids] = self.trajectory_pos[env_ids, 0]
            self.target_ee_rot6d[env_ids] = self.trajectory_rot6d[env_ids, 0]
            # Build height trajectory for reset envs (parallel to EE trajectory)
            if hasattr(self, 'height_trajectory'):
                self._build_height_trajectory(env_ids)
                self.target_base_height[env_ids] = self.height_trajectory[env_ids, 0]
            # Reproject z for ground_heading at reset (frame 0)
            if self.reference_frame == "ground_heading" and hasattr(self, 'target_base_height'):
                self.target_ee_pos[env_ids, :, 2] += self.target_base_height[env_ids].unsqueeze(-1)
            return

        if self.trajectory_mode is not None:
            # Trajectory mode: pre-sample waypoints and build full reference trajectory
            self._sample_and_build_trajectory(env_ids)
            self.trajectory_index[env_ids] = 0
            # Set initial targets to frame 0
            self.target_ee_pos[env_ids] = self.trajectory_pos[env_ids, 0]
            self.target_ee_rot6d[env_ids] = self.trajectory_rot6d[env_ids, 0]
            # Build height trajectory for reset envs (parallel to EE trajectory)
            if hasattr(self, 'height_trajectory'):
                self._build_height_trajectory(env_ids)
                self.target_base_height[env_ids] = self.height_trajectory[env_ids, 0]
            return

        # Existing single-point sampling path
        feasible_joints = self._sample_feasible_joints_with_rejection(env_ids)
        ee_pos_world, ee_quat_world = self._compute_fk(env_ids, feasible_joints)
        ee_pos_ref, ee_rot6d_ref = self._to_reference_frame(env_ids, ee_pos_world, ee_quat_world)

        self.target_arm_pos[env_ids] = feasible_joints
        self.target_ee_pos[env_ids] = ee_pos_ref
        self.target_ee_rot6d[env_ids] = ee_rot6d_ref

    def _sample_posture_command(self, env_ids: torch.Tensor) -> None:
        """Sample the per-episode trunk pitch for the given envs.

        Pitch is drawn uniformly from the configured degree ranges (default
        (0, 0), i.e. upright) and multiplied by ``posture_range_scale``. The
        commanded pitch restarts upright and slews toward this goal in
        ``_update_posture``. Roll is not commanded yet; the gravity buffers are
        full 3-vectors so it can be added without changing any shapes.
        """
        n = env_ids.numel()
        deg = math.pi / 180.0

        def _uniform(rng):
            lo, hi = float(rng[0]), float(rng[1])
            if lo == 0.0 and hi == 0.0:
                return torch.zeros(n, device=self.device)
            return torch.empty(n, device=self.device).uniform_(lo, hi) * deg

        pelvis = _uniform(self.pelvis_pitch_range_deg)
        if self.waist_pitch_range_deg is not None:
            torso = pelvis + _uniform(self.waist_pitch_range_deg)
        else:
            torso = _uniform(self.torso_pitch_range_deg)
        self._posture_goal_pitch[env_ids, 0] = self.posture_range_scale * pelvis
        self._posture_goal_pitch[env_ids, 1] = self.posture_range_scale * torso
        if self.walking_pelvis_pitch_range_deg is not None:
            walk_pelvis = _uniform(self.walking_pelvis_pitch_range_deg)
            walk_torso = walk_pelvis + _uniform(self.walking_waist_pitch_range_deg)
            self._posture_walk_goal_pitch[env_ids, 0] = self.posture_range_scale * walk_pelvis
            self._posture_walk_goal_pitch[env_ids, 1] = self.posture_range_scale * walk_torso
        self._posture_pitch[env_ids] = 0.0
        self._write_posture_gravity(env_ids)

    def _write_posture_gravity(self, env_ids=None) -> None:
        """Rebuild the gravity-direction commands from the commanded pitch."""
        sl = slice(None) if env_ids is None else env_ids
        for k, buf in ((0, self.target_pelvis_gravity), (1, self.target_torso_gravity)):
            theta = self._posture_pitch[sl, k]
            buf[sl, 0] = torch.sin(theta)
            buf[sl, 1] = 0.0
            buf[sl, 2] = -torch.cos(theta)

    def _update_posture(self) -> None:
        """Slew the commanded pitch toward its standing or walking goal."""
        goal = self._posture_goal_pitch
        if self.walking_pelvis_pitch_range_deg is not None:
            cmd = self.env.command_manager.commands
            standing = (cmd[:, :3].abs().sum(dim=-1) == 0.0).unsqueeze(-1)
            goal = torch.where(standing, goal, self._posture_walk_goal_pitch)
        max_step = self.posture_rate_deg_s * (math.pi / 180.0) * self.env.dt
        self._posture_pitch += (goal - self._posture_pitch).clamp(-max_step, max_step)
        self._write_posture_gravity()

    def _to_reference_frame(
        self, env_ids: torch.Tensor, ee_pos_world: torch.Tensor, ee_quat_world: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert end-effector poses from world frame to configured reference frame."""
        ref_pos, ref_quat = self._get_reference_pose(env_ids)
        ee_pos_ref = torch.zeros_like(ee_pos_world)
        ee_quat_ref = torch.zeros_like(ee_quat_world)
        for i in range(2):  # Left and right
            ee_pos_ref[:, i], ee_quat_ref[:, i] = world_to_reference_frame(
                ee_pos_world[:, i], ee_quat_world[:, i], ref_pos, ref_quat
            )

        ee_rot6d_ref = torch.zeros((ee_quat_ref.shape[0], 2, 6), device=self.device)
        for i in range(2):
            ee_rot6d_ref[:, i] = quat_to_rot6d(ee_quat_ref[:, i])

        return ee_pos_ref, ee_rot6d_ref

    def _get_reference_pose(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get configured reference-frame pose in world coordinates."""
        if self.reference_frame == "torso":
            ref_pos = self.env.simulator._rigid_body_pos[env_ids, self.torso_body_index]
            ref_quat = self.env.simulator._rigid_body_rot[env_ids, self.torso_body_index]
            return ref_pos, ref_quat

        if self.reference_frame == "base":
            ref_pos = self.env.simulator.robot_root_states[env_ids, :3]
            ref_quat = self.env.simulator.robot_root_states[env_ids, 3:7]
            return ref_pos, ref_quat

        if self.reference_frame == "heading":
            root_pos = self.env.simulator.robot_root_states[env_ids, :3]
            root_quat = self.env.simulator.robot_root_states[env_ids, 3:7]
            ref_quat = heading_quat_from_root_quat(root_quat)
            return root_pos, ref_quat

        if self.reference_frame == "ground_heading":
            root_pos = self.env.simulator.robot_root_states[env_ids, :3]
            root_quat = self.env.simulator.robot_root_states[env_ids, 3:7]
            ref_pos = root_pos.clone()
            ref_pos[:, 2] = 0.0
            ref_quat = heading_quat_from_root_quat(root_quat)
            return ref_pos, ref_quat

        if self.reference_frame == "world":
            ref_pos = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=self.target_ee_pos.dtype)
            ref_quat = torch.zeros((env_ids.numel(), 4), device=self.device, dtype=self.target_ee_pos.dtype)
            ref_quat[:, 3] = 1.0
            return ref_pos, ref_quat

        raise ValueError(
            f"Unsupported reference_frame '{self.reference_frame}'. "
            f"Expected 'base', 'torso', 'world', 'heading', or 'ground_heading'."
        )

    def _rotation_acceptance_mask(self, env_ids: torch.Tensor, ee_rot6d: torch.Tensor) -> torch.Tensor:
        """Accept samples whose left/right orientation deltas from default are within threshold.
        It is a placeholder for the potential future use
        """
        candidate_rotmat = rot6d_to_rotmat(ee_rot6d.reshape(-1, 6)).reshape(ee_rot6d.shape[0], 2, 3, 3)
        default_rotmat = self.target_ee_rotmat_default[env_ids]
        relative_rot = torch.matmul(default_rotmat.transpose(-2, -1), candidate_rotmat)

        trace = relative_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        rot_error = 3 - trace # trace(I - Rd^T R)

        return (rot_error <= 2).all(dim=1)
    
    def update_joint_limits(self):
        """Update effective joint limits based on configured weights."""
        hard_reset_joints = ["left_shoulder_roll_joint",
                             "right_shoulder_roll_joint"]
        
        self.arm_limits = self.env.simulator.dof_pos_limits[self.env.arm_dof_indices]

        # Apply hard reset joints -> if left-shoulder-roll then low-bound 0, if right-shoulder-roll then upperbound is 0
        for joint_name in hard_reset_joints:
            if joint_name in self.env.dof_names:
                joint_idx = self.env.dof_names.index(joint_name)
                if "left" in joint_name:
                    self.arm_limits[self.env.arm_dof_indices == joint_idx, 0] = 0.0
                elif "right" in joint_name:
                    self.arm_limits[self.env.arm_dof_indices == joint_idx, 1] = 0.0

        # Per-env, per-DOF sampling-range weights.
        # 1.0 => full simulator joint range, <1.0 => narrower range around center.
        self.joint_bounds_weight = torch.ones((self.num_envs, self.env.num_dof), device=self.device) * self.other_joint_bounds_weight
        limit_joint_names = [
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_shoulder_roll_joint",
            "left_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_shoulder_roll_joint",
            "right_wrist_yaw_joint",
        ]
        for joint_name in limit_joint_names:
            if joint_name in self.env.dof_names:
                joint_idx = self.env.dof_names.index(joint_name)
                self.joint_bounds_weight[:, joint_idx] = self.wrist_joint_bounds_weight       
                

    def _sample_feasible_joints_with_rejection(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample joint configurations with rejection sampling to avoid singularities."""
        # print(f"arm_limits: {self.arm_limits.cpu().numpy()}")
        default_arm_pos = self.env.default_dof_pos[env_ids][:, self.env.arm_dof_indices]
        # print(f"default_arm_pos: {default_arm_pos.cpu().numpy()}")
        num_samples = len(env_ids)
        num_arm_dof = len(self.env.arm_dof_indices)

        # Effective sampling bounds after applying per-DOF weights.
        # Use default_dof_pos as center; arm_half_range is the smaller distance
        # from default_dof_pos to either joint limit, scaled by 30%.
        dist_to_lower = default_arm_pos[0] - self.arm_limits[:, 0]  # [A]
        dist_to_upper = self.arm_limits[:, 1] - default_arm_pos[0]  # [A]
        arm_half_range = torch.min(dist_to_lower, dist_to_upper)    # [A]
        arm_center = default_arm_pos  # [K, A]
        arm_weight = self.joint_bounds_weight[env_ids][:, self.env.arm_dof_indices]  # [K, A]
        lower = arm_center - arm_half_range.unsqueeze(0) * arm_weight  # [K, A]
        upper = arm_center + arm_half_range.unsqueeze(0) * arm_weight  # [K, A]

        # Initialize with random samples
        samples = torch.rand((num_samples, num_arm_dof), device=self.device)
        # Sample around default arm pose with per-DOF weighted half-ranges.
        # Shapes:
        #   samples: [K, A], arm_weight: [K, A], arm_half_range: [A], arm_center: [K, A]
        samples = (
            (2.0 * samples - 1.0) * (arm_half_range.unsqueeze(0) * arm_weight)
            + arm_center
        )
        samples = torch.clamp(samples, min=lower, max=upper)

        # Rejection sampling: check manipulability
        for attempt in range(self.max_rejection_attempts):
            # Compute manipulability for current samples
            manipulability = self._compute_manipulability(samples, lower, upper)

            # Find samples that need resampling (low manipulability)
            bad_mask = manipulability < self.min_manipulability

            if not bad_mask.any():
                break  # All samples are good

            # Resample bad ones
            num_bad = bad_mask.sum().item()
            new_samples = torch.rand((num_bad, num_arm_dof), device=self.device)
            new_samples = new_samples * (upper[bad_mask] - lower[bad_mask]) + lower[bad_mask]
            samples[bad_mask] = new_samples


        # Add the summary of this rejection - how many samples were rejected and how many attempts were made
        # print(f"Rejection sampling: {bad_mask.sum().item()} out of {num_samples} samples were rejected after {attempt+1} attempts.")
        # print(f"samples: {samples.cpu().numpy()}")

        #NOTE(JS)
        # samples = self.env.default_dof_pos[env_ids][:, self.env.arm_dof_indices]  # --- IGNORE ---
        return samples

    def _compute_manipulability(
        self,
        joint_configs: torch.Tensor,
        lower_limits: torch.Tensor,
        upper_limits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute manipulability measure (simplified: joint distance from limits)."""
        # Simplified manipulability: distance from effective sampling limits
        # Better: compute actual Jacobian-based manipulability
        eps = 1e-6
        # Normalize joint positions to [0, 1]
        normalized = (joint_configs - lower_limits) / torch.clamp(upper_limits - lower_limits, min=eps)

        # Distance from limits (0 at limits, 0.5 at center)
        dist_from_limits = torch.min(normalized, 1.0 - normalized)

        # Manipulability: minimum distance across all joints
        manipulability = dist_from_limits.min(dim=1)[0]

        return manipulability

    def _compute_fk(self, env_ids: torch.Tensor, joint_configs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forward kinematics using simulator."""
        # Temporarily set joint positions and run FK
        original_dof_pos = self.env.simulator.dof_pos[env_ids].clone()
        original_dof_vel = self.env.simulator.dof_vel[env_ids].clone()

        # Ensure reset root states are pushed before FK queries.
        self.env.simulator.set_actor_root_state_tensor_robots(env_ids, self.env.simulator.robot_root_states)
        self.env.simulator.set_dof_state_tensor_robots(env_ids, self.env.simulator.dof_state)
        if hasattr(self.env.simulator, "write_state_updates"):
            self.env.simulator.write_state_updates()
        self.env.simulator.refresh_sim_tensors()

        # Set arm joints to target configuration
        row_ids = env_ids.unsqueeze(1)  # [K, 1]
        col_ids = self.env.arm_dof_indices.unsqueeze(0)  # [1, 14]
        self.env.simulator.dof_pos[row_ids, col_ids] = joint_configs
        self.env.simulator.dof_vel[row_ids, col_ids] = 0.0

        # Update simulator state to compute FK
        self.env.simulator.set_dof_state_tensor_robots(env_ids, self.env.simulator.dof_state)
        if hasattr(self.env.simulator, 'write_state_updates'):
            self.env.simulator.write_state_updates()
        self.env.simulator.refresh_sim_tensors()

        # Get end-effector poses
        ee_pos = self.env.simulator._rigid_body_pos[env_ids][:, self.ee_body_indices, :].clone()
        ee_quat = self.env.simulator._rigid_body_rot[env_ids][:, self.ee_body_indices, :].clone()

        invalid_mask = ~(torch.isfinite(ee_pos).all(dim=(1, 2)) & torch.isfinite(ee_quat).all(dim=(1, 2)))
        if invalid_mask.any():
            fallback_pos = torch.zeros((len(env_ids), 2, 3), device=self.device)
            fallback_pos[:, 0, 0] = 0.24  # Left arm: +X (left side)
            fallback_pos[:, 0, 1] = 0.23  # Left arm: +Y (left side)
            fallback_pos[:, 1, 0] = 0.24  # Right arm: +X (right side)
            fallback_pos[:, 1, 1] = -0.23  # Right arm: -Y (right side)
            fallback_pos[:, :, 2] = 0.0  # Both arms: Z height
            fallback_quat = torch.zeros((len(env_ids), 2, 4), device=self.device)
            fallback_quat[:, :, 3] = 1.0
            ee_pos[invalid_mask] = fallback_pos[invalid_mask]
            ee_quat[invalid_mask] = fallback_quat[invalid_mask]

        # Restore original joint positions
        self.env.simulator.dof_pos[env_ids] = original_dof_pos
        self.env.simulator.dof_vel[env_ids] = original_dof_vel
        self.env.simulator.set_dof_state_tensor_robots(env_ids, self.env.simulator.dof_state)
        if hasattr(self.env.simulator, 'write_state_updates'):
            self.env.simulator.write_state_updates()
        self.env.simulator.refresh_sim_tensors()

        return ee_pos, ee_quat

    def _sample_and_build_trajectory(self, env_ids: torch.Tensor) -> None:
        """Pre-sample waypoints and build the full reference trajectory.

        For smooth mode, waypoints are sampled in joint-space and FK'd to
        get EE poses. Consecutive waypoints are rejection-sampled to keep
        EE-space distances within thresholds (0.4m position, 45° rotation).
        If max attempts are exhausted, the sample is accepted anyway.
        """
        waypoints = []
        prev_joints = None
        prev_ee_pos = None  # [K, 2, 3]
        prev_ee_rotmat = None  # [K, 2, 3, 3]

        # Thresholds for smooth mode EE-space rejection
        max_ee_pos_dist = 0.4  # meters
        max_ee_rot_dist = math.radians(45)  # radians

        # Sample num_waypoints + 1 configs so that num_waypoints intervals
        # each produce a full transition (fence-post: W+1 posts for W segments).
        num_to_sample = (
            self.num_waypoints + 1
            if self.trajectory_mode == "smooth"
            else self.num_waypoints
        )
        rejection_stats = {"total": 0, "rejected": 0}

        for _ in range(num_to_sample):
            joints = self._sample_feasible_joints_with_rejection(env_ids)

            # Non-smooth mode: reject in joint-space (existing logic)
            if self.trajectory_mode == "non_smooth" and prev_joints is not None:
                for _attempt in range(self.max_rejection_attempts):
                    max_delta = (
                        (joints - prev_joints).abs().max(dim=-1).values
                    )  # [K]
                    bad_mask = max_delta > self.max_joint_step
                    if not bad_mask.any():
                        break
                    resampled = (
                        self._sample_feasible_joints_with_rejection(
                            env_ids[bad_mask]
                        )
                    )
                    joints[bad_mask] = resampled

            # Smooth mode: reject in EE-space
            if self.trajectory_mode == "smooth" and prev_ee_pos is not None:
                for _attempt in range(self.max_rejection_attempts):
                    # FK current candidate
                    ee_pos_w, ee_quat_w = self._compute_fk(env_ids, joints)
                    ee_pos_ref, ee_rot6d_ref = self._to_reference_frame(
                        env_ids, ee_pos_w, ee_quat_w
                    )
                    ee_rotmat = rot6d_to_rotmat(
                        ee_rot6d_ref.reshape(-1, 6)
                    ).reshape(-1, 2, 3, 3)

                    # Position distance: per-arm, take max across arms
                    pos_dist = torch.norm(
                        ee_pos_ref - prev_ee_pos, dim=-1
                    )  # [K, 2]
                    pos_bad = pos_dist.max(dim=-1).values > max_ee_pos_dist

                    # Rotation distance: geodesic angle per arm
                    # R_rel = prev^T @ curr, angle = arccos((tr-1)/2)
                    R_rel = (
                        prev_ee_rotmat.transpose(-2, -1) @ ee_rotmat
                    )  # [K, 2, 3, 3]
                    trace = (
                        R_rel[..., 0, 0]
                        + R_rel[..., 1, 1]
                        + R_rel[..., 2, 2]
                    )  # [K, 2]
                    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
                    rot_angle = torch.acos(cos_angle)  # [K, 2]
                    rot_bad = rot_angle.max(dim=-1).values > max_ee_rot_dist

                    bad_mask = pos_bad | rot_bad
                    rejection_stats["total"] += len(env_ids)
                    rejection_stats["rejected"] += bad_mask.sum().item()

                    if not bad_mask.any():
                        break
                    # Resample only the bad envs
                    resampled = (
                        self._sample_feasible_joints_with_rejection(
                            env_ids[bad_mask]
                        )
                    )
                    joints[bad_mask] = resampled
                else:
                    # Max attempts exhausted — accept anyway, recompute FK
                    ee_pos_w, ee_quat_w = self._compute_fk(env_ids, joints)
                    ee_pos_ref, ee_rot6d_ref = self._to_reference_frame(
                        env_ids, ee_pos_w, ee_quat_w
                    )
                    ee_rotmat = rot6d_to_rotmat(
                        ee_rot6d_ref.reshape(-1, 6)
                    ).reshape(-1, 2, 3, 3)

            # For smooth mode, compute FK for the first waypoint or update
            if self.trajectory_mode == "smooth":
                if prev_ee_pos is None:
                    # First waypoint: just compute FK
                    ee_pos_w, ee_quat_w = self._compute_fk(env_ids, joints)
                    ee_pos_ref, ee_rot6d_ref = self._to_reference_frame(
                        env_ids, ee_pos_w, ee_quat_w
                    )
                    ee_rotmat = rot6d_to_rotmat(
                        ee_rot6d_ref.reshape(-1, 6)
                    ).reshape(-1, 2, 3, 3)
                prev_ee_pos = ee_pos_ref
                prev_ee_rotmat = ee_rotmat

            waypoints.append(joints)
            prev_joints = joints

        waypoints = torch.stack(waypoints, dim=0)  # [W, K, A]

        if self.trajectory_mode == "smooth" and rejection_stats["total"] > 0:
            rate = (
                rejection_stats["rejected"] / rejection_stats["total"] * 100
            )
            print(
                f"  [EE rejection] {rejection_stats['rejected']}/"
                f"{rejection_stats['total']} samples rejected ({rate:.1f}%)"
            )

        if self.trajectory_mode == "non_smooth":
            self._build_nonsmooth_trajectory(env_ids, waypoints)
        elif self.trajectory_mode == "smooth":
            self._build_smooth_trajectory(env_ids, waypoints)

    def _build_nonsmooth_trajectory(self, env_ids: torch.Tensor, waypoints: torch.Tensor) -> None:
        """Build piecewise-constant trajectory: each waypoint held for sampling_interval steps."""
        for w in range(self.num_waypoints):
            ee_pos_world, ee_quat_world = self._compute_fk(env_ids, waypoints[w])
            ee_pos_ref, ee_rot6d_ref = self._to_reference_frame(env_ids, ee_pos_world, ee_quat_world)

            start = w * self.sampling_interval
            end = min((w + 1) * self.sampling_interval, self.total_steps)
            segment_len = end - start

            self.trajectory_pos[env_ids, start:end] = ee_pos_ref.unsqueeze(1).expand(-1, segment_len, -1, -1)
            self.trajectory_rot6d[env_ids, start:end] = ee_rot6d_ref.unsqueeze(1).expand(-1, segment_len, -1, -1)

        # Hold last waypoint for any remaining steps
        last_filled = self.num_waypoints * self.sampling_interval
        if last_filled < self.total_steps:
            self.trajectory_pos[env_ids, last_filled:] = self.trajectory_pos[env_ids, last_filled - 1:last_filled]
            self.trajectory_rot6d[env_ids, last_filled:] = self.trajectory_rot6d[env_ids, last_filled - 1:last_filled]

    def _interpolate_joints(self, waypoints: torch.Tensor, wp_times: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        """Linear interpolation in joint space, each joint dimension independently.

        Args:
            waypoints: [W, K, A] joint configs at waypoint times
            wp_times: [W] time indices for each waypoint
            frame_times: [T] time indices for output frames

        Returns:
            [K, T, A] interpolated joint configs
        """
        W, K, A = waypoints.shape

        clamped = frame_times.clamp(wp_times[0], wp_times[-1])
        idx = torch.searchsorted(wp_times, clamped).clamp(1, W - 1)
        t0 = wp_times[idx - 1]
        t1 = wp_times[idx]
        alpha = ((clamped - t0) / (t1 - t0)).clamp(0, 1)

        w0 = waypoints[idx - 1]  # [T, K, A]
        w1 = waypoints[idx]      # [T, K, A]
        alpha_expanded = alpha[:, None, None]  # [T, 1, 1]
        result = (1 - alpha_expanded) * w0 + alpha_expanded * w1  # [T, K, A]

        return result.permute(1, 0, 2)  # [K, T, A]

    def _build_smooth_trajectory(self, env_ids: torch.Tensor, waypoints: torch.Tensor) -> None:
        """Build smooth trajectory via task-space SE(3) interpolation.

        Instead of interpolating in joint-space and running FK per frame,
        we FK only at waypoints and interpolate EE poses in SE(3):
        - Position: linear or cosine (randomly chosen per segment)
        - Rotation: geodesic via Lie algebra (log-map, scale, exp-map)
        """
        W, K, _A = waypoints.shape

        # FK all waypoints to get EE poses [W, K, 2, 3] and [W, K, 2, 6]
        wp_pos_list = []
        wp_rot6d_list = []
        for w in range(W):
            ee_pos_w, ee_quat_w = self._compute_fk(env_ids, waypoints[w])
            ee_pos_ref, ee_rot6d_ref = self._to_reference_frame(
                env_ids, ee_pos_w, ee_quat_w
            )
            wp_pos_list.append(ee_pos_ref)      # [K, 2, 3]
            wp_rot6d_list.append(ee_rot6d_ref)  # [K, 2, 6]

        wp_pos = torch.stack(wp_pos_list, dim=0)      # [W, K, 2, 3]
        wp_rot6d = torch.stack(wp_rot6d_list, dim=0)  # [W, K, 2, 6]
        # Convert rot6d to rotmat for interpolation
        wp_rotmat = rot6d_to_rotmat(
            wp_rot6d.reshape(-1, 6)
        ).reshape(W, K, 2, 3, 3)

        # Interpolate each segment in task-space
        import random as _random

        num_segments = W - 1
        # Equal-duration segments based on sampling_interval
        seg_len = self.sampling_interval

        for seg in range(num_segments):
            start = seg * seg_len
            end = min((seg + 1) * seg_len, self.total_steps)
            frames = end - start
            if frames <= 0:
                continue

            # Randomly choose cosine or linear for this segment
            use_cosine = _random.random() < 0.5

            alpha_linear = torch.linspace(
                0.0, 1.0, frames, device=self.device
            )
            if use_cosine:
                alpha = 0.5 * (1.0 - torch.cos(alpha_linear * math.pi))
            else:
                alpha = alpha_linear

            # Position interpolation: (1-a)*p0 + a*p1
            p0 = wp_pos[seg]    # [K, 2, 3]
            p1 = wp_pos[seg + 1]
            alpha_pos = alpha[:, None, None, None]  # [T_seg, 1, 1, 1]
            interp_pos = (
                (1.0 - alpha_pos) * p0.unsqueeze(0)
                + alpha_pos * p1.unsqueeze(0)
            )  # [T_seg, K, 2, 3]

            # Rotation interpolation via Lie algebra
            R0 = wp_rotmat[seg]      # [K, 2, 3, 3]
            R1 = wp_rotmat[seg + 1]  # [K, 2, 3, 3]
            # Relative rotation: R0^T @ R1
            R_rel = R0.transpose(-2, -1) @ R1  # [K, 2, 3, 3]
            # Log map: axis-angle of R_rel
            omega = self._rotmat_log(R_rel)  # [K, 2, 3]

            alpha_rot = alpha[:, None, None, None]  # [T_seg, 1, 1, 1]
            # Scaled axis-angle per frame
            omega_scaled = alpha_rot * omega.unsqueeze(0)  # [T_seg, K, 2, 3]
            # Exp map and compose: R0 @ exp(alpha * log(R0^T R1))
            R_delta = self._rotmat_exp(omega_scaled)  # [T_seg, K, 2, 3, 3]
            R_interp = R0.unsqueeze(0) @ R_delta  # [T_seg, K, 2, 3, 3]

            # Convert rotmat back to rot6d
            interp_rot6d = R_interp[..., :, :2].transpose(
                -2, -1
            ).reshape(frames, K, 2, 6)

            # Write to trajectory buffers
            self.trajectory_pos[env_ids, start:end] = interp_pos.permute(
                1, 0, 2, 3
            )  # [K, T_seg, 2, 3]
            self.trajectory_rot6d[env_ids, start:end] = interp_rot6d.permute(
                1, 0, 2, 3
            )  # [K, T_seg, 2, 6]

        # Hold last waypoint for any remaining steps
        last_filled = num_segments * seg_len
        if last_filled < self.total_steps:
            self.trajectory_pos[env_ids, last_filled:] = (
                self.trajectory_pos[env_ids, last_filled - 1:last_filled]
            )
            self.trajectory_rot6d[env_ids, last_filled:] = (
                self.trajectory_rot6d[env_ids, last_filled - 1:last_filled]
            )

    def _build_height_trajectory(self, env_ids: torch.Tensor) -> None:
        """Sample height waypoints and build per-episode height trajectory.

        For each environment:
        - 70% probability: smooth (linear interpolation between waypoints)
        - 30% probability: step (piecewise-constant, held until next waypoint)

        First waypoint is biased toward nominal height (±0.03m).
        Consecutive waypoints are clamped to max_step difference.
        """
        K = env_ids.shape[0]
        W = self.num_waypoints

        if W == 0:
            # No waypoints possible — fill with nominal
            self.height_trajectory[env_ids] = self.height_nominal
            return

        # Determine per-env mode: smooth vs step
        smooth_mask = torch.rand(K, device=self.device) < self.height_smooth_prob  # [K]

        # Sample waypoints uniformly in [height_range_min, height_range_max]
        waypoints = (
            torch.rand(K, W, device=self.device)
            * (self.height_range_max - self.height_range_min)
            + self.height_range_min
        )  # [K, W]

        # Bias first waypoint toward nominal (±0.03m)
        first_wp_min = max(self.height_nominal - 0.03, self.height_range_min)
        first_wp_max = min(self.height_nominal + 0.03, self.height_range_max)
        waypoints[:, 0] = (
            torch.rand(K, device=self.device) * (first_wp_max - first_wp_min)
            + first_wp_min
        )

        # Clamp consecutive waypoint differences to height_max_step
        for i in range(1, W):
            diff = waypoints[:, i] - waypoints[:, i - 1]
            clamped_diff = diff.clamp(-self.height_max_step, self.height_max_step)
            waypoints[:, i] = waypoints[:, i - 1] + clamped_diff
            waypoints[:, i] = waypoints[:, i].clamp(self.height_range_min, self.height_range_max)

        # Build trajectory per env
        seg_len = self.sampling_interval

        # --- Smooth mode: linear interpolation between waypoints ---
        # --- Step mode: hold each waypoint constant until next ---
        trajectory = torch.full(
            (K, self.total_steps), self.height_nominal, dtype=torch.float32, device=self.device
        )

        for seg in range(W - 1):
            start = seg * seg_len
            end = min((seg + 1) * seg_len, self.total_steps)
            frames = end - start
            if frames <= 0:
                continue

            h0 = waypoints[:, seg]        # [K]
            h1 = waypoints[:, seg + 1]    # [K]

            # Smooth mode: linear interpolation
            alpha = torch.linspace(0.0, 1.0, frames, device=self.device)  # [frames]
            smooth_values = h0.unsqueeze(1) * (1.0 - alpha.unsqueeze(0)) + h1.unsqueeze(1) * alpha.unsqueeze(0)  # [K, frames]

            # Step mode: hold h0 constant
            step_values = h0.unsqueeze(1).expand(-1, frames)  # [K, frames]

            # Select based on per-env mode
            mask = smooth_mask.unsqueeze(1).expand(-1, frames)  # [K, frames]
            trajectory[:, start:end] = torch.where(mask, smooth_values, step_values)

        # Handle remaining steps after last segment (hold last waypoint)
        last_filled = (W - 1) * seg_len
        if last_filled < self.total_steps:
            # For smooth mode, the last waypoint is reached at the end of the last segment
            # For step mode, hold the last waypoint
            trajectory[:, last_filled:] = waypoints[:, -1].unsqueeze(1).expand(
                -1, self.total_steps - last_filled
            )

        # Clamp all values to configured range
        trajectory = trajectory.clamp(self.height_range_min, self.height_range_max)

        # Write to buffer
        self.height_trajectory[env_ids] = trajectory

    @staticmethod
    def _rotmat_log(R: torch.Tensor) -> torch.Tensor:
        """Logarithmic map SO(3) -> so(3). Returns axis-angle vector.

        Args:
            R: [..., 3, 3] rotation matrices.

        Returns:
            [..., 3] axis-angle vectors.
        """
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)  # [...]

        # Skew-symmetric part: (R - R^T) / (2 sin(theta))
        skew = R - R.transpose(-2, -1)  # [..., 3, 3]
        vec = torch.stack(
            [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1
        )

        # Handle small angles (theta ~ 0): scale ~ 0.5
        small = theta.abs() < 1e-6
        sin_theta = torch.sin(theta).clamp(min=1e-8)
        scale = theta / (2.0 * sin_theta)
        scale = torch.where(small, torch.ones_like(scale) * 0.5, scale)

        return vec * scale.unsqueeze(-1)

    @staticmethod
    def _rotmat_exp(omega: torch.Tensor) -> torch.Tensor:
        """Exponential map so(3) -> SO(3) via Rodrigues' formula.

        Args:
            omega: [..., 3] axis-angle vectors.

        Returns:
            R: [..., 3, 3] rotation matrices.
        """
        theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-8)
        axis = omega / theta

        # Skew-symmetric matrix from axis
        zero = torch.zeros_like(theta[..., 0])
        ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
        K = torch.stack([
            torch.stack([zero, -az, ay], dim=-1),
            torch.stack([az, zero, -ax], dim=-1),
            torch.stack([-ay, ax, zero], dim=-1),
        ], dim=-2)  # [..., 3, 3]

        sin_t = torch.sin(theta).unsqueeze(-1)   # [..., 1, 1]
        cos_t = torch.cos(theta).unsqueeze(-1)   # [..., 1, 1]

        I = torch.eye(3, device=omega.device, dtype=omega.dtype)
        # Broadcast I to match K shape
        I = I.expand_as(K)
        R = I + sin_t * K + (1.0 - cos_t) * (K @ K)

        # Handle near-zero rotations
        small = (theta.squeeze(-1) < 1e-6)
        if small.any():
            R[small] = I[small] + K[small]

        return R

    def _validate_trajectory_dataset(self, dataset: dict) -> None:
        """Validate a loaded trajectory dataset dictionary.

        Checks required keys, tensor shapes, trajectory count, and metadata
        compatibility with the current environment configuration.

        Args:
            dataset: Dictionary loaded from a .pt trajectory dataset file.

        Raises:
            ValueError: If the dataset is invalid or incompatible.
        """
        # Check required keys
        required_keys = {"trajectory_pos", "trajectory_rot6d", "metadata"}
        missing_keys = required_keys - set(dataset.keys())
        if missing_keys:
            raise ValueError(f"Trajectory dataset missing required keys: {sorted(missing_keys)}")

        pos = dataset["trajectory_pos"]
        rot6d = dataset["trajectory_rot6d"]
        metadata = dataset["metadata"]

        # Validate trajectory_pos shape: [N, T, 2, 3]
        if pos.ndim != 4 or pos.shape[2] != 2 or pos.shape[3] != 3:
            raise ValueError(
                f"Expected trajectory_pos shape [N, T, 2, 3], got {list(pos.shape)}"
            )

        # Validate trajectory_rot6d shape: [N, T, 2, 6]
        if rot6d.ndim != 4 or rot6d.shape[2] != 2 or rot6d.shape[3] != 6:
            raise ValueError(
                f"Expected trajectory_rot6d shape [N, T, 2, 6], got {list(rot6d.shape)}"
            )

        # Validate N and T consistency between tensors
        if pos.shape[0] != rot6d.shape[0] or pos.shape[1] != rot6d.shape[1]:
            raise ValueError(
                f"Inconsistent shapes: trajectory_pos {list(pos.shape)} vs "
                f"trajectory_rot6d {list(rot6d.shape)} — N and T must match"
            )

        # Validate at least 1 trajectory
        if pos.shape[0] < 1:
            raise ValueError("Trajectory dataset must contain at least 1 trajectory")

        # Validate total_steps matches environment
        dataset_total_steps = metadata.get("total_steps")
        if dataset_total_steps != self.total_steps:
            raise ValueError(
                f"Dataset total_steps={dataset_total_steps} does not match "
                f"env max_episode_length={self.total_steps}"
            )

        # Validate reference_frame matches configuration.
        # Heading and ground_heading are compatible with base-frame trajectories
        # since they only differ by roll/pitch/height rejection (negligible for
        # standing/walking humanoid).
        dataset_ref_frame = metadata.get("reference_frame")
        _COMPATIBLE_FRAMES = {"base", "heading", "ground_heading"}
        if dataset_ref_frame == self.reference_frame:
            pass  # exact match
        elif dataset_ref_frame in _COMPATIBLE_FRAMES and self.reference_frame in _COMPATIBLE_FRAMES:
            pass  # compatible frames
        else:
            raise ValueError(
                f"Dataset reference_frame='{dataset_ref_frame}' does not match "
                f"configured '{self.reference_frame}'"
            )

    def _load_trajectory_dataset(self, path: str) -> dict:
        """Load and validate a precomputed trajectory dataset from a .pt file.

        Args:
            path: File path to the .pt trajectory dataset.

        Returns:
            The loaded dataset dictionary.

        Raises:
            FileNotFoundError: If path is None/empty or the file does not exist.
        """
        if not path:
            raise FileNotFoundError(
                "trajectory_file must be specified when trajectory_mode is 'precomputed'"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Trajectory dataset file not found: {path}")

        dataset = torch.load(path, map_location="cpu", weights_only=True)
        self._validate_trajectory_dataset(dataset)

        self._cpu_trajectory_pos = dataset["trajectory_pos"]
        self._cpu_trajectory_rot6d = dataset["trajectory_rot6d"]
        self._num_dataset_trajectories = dataset["trajectory_pos"].shape[0]

        return dataset




    def step(self) -> None:
        """Update metrics with frame-aligned errors."""
        if self.trajectory_mode is not None:
            # Advance trajectory index (clamped to last frame)
            self.trajectory_index = (
                self.trajectory_index + 1
            ).clamp(max=self.total_steps - 1)
            # Update targets from trajectory
            env_ids_all = torch.arange(
                self.num_envs, device=self.device
            )
            self.target_ee_pos = self.trajectory_pos[
                env_ids_all, self.trajectory_index
            ]
            self.target_ee_rot6d = self.trajectory_rot6d[
                env_ids_all, self.trajectory_index
            ]
            # Update height target from trajectory
            if hasattr(self, 'height_trajectory'):
                self.target_base_height = self.height_trajectory[
                    env_ids_all, self.trajectory_index
                ]

            # For ground_heading with base-frame trajectories: reproject EE
            # z-targets using the commanded height. Base-frame z is relative to
            # pelvis; ground_heading z is relative to ground.
            if self.reference_frame == "ground_heading" and hasattr(self, 'target_base_height'):
                self.target_ee_pos[:, :, 2] += self.target_base_height.unsqueeze(-1)

        self._update_posture()

        self._update_tracking_metrics()

    def _update_tracking_metrics(self) -> None:
        """Compute frame-aligned EE pos/rot (and height) tracking metrics.

        Shared by ArmTrackingCommand and WorldFrameArmTrackingCommand so both
        compute identical metrics without duplicated code.
        """
        # Get current end-effector poses in world frame
        current_ee_pos_world = self.env.simulator._rigid_body_pos[:, self.ee_body_indices, :]
        current_ee_quat_world = self.env.simulator._rigid_body_rot[:, self.ee_body_indices, :]
        
        # Transform to reference frame to match targets
        env_ids = torch.arange(self.num_envs, device=self.device)
        ref_pos, ref_quat = self._get_reference_pose(env_ids)
        current_ee_pos = torch.zeros_like(current_ee_pos_world)
        current_ee_quat = torch.zeros_like(current_ee_quat_world)
        for i in range(2):
            current_ee_pos[:, i], current_ee_quat[:, i] = world_to_reference_frame(
                current_ee_pos_world[:, i], current_ee_quat_world[:, i], ref_pos, ref_quat
            )
        
        # Position error: [num_envs, 2] for left/right, plus aggregate.
        ee_pos_error_per_arm = torch.norm(self.target_ee_pos - current_ee_pos, dim=-1)
        self.metrics["ee_pos_error_left"] = ee_pos_error_per_arm[:, 0].mean()
        self.metrics["ee_pos_error_right"] = ee_pos_error_per_arm[:, 1].mean()
        self.metrics["ee_pos_error"] = ee_pos_error_per_arm.mean()
        
        # Orientation error (Frobenius norm: tr(I - R1^T R2))
        current_ee_rotmat = quat_to_rotmat(current_ee_quat.reshape(-1, 4)).reshape(
            current_ee_quat.shape[0], 2, 3, 3
        )
        target_ee_rotmat = rot6d_to_rotmat(self.target_ee_rot6d.reshape(-1, 6)).reshape(
            self.target_ee_rot6d.shape[0], 2, 3, 3
        )
        
        R1T_R2 = torch.matmul(current_ee_rotmat.transpose(-2, -1), target_ee_rotmat)
        trace_R1T_R2 = R1T_R2.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        ee_rot_error_per_arm = 3.0 - trace_R1T_R2
        self.metrics["ee_rot_error_left"] = ee_rot_error_per_arm[:, 0].mean()
        self.metrics["ee_rot_error_right"] = ee_rot_error_per_arm[:, 1].mean()
        self.metrics["ee_rot_error"] = ee_rot_error_per_arm.mean()
        
        # # Joint tracking error (for reference)
        # current_arm_pos = self.env.simulator.dof_pos[:, self.env.arm_dof_indices]
        # joint_error = torch.norm(self.target_arm_pos - current_arm_pos, dim=-1)
        # self.metrics["arm_tracking_error"] = joint_error.mean()

        # Height tracking metrics
        if hasattr(self, 'height_trajectory') and self.trajectory_mode is not None:
            current_height = self.env.simulator.robot_root_states[:, 2]
            self.metrics["base_height_error"] = torch.abs(
                current_height - self.target_base_height
            ).mean()
            self.metrics["target_base_height_mean"] = self.target_base_height.mean()

    @property
    def command(self) -> torch.Tensor:
        """Return SE(3) command for observations.
        
        Returns:
            Tensor of shape [num_envs, 18]: [left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6)]
        """
        cmd = torch.cat([
            self.target_ee_pos[:, 0],      # Left position (3)
            self.target_ee_rot6d[:, 0],    # Left rotation (6)
            self.target_ee_pos[:, 1],      # Right position (3)
            self.target_ee_rot6d[:, 1],    # Right rotation (6)
        ], dim=-1)
        
        # Safety check: replace NaN/Inf with zeros
        if torch.isnan(cmd).any() or torch.isinf(cmd).any():
            cmd = torch.nan_to_num(cmd, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return cmd
