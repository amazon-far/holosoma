"""
Evaluation script for collecting trajectory metrics:
1. Energy (per joint and average)
2. Foot slippage (left, right, average)
3. Foot contact force (left, right, average, std)
4. Action rate (per joint and average)
5. Action smoothness / jerk (per joint and average)
6. Joint acceleration (per joint and average)
7. Base angular velocity tracking error
8. Reference energy (PD approximation, per joint and average)
9. Motion tracking - GLOBAL frame (world coordinates, per body - min, max, average)
10. Motion tracking - LOCAL frame (robot anchor relative, per body - min, max, average)

Usage: 
    python src/holosoma/holosoma/eval_metric.py \
        --checkpoint=/path/to/model.pt \
        --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
        --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import tyro
from loguru import logger
import torch
import numpy as np

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


class MetricsCollector:
    """Collects and stores trajectory metrics during evaluation."""
    
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.num_dofs = env.num_dofs
        self.dof_names = env.dof_names
        
        # Get feet indices from body names
        if hasattr(env, 'feet_indices'):
            self.feet_indices = env.feet_indices
        else:
            # Find feet indices by body names
            body_names = env.simulator._body_list if hasattr(env.simulator, '_body_list') else []
            left_foot_idx = None
            right_foot_idx = None
            
            for idx, name in enumerate(body_names):
                if 'left_ankle_roll_link' in name:
                    left_foot_idx = idx
                elif 'right_ankle_roll_link' in name:
                    right_foot_idx = idx
            
            if left_foot_idx is not None and right_foot_idx is not None:
                self.feet_indices = [left_foot_idx, right_foot_idx]
                logger.info(f"Found feet indices: left={left_foot_idx}, right={right_foot_idx}")
            else:
                logger.warning("Could not find feet indices, using default [0, 1]")
                self.feet_indices = [0, 1]
        
        # Get body names for motion tracking (all bodies excluding contact points)
        if hasattr(env.simulator, '_body_list'):
            body_list = env.simulator._body_list
            exclude_names = ['left_foot_contact_point', 'right_foot_contact_point']
            self.body_indices_for_tracking = [
                i for i, name in enumerate(body_list) 
                if name not in exclude_names
            ]
            self.body_names = [body_list[i] for i in self.body_indices_for_tracking]
            logger.info(f"Tracking {len(self.body_names)}/{len(body_list)} bodies for motion tracking (excluding contact points)")
        else:
            self.body_names = []
            self.body_indices_for_tracking = []
        
        # Initialize storage for metrics per step
        self.reset_buffers()
        
    def reset_buffers(self):
        """Reset all metric buffers."""
        # Energy: |torque * joint_vel| per joint per step
        self.energy_per_step = []  # List of [num_envs, num_dofs] tensors
        
        # Foot slippage: xy velocity when in contact
        self.foot_slippage_left_per_step = []  # List of [num_envs] tensors
        self.foot_slippage_right_per_step = []  # List of [num_envs] tensors
        
        # Foot contact force: force magnitude for each foot
        self.foot_contact_force_left_per_step = []  # List of [num_envs] tensors
        self.foot_contact_force_right_per_step = []  # List of [num_envs] tensors
        
        # Action rate: |action_t - action_t-1| per joint per step
        self.action_rate_per_step = []  # List of [num_envs, num_dofs] tensors
        self.prev_actions = None  # Track previous actions manually
        self.prev_prev_actions = None  # Track t-2 actions for smoothness
        
        # Action smoothness (jerk): (action_t - 2*action_t-1 + action_t-2)^2
        self.action_smoothness_per_step = []  # List of [num_envs, num_dofs] tensors
        
        # Joint acceleration: |dof_vel_t - dof_vel_t-1| / dt
        self.joint_acc_per_step = []  # List of [num_envs, num_dofs] tensors
        self.prev_dof_vel = None
        
        # Base angular velocity tracking error
        self.base_ang_vel_error_per_step = []  # List of [num_envs] tensors
        
        # Reference energy (PD approximation)
        self.ref_energy_per_step = []  # List of [num_envs, num_dofs] tensors
        
        # Motion tracking (global frame): distance in world coordinates
        self.motion_tracking_global_per_step = []  # List of [num_envs, num_bodies, 3] tensors
        
        # Motion tracking (local frame): distance in robot anchor-relative coordinates
        self.motion_tracking_local_per_step = []  # List of [num_envs, num_bodies, 3] tensors
        
        # Track step count
        self.step_count = 0
        
    def collect_step_metrics(self, skip_padding=False, padding_frames=100):
        """Collect metrics for the current step.
        
        Args:
            skip_padding: If True, skip collecting metrics during padding frames
            padding_frames: Number of padding frames to skip at the beginning
        """
        env = self.env
        
        # Skip padding frames if requested
        if skip_padding and hasattr(env, 'command_manager'):
            motion_cmd = env.command_manager.get_state("motion_command")
            if motion_cmd is not None and hasattr(motion_cmd, 'time_steps'):
                current_timestep = motion_cmd.time_steps[0].item()
                if current_timestep < padding_frames:
                    # Skip this step - don't collect metrics during padding
                    return
        
        # 1. Energy: |torque * joint_vel|
        # Following the reference: torch.norm(torch.abs(torque * joint_vel), dim=-1)
        # Here we store per-joint to get detailed breakdown
        # Get torques from action manager
        action_term_names = list(env.action_manager._term_instances.keys())
        if len(action_term_names) > 0:
            action_term = env.action_manager._term_instances[action_term_names[0]]
            if hasattr(action_term, 'torques'):
                torques = action_term.torques
            else:
                torques = torch.zeros(env.num_envs, env.num_dofs, device=env.device)
        else:
            torques = torch.zeros(env.num_envs, env.num_dofs, device=env.device)
        
        energy = torch.abs(torques * env.simulator.dof_vel)  # [num_envs, num_dofs]
        self.energy_per_step.append(energy.clone().cpu())
        
        # 2. Foot slippage: velocity when in contact with ground
        # is_contact = contact_force > 1.0, foot_planar_velocity = norm(velocity[:, :2])
        is_contact = torch.norm(env.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0  # [num_envs, 2]
        foot_vel = env.simulator._rigid_body_vel[:, self.feet_indices, :2]  # [num_envs, 2, 2] (2 feet, xy vel)
        foot_planar_velocity = torch.linalg.norm(foot_vel, dim=-1)  # [num_envs, 2]
        
        # Left foot (index 0), Right foot (index 1)
        slippage = is_contact * foot_planar_velocity  # [num_envs, 2]
        self.foot_slippage_left_per_step.append(slippage[:, 0].clone().cpu())
        self.foot_slippage_right_per_step.append(slippage[:, 1].clone().cpu())
        
        # 3. Foot contact force: magnitude of contact force
        foot_contact_forces = torch.norm(env.simulator.contact_forces[:, self.feet_indices, :], dim=-1)  # [num_envs, 2]
        self.foot_contact_force_left_per_step.append(foot_contact_forces[:, 0].clone().cpu())
        self.foot_contact_force_right_per_step.append(foot_contact_forces[:, 1].clone().cpu())
        
        # 4. Action rate: |action_t - action_t-1|
        # Get current actions from action manager
        current_actions = env.action_manager._action
        
        if self.prev_actions is not None:
            action_rate = torch.abs(current_actions - self.prev_actions)  # [num_envs, num_dofs]
            self.action_rate_per_step.append(action_rate.clone().cpu())
        else:
            # First step: no previous action, so action_rate is 0
            action_rate = torch.zeros_like(current_actions)
            self.action_rate_per_step.append(action_rate.clone().cpu())
        
        # 5. Action smoothness (jerk): (action_t - 2*action_t-1 + action_t-2)^2
        if self.prev_actions is not None and self.prev_prev_actions is not None:
            action_acc = current_actions - 2 * self.prev_actions + self.prev_prev_actions  # [num_envs, num_dofs]
            action_smoothness = action_acc ** 2  # [num_envs, num_dofs]
            self.action_smoothness_per_step.append(action_smoothness.clone().cpu())
        else:
            # First two steps: no smoothness calculation
            action_smoothness = torch.zeros_like(current_actions)
            self.action_smoothness_per_step.append(action_smoothness.clone().cpu())
        
        # 6. Joint acceleration: |dof_vel_t - dof_vel_t-1| / dt
        if self.prev_dof_vel is not None:
            joint_acc = torch.abs(env.simulator.dof_vel - self.prev_dof_vel) / env.dt  # [num_envs, num_dofs]
            self.joint_acc_per_step.append(joint_acc.clone().cpu())
        else:
            joint_acc = torch.zeros_like(env.simulator.dof_vel)
            self.joint_acc_per_step.append(joint_acc.clone().cpu())
        self.prev_dof_vel = env.simulator.dof_vel.clone()
        
        # 7. Base angular velocity tracking error & Reference energy
        if hasattr(env, 'command_manager'):
            motion_cmd = env.command_manager.get_state("motion_command")
            if motion_cmd is not None and hasattr(motion_cmd, 'body_ang_vel_w'):
                # Get reference base angular velocity (pelvis = body index 0)
                ref_base_ang_vel = motion_cmd.body_ang_vel_w[:, 0, :]  # [num_envs, 3]
                # Get actual base angular velocity
                actual_base_ang_vel = env.simulator.robot_root_states[:, 10:13]  # [num_envs, 3]
                base_ang_vel_error = torch.norm(ref_base_ang_vel - actual_base_ang_vel, dim=-1)  # [num_envs]
                self.base_ang_vel_error_per_step.append(base_ang_vel_error.clone().cpu())
                
                # 8. Reference energy (PD controller approximation)
                # Estimate torque needed to track reference: τ ≈ Kp*(q_ref - q) + Kd*(dq_ref - dq)
                ref_dof_pos = motion_cmd.joint_pos  # [num_envs, num_dofs]
                ref_dof_vel = motion_cmd.joint_vel  # [num_envs, num_dofs]
                
                # Get Kp and Kd from action term
                action_term_names = list(env.action_manager._term_instances.keys())
                if len(action_term_names) > 0:
                    action_term = env.action_manager._term_instances[action_term_names[0]]
                    if hasattr(action_term, 'p_gains') and hasattr(action_term, 'd_gains'):
                        Kp = action_term.p_gains  # [num_dofs]
                        Kd = action_term.d_gains  # [num_dofs]
                        
                        # Estimate reference tracking torque
                        pos_error = ref_dof_pos - env.simulator.dof_pos
                        vel_error = ref_dof_vel - env.simulator.dof_vel
                        ref_torque_approx = Kp * pos_error + Kd * vel_error  # [num_envs, num_dofs]
                        
                        # Reference energy: |τ_ref × dq_ref|
                        ref_energy = torch.abs(ref_torque_approx * ref_dof_vel)  # [num_envs, num_dofs]
                        self.ref_energy_per_step.append(ref_energy.clone().cpu())
        
        # Update previous actions for next step
        self.prev_prev_actions = self.prev_actions
        self.prev_actions = current_actions.clone()
        
        # 9. Motion tracking (global frame): distance in world coordinates (30 bodies)
        # 10. Motion tracking (local frame): distance in robot anchor-relative coordinates (30 bodies)
        if hasattr(env, 'command_manager') and len(self.body_indices_for_tracking) > 0:
            try:
                motion_cmd = env.command_manager.get_state("motion_command")
                if motion_cmd is not None and hasattr(motion_cmd, 'motion'):
                    # Get ALL body positions from motion and simulator (30 bodies, excluding contact points)
                    # motion.body_pos_w[time_steps] gives [num_envs, num_bodies, 3] but needs env_origins added
                    ref_body_pos_all = (
                        motion_cmd.motion.body_pos_w[motion_cmd.time_steps] 
                        + env.simulator.scene.env_origins[:, None, :]
                    )  # [num_envs, 32, 3]
                    robot_body_pos_all = env.simulator._rigid_body_pos  # [num_envs, 32, 3]
                    
                    # Filter to exclude contact points
                    ref_body_pos = ref_body_pos_all[:, self.body_indices_for_tracking, :]  # [num_envs, 30, 3]
                    robot_body_pos = robot_body_pos_all[:, self.body_indices_for_tracking, :]  # [num_envs, 30, 3]
                    
                    # Global frame: difference in world coordinates (all 30 bodies)
                    motion_tracking_global = ref_body_pos - robot_body_pos  # [num_envs, 30, 3]
                    self.motion_tracking_global_per_step.append(motion_tracking_global.clone().cpu())
                    
                    # Local frame (beyondmimic style): compute for ALL 30 bodies
                    # Same logic as wbt.py lines 491-538, but for all bodies
                    from holosoma.utils.rotations import quat_apply, quat_inverse, quat_mul, yaw_quat
                    
                    # Get reference body poses (use root at episode start, else configured ref body)
                    use_root = (env.episode_length_buf == 0).unsqueeze(1).float()
                    ref_pos_w = motion_cmd.root_pos_w * use_root + motion_cmd.ref_pos_w * (1 - use_root)
                    ref_quat_w = motion_cmd.root_quat_w * use_root + motion_cmd.ref_quat_w * (1 - use_root)
                    robot_ref_pos_w = motion_cmd.robot_root_pos_w * use_root + motion_cmd.robot_ref_pos_w * (1 - use_root)
                    robot_ref_quat_w = motion_cmd.robot_root_quat_w * use_root + motion_cmd.robot_ref_quat_w * (1 - use_root)
                    
                    # Repeat for all 30 bodies
                    num_bodies = len(self.body_indices_for_tracking)
                    ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, num_bodies, 1)
                    ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, num_bodies, 1)
                    robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1)
                    robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1)
                    
                    # Compute yaw-only rotation difference
                    delta_quat_w = yaw_quat(
                        quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
                    )
                    
                    # Compute relative body positions (beyondmimic style)
                    delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
                    delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences only
                    body_pos_relative_w_all = (
                        robot_ref_pos_w_repeat
                        + delta_pos_w_height
                        + quat_apply(delta_quat_w, ref_body_pos - ref_pos_w_repeat, w_last=True)
                    )
                    
                    # Local tracking error (all 30 bodies)
                    motion_tracking_local = body_pos_relative_w_all - robot_body_pos  # [num_envs, 30, 3]
                    self.motion_tracking_local_per_step.append(motion_tracking_local.clone().cpu())
            except Exception as e:
                logger.warning(f"Failed to compute motion tracking metrics: {e}")
        
        self.step_count += 1
        
    def compute_summary_statistics(self):
        """Compute summary statistics from collected metrics."""
        results = {}
        
        if self.step_count == 0:
            logger.warning("No steps collected!")
            return results
            
        # ===================== 1. Energy =====================
        if len(self.energy_per_step) > 0:
            # Stack: [num_steps, num_envs, num_dofs]
            energy_stack = torch.stack(self.energy_per_step, dim=0)
            
            # Mean energy per step (average over all steps and envs)
            mean_energy_per_step = energy_stack.mean(dim=0).mean(dim=0)  # [num_dofs]
            
            # Average energy (mean over all joints)
            avg_energy = mean_energy_per_step.mean()
            
            results['energy'] = {
                'per_joint': {self.dof_names[i]: mean_energy_per_step[i].item() for i in range(len(self.dof_names))},
                'average': avg_energy.item(),
            }
            
        # ===================== 2. Foot Slippage =====================
        if len(self.foot_slippage_left_per_step) > 0:
            left_slip = torch.stack(self.foot_slippage_left_per_step, dim=0)  # [num_steps, num_envs]
            right_slip = torch.stack(self.foot_slippage_right_per_step, dim=0)
            
            # Sum over trajectory
            left_slip_sum = left_slip.sum(dim=0).mean()  # mean over envs
            right_slip_sum = right_slip.sum(dim=0).mean()
            avg_slip = (left_slip_sum + right_slip_sum) / 2
            
            results['foot_slippage'] = {
                'left': left_slip_sum.item(),
                'right': right_slip_sum.item(),
                'average': avg_slip.item(),
            }
            
        # ===================== 3. Foot Contact Force =====================
        if len(self.foot_contact_force_left_per_step) > 0:
            left_force = torch.stack(self.foot_contact_force_left_per_step, dim=0)  # [num_steps, num_envs]
            right_force = torch.stack(self.foot_contact_force_right_per_step, dim=0)
            
            # Mean over trajectory (average contact force per step)
            left_force_mean = left_force.mean(dim=0).mean()  # mean over envs
            right_force_mean = right_force.mean(dim=0).mean()
            avg_force = (left_force_mean + right_force_mean) / 2
            
            # Max force during trajectory
            left_force_max = left_force.max(dim=0)[0].mean()
            right_force_max = right_force.max(dim=0)[0].mean()
            
            # Standard deviation (variability of contact force)
            left_force_std = left_force.std(dim=0).mean()  # std over time, mean over envs
            right_force_std = right_force.std(dim=0).mean()
            avg_force_std = (left_force_std + right_force_std) / 2
            
            results['foot_contact_force'] = {
                'left_mean': left_force_mean.item(),
                'right_mean': right_force_mean.item(),
                'average_mean': avg_force.item(),
                'left_max': left_force_max.item(),
                'right_max': right_force_max.item(),
                'left_std': left_force_std.item(),
                'right_std': right_force_std.item(),
                'average_std': avg_force_std.item(),
            }
            
        # ===================== 4. Action Rate =====================
        if len(self.action_rate_per_step) > 1:  # Need at least 2 steps
            action_rate_stack = torch.stack(self.action_rate_per_step, dim=0)  # [num_steps, num_envs, num_dofs]
            
            # Skip first step (action_rate is 0 for first step) and compute mean
            per_joint_action_rate = action_rate_stack[1:].mean(dim=0).mean(dim=0)  # [num_dofs]
            avg_action_rate = per_joint_action_rate.mean()
            
            results['action_rate'] = {
                'per_joint': {self.dof_names[i]: per_joint_action_rate[i].item() for i in range(len(self.dof_names))},
                'average': avg_action_rate.item(),
            }
            
        # ===================== 5. Action Smoothness (Jerk) =====================
        if len(self.action_smoothness_per_step) > 2:  # Need at least 3 steps
            smoothness_stack = torch.stack(self.action_smoothness_per_step, dim=0)  # [num_steps, num_envs, num_dofs]
            
            # Skip first two steps and compute mean
            per_joint_smoothness = smoothness_stack[2:].mean(dim=0).mean(dim=0)  # [num_dofs]
            mean_smoothness = per_joint_smoothness.mean()
            
            results['action_smoothness'] = {
                'per_joint': {self.dof_names[i]: per_joint_smoothness[i].item() for i in range(len(self.dof_names))},
                'average': mean_smoothness.item(),
            }
            
        # ===================== 6. Joint Acceleration =====================
        if len(self.joint_acc_per_step) > 1:
            joint_acc_stack = torch.stack(self.joint_acc_per_step, dim=0)  # [num_steps, num_envs, num_dofs]
            
            per_joint_acc = joint_acc_stack[1:].mean(dim=0).mean(dim=0)  # [num_dofs]
            mean_joint_acc = per_joint_acc.mean()
            
            results['joint_acceleration'] = {
                'per_joint': {self.dof_names[i]: per_joint_acc[i].item() for i in range(len(self.dof_names))},
                'average': mean_joint_acc.item(),
            }
            
        # ===================== 7. Base Angular Velocity Error =====================
        if len(self.base_ang_vel_error_per_step) > 0:
            base_ang_vel_error_stack = torch.stack(self.base_ang_vel_error_per_step, dim=0)  # [num_steps, num_envs]
            mean_base_ang_vel_error = base_ang_vel_error_stack.mean()
            
            results['base_ang_vel_error'] = {
                'average': mean_base_ang_vel_error.item(),
            }
            
        # ===================== 8. Reference Energy (PD Approximation) =====================
        if len(self.ref_energy_per_step) > 0:
            ref_energy_stack = torch.stack(self.ref_energy_per_step, dim=0)  # [num_steps, num_envs, num_dofs]
            
            # Mean reference energy per joint
            per_joint_ref_energy = ref_energy_stack.mean(dim=0).mean(dim=0)  # [num_dofs]
            mean_ref_energy = per_joint_ref_energy.mean()
            
            results['ref_energy'] = {
                'per_joint': {self.dof_names[i]: per_joint_ref_energy[i].item() for i in range(len(self.dof_names))},
                'average': mean_ref_energy.item(),
            }
            
        # ===================== 9. Motion Tracking (Global Frame) =====================
        if len(self.motion_tracking_global_per_step) > 0:
            # Stack: [num_steps, num_envs, num_bodies, 3]
            motion_stack = torch.stack(self.motion_tracking_global_per_step, dim=0)
            
            # Compute distance (L2 norm) per body
            motion_dist = torch.norm(motion_stack, dim=-1)  # [num_steps, num_envs, num_bodies]
            
            # Statistics over trajectory (per body)
            # Mean over steps and envs
            per_body_mean = motion_dist.mean(dim=0).mean(dim=0)  # [num_bodies]
            per_body_min = motion_dist.min(dim=0)[0].mean(dim=0)  # [num_bodies]
            per_body_max = motion_dist.max(dim=0)[0].mean(dim=0)  # [num_bodies]
            
            # Overall statistics
            overall_mean = per_body_mean.mean()
            overall_min = per_body_min.min()
            overall_max = per_body_max.max()
            
            # Per-body results
            num_bodies = len(self.body_names) if len(self.body_names) > 0 else per_body_mean.shape[0]
            per_body_results = {}
            for i in range(per_body_mean.shape[0]):
                body_name = self.body_names[i] if i < len(self.body_names) else f"body_{i}"
                per_body_results[body_name] = {
                    'mean': per_body_mean[i].item(),
                    'min': per_body_min[i].item(),
                    'max': per_body_max[i].item(),
                }
            
            results['motion_tracking_global'] = {
                'per_body': per_body_results,
                'overall': {
                    'mean': overall_mean.item(),
                    'min': overall_min.item(),
                    'max': overall_max.item(),
                },
            }
            
        # ===================== 10. Motion Tracking (Local Frame - Robot Anchor Relative) =====================
        if len(self.motion_tracking_local_per_step) > 0:
            # Stack: [num_steps, num_envs, num_bodies, 3]
            motion_local_stack = torch.stack(self.motion_tracking_local_per_step, dim=0)
            
            # Compute distance (L2 norm) per body
            motion_local_dist = torch.norm(motion_local_stack, dim=-1)  # [num_steps, num_envs, num_bodies]
            
            # Statistics
            per_body_local_mean = motion_local_dist.mean(dim=0).mean(dim=0)  # [num_bodies]
            per_body_local_min = motion_local_dist.min(dim=0)[0].mean(dim=0)
            per_body_local_max = motion_local_dist.max(dim=0)[0].mean(dim=0)
            
            # Overall statistics
            overall_local_mean = per_body_local_mean.mean()
            overall_local_min = per_body_local_min.min()
            overall_local_max = per_body_local_max.max()
            
            # Per-body results
            per_body_local_results = {}
            for i in range(per_body_local_mean.shape[0]):
                body_name = self.body_names[i] if i < len(self.body_names) else f"body_{i}"
                per_body_local_results[body_name] = {
                    'mean': per_body_local_mean[i].item(),
                    'min': per_body_local_min[i].item(),
                    'max': per_body_local_max[i].item(),
                }
            
            results['motion_tracking_local'] = {
                'per_body': per_body_local_results,
                'overall': {
                    'mean': overall_local_mean.item(),
                    'min': overall_local_min.item(),
                    'max': overall_local_max.item(),
                },
            }
            
        results['metadata'] = {
            'num_steps': self.step_count,
            'num_envs': self.num_envs,
            'num_dofs': self.num_dofs,
        }
        
        return results
    
    def save_results(self, save_path: Path, results: dict):
        """Save results to a text file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAJECTORY EVALUATION METRICS\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            if 'metadata' in results:
                f.write("METADATA\n")
                f.write("-" * 40 + "\n")
                for key, value in results['metadata'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # 1. Energy
            if 'energy' in results:
                f.write("1. ENERGY (|torque * joint_velocity| - mean per step)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average energy across all joints: {results['energy']['average']:.4f}\n")
                f.write("\n  Per-joint mean energy (averaged over steps):\n")
                for joint_name, energy in results['energy']['per_joint'].items():
                    f.write(f"    {joint_name}: {energy:.4f}\n")
                f.write("\n")
            
            # 2. Foot Slippage
            if 'foot_slippage' in results:
                f.write("2. FOOT SLIPPAGE (velocity when in contact)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Left foot (sum over trajectory): {results['foot_slippage']['left']:.4f}\n")
                f.write(f"  Right foot (sum over trajectory): {results['foot_slippage']['right']:.4f}\n")
                f.write(f"  Average: {results['foot_slippage']['average']:.4f}\n")
                f.write("\n")
            
            # 3. Foot Contact Force
            if 'foot_contact_force' in results:
                f.write("3. FOOT CONTACT FORCE\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Left foot (mean): {results['foot_contact_force']['left_mean']:.4f}\n")
                f.write(f"  Right foot (mean): {results['foot_contact_force']['right_mean']:.4f}\n")
                f.write(f"  Average (mean): {results['foot_contact_force']['average_mean']:.4f}\n")
                f.write(f"  Left foot (max): {results['foot_contact_force']['left_max']:.4f}\n")
                f.write(f"  Right foot (max): {results['foot_contact_force']['right_max']:.4f}\n")
                f.write(f"  Left foot (std): {results['foot_contact_force']['left_std']:.4f}\n")
                f.write(f"  Right foot (std): {results['foot_contact_force']['right_std']:.4f}\n")
                f.write(f"  Average (std): {results['foot_contact_force']['average_std']:.4f}\n")
                f.write("\n")
            
            # 4. Action Rate
            if 'action_rate' in results:
                f.write("4. ACTION RATE (|action_t - action_t-1|)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average action rate: {results['action_rate']['average']:.6f}\n")
                f.write("\n  Per-joint action rate:\n")
                for joint_name, rate in results['action_rate']['per_joint'].items():
                    f.write(f"    {joint_name}: {rate:.6f}\n")
                f.write("\n")
            
            # 5. Action Smoothness (Jerk)
            if 'action_smoothness' in results:
                f.write("5. ACTION SMOOTHNESS - Jerk ((action_t - 2*action_t-1 + action_t-2)^2)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average smoothness (lower is better): {results['action_smoothness']['average']:.6f}\n")
                f.write("\n  Per-joint smoothness:\n")
                for joint_name, smoothness in results['action_smoothness']['per_joint'].items():
                    f.write(f"    {joint_name}: {smoothness:.6f}\n")
                f.write("\n")
            
            # 6. Joint Acceleration
            if 'joint_acceleration' in results:
                f.write("6. JOINT ACCELERATION (|dof_vel_t - dof_vel_t-1| / dt)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average joint acceleration: {results['joint_acceleration']['average']:.4f} rad/s^2\n")
                f.write("\n  Per-joint acceleration:\n")
                for joint_name, acc in results['joint_acceleration']['per_joint'].items():
                    f.write(f"    {joint_name}: {acc:.4f}\n")
                f.write("\n")
            
            # 7. Base Angular Velocity Error
            if 'base_ang_vel_error' in results:
                f.write("7. BASE ANGULAR VELOCITY TRACKING ERROR\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average error (L2 norm): {results['base_ang_vel_error']['average']:.6f} rad/s\n")
                f.write("\n")
            
            # 8. Reference Energy (PD Approximation)
            if 'ref_energy' in results:
                f.write("8. REFERENCE ENERGY (PD Controller Approximation)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average reference energy: {results['ref_energy']['average']:.4f}\n")
                f.write("  Note: Estimated using torque_ref ~= Kp*(q_ref-q) + Kd*(dq_ref-dq)\n")
                f.write("\n  Per-joint reference energy:\n")
                for joint_name, energy in results['ref_energy']['per_joint'].items():
                    f.write(f"    {joint_name}: {energy:.4f}\n")
                f.write("\n")
            
            # 9. Motion Tracking (Global)
            if 'motion_tracking_global' in results:
                f.write("9. MOTION TRACKING - GLOBAL FRAME (world coordinates)\n")
                f.write("-" * 40 + "\n")
                overall = results['motion_tracking_global']['overall']
                f.write(f"  Overall mean distance: {overall['mean']:.6f} m\n")
                f.write(f"  Overall min distance: {overall['min']:.6f} m\n")
                f.write(f"  Overall max distance: {overall['max']:.6f} m\n")
                f.write("\n  Per-body statistics:\n")
                for body_name, stats in results['motion_tracking_global']['per_body'].items():
                    f.write(f"    {body_name}:\n")
                    f.write(f"      mean: {stats['mean']:.6f} m, min: {stats['min']:.6f} m, max: {stats['max']:.6f} m\n")
                f.write("\n")
            
            # 10. Motion Tracking (Local)
            if 'motion_tracking_local' in results:
                f.write("10. MOTION TRACKING - LOCAL FRAME (robot anchor relative)\n")
                f.write("-" * 40 + "\n")
                overall_local = results['motion_tracking_local']['overall']
                f.write(f"  Overall mean distance: {overall_local['mean']:.6f} m\n")
                f.write(f"  Overall min distance: {overall_local['min']:.6f} m\n")
                f.write(f"  Overall max distance: {overall_local['max']:.6f} m\n")
                f.write("\n  Per-body statistics:\n")
                for body_name, stats in results['motion_tracking_local']['per_body'].items():
                    f.write(f"    {body_name}:\n")
                    f.write(f"      mean: {stats['mean']:.6f} m, min: {stats['min']:.6f} m, max: {stats['max']:.6f} m\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Results saved to {save_path}")


def evaluate_with_metrics(algo: BaseAlgo, env, num_steps: int | None, metrics_collector: MetricsCollector):
    """Run evaluation while collecting metrics.
    
    Uses algo's inference policy to evaluate the agent.
    If num_steps is None, runs until motion ends.
    """
    
    # Set environment to evaluation mode
    if hasattr(env, 'set_is_evaluating'):
        env.set_is_evaluating()
    
    # Reset environment
    obs_dict = env.reset_all()
    for key in obs_dict:
        obs_dict[key] = obs_dict[key].to(algo.device)
    
    # Get inference policy from algo
    eval_policy = algo.get_inference_policy(device=str(algo.device))
    
    if num_steps is None:
        logger.info("Starting evaluation until motion ends...")
    else:
        logger.info(f"Starting evaluation for {num_steps} steps...")
    
    step = 0
    prev_timestep = 0  # Track previous timestep to detect reset
    motion_completed_once = False  # Track if motion completed at least once
    
    while True:
        with torch.inference_mode():
            # Get actions from policy
            actions = eval_policy(obs_dict)
        
        # Step the environment
        actor_state_for_env = {"actions": actions}
        obs_dict, rewards, dones, infos = env.step(actor_state_for_env)
        
        for key in obs_dict:
            obs_dict[key] = obs_dict[key].to(algo.device)
        
        # Collect metrics AFTER the step
        metrics_collector.collect_step_metrics()
        
        step += 1
        
        if step % 100 == 0:
            if num_steps is not None:
                logger.info(f"Step {step}/{num_steps}")
            else:
                # Log motion progress
                if hasattr(env, 'command_manager'):
                    motion_cmd = env.command_manager.get_state("motion_command")
                    if motion_cmd is not None and hasattr(motion_cmd, 'time_steps') and hasattr(motion_cmd.motion, 'time_step_total'):
                        current_timestep = motion_cmd.time_steps[0].item()
                        total_timesteps = motion_cmd.motion.time_step_total
                        logger.info(f"Step {step} - Motion progress: {current_timestep}/{total_timesteps}")
                    else:
                        logger.info(f"Step {step}")
                else:
                    logger.info(f"Step {step}")
        
        # Check if motion ended by detecting timestep reset (decrease)
        motion_ended = False
        if hasattr(env, 'command_manager'):
            motion_cmd = env.command_manager.get_state("motion_command")
            if motion_cmd is not None and hasattr(motion_cmd, 'time_steps') and hasattr(motion_cmd.motion, 'time_step_total'):
                current_timestep = motion_cmd.time_steps[0].item()
                total_timesteps = motion_cmd.motion.time_step_total
                
                # Check if motion reached near the end
                if current_timestep >= total_timesteps - 5:
                    motion_completed_once = True
                
                # Detect reset: timestep decreased significantly (motion restarted)
                if motion_completed_once and current_timestep < prev_timestep - 10:
                    motion_ended = True
                    logger.info(f"Motion completed and reset detected at step {step} (timestep went from {prev_timestep} to {current_timestep})")
                
                prev_timestep = current_timestep
        
        # Also check time_out_buf as fallback
        if hasattr(env, 'time_out_buf') and env.time_out_buf.any():
            motion_ended = True
            logger.info(f"time_out_buf triggered at step {step}")
        
        if motion_ended:
            logger.info(f"Motion ended at step {step}")
            break
        
        # If num_steps is specified, check if we've reached it
        if num_steps is not None and step >= num_steps:
            logger.info(f"Reached maximum steps: {num_steps}")
            break
    
    logger.info(f"Evaluation completed. Total steps: {metrics_collector.step_count}")


def run_eval_with_metrics(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
):
    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval_metrics")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = Path(checkpoint_path).parent
    
    # Get number of evaluation steps
    # If max_eval_steps is None, 0, or negative, run until motion ends
    num_eval_steps = tyro_config.training.max_eval_steps
    if num_eval_steps is None or num_eval_steps <= 0:
        num_eval_steps = None
        logger.info("Running evaluation until motion ends (no step limit)")
    else:
        logger.info(f"Running evaluation for {num_eval_steps} steps")

    # Initialize metrics collector
    metrics_collector = MetricsCollector(env, device)
    
    # Run evaluation with metrics collection
    evaluate_with_metrics(algo, env, num_eval_steps, metrics_collector)
    
    # Compute summary statistics
    results = metrics_collector.compute_summary_statistics()
    
    # Save results
    # Default: save to checkpoint directory
    metrics_dir = checkpoint_dir / "metrics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_num = checkpoint_path.split('/')[-1].split('_')[-1].split('.')[0]
    save_path = metrics_dir / f"trajectory_metrics_ckpt_{ckpt_num}_{timestamp}.txt"
    
    metrics_collector.save_results(save_path, results)
    
    # Print summary to console
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    if 'energy' in results:
        logger.info(f"Energy - Average per step: {results['energy']['average']:.4f}")
    
    if 'foot_slippage' in results:
        logger.info(f"Foot Slippage - Left: {results['foot_slippage']['left']:.4f}, Right: {results['foot_slippage']['right']:.4f}, Avg: {results['foot_slippage']['average']:.4f}")
    
    if 'foot_contact_force' in results:
        logger.info(f"Foot Contact Force - Left: {results['foot_contact_force']['left_mean']:.4f}, Right: {results['foot_contact_force']['right_mean']:.4f}, Avg: {results['foot_contact_force']['average_mean']:.4f}")
    
    if 'action_rate' in results:
        logger.info(f"Action Rate - Average: {results['action_rate']['average']:.6f}")
    
    if 'motion_tracking_global' in results:
        overall_global = results['motion_tracking_global']['overall']
        logger.info(f"Motion Tracking (Global) - Mean: {overall_global['mean']:.6f}m, Min: {overall_global['min']:.6f}m, Max: {overall_global['max']:.6f}m")
    
    if 'motion_tracking_local' in results:
        overall_local = results['motion_tracking_local']['overall']
        logger.info(f"Motion Tracking (Local) - Mean: {overall_local['mean']:.6f}m, Min: {overall_local['min']:.6f}m, Max: {overall_local['max']:.6f}m")
    
    logger.info("=" * 60)

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_metrics(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
