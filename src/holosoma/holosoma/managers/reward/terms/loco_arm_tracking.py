"""Reward terms for the combined loco-arm-tracking environment.

Each function takes ``env`` as its first argument plus optional named parameters
and returns a ``torch.Tensor`` of shape ``[num_envs]``.

This module is self-contained — it does not import from the sibling
arm_tracking or locomotion reward term modules — so that the
loco_arm_tracking environment remains fully independent.

Note: ``penalty_base_lin_vel_xy`` is explicitly excluded since base linear
velocity is now a desired behavior in this combined environment.
"""

from __future__ import annotations

import torch

from holosoma.utils.rotation_6d import (
    heading_quat_from_root_quat,
    quat_to_rotmat,
    rot6d_to_rotmat,
    world_to_reference_frame,
)
from holosoma.utils.rotations import quat_rotate_inverse


def _get_arm_command_state(env):
    """Retrieve the arm command state from the command manager."""
    arm_command_state = env.command_manager.get_state("arm_command")
    if arm_command_state is None:
        raise AttributeError(
            "arm_command is not registered with the command manager."
        )
    return arm_command_state


def _get_current_ee_pose_in_command_frame(env):
    """Compute current EE poses in the arm command reference frame.

    Returns:
        current_ee_pos: [num_envs, 2, 3]
        current_ee_quat: [num_envs, 2, 4] (xyzw)
    """
    arm_command_state = _get_arm_command_state(env)

    current_ee_pos_world = env.simulator._rigid_body_pos[
        :, arm_command_state.ee_body_indices, :
    ]
    current_ee_quat_world = env.simulator._rigid_body_rot[
        :, arm_command_state.ee_body_indices, :
    ]

    if arm_command_state.reference_frame == "torso":
        torso_body_index = getattr(arm_command_state, "torso_body_index", None)
        if torso_body_index is None:
            torso_body_index = env.simulator.find_rigid_body_indice("torso_link")
        ref_pos = env.simulator._rigid_body_pos[:, torso_body_index]
        ref_quat = env.simulator._rigid_body_rot[:, torso_body_index]
    elif arm_command_state.reference_frame == "base":
        ref_pos = env.simulator.robot_root_states[:, :3]
        ref_quat = env.simulator.robot_root_states[:, 3:7]
    elif arm_command_state.reference_frame == "heading":
        root_quat = env.simulator.robot_root_states[:, 3:7]
        ref_pos = env.simulator.robot_root_states[:, :3]
        ref_quat = heading_quat_from_root_quat(root_quat)
    elif arm_command_state.reference_frame == "ground_heading":
        root_pos = env.simulator.robot_root_states[:, :3]
        root_quat = env.simulator.robot_root_states[:, 3:7]
        ref_pos = root_pos.clone()
        ref_pos[:, 2] = 0.0
        ref_quat = heading_quat_from_root_quat(root_quat)
    elif arm_command_state.reference_frame == "world":
        return current_ee_pos_world, current_ee_quat_world
    else:
        raise ValueError(
            f"Unsupported reference_frame '{arm_command_state.reference_frame}'. "
            f"Expected 'base', 'torso', 'world', 'heading', or 'ground_heading'."
        )

    current_ee_pos = torch.zeros_like(current_ee_pos_world)
    current_ee_quat = torch.zeros_like(current_ee_quat_world)
    for arm_idx in range(current_ee_pos_world.shape[1]):
        current_ee_pos[:, arm_idx], current_ee_quat[:, arm_idx] = (
            world_to_reference_frame(
                current_ee_pos_world[:, arm_idx],
                current_ee_quat_world[:, arm_idx],
                ref_pos,
                ref_quat,
            )
        )
    return current_ee_pos, current_ee_quat


def _tracking_errors(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (position squared error, rotational trace error), both [num_envs].

    Position error: mean over left/right arm of squared L2 distance.
    Rotational error: tr(I - R_current^T R_target) = 3 - tr(R_current^T R_target),
    averaged over both arms.
    """
    arm_command_state = _get_arm_command_state(env)
    current_ee_pos, current_ee_quat = _get_current_ee_pose_in_command_frame(env)

    # Position error: mean over left/right arm of squared L2 distance.
    pos_error = torch.sum(
        torch.square(current_ee_pos - arm_command_state.target_ee_pos), dim=-1
    ).mean(dim=-1)

    # Rotational error: tr(I - R_current^T R_target) = 3 - tr(R_current^T R_target)
    current_ee_rotmat = quat_to_rotmat(
        current_ee_quat.reshape(-1, 4)
    ).reshape(current_ee_quat.shape[0], 2, 3, 3)
    target_ee_rotmat = rot6d_to_rotmat(
        arm_command_state.target_ee_rot6d.reshape(-1, 6)
    ).reshape(arm_command_state.target_ee_rot6d.shape[0], 2, 3, 3)
    rel = torch.matmul(current_ee_rotmat.transpose(-2, -1), target_ee_rotmat)
    trace_rel = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rot_error = (3.0 - trace_rel).mean(dim=-1)

    return pos_error, rot_error


def _tracking_pos_l1_error(env) -> torch.Tensor:
    """Return L1 position error: mean over arms of ||e||_1, shape [num_envs]."""
    arm_command_state = _get_arm_command_state(env)
    current_ee_pos, _ = _get_current_ee_pose_in_command_frame(env)
    l1_per_arm = torch.sum(torch.abs(current_ee_pos - arm_command_state.target_ee_pos), dim=-1)
    return l1_per_arm.mean(dim=-1)


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile."""

    def cubic_bezier_interpolation(
        y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)
    stance = cubic_bezier_interpolation(
        torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x
    )
    swing = cubic_bezier_interpolation(
        torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1
    )
    return torch.where(x <= 0.5, stance, swing)


def tracking_lin_vel(env, tracking_sigma: float = 0.25, **kwargs) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes).

    Uses exponential reward: exp(-||v_actual - v_cmd||² / sigma)
    where v_actual is body-frame base linear velocity xy and v_cmd is the
    locomotion command.

    Args:
        env: The environment instance.
        tracking_sigma: Sigma for exponential reward scaling.

    Returns:
        Reward tensor [num_envs].
    """
    commands = env.command_manager.commands
    lin_vel_world = env.simulator.robot_root_states[:, 7:10]
    lin_vel_body = quat_rotate_inverse(env.simulator.base_quat, lin_vel_world, w_last=True)
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - lin_vel_body[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def tracking_ang_vel(env, tracking_sigma: float = 0.25, **kwargs) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw).

    Uses exponential reward: exp(-||ω_actual - ω_cmd||² / sigma)
    where ω_actual is body-frame base angular velocity z and ω_cmd is the
    locomotion command.

    Args:
        env: The environment instance.
        tracking_sigma: Sigma for exponential reward scaling.

    Returns:
        Reward tensor [num_envs].
    """
    commands = env.command_manager.commands
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    ang_vel_body = quat_rotate_inverse(env.simulator.base_quat, ang_vel_world, w_last=True)
    ang_vel_error = torch.square(commands[:, 2] - ang_vel_body[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def tracking_ee_pos_phase_gated(
    env,
    tracking_sigma: float = 0.01,
    touchdown_gate_sigma_scale: float = 4.0,
    touchdown_gate_phase_width: float = 0.20,
    **kwargs,
) -> torch.Tensor:
    """EE position tracking with sigma scaled around foot touchdown.

    Identical to ``tracking_ee_pos`` when ``touchdown_gate_sigma_scale=1.0``.
    The scale is a direct multiplier on sigma at peak touchdown:
    ``sigma_eff = tracking_sigma * lerp(1.0, touchdown_gate_sigma_scale, gate)``

    The gate is constructed from the gait phase clock (exogenous, ungameable):
    ``gate = max(exp(-(1 - cos(phi_L - pi)) / width²),
                 exp(-(1 - cos(phi_R - pi)) / width²))``
    which peaks at 1.0 at touchdown (phi = ±pi) and decays to ~0 at mid-swing.

    Examples:
        - 4.0 (default): 4× sigma at touchdown (relaxed, tolerant of impact)
        - 1.0: no phase gating (constant sigma)
        - 0.5: sigma/2 at touchdown (2× tighter during impact)

    Args:
        env: The environment instance.
        tracking_sigma: Baseline sigma (same as ``tracking_ee_pos``).
            Curriculum-annealed; the gate scales relative to this value.
        touchdown_gate_sigma_scale: Direct sigma multiplier at peak touchdown.
            Must be > 0. Values > 1 relax, values < 1 tighten.
            Set to 1.0 to disable phase gating entirely.
        touchdown_gate_phase_width: Gaussian width (radians) of the
            touchdown gate. Controls what fraction of the gait cycle is
            covered. 0.20 rad ≈ 12% FWHM ≈ 30-80ms at typical gait freq.

    Returns:
        Reward tensor [num_envs].
    """
    pos_error, _ = _tracking_errors(env)

    if touchdown_gate_sigma_scale == 1.0:
        return torch.exp(-pos_error / tracking_sigma)

    gait_state = env.command_manager.get_state("locomotion_gait")
    phi = gait_state.phase  # [num_envs, 2], radians in [-pi, pi]

    # Periodic distance² to touchdown (phi = ±pi): 0 at touchdown, 2 at peak swing.
    d2 = 1.0 - torch.cos(phi - torch.pi)
    gate_per_foot = torch.exp(-d2 / (touchdown_gate_phase_width ** 2))
    gate = gate_per_foot.max(dim=1).values  # [num_envs]

    # Direct multiplier: lerp from 1.0 (mid-swing) to scale (touchdown)
    sigma_eff = tracking_sigma * (1.0 + (touchdown_gate_sigma_scale - 1.0) * gate)
    return torch.exp(-pos_error / sigma_eff)


def tracking_ee_pos_combined_norm_phase_gated(
    env,
    sq_sigma: float = 0.001,
    l1_sigma: float = 0.01,
    touchdown_gate_sigma_scale: float = 4.0,
    touchdown_gate_phase_width: float = 0.20,
    **kwargs,
) -> torch.Tensor:
    """Combined L2²+L1 position tracking with sigma scaled around foot touchdown.

    Identical to the arm_tracking ``tracking_ee_pos_combined_norm`` when
    ``touchdown_gate_sigma_scale=1.0``. The scale is a direct multiplier on
    sigma at peak touchdown (same semantics as ``tracking_ee_pos_phase_gated``).

    r = exp(-(||e||² / sq_sigma_eff + ||e||₁ / (√3 · l1_sigma_eff)))

    Sigma scaling is a direct multiplier:
    ``sigma_eff = sigma * lerp(1.0, touchdown_gate_sigma_scale, gate)``

    Examples:
        - 4.0 (default): 4× sigma at touchdown (relaxed)
        - 1.0: no phase gating (constant sigma)
        - 0.5: sigma/2 at touchdown (2× tighter)

    Args:
        env: The environment instance.
        sq_sigma: Baseline squared-error sigma. Curriculum-annealed.
        l1_sigma: Baseline L1-error sigma. Curriculum-annealed.
        touchdown_gate_sigma_scale: Direct sigma multiplier at peak touchdown.
            Must be > 0. Values > 1 relax, values < 1 tighten.
            Set to 1.0 to disable phase gating entirely.
        touchdown_gate_phase_width: Gaussian width (radians) of the
            touchdown gate. 0.20 rad ≈ 12% FWHM ≈ 30-80ms at typical freq.
    """
    import math

    pos_error_sq, _ = _tracking_errors(env)
    pos_error_l1 = _tracking_pos_l1_error(env)

    # l1_sigma <= 0 disables the L1 term entirely (pure L2sq behaviour),
    # avoiding a division-by-zero. This lets the L1 contribution be switched
    # off from the CLI via --finetune-l1-sigma 0.
    use_l1 = l1_sigma > 0.0

    if touchdown_gate_sigma_scale == 1.0:
        l1_contrib = (
            pos_error_l1 / (math.sqrt(3.0) * l1_sigma) if use_l1 else 0.0
        )
        return torch.exp(-pos_error_sq / sq_sigma - l1_contrib)

    gait_state = env.command_manager.get_state("locomotion_gait")
    phi = gait_state.phase  # [num_envs, 2]

    d2 = 1.0 - torch.cos(phi - torch.pi)
    gate_per_foot = torch.exp(-d2 / (touchdown_gate_phase_width ** 2))
    gate = gate_per_foot.max(dim=1).values  # [num_envs]

    # Direct multiplier: lerp from 1.0 (mid-swing) to scale (touchdown)
    scale = 1.0 + (touchdown_gate_sigma_scale - 1.0) * gate
    sq_sigma_eff = sq_sigma * scale

    if use_l1:
        l1_sigma_eff = l1_sigma * scale
        l1_contrib = pos_error_l1 / (math.sqrt(3.0) * l1_sigma_eff)
    else:
        l1_contrib = 0.0

    return torch.exp(-pos_error_sq / sq_sigma_eff - l1_contrib)


def tracking_ee_rot(env, tracking_sigma: float = 0.20, **kwargs) -> torch.Tensor:
    """Reward end-effector rotation tracking using Frobenius/trace rotational error.

    Uses exp(-||rot_error||² / sigma) averaged over both arms.

    Args:
        env: The environment instance.
        tracking_sigma: Sigma for exponential reward scaling.

    Returns:
        Reward tensor [num_envs].
    """
    _, rot_error = _tracking_errors(env)
    return torch.exp(-rot_error / tracking_sigma)


def feet_phase(
    env, swing_height: float = 0.09, tracking_sigma: float = 0.008, **kwargs
) -> torch.Tensor:
    """Reward for tracking desired foot height based on gait phase.

    Uses the gait phase clock to determine expected foot height via cubic
    Bézier interpolation and rewards matching actual foot heights.

    Args:
        env: The environment instance.
        swing_height: Maximum height during swing phase.
        tracking_sigma: Sigma for exponential reward scaling.

    Returns:
        Reward tensor [num_envs].
    """
    # Get foot heights (relative to terrain)
    foot_z_left = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 0]
    foot_z_right = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 1]

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    rz_left = _expected_foot_height(gait_state.phase[:, 0], swing_height)
    rz_right = _expected_foot_height(gait_state.phase[:, 1], swing_height)

    # Calculate height tracking errors
    error_left = torch.square(foot_z_left - rz_left)
    error_right = torch.square(foot_z_right - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right

    return torch.exp(-total_error / tracking_sigma)


def _posture_gravity_target(env, attr: str) -> torch.Tensor | None:
    """Commanded gravity direction for a trunk body, or None if not commanded.

    Returns the [num_envs, 3] target set by ArmTrackingCommand. A config that
    does not command posture leaves the buffer upright, and callers fall back
    to the plain uprightness error, so existing experiments are unchanged.
    """
    state = env.command_manager.get_state("arm_command")
    return getattr(state, attr, None) if state is not None else None


def _gravity_tracking_error(projected_gravity: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
    """Squared error between measured and commanded gravity, horizontal part.

    ``g_xy`` fully parameterises tilt below 90 degrees: its magnitude is
    sin(tilt) and its direction is the tilt direction. With an upright target
    the target's horizontal part is zero and this reduces EXACTLY to the
    previous ``||g_xy||^2``, so sigma keeps its meaning and upright-commanded
    runs are bit-identical.
    """
    if target is None:
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    return torch.sum(torch.square(projected_gravity[:, :2] - target[:, :2]), dim=1)


def tracking_orientation(env, tracking_sigma: float = 0.10, **kwargs) -> torch.Tensor:
    """Reward base (pelvis) uprightness using exponential kernel.

    Computes exp(-error / sigma) where error = ||g_xy||^2 is the squared
    magnitude of the projected gravity vector's x/y components in the base
    frame. When perfectly upright error=0 → reward=1. The sigma parameter
    controls the tolerance: at error=sigma the reward is ~0.37.

    For reference, ||g_xy||^2 ≈ sin²(tilt_angle), so:
      - sigma=0.25 → reward ~0.37 at ~30° tilt
      - sigma=0.10 → reward ~0.37 at ~18° tilt
      - sigma=0.50 → reward ~0.37 at ~45° tilt
    """
    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(env.simulator.base_quat, gravity, w_last=True)
    target = _posture_gravity_target(env, "target_pelvis_gravity")
    error = _gravity_tracking_error(projected_gravity, target)
    return torch.exp(-error / tracking_sigma)


def tracking_torso_orientation(env, tracking_sigma: float = 0.10, **kwargs) -> torch.Tensor:
    """Reward torso (last waist link) uprightness using exponential kernel.

    Same as tracking_orientation but measured at the torso_link frame (the
    body after waist_pitch_joint) rather than the pelvis. This captures the
    combined effect of all three waist joints (yaw/roll/pitch) and penalises
    torso lean even when the pelvis is upright.

    Sigma semantics are identical to tracking_orientation.
    """
    torso_body_index = env.simulator.find_rigid_body_indice("torso_link")
    torso_quat = env.simulator._rigid_body_rot[:, torso_body_index]

    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(torso_quat, gravity, w_last=True)
    target = _posture_gravity_target(env, "target_torso_gravity")
    error = _gravity_tracking_error(projected_gravity, target)
    return torch.exp(-error / tracking_sigma)


def penalty_arm_action_rate(env, **kwargs) -> torch.Tensor:
    """Penalize action delta of arm joints between consecutive policy steps.

    Computes ||Δa_arm||² using arm_dof_indices to select arm components
    from the action difference.

    Args:
        env: The environment instance.

    Returns:
        Reward tensor [num_envs].
    """
    prev_action = env.action_manager.prev_action
    curr_action = env.action_manager.action
    arm_indices = env.arm_dof_indices
    prev_arm = prev_action[:, arm_indices]
    curr_arm = curr_action[:, arm_indices]
    return torch.sum(torch.square(prev_arm - curr_arm), dim=1)


def penalty_leg_action_rate(env, **kwargs) -> torch.Tensor:
    """Penalize action delta of leg joints between consecutive policy steps.

    Computes ||Δa_leg||² using leg_dof_indices to select leg components
    from the action difference.

    Args:
        env: The environment instance.

    Returns:
        Reward tensor [num_envs].
    """
    prev_action = env.action_manager.prev_action
    curr_action = env.action_manager.action
    leg_indices = env.leg_dof_indices
    prev_leg = prev_action[:, leg_indices]
    curr_leg = curr_action[:, leg_indices]
    return torch.sum(torch.square(prev_leg - curr_leg), dim=1)


def penalty_dof_pos_limits(env, soft_dof_pos_limit: float = 0.95, **kwargs) -> torch.Tensor:
    """Penalize joint positions outside soft limits.

    Computes the sum of violations beyond soft limits (defined as a fraction
    of the hard limit range).

    Args:
        env: The environment instance.
        soft_dof_pos_limit: Soft limit as fraction of hard limit range.

    Returns:
        Reward tensor [num_envs].
    """
    limits_source = getattr(env.simulator, "hard_dof_pos_limits", env.simulator.dof_pos_limits)
    m = (limits_source[:, 0] + limits_source[:, 1]) / 2.0
    r = limits_source[:, 1] - limits_source[:, 0]
    lower_soft = m - 0.5 * r * soft_dof_pos_limit
    upper_soft = m + 0.5 * r * soft_dof_pos_limit

    q = env.simulator.dof_pos
    out_of_limits = -(q - lower_soft).clip(max=0.0)
    out_of_limits += (q - upper_soft).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def penalty_termination(env, **kwargs) -> torch.Tensor:
    """One-time penalty for non-timeout termination.

    Returns 1.0 for terminated envs (reset_buf & ~time_out_buf).

    Args:
        env: The environment instance.

    Returns:
        Reward tensor [num_envs].
    """
    return (env.reset_buf * ~env.time_out_buf).float()


def penalty_close_feet_xy(env, close_feet_threshold: float = 0.15, **kwargs) -> torch.Tensor:
    """Penalize when feet are too close together in the xy plane.

    Uses yaw-aligned lateral distance between feet. Returns 1.0 for envs
    where feet are closer than threshold, 0.0 otherwise.

    Args:
        env: The environment instance.
        close_feet_threshold: Minimum lateral distance between feet.

    Returns:
        Reward tensor [num_envs].
    """
    from holosoma.utils.rotations import quat_apply

    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    base_forward = quat_apply(env.base_quat, forward, w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )
    return (feet_distance < close_feet_threshold).float()


def penalty_feet_ori(env, **kwargs) -> torch.Tensor:
    """Penalize feet orientation deviation from flat.

    Returns the sum of projected gravity xy-norm for both feet.

    Args:
        env: The environment instance.

    Returns:
        Reward tensor [num_envs].
    """
    gravity = torch.zeros((env.num_envs, 3), device=env.device)
    gravity[:, 2] = -1.0
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)
    return (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
    )


def penalty_arm_residual_l2(env, **kwargs) -> torch.Tensor:
    """Penalize arm action magnitude.

    For the GAC-enriched variant: penalizes ``||arm_residuals||²``, the
    residual correction applied on top of the controller output.

    For the baseline variant (no controller): penalizes ``||arm_action||²``,
    the raw policy action on arm DOFs. This keeps arm actions small, which
    is a natural regularizer towards the default pose since the joint target
    is ``action * action_scale + default_dof_pos``.

    Args:
        env: The environment instance.

    Returns:
        Reward tensor [num_envs].
    """
    # Prefer the GAC-style residual buffer if the active action term exposes it.
    for _, term in env.action_manager.iter_terms():
        arm_residuals = getattr(term, "arm_residuals", None)
        if arm_residuals is not None:
            return torch.sum(torch.square(arm_residuals), dim=-1)

    # Fallback: penalize arm DOF components of the current action (baseline).
    actions = env.action_manager.action
    arm_indices = env.arm_dof_indices
    return torch.sum(torch.square(actions[:, arm_indices]), dim=-1)


def alive(env, **kwargs) -> torch.Tensor:
    """Dense survival signal — returns constant 1.0 for all envs.

    Args:
        env: The environment instance.

    Returns:
        Reward tensor [num_envs].
    """
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)


def penalty_foot_touchdown_shock(
    env, contact_threshold: float = 5.0, vz_threshold: float = 0.25, **kwargs
) -> torch.Tensor:
    """Penalty for downward foot velocity at the instant of touchdown.

    Detects the first simulation tick where each foot transitions from
    no-contact to contact (vertical force exceeds *contact_threshold*).
    At that tick, penalizes the squared downward speed exceeding
    *vz_threshold*.

    Requires ``env._prev_foot_contact`` buffer (bool, [num_envs, 2])
    initialized in the manager's ``_init_buffers`` and reset in
    ``_reset_buffers_callback``.

    Args:
        env: The environment instance.
        contact_threshold: Vertical force (N) to consider foot in contact.
        vz_threshold: Dead-zone for downward speed (m/s) below which no
            penalty is applied.

    Returns:
        Penalty tensor [num_envs].
    """
    # Vertical contact force per foot: [num_envs, 2]
    fz = env.simulator.contact_forces[:, env.feet_indices, 2]
    in_contact = fz > contact_threshold

    # Touchdown: just entered contact this tick
    prev = env._prev_foot_contact
    touchdown = in_contact & (~prev)

    # Foot vertical velocity (world z): [num_envs, 2]
    foot_vz = env.simulator._rigid_body_vel[:, env.feet_indices, 2]

    # Downward speed above threshold (vz < 0 means moving down)
    downward_speed = torch.clamp(-foot_vz - vz_threshold, min=0.0)

    cost = touchdown.float() * downward_speed.square()

    # Update memory
    env._prev_foot_contact[:] = in_contact

    return torch.sum(cost, dim=1)


def _foot_fz_peak_over_interval(env) -> torch.Tensor:
    """Per-foot peak positive vertical contact force over the control interval.

    Physics runs at ``control_decimation`` substeps per control step, but a
    single force sample (``contact_forces``) can miss a peak that occurred on an
    earlier substep. IsaacSim stores a per-substep history in
    ``contact_forces_history`` (index 0 = most recent), so the maximum over the
    most-recent ``control_decimation`` samples recovers the true interval peak.

    Returns:
        Peak vertical force per foot, clamped at >= 0: [num_envs, num_feet].

    Raises:
        ValueError: if the available history is shorter than the control
            decimation (an incomplete interval could silently miss impacts).
    """
    decimation = env.simulator.simulator_config.sim.control_decimation
    history = env.simulator.contact_forces_history  # [env, hist, body, 3]
    available = history.shape[1]
    if available < decimation:
        raise ValueError(
            f"contact_forces_history length ({available}) is shorter than "
            f"control_decimation ({decimation}); the peak-force penalty would "
            "miss the earliest substep(s) of the control interval. Increase "
            "contact_sensor_history_length to cover the control interval."
        )

    # Most-recent full control interval: [env, decimation, num_feet]
    fz_history = history[:, :decimation, :, :][:, :, env.feet_indices, 2]
    # Peak positive vertical force per foot over the interval: [env, num_feet]
    return torch.clamp(fz_history, min=0.0).amax(dim=1)


def penalty_foot_contact_force(
    env, fz_threshold: float = 400.0, **kwargs
) -> torch.Tensor:
    """Penalize excessive vertical contact force on feet.

    Uses the peak positive vertical force per foot across the *entire* control
    interval (all physics substeps), not just the final sample. This avoids
    missing an impact peak that occurs on an earlier substep. Penalizes the
    squared excess force (normalized by threshold for scale).

    This is more robust than velocity-based touchdown penalties because it
    directly penalizes what damages hardware — high impact forces — regardless
    of how the force arose (fast descent, stiff contact, etc.).

    Feet are treated independently: two simultaneous impacts remain visible as
    two costs (summed, no cross-foot maximum).

    Args:
        env: The environment instance.
        fz_threshold: Vertical force (N) above which penalty applies.
            Default 400N is ~2.33x half body static weight for G1 (35kg).

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    # Peak vertical force per foot over the control interval: [num_envs, num_feet]
    fz_peak = _foot_fz_peak_over_interval(env)

    # Excess above threshold, normalized by threshold for unit-scale
    excess = torch.clamp(fz_peak - fz_threshold, min=0.0) / fz_threshold

    return torch.sum(excess.square(), dim=1)


def penalty_foot_pretouchdown_velocity(
    env,
    contact_threshold: float = 5.0,
    clearance_threshold: float = 0.05,
    safe_downward_speed: float = 0.15,
    min_cmd_speed: float = 0.03,
    **kwargs,
) -> torch.Tensor:
    """Penalize fast downward foot velocity in the pre-touchdown approach.

    Unlike ``penalty_foot_touchdown_shock`` (which fires *at* the contact tick
    using a post-contact velocity), this shapes the descent *before* contact:
    it penalizes downward foot speed while a foot is airborne and close to the
    ground. Gating is computed independently per foot (design §5.1):

        active_i = airborne_i & (clearance_i < clearance_threshold)
                   & moving_command

    - ``airborne_i``: foot not in contact (vertical force < contact_threshold),
      so planted stance feet are never penalized.
    - ``clearance_i < clearance_threshold``: only the near-ground final
      approach, not the full swing arc (mid-swing descent is normal).
    - ``moving_command``: only when the robot is commanded to walk; standing
      receives no velocity-shaping penalty.

    The cost is the squared downward speed beyond a safe dead-zone, so slow
    normal landings cost nothing. Feet are summed independently (no cross-foot
    max, avoiding a global two-foot phase window).

    Intended as a low-weight fine-tune on a stable walking checkpoint (design
    §5.2), not from scratch.

    Args:
        env: The environment instance.
        contact_threshold: Vertical force (N) below which a foot is airborne.
        clearance_threshold: Foot height above terrain (m) below which the
            near-ground approach gate is active (design: 0.04-0.06 m).
        safe_downward_speed: Dead-zone downward speed (m/s); descent slower
            than this is not penalized (design: 0.10-0.20 m/s).
        min_cmd_speed: Minimum commanded linear speed (m/s) to activate.

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    # Airborne per foot: vertical contact force below threshold. [num_envs, 2]
    fz = env.simulator.contact_forces[:, env.feet_indices, 2]
    airborne = fz < contact_threshold

    # Foot clearance above terrain per foot. [num_envs, 2]
    feet_heights = env.terrain_manager.get_state("locomotion_terrain").feet_heights
    near_ground = feet_heights < clearance_threshold

    # Moving command gate (broadcast over feet). [num_envs, 1]
    commands = env.command_manager.commands
    cmd_speed = torch.linalg.norm(commands[:, :2], dim=-1, keepdim=True)
    moving_command = cmd_speed > min_cmd_speed

    active = airborne & near_ground & moving_command

    # Foot vertical velocity (world z): [num_envs, 2]. vz < 0 = descending.
    foot_vz = env.simulator._rigid_body_vel[:, env.feet_indices, 2]
    downward_excess = torch.clamp(-foot_vz - safe_downward_speed, min=0.0)

    cost = active.float() * downward_excess.square()

    return torch.sum(cost, dim=1)


def penalty_lateral_velocity_quadratic(
    env,
    min_cmd_speed: float = 0.03,
    **kwargs,
) -> torch.Tensor:
    """Quadratic penalty on perpendicular velocity relative to command direction.

    When the commanded linear speed is above *min_cmd_speed*, decomposes the
    body-frame velocity into parallel and perpendicular components relative to
    the command direction, and returns ||v_perp||².

    This is a pure penalty (non-negative), intended to be used with a negative
    weight. Unlike the exponential variant, it scales linearly with drift
    magnitude — no cliff-edge behaviour at small sigma.

    Returns 0.0 when standing (command direction ill-defined).

    Args:
        env: The environment instance.
        min_cmd_speed: Minimum commanded speed to activate directional decomposition.

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    commands = env.command_manager.commands
    c_xy = commands[:, :2]

    lin_vel_world = env.simulator.robot_root_states[:, 7:10]
    lin_vel_body = quat_rotate_inverse(env.simulator.base_quat, lin_vel_world, w_last=True)
    v_xy = lin_vel_body[:, :2]

    cmd_speed = torch.linalg.norm(c_xy, dim=-1, keepdim=True)
    cmd_dir = c_xy / torch.clamp(cmd_speed, min=1e-6)

    v_parallel = torch.sum(v_xy * cmd_dir, dim=-1)
    v_perp = v_xy - v_parallel.unsqueeze(-1) * cmd_dir
    perp_sq = torch.sum(v_perp.square(), dim=-1)

    moving_cmd = (cmd_speed.squeeze(-1) > min_cmd_speed).float()
    return moving_cmd * perp_sq


def penalty_turn_in_place_leakage(
    env,
    lin_cmd_threshold: float = 0.03,
    yaw_cmd_threshold: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    """Penalize linear velocity during pure yaw (turn-in-place) commands.

    Activates only when the commanded linear speed is near zero AND the
    commanded yaw rate exceeds a threshold — i.e., the robot should be
    spinning in place without translating.

    Uses instantaneous body-frame planar velocity.

    Args:
        env: The environment instance.
        lin_cmd_threshold: Max linear command norm to consider "near zero".
        yaw_cmd_threshold: Min absolute yaw command to consider "turning".

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    commands = env.command_manager.commands
    c_xy = commands[:, :2]
    wz_cmd = commands[:, 2]

    lin_cmd_norm = torch.linalg.norm(c_xy, dim=-1)

    # Turn-in-place indicator: low linear command AND significant yaw command
    turn_mask = (
        (lin_cmd_norm < lin_cmd_threshold) & (torch.abs(wz_cmd) > yaw_cmd_threshold)
    ).float()

    lin_vel_world = env.simulator.robot_root_states[:, 7:10]
    lin_vel_body = quat_rotate_inverse(env.simulator.base_quat, lin_vel_world, w_last=True)
    v_xy = lin_vel_body[:, :2]

    return turn_mask * torch.sum(v_xy.square(), dim=-1)


def penalty_straight_walk_yaw_leakage(
    env,
    lin_cmd_threshold: float = 0.03,
    yaw_cmd_threshold: float = 0.03,
    **kwargs,
) -> torch.Tensor:
    """Penalize yaw velocity during pure linear (straight-walk) commands.

    Activates only when the commanded yaw rate is near zero AND the
    commanded linear speed exceeds a threshold — i.e., the robot should
    be walking straight without turning.

    Uses instantaneous body-frame yaw rate.

    Args:
        env: The environment instance.
        lin_cmd_threshold: Min linear command norm to consider "walking".
        yaw_cmd_threshold: Max absolute yaw command to consider "near zero".

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    commands = env.command_manager.commands
    c_xy = commands[:, :2]
    wz_cmd = commands[:, 2]

    lin_cmd_norm = torch.linalg.norm(c_xy, dim=-1)

    # Straight-walk indicator: significant linear command AND low yaw command
    straight_mask = (
        (lin_cmd_norm > lin_cmd_threshold) & (torch.abs(wz_cmd) < yaw_cmd_threshold)
    ).float()

    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    ang_vel_body = quat_rotate_inverse(env.simulator.base_quat, ang_vel_world, w_last=True)
    wz = ang_vel_body[:, 2]

    return straight_mask * wz.square()


def _get_gait_avg_velocity_state(env):
    """Retrieve the GaitAveragedVelocity command term, or None if absent."""
    return env.command_manager.get_state("gait_avg_velocity")


def penalty_straight_walk_yaw_leakage_gait_avg(
    env,
    lin_cmd_threshold: float = 0.03,
    yaw_cmd_threshold: float = 0.03,
    instant_weight: float = 0.15,
    **kwargs,
) -> torch.Tensor:
    """Gait-averaged yaw-leakage penalty during straight-walk commands.

    A bipedal gait must yaw back and forth each step, so the *net* per-stride
    yaw drift is the quantity worth suppressing, not the instantaneous yaw
    rate. This penalizes the gait-cycle running mean of body-frame yaw rate
    (intra-stride oscillation cancels in the mean), plus a small instantaneous
    term for a dense per-step gradient and an amplitude sanity cap.

        penalty = mask * (wz_running_mean² + instant_weight * wz_instant²)

    The running mean is maintained by the ``gait_avg_velocity`` command term
    with a warmup guard (falls back to the previous full-cycle mean during the
    noisy early-stride window). Falls back to the instantaneous-only penalty if
    that term is not registered.

    Args:
        env: The environment instance.
        lin_cmd_threshold: Min linear command norm to consider "walking".
        yaw_cmd_threshold: Max absolute yaw command to consider "near zero".
        instant_weight: Weight of the small instantaneous companion term,
            relative to the averaged term (default 0.15).

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    commands = env.command_manager.commands
    c_xy = commands[:, :2]
    wz_cmd = commands[:, 2]
    lin_cmd_norm = torch.linalg.norm(c_xy, dim=-1)
    straight_mask = (
        (lin_cmd_norm > lin_cmd_threshold) & (torch.abs(wz_cmd) < yaw_cmd_threshold)
    ).float()

    state = _get_gait_avg_velocity_state(env)
    if state is None or state.wz_running_mean is None:
        # Fallback: instantaneous-only (matches non-averaged behavior).
        ang_vel_world = env.simulator.robot_root_states[:, 10:13]
        ang_vel_body = quat_rotate_inverse(
            env.simulator.base_quat, ang_vel_world, w_last=True
        )
        return straight_mask * ang_vel_body[:, 2].square()

    avg_term = state.wz_running_mean.square()
    instant_term = state.wz_instant.square()
    return straight_mask * (avg_term + instant_weight * instant_term)


def penalty_lateral_velocity_gait_avg(
    env,
    min_cmd_speed: float = 0.03,
    instant_weight: float = 0.15,
    **kwargs,
) -> torch.Tensor:
    """Gait-averaged lateral-velocity (orthogonal drift) penalty while walking.

    Weight shifts during a bipedal gait cause inherent side-to-side pelvis
    sway whose per-stride mean is ~zero; the net cross-track drift is the real
    target. Penalizes the gait-cycle running mean of the signed lateral
    (perpendicular-to-command) body velocity, plus a small instantaneous term
    for dense gradient and amplitude sanity.

        penalty = moving * (v_perp_running_mean² + instant_weight * v_perp_instant²)

    Falls back to the instantaneous quadratic lateral penalty if the
    ``gait_avg_velocity`` command term is not registered.

    Args:
        env: The environment instance.
        min_cmd_speed: Min commanded speed to activate directional decomposition.
        instant_weight: Weight of the instantaneous companion term, relative to
            the averaged term (default 0.15).

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    state = _get_gait_avg_velocity_state(env)
    if state is None or state.v_perp_running_mean is None:
        # Fallback: instantaneous quadratic lateral penalty.
        return penalty_lateral_velocity_quadratic(env, min_cmd_speed=min_cmd_speed)

    commands = env.command_manager.commands
    cmd_speed = torch.linalg.norm(commands[:, :2], dim=-1)
    moving = (cmd_speed > min_cmd_speed).float()

    avg_term = state.v_perp_running_mean.square()
    instant_term = state.v_perp_instant.square()
    return moving * (avg_term + instant_weight * instant_term)


def penalty_standing_ang_vel(
    env,
    lin_cmd_threshold: float = 0.03,
    yaw_cmd_threshold: float = 0.03,
    **kwargs,
) -> torch.Tensor:
    """Penalize angular velocity when all locomotion commands are near zero (standing).

    Activates only when both the commanded linear speed AND the commanded
    yaw rate are below their respective thresholds — i.e., the robot should
    be standing still without any rotation.

    Uses instantaneous body-frame yaw rate.

    Args:
        env: The environment instance.
        lin_cmd_threshold: Max linear command norm to consider "standing".
        yaw_cmd_threshold: Max absolute yaw command to consider "standing".

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    commands = env.command_manager.commands
    c_xy = commands[:, :2]
    wz_cmd = commands[:, 2]

    lin_cmd_norm = torch.linalg.norm(c_xy, dim=-1)

    # Standing indicator: both linear and yaw commands near zero
    stand_mask = (
        (lin_cmd_norm < lin_cmd_threshold) & (torch.abs(wz_cmd) < yaw_cmd_threshold)
    ).float()

    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    ang_vel_body = quat_rotate_inverse(env.simulator.base_quat, ang_vel_world, w_last=True)
    wz = ang_vel_body[:, 2]

    return stand_mask * wz.square()


def _get_heading_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract scalar yaw (heading) from w-last quaternion. Shape: [N]."""
    # Forward direction = R @ [1, 0, 0]
    q_vec = quat[:, :3]
    q_w = quat[:, 3]
    # Simplified quat_rotate for unit-x: fx = 2w²-1 + 2*qx², fy = 2*(qx*qy + w*qz)
    fx = 2.0 * (q_w * q_w + q_vec[:, 0] * q_vec[:, 0]) - 1.0
    fy = 2.0 * (q_vec[:, 0] * q_vec[:, 1] + q_w * q_vec[:, 2])
    return torch.atan2(fy, fx)


def penalty_heading_hold(
    env,
    lin_cmd_threshold: float = 0.05,
    yaw_cmd_threshold: float = 0.03,
    temperature: float = 0.02,
    warmup_steps: int = 10,
    **kwargs,
) -> torch.Tensor:
    """Penalize accumulated yaw drift from the heading at command-latch time.

    When the velocity command indicates straight walking (significant linear
    command, near-zero yaw command), the penalty latches the robot's heading
    at the moment the straight-walk regime starts. Subsequent steps penalize
    the squared angular distance from that latched heading.

    This is stronger than instantaneous yaw-rate penalties because it targets
    the *integrated* drift — the actual failure mode — rather than the rate.
    Transient intra-stride yaw oscillations (which average to zero) don't
    trigger this penalty much, but net drift accumulates and gets penalized
    increasingly.

    State is stored on ``env._heading_hold_state`` (lazy-initialized).

    The penalty uses the same soft sigmoid mask as the soft-gated variants,
    applied to the drift² term. A warmup window (default 10 steps) after
    latch suppresses the penalty while the robot transitions into straight
    walking, avoiding penalizing the initial alignment correction.

    Args:
        env: The environment instance.
        lin_cmd_threshold: Center of the linear command activation sigmoid.
        yaw_cmd_threshold: Center of the yaw command deactivation sigmoid.
        temperature: Sigmoid temperature controlling transition sharpness.
        warmup_steps: Steps after latch before penalty activates (grace period).

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    device = env.device
    N = env.num_envs

    # Lazy-initialize state buffers
    if not hasattr(env, "_heading_hold_state"):
        env._heading_hold_state = {
            "latched_yaw": torch.zeros(N, device=device),
            "is_latched": torch.zeros(N, dtype=torch.bool, device=device),
            "steps_since_latch": torch.zeros(N, dtype=torch.long, device=device),
            "prev_straight": torch.zeros(N, dtype=torch.bool, device=device),
        }

    state = env._heading_hold_state

    # Current commands
    commands = env.command_manager.commands
    c_xy = commands[:, :2]
    wz_cmd = commands[:, 2]
    lin_cmd_norm = torch.linalg.norm(c_xy, dim=-1)

    # Determine if currently in straight-walk regime (hard, for latching logic)
    is_straight = (lin_cmd_norm > lin_cmd_threshold) & (
        torch.abs(wz_cmd) < yaw_cmd_threshold
    )

    # Current heading
    current_yaw = _get_heading_from_quat(env.simulator.base_quat)

    # Detect rising edge: was NOT straight, now IS straight → latch
    rising_edge = is_straight & ~state["prev_straight"]

    # Detect env resets
    if hasattr(env, "reset_buf"):
        reset_mask = env.reset_buf.bool()
    else:
        reset_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Latch on rising edge
    if rising_edge.any():
        state["latched_yaw"][rising_edge] = current_yaw[rising_edge]
        state["is_latched"][rising_edge] = True
        state["steps_since_latch"][rising_edge] = 0

    # Invalidate on leaving straight-walk or reset
    invalidate = (~is_straight) | reset_mask
    if invalidate.any():
        state["is_latched"][invalidate] = False
        state["steps_since_latch"][invalidate] = 0

    # Increment step counter for latched envs
    state["steps_since_latch"][state["is_latched"]] += 1

    # Update previous state
    state["prev_straight"] = is_straight.clone()

    # Compute angular drift (wrapped to [-pi, pi])
    delta_yaw = current_yaw - state["latched_yaw"]
    delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

    # Soft mask (same as penalty_straight_walk_yaw_leakage_soft)
    lin_gate = torch.sigmoid((lin_cmd_norm - lin_cmd_threshold) / temperature)
    yaw_gate = 1.0 - torch.sigmoid(
        (torch.abs(wz_cmd) - yaw_cmd_threshold) / temperature
    )
    soft_mask = lin_gate * yaw_gate

    # Warmup mask: suppress for the first N steps after latch
    warmed_up = (state["steps_since_latch"] > warmup_steps).float()

    # Final penalty: only for latched, warmed-up envs
    active = state["is_latched"].float() * warmed_up

    return soft_mask * active * delta_yaw.square()


def penalty_ee_body_twist(env, **kwargs) -> torch.Tensor:
    """Penalize end-effector velocity (body-frame twist relative to reference frame).

    Uses ground-truth V_RE_E = J_RE_E @ q_dot_true (pelvis linvel not zeroed)
    when a controller is available. Otherwise computes the EE twist directly
    from rigid-body velocities (world-frame) transformed into the EE body frame
    relative to the ground-heading reference frame.

    Returns:
        Penalty tensor [num_envs] (non-negative, higher = worse).
    """
    for _, term in env.action_manager.iter_terms():
        controller = getattr(term, "base_controller", None)
        if controller is not None:
            twist_true = getattr(controller, "ee_body_twist_true", None)
            if twist_true is not None:
                return torch.sum(twist_true[0].square(), dim=-1) + torch.sum(twist_true[1].square(), dim=-1)

    # Fallback: compute EE twist from rigid body velocities when no controller
    # is available (e.g. baseline policy without GAC/H-inf).
    return _compute_ee_twist_from_rigid_body(env)


def _compute_ee_twist_from_rigid_body(env) -> torch.Tensor:
    """Compute EE body-frame twist relative to ground-heading frame from rigid body vels.

    This is the fallback for environments without a model-based controller (baseline).
    Computes V_RE_E (6D twist of EE relative to reference R, expressed in EE frame E)
    using the simulator's rigid body linear and angular velocities.

    The relative twist accounts for the position offset between reference and EE:
        v_rel_w = v_WE - (v_WR + ω_WR × (p_WE - p_WR))
        ω_rel_w = ω_WE - ω_WR
        V_RE_E = R_WE^T @ [v_rel_w; ω_rel_w]
    """
    arm_command_state = _get_arm_command_state(env)
    ee_indices = arm_command_state.ee_body_indices  # [2] (left, right)

    # EE world-frame velocities and positions
    ee_lin_vel_w = env.simulator._rigid_body_vel[:, ee_indices, :]      # [N, 2, 3]
    ee_ang_vel_w = env.simulator._rigid_body_ang_vel[:, ee_indices, :]  # [N, 2, 3]
    ee_pos_w = env.simulator._rigid_body_pos[:, ee_indices, :]          # [N, 2, 3]

    # Reference frame (ground-heading) state:
    root_states = env.simulator.robot_root_states
    ref_pos_w = root_states[:, :3].clone()         # pelvis position [N, 3]
    ref_pos_w[:, 2] = 0.0                          # ground-heading: z = 0
    ref_lin_vel_w = root_states[:, 7:10].clone()   # pelvis linear velocity [N, 3]
    ref_lin_vel_w[:, 2] = 0.0                      # ground-heading: no vertical motion
    # Ground-heading angular velocity: only yaw component
    ref_ang_vel_w = torch.zeros(env.num_envs, 3, device=env.device)
    ref_ang_vel_w[:, 2] = root_states[:, 10:13][:, 2]  # omega_z

    # EE quaternions for body-frame rotation
    ee_quat_w = env.simulator._rigid_body_rot[:, ee_indices, :]  # [N, 2, 4] xyzw

    penalty = torch.zeros(env.num_envs, device=env.device)

    for arm_idx in range(2):
        # Position offset from reference to EE
        p_RE_w = ee_pos_w[:, arm_idx] - ref_pos_w  # [N, 3]

        # Linear velocity of reference frame at EE location:
        # v_R_at_E = v_R + ω_R × (p_E - p_R)
        v_ref_at_ee = ref_lin_vel_w + torch.cross(ref_ang_vel_w, p_RE_w, dim=-1)

        # Relative velocities in world frame
        v_rel_w = ee_lin_vel_w[:, arm_idx] - v_ref_at_ee  # [N, 3]
        w_rel_w = ee_ang_vel_w[:, arm_idx] - ref_ang_vel_w  # [N, 3]

        # Transform to EE body frame: v_E = R_WE^T @ v_W
        v_rel_E = quat_rotate_inverse(ee_quat_w[:, arm_idx], v_rel_w, w_last=True)  # [N, 3]
        w_rel_E = quat_rotate_inverse(ee_quat_w[:, arm_idx], w_rel_w, w_last=True)  # [N, 3]

        # Sum of squared twist components
        penalty = penalty + v_rel_E.square().sum(dim=-1) + w_rel_E.square().sum(dim=-1)

    return penalty

