"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_dir: str = ""
    """Directory (or comma-separated directories) of .npz motion files.
    When non-empty, takes precedence over motion_file."""

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    start_at_timestep_zero_prob: float = 0.0
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    enable_default_pose_prepend: bool = False
    """If True, prepend interpolated frames from the default pose into the motion's first frame,
    giving the policy a trackable lead-in. Every clip gets its own.
    IsaacSim only: the splice needs a backend whose state write runs forward kinematics."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the prepended interpolation phase.
    Only used if enable_default_pose_prepend is True. Rounded to a whole number of policy steps;
    a value that is not a multiple of dt logs a warning and the rounded duration is used."""

    enable_default_pose_append: bool = False
    """If True, append interpolated frames from the motion's last frame back to the default pose,
    giving the policy a trackable lead-out. Every clip gets its own.
    IsaacSim only: the splice needs a backend whose state write runs forward kinematics."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the appended interpolation phase.
    Only used if enable_default_pose_append is True. Rounded to a whole number of policy steps;
    a value that is not a multiple of dt logs a warning and the rounded duration is used."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)
