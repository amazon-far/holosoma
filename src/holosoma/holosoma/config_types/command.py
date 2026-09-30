"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

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

    shard_motions: bool = False
    """When True and using motion_dir with multi-GPU training, each GPU loads only its
    shard of the motion files (round-robin by global rank). Reduces memory and load time
    proportionally to the number of GPUs. Each GPU's environments sample only from their
    local shard."""

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    start_at_timestep_zero_prob: float = 0.0
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    default_pose_transition_strategy: Literal["none", "interpolation", "learned"] = "none"
    """Strategy for the default-to-motion-start transition at the beginning of each clip:
      - 'none': no special handling (default, backward compatible).
      - 'interpolation': prepend interpolated frames from default standing pose to motion
        start. The motion data is physically augmented with smooth transition frames.
        Uses default_pose_prepend_duration_s for timing.
      - 'learned': at episode resets starting at frame 0, the robot is initialized in the
        default standing pose but the motion command starts at frame 0 of the original clip.
        The policy must learn the transition autonomously without an explicit trajectory.
    When set to 'interpolation', enable_default_pose_prepend is implicitly enabled."""

    enable_default_pose_prepend: bool = False
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track.
    Also enabled implicitly when default_pose_transition_strategy='interpolation'."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True or strategy is 'interpolation'."""

    enable_default_pose_append: bool = False
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    recompute_velocities_from_positions: bool = False
    """Recompute joint and body velocities from positions within each source clip."""

    # adaptive sampling parameters
    adaptive_kernel_size: int = 1
    """Kernel size for smoothing failure bin counts in adaptive sampling."""

    adaptive_lambda: float = 0.8
    """Exponential decay factor for the non-causal smoothing kernel."""

    adaptive_alpha: float = 0.001
    """EMA coefficient used when accumulating per-bin failure counts."""

    adaptive_uniform_ratio: float = 0.1
    """Fraction of uniform exploration mixed into adaptive distributions."""

    motion_sampling_top1_prob_cap: float = 0.0
    """Maximum per-clip sampling probability; 0 disables the cap."""

    motion_exclude_filename_substrings: tuple[str, ...] = ()
    """Case-sensitive filename substrings to exclude from a motion directory."""

    failure_weighted_sampler: bool = False
    """Use per-bin failure rates instead of failure counts in the two-layer sampler."""

    failure_rate_max_over_mean: float = 200.0
    """Cap each failure rate at this multiple of the mean valid-bin rate."""

    failure_counts_monotonic: bool = False
    """Use monotonic episode/failure counters instead of decayed counts."""

    # multi-motion adaptive sampling
    adaptive_motion_weighting: Literal["uniform", "failure"] = "uniform"
    """How motion_id is sampled at reset when training on a multi-clip dataset:
      - 'uniform' (default, backward compatible): uniform random across clips, regardless of failure rate.
      - 'failure': multinomial weighted by per-clip failure statistics.
    """

    adaptive_phase_per_motion: bool = False
    """When True, the adaptive timestep sampler is per-motion: bins are partitioned per clip,
    failure attribution is keyed by (motion_id, bin_within_motion), and phase sampling is conditional
    on the chosen motion_id. Only meaningful when use_adaptive_timesteps_sampler is True."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)
