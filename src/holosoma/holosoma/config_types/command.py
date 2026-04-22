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

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    start_at_timestep_zero_prob: float = 0.2
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.95
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    enable_default_pose_prepend: bool = True
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = True
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)


@dataclass(frozen=True)
class MultiMotionConfig:
    """Configuration for multi-motion training (PhUMA).

    Instead of a single motion file, loads a directory of .npz motion files
    and maintains a pool of motions on GPU. Each env tracks a randomly
    assigned motion from the pool.
    """

    motion_dir: str
    """Directory containing .npz motion files (searched recursively)."""

    split_file: str = ""
    """Optional path to a split file (one relative path per line, no extension).
    Only motions listed in this file will be used. If empty, all .npz files are used."""

    pool_size: int = 256
    """Number of motions to hold on GPU simultaneously."""

    pool_ordered: bool = False
    """If True, load motion files into pool slots in sorted-path order, so that
    `motion_index == sorted_file_index`. Requires `pool_size == len(files)`.
    Used when motion_index must line up with an external ordering (e.g. terrain
    tiles in multi-mesh training)."""

    max_motions: int = 0
    """Cap the number of discovered motion files to the first N (sorted). 0
    (default) keeps all. Useful for quick tests. Must agree with
    `TerrainTermCfg.obj_max_tiles` when `bind_terrain_tiles=True`."""

    bind_terrain_tiles: bool = False
    """If True, at reset time bind each env to the grid cell whose column matches
    the sampled motion_index. Requires the active terrain to be a multi-mesh
    load_obj terrain (i.e. `terrain.terrain_term.obj_dir` is set). A random row
    within the grid is chosen per-env each reset, and `scene.env_origins` is
    updated accordingly so the motion plays on the matching scene mesh tile.
    Must be used with `pool_ordered=True`."""

    resample_interval: int = 0
    """Resample the motion pool from disk every N env steps (0 = never).
    When triggered, the entire pool is replaced with new random motions.
    Envs continue with their current motion until their next reset."""

    min_motion_length: int = 30
    """Minimum motion length in frames. Shorter motions are skipped."""

    body_name_ref: list[str] = field(default_factory=list)
    """Body name of the reference frame (e.g., torso_link)."""

    body_names_to_track: list[str] = field(default_factory=list)
    """Key body names to track, used for reward/termination computation."""

    start_at_timestep_zero_prob: float = 0.2
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.95
    """When starting at timestep 0, probability of freezing motion counter at 0."""

    enable_default_pose_prepend: bool = False
    """Multi-motion: default pose prepend is disabled by default."""

    enable_default_pose_append: bool = False
    """Multi-motion: default pose append is disabled by default."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration of pre-appended interpolation phase."""

    default_pose_append_duration_s: float = 2.0
    """Duration of post-appended interpolation phase."""

    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)
