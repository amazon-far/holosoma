"""Configuration types for viser visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from holosoma_retargeting.xsens.morphology_adaptation import XsensRootMotionConfig


@dataclass(frozen=True)
class ViserConfig:
    """Configuration for viser player visualization.

    This follows the pattern from holosoma's config_types.
    Uses a flat structure with default values.
    """

    qpos_npz: str = "rt_results/OMOMO_new/box_parallel/sub8_largebox_051_original.npz"
    """Path to .npz file with qpos data."""

    robot_urdf: str = "models/g1/g1_29dof.urdf"
    """Path to robot URDF file."""

    object_urdf: str | None = None
    """Path to object URDF file (optional)."""

    fps: int = 30
    """Frames per second for playback."""

    assume_object_in_qpos: bool = True
    """Whether object pose is included in qpos array."""

    loop: bool = False
    """Whether to loop playback."""

    show_meshes: bool = True
    """Whether to show mesh visualizations."""

    camera_follow: bool = False
    """Whether the interactive camera initially follows the displayed actors."""

    grid_width: float | None = None
    """Optional minimum grid width. None derives it entirely from motion bounds."""

    grid_height: float | None = None
    """Optional minimum grid height. None derives it entirely from motion bounds."""

    grid_padding: float = 1.0
    """Horizontal padding around the complete motion bounds, in metres."""

    visual_fps_multiplier: int = 2
    """Visual FPS multiplier for interpolation."""

    playback_speed: float = 1.0
    """Initial playback speed relative to real time."""

    record_video: bool = False
    """Whether to record the Viser playback to a video file."""

    record_path: str | None = None
    """Optional recording output path. Defaults beside the source motion with a .mp4 suffix."""

    record_width: int = 1280
    """Rendered recording width in pixels."""

    record_height: int = 720
    """Rendered recording height in pixels."""

    record_fps: int | None = None
    """FPS for the recorded video. Defaults to the motion FPS."""

    record_start_frame: int = 0
    """First source frame to record."""

    record_end_frame: int | None = None
    """Last source frame to record, inclusive. Defaults to the final frame."""

    record_stride: int = 1
    """Source-frame stride for recording."""

    record_connect_timeout: float = 120.0
    """Seconds to wait for a browser client before recording."""

    record_start_delay: float = 3.0
    """Seconds to wait after a client connects before recording starts."""

    record_settle_time: float = 0.0
    """Seconds to wait after each scene update before capturing a rendered frame."""

    record_warmup_renders: int = 0
    """Number of throwaway renders after each scene update before saving the frame."""

    record_transport_format: str = "jpeg"
    """Browser render transport format: jpeg or png."""

    record_exit_after: bool = False
    """Exit the process after recording completes."""

    min_fps: int = 1
    """Minimum FPS setting."""

    max_fps: int = 240
    """Maximum FPS setting."""

    min_interp_mult: int = 1
    """Minimum interpolation multiplier."""

    max_interp_mult: int = 8
    """Maximum interpolation multiplier."""


@dataclass(frozen=True)
class XsensViserConfig(ViserConfig):
    """Viser configuration for Xsens and combined robot/Xsens playback."""

    actor_modes: tuple[Literal["robot", "xsens", "g1_xsens", "all"], ...] = ("robot",)
    """Actors to compose in one scene. ``all`` expands to all three actor types."""

    xsens_hdf5: str | None = None
    """Xsens HDF5 motion shared by the xsens and g1_xsens actors."""

    xsens_usd: str | None = None
    """Recording-specific Xsens USDA override; defaults to the HDF5 sibling model."""

    g1_xsens_usd: str | None = None
    """G1-proportioned Xsens USDA override; defaults to the packaged demo result."""

    g1_xsens_root_motion: XsensRootMotionConfig = field(default_factory=XsensRootMotionConfig)
    """Floating-base translation policy for the G1-proportioned Xsens actor."""

    actor_spacing_m: float = 2.0
    """Lateral center-to-center spacing for the centered side-by-side actor layout."""

    xsens_target_fps: float | None = None
    """Optional HDF5 pre-sampling rate. None preserves the native timestamps."""

    xsens_frame_indices: tuple[int, ...] | None = None
    """Sparse post-resampling frames to play as a uniformly timed storyboard."""

    show_xsens_meshes: bool = True
    """Whether Xsens avatar meshes are initially visible."""

    show_xsens_landmarks: bool = False
    """Whether calibrated Xsens landmarks are initially visible."""

    show_tennis_racket: bool = True
    """Whether the tracked tennis racket is initially visible."""
