"""Configuration types for viser visualization."""

from __future__ import annotations

from dataclasses import dataclass


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

    grid_width: float = 8.0
    """Grid width for visualization."""

    grid_height: float = 8.0
    """Grid height for visualization."""

    visual_fps_multiplier: int = 2
    """Visual FPS multiplier for interpolation."""

    record_video: bool = False
    """Whether to record the Viser playback to a video file."""

    record_path: str = "viser_player_recording.mp4"
    """Output path for recorded video. Extension may be .mp4 or .gif."""

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
