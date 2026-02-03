"""Video recording configuration types for holosoma simulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

from pydantic import Field
from typing_extensions import Annotated


@dataclass(frozen=True)
class SphericalCameraConfig:
    """Configures spherical coordinate camera positioning.

    Uses distance, azimuth, and elevation to position camera relative to tracked object.
    """

    type: Literal["spherical"] = "spherical"
    """Camera configuration type identifier."""

    distance: float = 3.0
    """Distance from the tracked object in meters."""

    azimuth: float = 90.0
    """Horizontal angle in degrees, where 0 = +X axis."""

    elevation: float = 20.0
    """Vertical angle in degrees, where 0 = horizontal."""


@dataclass(frozen=True)
class CartesianCameraConfig:
    """Configures cartesian offset camera positioning.

    Uses XYZ offsets to position camera relative to tracked object.
    """

    type: Literal["cartesian"] = "cartesian"
    """Camera configuration type identifier."""

    offset: list[float] = field(default_factory=lambda: [2.0, 2.0, 1.0])
    """Camera position offset from tracked object [x, y, z] in meters."""

    target_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.3])
    """Camera target offset from tracked object [x, y, z] in meters."""


@dataclass(frozen=True)
class FixedCameraConfig:
    """Configures fixed world position camera.

    Uses absolute world coordinates for camera position and target.
    """

    type: Literal["fixed"] = "fixed"
    """Camera configuration type identifier."""

    position: list[float] = field(default_factory=lambda: [5.0, 5.0, 3.0])
    """Absolute camera position in world coordinates [x, y, z]."""

    target: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    """Absolute camera target position in world coordinates [x, y, z]."""


# Discriminated union for camera configurations
CameraConfig = Annotated[
    Union[SphericalCameraConfig, CartesianCameraConfig, FixedCameraConfig], Field(discriminator="type")
]


@dataclass(frozen=True)
class VideoConfig:
    """Configures video recording across all holosoma simulators.

    Provides unified interface for video recording settings compatible with
    IsaacGym, MuJoCo, and IsaacSim simulators.
    """

    enabled: bool = True
    """Whether video recording is enabled."""

    interval: int = 10
    """Record video every N episodes. Set to 1 to record every episode."""

    width: int = 640
    """Video frame width in pixels."""

    height: int = 360
    """Video frame height in pixels."""

    playback_rate: float = 1.0
    """Video playback speed relative to the simulation speed (frames captured at control frequency).
        1.0 = real-time, 2.0 = 2x faster, 0.5 = slow motion.
    """

    camera_smoothing: float = 0.95
    """Camera smoothing factor for reducing shake (0.0 = no smoothing, 0.99 = very smooth)."""

    output_format: str = "h264"
    """Video output format ('mp4' or 'h264'). 'h264' provides better browser compatibility."""

    save_dir: str | None = None
    """Directory to save video files. If None, uses simulator's default (experiment directory)."""

    upload_to_wandb: bool = True
    """Whether to upload videos to wandb for logging if wandb is enabled."""

    show_command_overlay: bool = True
    """Whether to overlay robot command information on video frames."""

    record_env_id: int = 0
    """Which environment to record (for multi-environment simulations)."""

    camera: CameraConfig = field(default_factory=CartesianCameraConfig)
    """Camera configuration with automatic type discrimination based on 'type' field."""

    tracking_body_name: str | None = None
    """Name of the robot body to track. If None or 'auto', will try default body names in order."""

    use_recording_thread: bool = False
    """Whether to use background thread for video recording (MuJoCo-only)"""

    recording_thread_fps: float = 30.0
    """Target FPS for threaded recording."""

    recording_queue_size: int = 100
    """Maximum size of the recording queue for threaded mode."""

    vertical_fov: float = 45.0
    """Camera vertical field of view in degrees. Used for approximate consistency across simulators."""

    def get_aspect_ratio(self) -> float:
        """Calculate aspect ratio from video dimensions."""
        return self.width / self.height

    def get_horizontal_fov(self) -> float:
        """Convert vertical FOV to horizontal FOV for simulators that need it."""
        # lazy imports since this is a config type file
        import math

        aspect_ratio = self.get_aspect_ratio()
        v_fov_rad = math.radians(self.vertical_fov)
        h_fov_rad = 2 * math.atan(math.tan(v_fov_rad / 2) * aspect_ratio)
        return math.degrees(h_fov_rad)
