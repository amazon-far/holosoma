"""Viewer configuration types for holosoma simulators."""

from __future__ import annotations

from dataclasses import dataclass

from holosoma.config_types.video import CameraConfig


@dataclass(frozen=True)
class ViewerConfig:
    """Configuration for interactive viewer camera behavior.

    This is separate from video recording camera config to allow independent
    configuration of viewer and recording cameras.

    Parameters
    ----------
    enable_tracking : bool, default=False
        Enable camera tracking in the interactive viewer.
    camera : CameraConfig | None, default=None
        Camera configuration (FixedCameraConfig | SphericalCameraConfig | CartesianCameraConfig).
        Must be provided if enable_tracking=True.

    Examples
    --------
    Enable viewer camera with default spherical tracking:

    >>> from holosoma.config_types.video import SphericalCameraConfig
    >>> viewer = ViewerConfig(
    ...     enable_tracking=True,
    ...     camera=SphericalCameraConfig(),  # Uses all defaults
    ... )

    Enable viewer camera with custom settings:

    >>> viewer = ViewerConfig(
    ...     enable_tracking=True,
    ...     camera=SphericalCameraConfig(
    ...         distance=5.0,
    ...         azimuth=90.0,
    ...         elevation=30.0,
    ...         smoothing=0.9,
    ...         tracking_body_name="#base",
    ...     ),
    ... )
    """

    enable_tracking: bool = False
    """Enable camera tracking in the interactive viewer."""

    camera: CameraConfig | None = None
    """Camera configuration. Must be provided if enable_tracking=True."""
