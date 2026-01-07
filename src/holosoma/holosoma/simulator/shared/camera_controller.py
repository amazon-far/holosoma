"""Unified camera positioning for viewer and video recording.

This module provides a shared camera controller that supports multiple camera modes
(Fixed, Spherical, Cartesian) with optional robot tracking and camera smoothing.
The controller provides consistent camera behavior for both interactive viewer
and video recording use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from holosoma.config_types.video import (
    CameraConfig,
    CartesianCameraConfig,
    FixedCameraConfig,
    SphericalCameraConfig,
)

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


@dataclass
class CameraParameters:
    """Standardized camera parameters for simulator APIs.

    This class provides a uniform interface for camera positioning that can be
    used across different simulator backends (MuJoCo, IsaacGym, IsaacSim).

    Parameters
    ----------
    position : tuple[float, float, float]
        Camera position in world coordinates (x, y, z).
    target : tuple[float, float, float]
        Camera target point in world coordinates (x, y, z).
    distance : float
        Distance from camera to target in meters.
    azimuth : float
        Horizontal angle in degrees (0 = +X axis).
    elevation : float
        Vertical angle in degrees (0 = horizontal).
    """

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


class CameraController:
    """Unified camera positioning for viewer and video recording.

    This class provides consistent camera tracking behavior across all use cases.
    It supports multiple camera modes (Fixed, Spherical, Cartesian) with optional
    robot tracking and camera smoothing.

    All camera-specific settings (smoothing, tracking_body_name) come from the
    camera config itself, keeping the interface clean and unambiguous.

    Parameters
    ----------
    config : CameraConfig
        Camera configuration (FixedCameraConfig | SphericalCameraConfig | CartesianCameraConfig).
        Config contains all camera-specific settings including smoothing and tracking_body_name.
    simulator : BaseSimulator
        Reference to simulator for accessing robot state.

    Attributes
    ----------
    config : CameraConfig
        Camera configuration.
    simulator : BaseSimulator
        Reference to simulator.
    robot_body_id : int | None
        Resolved robot body ID for tracking (None for fixed cameras).
    smoothed_cam_pos : tuple[float, float, float] | None
        Smoothed camera position for reducing shake.
    smoothed_cam_target : tuple[float, float, float] | None
        Smoothed camera target for reducing shake.
    """

    # Default tracking body names to try in order of preference
    DEFAULT_TRACKING_BODY_NAMES = [
        "Trunk",  # Common in humanoid robots
        "torso",  # Standard anatomical naming
        "base_link",  # ROS standard naming convention
        "Waist",  # Alternative trunk naming
        "pelvis",  # Anatomical naming for hip area
        "hip",  # Simple hip naming
        "base",  # Generic base naming
    ]

    def __init__(
        self,
        config: CameraConfig,
        simulator: BaseSimulator,
    ):
        """Initialize camera controller.

        Parameters
        ----------
        config : CameraConfig
            Camera configuration.
        simulator : BaseSimulator
            Reference to simulator for accessing robot state.
        """
        self.config = config
        self.simulator = simulator

        # State
        self.robot_body_id: int | None = None
        self.smoothed_cam_pos: tuple[float, float, float] | None = None
        self.smoothed_cam_target: tuple[float, float, float] | None = None

    def update(self, robot_pos: tuple[float, float, float] | None = None) -> CameraParameters:
        """Calculate current camera parameters.

        Parameters
        ----------
        robot_pos : tuple[float, float, float] | None, default=None
            Optional pre-fetched robot position. If None, will fetch from simulator.

        Returns
        -------
        CameraParameters
            Calculated camera parameters for simulator API.

        Raises
        ------
        RuntimeError
            If robot body not resolved for tracking camera modes.
        ValueError
            If unsupported camera config type is provided.
        """
        config: CameraConfig = self.config

        if isinstance(config, FixedCameraConfig):
            # Fixed camera position - no robot tracking
            position: tuple[float, float, float] = (
                float(config.position[0]),
                float(config.position[1]),
                float(config.position[2]),
            )
            target: tuple[float, float, float] = (
                float(config.target[0]),
                float(config.target[1]),
                float(config.target[2]),
            )
            distance, azimuth, elevation = self._cartesian_to_spherical(position, target)

            return CameraParameters(
                position=position,
                target=target,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
            )

        if self.robot_body_id is None and isinstance(config, (SphericalCameraConfig, CartesianCameraConfig)):
            self._resolve_tracking_body()

        if isinstance(config, SphericalCameraConfig):
            # Spherical camera positioning relative to robot
            if self.robot_body_id is None:
                raise RuntimeError("Robot body not resolved for spherical camera mode")

            # Get robot position
            if robot_pos is None:
                robot_pos = self._get_robot_position()

            # Apply smoothing to target using camera's smoothing parameter
            # Note: SphericalCameraConfig will need smoothing field added
            smoothing = getattr(config, "smoothing", 0.95)  # Default for backward compatibility

            if self.smoothed_cam_target is None:
                self.smoothed_cam_target = robot_pos
            else:
                self.smoothed_cam_target = (
                    smoothing * self.smoothed_cam_target[0] + (1 - smoothing) * robot_pos[0],
                    smoothing * self.smoothed_cam_target[1] + (1 - smoothing) * robot_pos[1],
                    smoothing * self.smoothed_cam_target[2] + (1 - smoothing) * robot_pos[2],
                )

            # Calculate position from spherical coordinates
            position = self._spherical_to_cartesian(
                config.distance, config.azimuth, config.elevation, self.smoothed_cam_target
            )

            return CameraParameters(
                position=position,
                target=self.smoothed_cam_target,
                distance=config.distance,
                azimuth=config.azimuth,
                elevation=config.elevation,
            )

        if isinstance(config, CartesianCameraConfig):
            # Cartesian offset positioning relative to robot
            if self.robot_body_id is None:
                raise RuntimeError("Robot body not resolved for cartesian camera mode")

            # Get robot position
            if robot_pos is None:
                robot_pos = self._get_robot_position()

            # Calculate desired camera positions
            target_cam_pos: tuple[float, float, float] = (
                robot_pos[0] + config.offset[0],
                robot_pos[1] + config.offset[1],
                robot_pos[2] + config.offset[2],
            )
            target_cam_target: tuple[float, float, float] = (
                robot_pos[0] + config.target_offset[0],
                robot_pos[1] + config.target_offset[1],
                robot_pos[2] + config.target_offset[2],
            )

            # Apply smoothing
            smoothing = getattr(config, "smoothing", 0.95)  # Default for backward compatibility
            smoothed_cam_pos, smoothed_cam_target = self._apply_camera_smoothing(
                target_cam_pos, target_cam_target, smoothing
            )

            # Convert to spherical for simulators that need it
            distance, azimuth, elevation = self._cartesian_to_spherical(smoothed_cam_pos, smoothed_cam_target)

            return CameraParameters(
                position=smoothed_cam_pos,
                target=smoothed_cam_target,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
            )

        raise ValueError(f"Unsupported camera config type: {type(self.config)}")

    def reset(self) -> None:
        """Reset camera smoothing state.

        This should be called at the start of new episodes to prevent
        smoothing artifacts from carrying over.
        """
        self.smoothed_cam_pos = None
        self.smoothed_cam_target = None

    def _resolve_tracking_body(self) -> None:
        """Resolve robot tracking body using automatic fallback logic.

        Tries body names in order using the simulator's find_rigid_body_indice()
        method. First tries the explicitly configured tracking_body_name (if set),
        then falls back to default body names.

        Raises
        ------
        ValueError
            If no suitable tracking body is found.
        """
        # Get tracking_body_name from config (with fallback for configs without this field)
        tracking_body_name = getattr(self.config, "tracking_body_name", "auto")

        # Build list of names to try: explicit name first (if set), then defaults
        if tracking_body_name and tracking_body_name != "auto":
            names_to_try = [tracking_body_name] + self.DEFAULT_TRACKING_BODY_NAMES
        else:
            names_to_try = self.DEFAULT_TRACKING_BODY_NAMES

        # Try each name in order until one works
        for name in names_to_try:
            try:
                self.robot_body_id = self.simulator.find_rigid_body_indice(name)
                logger.debug(f"Camera controller resolved tracking body: {name}")
                return
            except Exception:  # noqa: S112, PERF203
                continue

        # If no names worked, fail with helpful error
        raise ValueError(
            f"No suitable tracking body found. Tried: {names_to_try}. "
            f"Please set tracking_body_name explicitly to a valid body name in your robot model."
        )

    def _get_robot_position(self) -> tuple[float, float, float]:
        """Get current robot position from simulator.

        Uses the unified BaseSimulator interface to get robot position.
        For video recording, uses the record_env_id. For viewer, typically uses env 0.

        Returns
        -------
        tuple[float, float, float]
            Robot position (x, y, z) in world coordinates.
        """
        # For now, default to env 0 (viewer case)
        # Video recorder will need to pass record_env_id if different
        env_id = 0
        robot_pos = self.simulator.robot_root_states[env_id, :3]  # [x, y, z]
        return float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])

    def _apply_camera_smoothing(
        self,
        target_cam_pos: tuple[float, float, float],
        target_cam_target: tuple[float, float, float],
        smoothing: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Apply exponential moving average smoothing to camera positions.

        Parameters
        ----------
        target_cam_pos : tuple[float, float, float]
            Target camera position (x, y, z).
        target_cam_target : tuple[float, float, float]
            Target camera target (x, y, z).
        smoothing : float
            Smoothing factor (0.0 = no smoothing, 0.99 = very smooth).

        Returns
        -------
        tuple[tuple[float, float, float], tuple[float, float, float]]
            Tuple of (smoothed_camera_position, smoothed_camera_target).
        """
        # Apply smoothing to reduce camera shaking
        if self.smoothed_cam_pos is None or self.smoothed_cam_target is None:
            # Initialize smoothed positions on first update
            self.smoothed_cam_pos = target_cam_pos
            self.smoothed_cam_target = target_cam_target
        else:
            # Smooth camera position using exponential moving average
            self.smoothed_cam_pos = (
                smoothing * self.smoothed_cam_pos[0] + (1 - smoothing) * target_cam_pos[0],
                smoothing * self.smoothed_cam_pos[1] + (1 - smoothing) * target_cam_pos[1],
                smoothing * self.smoothed_cam_pos[2] + (1 - smoothing) * target_cam_pos[2],
            )

            # Smooth camera target
            self.smoothed_cam_target = (
                smoothing * self.smoothed_cam_target[0] + (1 - smoothing) * target_cam_target[0],
                smoothing * self.smoothed_cam_target[1] + (1 - smoothing) * target_cam_target[1],
                smoothing * self.smoothed_cam_target[2] + (1 - smoothing) * target_cam_target[2],
            )

        return self.smoothed_cam_pos, self.smoothed_cam_target

    @staticmethod
    def _cartesian_to_spherical(
        position: tuple[float, float, float], target: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Convert cartesian position to spherical coordinates relative to target.

        Parameters
        ----------
        position : tuple[float, float, float]
            Camera position (x, y, z).
        target : tuple[float, float, float]
            Camera target position (x, y, z).

        Returns
        -------
        tuple[float, float, float]
            Spherical coordinates (distance, azimuth_degrees, elevation_degrees).
        """
        offset = np.array(position) - np.array(target)
        distance = np.linalg.norm(offset)
        azimuth = np.degrees(np.arctan2(offset[1], offset[0]))
        elevation = np.degrees(np.arcsin(offset[2] / distance)) if distance > 0 else 0
        return float(distance), float(azimuth), float(elevation)

    @staticmethod
    def _spherical_to_cartesian(
        distance: float, azimuth_degrees: float, elevation_degrees: float, target: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Convert spherical coordinates to cartesian position relative to target.

        Parameters
        ----------
        distance : float
            Distance from target in meters.
        azimuth_degrees : float
            Horizontal angle in degrees (0 = +X axis).
        elevation_degrees : float
            Vertical angle in degrees (0 = horizontal).
        target : tuple[float, float, float]
            Camera target position (x, y, z).

        Returns
        -------
        tuple[float, float, float]
            Camera position (x, y, z) in world coordinates.
        """
        azimuth_rad = np.radians(azimuth_degrees)
        elevation_rad = np.radians(elevation_degrees)

        offset_x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        offset_y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        offset_z = distance * np.sin(elevation_rad)

        return (
            float(target[0] + offset_x),
            float(target[1] + offset_y),
            float(target[2] + offset_z),
        )
