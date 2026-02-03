"""IsaacSim-specific video recording implementation.

This module provides the IsaacSim implementation of the video recording interface,
using omni.replicator.core for efficient rendering and supporting both synchronous
and asynchronous recording modes with full feature parity to IsaacGym and MuJoCo.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

import torch
from loguru import logger


from isaaclab.utils.math import (
    create_rotation_matrix_from_view,
    quat_from_matrix,
)

from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim


class IsaacSimVideoRecorder(VideoRecorderInterface):
    """IsaacSim-specific video recording implementation with replicator integration.

    This class implements video recording for IsaacSim simulations using omni.replicator.core
    for efficient rendering. It provides full feature parity with IsaacGym and MuJoCo
    implementations while leveraging IsaacSim's native rendering capabilities.

    The implementation supports:
    - All camera modes (spherical, cartesian, fixed)
    - Robot tracking with smoothing
    - Command overlays
    - Both synchronous and threaded recording
    - Unified video encoding and wandb integration

    Parameters
    ----------
    config : VideoConfig
        Video recording configuration settings.
    simulator : IsaacSim
        Reference to the IsaacSim simulator instance.
    """

    def __init__(self, config, simulator: IsaacSim) -> None:
        """Initialize IsaacSim video recorder.

        Parameters
        ----------
        config : VideoConfig
            Video recording configuration settings.
        simulator : IsaacSim
            The IsaacSim simulator instance.
        """
        super().__init__(config, simulator)

        # Override typing for mypy
        self.simulator: IsaacSim = simulator

        # IsaacSim-specific attributes using replicator
        self._render_product = None
        self._rgb_annotator = None
        self._camera_prim_path: str | None = None

        logger.info(
            f"IsaacSim video recorder initialized - {'threaded' if config.use_recording_thread else 'synchronous'} mode"
        )
        logger.info(f"Camera config: {config.camera.type}")
        logger.info(f"Video dimensions: {config.width}x{config.height}")

    def setup_recording(self) -> None:
        """Initialize IsaacSim video recording using omni.replicator.core.

        Creates a USD camera prim and replicator render product for video capture,
        following the same pattern used in IsaacLab environments.

        Raises
        ------
        RuntimeError
            If camera creation fails or replicator setup encounters issues.
        """
        super().setup_recording()

        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        # Create camera prim path for recording environment
        self._camera_prim_path = f"/World/envs/env_{self.config.record_env_id}/VideoCamera"

        # Create camera prim directly in USD stage
        stage = omni.usd.get_context().get_stage()
        self.camera_prim = UsdGeom.Camera.Define(stage, self._camera_prim_path)

        # Calculate camera parameters from config vertical FOV
        # For now, given different camera APIs, this is approximately similar across simulators...
        aspect_ratio = self.config.get_aspect_ratio()
        vertical_aperture = 24.0  # Standard 35mm equivalent
        horizontal_aperture = vertical_aperture * aspect_ratio

        # Calculate focal length
        focal_length = self._calculate_focal_length_from_fov(
            self.config.vertical_fov,
            vertical_aperture,  # Use vertical aperture for vertical FOV
        )

        # Set calculated camera properties
        self.camera_prim.GetFocalLengthAttr().Set(focal_length)
        self.camera_prim.GetClippingRangeAttr().Set((0.1, 1000.0))
        self.camera_prim.GetHorizontalApertureAttr().Set(horizontal_aperture)
        self.camera_prim.GetVerticalApertureAttr().Set(vertical_aperture)

        # Create a view for the sensor
        from isaacsim.core.prims import XFormPrim

        self._view = XFormPrim(self._camera_prim_path, reset_xform_properties=True)

        self._view.initialize()

        # Create render product using replicator
        resolution = (self.config.width, self.config.height)
        self._render_product = rep.create.render_product(self._camera_prim_path, resolution)

        # Create RGB annotator
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
            "rgb", device=self.simulator.device, do_array_copy=False
        )
        self._rgb_annotator.attach([self._render_product])

        logger.debug(f"Created replicator camera at: {self._camera_prim_path}")
        logger.debug(f"Render product resolution: {resolution}")
        logger.debug("IsaacSim video recording setup completed")

    def _capture_frame_impl(self) -> None:
        """Capture frame using replicator annotator.

        Updates camera position to track the robot, captures frame using replicator,
        applies command overlays, and stores the frame in the recording buffer.

        Raises
        ------
        RuntimeError
            If frame capture fails due to rendering or annotator issues.
        """
        if self._rgb_annotator is None:
            logger.warning("Cannot capture frame - annotator not initialized")
            return

        logger.debug(f"Recording frame at step {getattr(self.simulator, '_sim_step_counter', 0)}")

        try:
            # Update camera position to track robot using shared logic
            self._update_camera_position()

            # Get RGB data from replicator annotator
            rgb_data = self._rgb_annotator.get_data()

            # Convert to numpy array (handle bytes, numpy arrays, and warp arrays)
            if isinstance(rgb_data, np.ndarray):
                # Already a numpy array
                pass  # rgb_data is already in the correct format
            elif hasattr(rgb_data, "numpy"):
                # Warp array (when do_array_copy=False) - convert to numpy
                rgb_data = rgb_data.numpy()
            else:
                # Bytes data (default behavior) - convert to array
                rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)

            # note: initially the renderer is warming up and returns empty data
            if rgb_data.size == 0:
                rgb_array = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            else:
                rgb_array = rgb_data[:, :, :3]

            # Apply command overlay using shared logic
            frame_with_overlay = self._apply_command_overlay(rgb_array)

            # Add frame to buffer using shared method
            self._add_frame(frame_with_overlay)

        except Exception as e:
            raise RuntimeError(f"IsaacSim frame capture failed: {e}") from e

    def _update_camera_position(self) -> None:
        """Update camera position to track the robot.

        Uses the shared camera calculation system for unified camera positioning
        across all camera modes, then applies the results to the USD camera prim.
        """
        if self._camera_prim_path is None:
            return

        try:
            import omni.usd

            # Use shared camera calculation method
            camera_params = self._calculate_camera_parameters()

            # Extract position and target from shared calculation
            position = torch.tensor([camera_params["position"]], device=self.simulator.device, dtype=torch.float32)
            target = torch.tensor([camera_params["target"]], device=self.simulator.device, dtype=torch.float32)

            up_axis = "Z"
            orientations = quat_from_matrix(
                create_rotation_matrix_from_view(position, target, up_axis, device=self.simulator.device)
            )

            self._view.set_world_poses(
                position,
                orientations,
                torch.tensor([self.config.record_env_id], device=self.simulator.device, dtype=torch.int32),
            )

        except Exception as e:
            logger.warning(f"Failed to update camera position: {e}")

    def cleanup(self) -> None:
        """Clean up video recording resources.

        Releases replicator resources and clears frame buffers.
        """
        super().cleanup()

        # Clean up replicator resources
        if self._rgb_annotator is not None:
            try:
                # Detach annotator from render product
                if self._render_product is not None:
                    self._rgb_annotator.detach([self._render_product])
                self._rgb_annotator = None
            except Exception as e:
                logger.warning(f"Error cleaning up RGB annotator: {e}")

        # Clear render product
        if self._render_product is not None:
            try:
                # Render product cleanup is handled by replicator
                self._render_product = None
            except Exception as e:
                logger.warning(f"Error cleaning up render product: {e}")

        # Clear camera prim path
        self._camera_prim_path = None

        logger.debug("IsaacSim video recorder cleanup completed")

    def _start_persistent_thread(self) -> None:
        """Start persistent recording thread for IsaacSim.

        Unlike IsaacGym which doesn't support threading, IsaacSim can handle
        threaded recording similar to MuJoCo.
        """
        if self.thread_active:
            logger.warning("Video recording thread already active")
            return

        logger.debug("Starting IsaacSim video recording thread")

        self.recording_thread = threading.Thread(
            target=self._recording_thread_worker, daemon=True, name="IsaacSimVideoRecorder"
        )
        self.recording_thread.start()
        self.thread_active = True

        logger.debug("IsaacSim video recording thread started")

    def _calculate_focal_length_from_fov(self, vertical_fov_degrees: float, aperture_mm: float) -> float:
        """Calculate focal length from vertical FOV

        Parameters
        ----------
        vertical_fov_degrees : float
            Vertical field of view in degrees (matches MuJoCo fovy).
        aperture_mm : float
            Horizontal aperture in millimeters.

        Returns
        -------
        float
            Focal length in millimeters.
        """
        import math

        vertical_fov_rad = math.radians(vertical_fov_degrees)
        return aperture_mm / (2 * math.tan(vertical_fov_rad / 2))

    def _get_robot_position(self) -> tuple[float, float, float]:
        """Get robot position from the recording environment.

        Uses IsaacSim's robot root states to get position consistently.

        Returns
        -------
        tuple[float, float, float]
            Robot position (x, y, z) in the recording environment.
        """
        record_env_id = self.config.record_env_id

        # Get robot position from IsaacSim simulator
        if hasattr(self.simulator, "robot_root_states"):
            robot_pos = self.simulator.robot_root_states[record_env_id, :3]  # [x, y, z]
            return float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
        else:
            # Fallback to base simulator interface
            return super()._get_robot_position()
