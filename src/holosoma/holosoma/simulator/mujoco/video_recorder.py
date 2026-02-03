"""MuJoCo-specific video recording implementation.

This module provides the MuJoCo implementation of the video recording interface,
supporting both holosoma_inference-style spherical camera positioning and holosoma-style
cartesian offset positioning with unified sync/async architecture using
thread-local rendering.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger

import mujoco
from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.mujoco import MuJoCo


class MuJoCoVideoRecorder(VideoRecorderInterface):
    """MuJoCo-specific video recording implementation with unified sync/async architecture.

    Uses property-based lazy renderer creation to provide transparent thread-local
    rendering. The same _capture_frame_impl() method works on both main thread
    (sync mode) and recording thread (async mode).

    Parameters
    ----------
    config : VideoConfig
        Video recording configuration settings.
    simulator : MuJoCo
        Reference to the MuJoCo simulator instance.
    """

    def __init__(self, config, simulator: MuJoCo) -> None:
        super().__init__(config, simulator)

        # Override typing for mypy
        self.simulator: MuJoCo = simulator

        # Thread-local renderer storage (created lazily via properties)
        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None

        logger.info(
            f"MuJoCo video recorder initialized - {'threaded' if config.use_recording_thread else 'synchronous'} mode"
        )
        logger.info(f"Camera config: {config.camera.type}")
        logger.info(f"Video dimensions: {config.width}x{config.height}")

    @property
    def renderer(self) -> mujoco.Renderer:
        """Lazy-create MuJoCo renderer on current thread."""
        if self._renderer is None:
            logger.debug(f"Creating MuJoCo renderer on thread: {threading.current_thread().name}")
            self._renderer = mujoco.Renderer(
                self.simulator.root_model, height=self.config.height, width=self.config.width
            )
        return self._renderer

    @property
    def camera(self) -> mujoco.MjvCamera:
        """Lazy-create MuJoCo camera on current thread."""
        if self._camera is None:
            logger.debug(f"Creating MuJoCo camera on thread: {threading.current_thread().name}")
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._camera)
        return self._camera

    def setup_recording(self) -> None:
        """Initialize MuJoCo video recording system.

        Overrides the model's global FOV with the config value for consistency.
        """
        super().setup_recording()

        # Override model's global FOV with config value
        # Note: This also changes the viewport, and ideally we set just the recording camera...
        assert self.simulator.root_model is not None
        if hasattr(self.simulator.root_model, "vis") and hasattr(self.simulator.root_model.vis, "global_"):
            self.simulator.root_model.vis.global_.fovy = self.config.vertical_fov
        else:
            raise RuntimeError(f"Failed to setup video recording camera FOV: {self.config}")

        # Set near clipping plane to prevent robot clipping with large terrains
        # The actual near distance = znear * model.stat.extent
        # For camera_near_plane=0.01m and typical extent, this gives good results
        if hasattr(self.simulator.root_model, "vis") and hasattr(self.simulator.root_model.vis, "map"):
            # Store original value for reference
            znear = 0.01
            original_znear = self.simulator.root_model.vis.map.znear

            # Calculate znear multiplier: we want actual_near = camera_near_plane
            # actual_near = znear * extent, so znear = camera_near_plane / extent
            extent = self.simulator.root_model.stat.extent
            self.simulator.root_model.vis.map.znear = znear / extent

            logger.info(
                f"Video camera near plane: {znear}m "
                f"(znear: {original_znear:.4f} -> {self.simulator.root_model.vis.map.znear:.4f}, "
                f"extent: {extent:.2f}m)"
            )

    def _resolve_tracking_body(self) -> None:
        """MuJoCo-specific tracking body resolution.

        Calls the shared resolution logic and then sets robot_model for renderer.

        Raises
        ------
        ValueError
            If no suitable tracking body is found.
        """
        # Call shared resolution logic
        super()._resolve_tracking_body()

        # Set robot_model for MuJoCo renderer compatibility
        self.robot_model = self.simulator.root_model

    def _capture_frame_impl(self) -> None:
        """Unified frame capture implementation - works on any thread.

        Uses property-based lazy renderer creation to work transparently
        on both main thread (sync mode) and recording thread (async mode).

        Raises
        ------
        RuntimeError
            If frame capture fails due to rendering issues.
        """
        # Get render data via backend (handles GPU→CPU sync for WarpBackend)
        # Always use world_id=0 to record from first environment
        render_data = self.simulator.backend.get_render_data(world_id=0)

        # Update camera position (properties create renderer/camera if needed)
        self._update_camera_position(self.camera)

        # Render frame using thread-appropriate renderer
        self.renderer.update_scene(render_data, camera=self.camera)
        frame = self.renderer.render()

        if frame is None:
            raise RuntimeError("MuJoCo renderer returned None frame")

        # Convert to RGB if needed (MuJoCo returns RGB by default)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = frame
        else:
            raise RuntimeError(f"Unexpected frame shape: {frame.shape}")

        # Apply command overlay using shared logic
        frame_with_overlay = self._apply_command_overlay(frame_rgb)

        # Add frame to buffer using shared method
        self._add_frame(frame_with_overlay)

    def _update_camera_position(
        self, camera: mujoco.MjvCamera, robot_pos: tuple[float, float, float] | None = None
    ) -> None:
        """Update camera position based on camera mode and configuration."""

        # Use shared camera calculation method
        camera_params = self._calculate_camera_parameters(robot_pos)

        # Extract parameters from shared calculation
        target = camera_params["target"]
        distance = camera_params["distance"]
        azimuth = camera_params["azimuth"]
        elevation = camera_params["elevation"]

        # Apply to MuJoCo camera using spherical coordinates
        camera.lookat[:] = target
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = -elevation

    def cleanup(self) -> None:
        """Clean up video recording resources.

        Releases MuJoCo renderer, camera, clears frame buffers, and
        shuts down persistent recording thread if active.
        """
        super().cleanup()

        # Clean up MuJoCo resources
        self._renderer = None
        self._camera = None
        self.robot_body_id = None
