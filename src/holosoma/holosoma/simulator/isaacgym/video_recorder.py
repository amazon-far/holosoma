"""IsaacGym-specific video recording implementation.

This module provides the IsaacGym implementation of the video recording interface,
updated to work with the new threading system and explicit camera configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from isaacgym import gymapi
from loguru import logger

from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

if TYPE_CHECKING:
    from holosoma.simulator.isaacgym.isaacgym import IsaacGym


class IsaacGymVideoRecorder(VideoRecorderInterface):
    """IsaacGym-specific video recording implementation.

    This class implements video recording for IsaacGym simulations by creating
    camera sensors, capturing frames during simulation, and encoding videos
    with command overlays and robot tracking capabilities.

    The implementation is optimized for performance by only recording from
    environment 0 and includes camera smoothing to reduce video shake.

    Parameters
    ----------
    config : VideoConfig
        Video recording configuration settings.
    simulator : IsaacGym
        Reference to the IsaacGym simulator instance.
    """

    def __init__(self, config, simulator: IsaacGym) -> None:
        """Initialize IsaacGym video recorder.

        Parameters
        ----------
        config : VideoConfig
            Video recording configuration settings.
        simulator : IsaacGym
            The IsaacGym simulator instance.
        """
        super().__init__(config, simulator)

        # Override typing for mypy
        self.simulator: IsaacGym = simulator

        # IsaacGym-specific attributes
        self.camera_handles: list[int] = []

    def setup_recording(self) -> None:
        """Initialize IsaacGym video recording system.

        Creates camera sensors for the recording environment and configures
        them for optimal video capture.

        Raises
        ------
        RuntimeError
            If camera creation fails or graphics are not available for headless recording.
        """
        super().setup_recording()

        # Create camera for recording environment
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.config.width
        camera_props.height = self.config.height
        # Convert vertical FOV to horizontal FOV for IsaacGym
        horizontal_fov = self.config.get_horizontal_fov()
        camera_props.horizontal_fov = horizontal_fov

        env_ptr = self.simulator.envs[self.config.record_env_id]
        camera_handle = self.simulator.gym.create_camera_sensor(env_ptr, camera_props)

        if camera_handle >= 0:
            self.camera_handles.append(camera_handle)
        else:
            raise RuntimeError(f"Failed to create video camera, config: {self.config}")

    def _start_persistent_thread(self) -> None:
        raise NotImplementedError("IsaacGym does not support recording video on a separate thread")

    def _capture_frame_impl(self) -> None:
        """Implementation-specific frame capture logic.

        Updates camera position to track the robot, renders the current frame,
        applies command overlays, and stores the frame in the recording buffer.

        Raises
        ------
        RuntimeError
            If frame capture fails due to rendering or memory issues.
        """
        if not self.camera_handles:
            logger.warning("Cannot capture frame - no cameras available")
            return

        logger.debug(f"Recording frame at step {getattr(self.simulator, 'step_counter', 0)}")

        # Ensure simulation results are fetched before rendering
        self.simulator.gym.fetch_results(self.simulator.sim, True)

        # Update camera to track robot
        self._update_camera_position()

        # Render frame
        self.simulator.gym.step_graphics(self.simulator.sim)
        self.simulator.gym.render_all_camera_sensors(self.simulator.sim)
        self.simulator.gym.start_access_image_tensors(self.simulator.sim)

        try:
            # Get camera image from recording environment
            record_env_id = self.config.record_env_id
            env_ptr = self.simulator.envs[record_env_id]
            camera_handle = self.camera_handles[0]

            image = self.simulator.gym.get_camera_image(self.simulator.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)

            if image is None:
                raise RuntimeError("Video recording camera image is None")

            # Convert raw image data to numpy array
            image_array = np.frombuffer(image, dtype=np.uint8).reshape(self.config.height, self.config.width, 4)

            # Convert BGRA to RGB (remove alpha channel)
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGB)

            # Apply command overlay using shared logic
            image_with_overlay = self._apply_command_overlay(image_rgb)

            # Add frame to buffer using shared method
            self._add_frame(image_with_overlay)

        finally:
            self.simulator.gym.end_access_image_tensors(self.simulator.sim)

    def cleanup(self) -> None:
        """Clean up video recording resources.

        Releases camera handles and clears frame buffers.
        """
        super().cleanup()

        # Camera handles are managed by IsaacGym and will be cleaned up
        # when the simulator is destroyed
        self.camera_handles = []

    def _update_camera_position(self) -> None:
        """Update camera position to track the robot.

        Uses the shared camera calculation system for unified camera positioning.
        """
        if not self.camera_handles:
            return

        camera_handle = self.camera_handles[0]
        if camera_handle < 0:
            return

        # Use shared camera calculation method
        camera_params = self._calculate_camera_parameters()

        # Extract position and target from shared calculation
        position = camera_params["position"]
        target = camera_params["target"]

        # Convert to IsaacGym Vec3 format and update camera location
        gym_cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        gym_cam_target = gymapi.Vec3(target[0], target[1], target[2])

        record_env_id = self.config.record_env_id
        env_ptr = self.simulator.envs[record_env_id]
        self.simulator.gym.set_camera_location(camera_handle, env_ptr, gym_cam_pos, gym_cam_target)
