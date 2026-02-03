"""Base video recording interface for holosoma simulators with threading support.

This module provides the abstract base class for video recording functionality
that can be implemented by different simulators (IsaacGym, MuJoCo, IsaacSim).
The interface standardizes video recording operations while allowing for
simulator-specific optimizations.
"""

from __future__ import annotations

import threading
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.config_types.video import CameraConfig, CartesianCameraConfig, FixedCameraConfig, SphericalCameraConfig
from holosoma.utils.video_utils import create_video, format_command_labels, overlay_text_on_image

if TYPE_CHECKING:
    from holosoma.config_types.video import VideoConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class VideoRecorderInterface(ABC):
    """Abstract base class for simulator-specific video recording.

    This interface defines the contract that all video recorders must implement
    to provide consistent video recording functionality across different simulators.
    Each simulator can implement this interface with its own optimizations while
    maintaining a unified API for tasks and environments.

    The video recorder handles the complete lifecycle of video recording:
    - Setup and initialization of cameras and rendering systems
    - Episode-based recording management
    - Frame capture and processing
    - Video encoding and output
    - Resource cleanup

    Parameters
    ----------
    config : VideoConfig
        Configuration object containing all video recording settings.
    simulator : BaseSimulator
        Reference to the simulator instance for accessing simulation state.
    """

    # Default tracking body names to try in order of preference
    DEFAULT_TRACKING_BODY_NAMES = [
        "Trunk",  # Common in humanoid robots (e.g., user's example)
        "torso",  # Standard anatomical naming
        "base_link",  # ROS standard naming convention
        "Waist",  # Alternative trunk naming (e.g., user's example)
        "pelvis",  # Anatomical naming for hip area
        "hip",  # Simple hip naming
        "base",  # Generic base naming
    ]

    def __init__(self, config: VideoConfig, simulator: BaseSimulator) -> None:
        """Initialize the video recorder with configuration and simulator reference.

        Parameters
        ----------
        config : VideoConfig
            Video recording configuration settings.
        simulator : object
            The simulator instance (typed as object to avoid circular imports).
        """
        self.config = config
        self.simulator = simulator
        self._is_recording = False
        self._current_episode = 0
        self._total_episodes = 0

        # Shared frame buffer for all simulators
        self.video_frames: list[npt.NDArray[np.uint8]] = []

        # Camera smoothing state for robot tracking
        self.smoothed_cam_pos: tuple[float, float, float] | None = None
        self.smoothed_cam_target: tuple[float, float, float] | None = None

        # Shared tracking body resolution
        self.robot_body_id: int | None = None

        # Frame decimation counter for capturing at control frequency
        self._frame_counter: int = 0

        # Performance timing statistics
        self._frame_times: list[float] = []

        # Threading components (shared across all simulators)
        self.recording_thread: Thread | None = None
        self.stop_recording_event: threading.Event | None = None
        self.thread_active = False

        if config.use_recording_thread:
            self._setup_threaded_recording()

    def _resolve_tracking_body(self) -> None:
        """Resolve tracking body using automatic resolution with exception handling.

        This shared method tries body names in order using the simulator's
        find_rigid_body_indice() method and handles exceptions gracefully.

        Raises
        ------
        ValueError
            If no suitable tracking body is found.
        """
        # Build list of names to try: explicit name first (if set), then defaults
        if self.config.tracking_body_name and self.config.tracking_body_name != "auto":
            names_to_try = [self.config.tracking_body_name] + self.DEFAULT_TRACKING_BODY_NAMES
        else:
            names_to_try = self.DEFAULT_TRACKING_BODY_NAMES

        # Try each name in order until one works
        for name in names_to_try:
            try:
                self.robot_body_id = self.simulator.find_rigid_body_indice(name)
                return
            except Exception:  # noqa: S112, PERF203
                continue

        # If no names worked, fail with helpful error
        raise ValueError(
            f"No suitable tracking body found. Tried: {names_to_try}. "
            f"Please set tracking_body_name explicitly to a valid body name in your robot model."
        )

    def _setup_threaded_recording(self) -> None:
        """Shared threading setup logic."""
        self.render_signal_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.episode_complete_event = threading.Event()

    def setup_recording(self) -> None:
        """Initialize video recording system.

        This method is called once during simulator initialization.

        Raises
        ------
        RuntimeError
            If video recording setup fails due to MuJoCo initialization issues.
        ValueError
            If tracking body cannot be resolved for tracking modes.
        """
        try:
            # Only create camera for the recording environment (typically env 0)
            if self.config.record_env_id >= self.simulator.num_envs:
                raise RuntimeError(
                    f"Record environment ID {self.config.record_env_id} exceeds available environments "
                    f"({self.simulator.num_envs})"
                )

            # Resolve tracking body for tracking modes
            if isinstance(self.config.camera, (SphericalCameraConfig, CartesianCameraConfig)):
                self._resolve_tracking_body()

            # Start persistent thread for threaded mode
            if self.config.use_recording_thread:
                self._start_persistent_thread()
            logger.info(f"Video recording enabled, saving videos to {self._get_save_directory()}")
        except Exception as e:
            raise RuntimeError(f"Video recording setup failed: {e}") from e

    def _start_persistent_thread(self) -> None:
        """Start persistent recording thread that runs for entire simulation."""
        if self.thread_active:
            return

        logger.debug("Starting video recording thread")

        self.recording_thread = Thread(target=self._recording_thread_worker, daemon=True, name="VideoRecorder")
        self.recording_thread.start()
        self.thread_active = True

        logger.debug("Video recording thread started")

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start_recording(self, episode_id: int) -> None:
        """Start recording for a specific episode.

        This method is called at the beginning of episodes that should be recorded
        based on the video_interval configuration. It should prepare the recording
        system for capturing frames during the episode.

        Parameters
        ----------
        episode_id : int
            The ID of the episode being recorded.

        Raises
        ------
        RuntimeError
            If recording cannot be started due to system issues.
        """
        # Set recording state - this method now owns the recording flag
        self._is_recording = True
        self._start_recording(episode_id)

    def capture_frame(self, env_id: int = 0) -> None:
        """Shared frame capture dispatch logic.

        This method is called during each simulation step and handles the logic
        for determining whether to actually capture a frame based on recording
        state, environment filtering, and control decimation.

        Frames are captured at control frequency (not physics frequency) by
        only capturing on every Nth physics step, where N = control_decimation.

        Parameters
        ----------
        env_id : int, default=0
            The environment ID where the frame is being captured.
        """
        # Only capture frames when recording is active and from the correct environment
        if not self._is_recording or env_id != self.config.record_env_id:
            return

        # Increment frame counter for decimation tracking
        self._frame_counter += 1

        # Only capture frame at control frequency (every control_decimation physics steps)
        control_decimation = self.simulator.simulator_config.sim.control_decimation
        if self._frame_counter % control_decimation != 0:
            return

        if self.config.use_recording_thread:
            # Signal the recording thread to capture a frame
            if hasattr(self, "render_signal_event"):
                self.render_signal_event.set()
        else:
            # Capture frame directly on main thread with timing
            start_time = time.perf_counter()
            self._capture_frame_impl()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._frame_times.append(elapsed_ms)

    @abstractmethod
    def _capture_frame_impl(self) -> None:
        """Simulator-specific synchronous frame capture.

        This method should:
        - Update camera position if tracking is enabled
        - Render the current simulation state
        - Apply any overlays (command information, etc.)
        - Store the frame in the recording buffer

        The method should be optimized for performance as it's called frequently
        during simulation.

        Raises
        ------
        RuntimeError
            If frame capture fails due to rendering or memory issues.
        """

    def _recording_thread_worker(self) -> None:
        try:
            assert self.stop_recording_event is not None
            while not self.stop_recording_event.is_set():
                # Check if we should stop recording this episode (moved outside render signal check)
                if not self._is_recording and hasattr(self, "episode_complete_event"):
                    # Episode ended, signal completion and continue waiting for next episode
                    logger.warning("Signalling: episode_complete_event")
                    self.episode_complete_event.set()

                # Wait for signal to capture a frame. One second ought to be sufficient (1 FPS),
                # otherwise misses frames. Ideally we increase robustness...
                logger.debug("Waiting: render_signal_event")
                if self.render_signal_event.wait(timeout=1.0):
                    self.render_signal_event.clear()

                    # Only capture if still recording, with timing
                    if self._is_recording:
                        start_time = time.perf_counter()
                        self._capture_frame_impl()
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        self._frame_times.append(elapsed_ms)

        except Exception as e:
            # Log exceptions from the threads, don't break main thread though
            logger.error(f"Recording thread error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def stop_recording(self) -> None:
        """Stop recording and save the video.

        This method is called at the end of recorded episodes. It should:
        - Finalize the current recording
        - Encode frames into video format
        - Save video to disk
        - Upload to wandb if configured
        - Clear frame buffers for next recording

        Raises
        ------
        RuntimeError
            If video encoding or saving fails.
        """
        # Guard against double-stop
        if not self._is_recording:
            return

        # Clear recording state - this method now owns the recording flag
        self._is_recording = False
        self._stop_recording()

    def cleanup(self) -> None:
        """Clean up video recording resources.

        This method should release all resources used by the video recorder:
        - Close camera handles
        - Free memory buffers
        - Clean up temporary files
        - Release rendering contexts

        This method is called during simulator shutdown.
        """
        self.stop_recording()

        # Shutdown persistent thread if active
        if self.config.use_recording_thread:
            self._cleanup_persistent_thread()

        # Clear frame buffer using shared method
        self._clear_frame_buffer()

    def should_record_episode(self, total_episodes: int) -> bool:
        """Determine if the current total episode count should trigger recording.

        This method implements the standard logic for deciding whether to record
        based on the video_interval configuration and total episode count.

        Parameters
        ----------
        total_episodes : int
            The total number of episodes completed across all environments.

        Returns
        -------
        bool
            True if the episode should be recorded, False otherwise.
        """
        if not self.config.enabled:
            return False

        return total_episodes % self.config.interval == 0

    def on_episode_start(self, env_id: int) -> None:
        """Handle episode start event.

        This method is called when an episode starts in a specific environment.
        It tracks total episodes across all environments and only records from
        the configured record_env_id.

        Parameters
        ----------
        env_id : int
            The environment ID where the episode is starting.
        """
        # Only record from the specified environment
        if env_id != self.config.record_env_id:
            return

        # Increment total episode count (across all environments)
        self._total_episodes += 1

        if self.should_record_episode(self._total_episodes):
            self._current_episode = self._total_episodes
            self.start_recording(self._total_episodes)

    def on_episode_end(self, env_id: int) -> None:
        """Handle episode end event.

        This method is called when an episode ends and stops recording
        if it was active for the current episode.
        """
        # Only stop for the specified environment
        if env_id != self.config.record_env_id:
            return

        # Let stop_recording() handle the recording state check
        self.stop_recording()

    @property
    def is_recording(self) -> bool:
        """Check if recording is currently active.

        Returns
        -------
        bool
            True if currently recording, False otherwise.
        """
        return self._is_recording

    def _get_save_directory(self) -> Path:
        """Get the directory for saving video files.

        Returns
        -------
        Path
            Path object for the video save directory.
        """
        if self.config.save_dir is not None:
            return Path(self.config.save_dir)

        return Path("logs/videos")

    # ===== Shared Frame Buffer Management =====

    def _clear_frame_buffer(self) -> None:
        """Clear the video frame buffer."""
        self.video_frames = []

    def _add_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Add a frame to the video buffer.

        Parameters
        ----------
        frame : npt.NDArray[np.uint8]
            RGB frame to add to the buffer, shape (H, W, 3).
        """
        self.video_frames.append(frame)

    def _get_frame_count(self) -> int:
        """Get the number of frames in the buffer.

        Returns
        -------
        int
            Number of frames currently in the buffer.
        """
        return len(self.video_frames)

    # ===== Shared Video Encoding and Saving =====

    def _encode_and_save_video(self) -> None:
        """Encode frames and save video using shared logic.

        This method handles the common video encoding and saving logic
        that is the same across all simulators.

        Raises
        ------
        RuntimeError
            If video encoding or saving fails.
        """
        if not self.video_frames:
            return

        try:
            # Convert frames to numpy array
            video_array = np.array(self.video_frames)
            video_array_uint8 = video_array.astype(np.uint8)

            # Calculate actual video FPS based on control frequency and playback rate
            # Frames are captured at control_frequency = sim_fps / control_decimation
            # To achieve desired playback rate: actual_fps = control_frequency * playback_rate
            sim_config = self.simulator.simulator_config.sim
            control_frequency = sim_config.fps / sim_config.control_decimation
            display_fps = control_frequency * self.config.playback_rate

            # Get save directory
            save_dir = self._get_save_directory()
            save_dir.mkdir(parents=True, exist_ok=True)

            # Create and save video
            create_video(
                video_frames=video_array_uint8,
                fps=display_fps,
                save_dir=save_dir,
                output_format=self.config.output_format,
                wandb_logging=self.config.upload_to_wandb,
                episode_id=self._current_episode,
            )

        except Exception as e:
            raise RuntimeError(f"Video encoding failed: {e}") from e
        finally:
            # Clear frame buffer
            self._clear_frame_buffer()

    # ===== Shared Robot Tracking Logic =====

    def _get_robot_position(self) -> tuple[float, float, float]:
        """Get robot position from the recording environment.

        Uses the unified BaseSimulator interface to get robot position
        consistently across all simulators.

        Returns
        -------
        tuple[float, float, float]
            Robot position (x, y, z) in the recording environment.
        """
        record_env_id = self.config.record_env_id
        robot_pos = self.simulator.robot_root_states[record_env_id, :3]  # [x, y, z]
        return float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])

    # ===== Shared Camera Calculation Logic =====

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
            Distance from target.
        azimuth_degrees : float
            Horizontal angle in degrees.
        elevation_degrees : float
            Vertical angle in degrees.
        target : tuple[float, float, float]
            Camera target position (x, y, z).

        Returns
        -------
        tuple[float, float, float]
            Camera position (x, y, z).
        """
        azimuth_rad = np.radians(azimuth_degrees)
        elevation_rad = np.radians(elevation_degrees)

        offset_x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        offset_y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        offset_z = distance * np.sin(elevation_rad)

        return (target[0] + offset_x, target[1] + offset_y, target[2] + offset_z)

    def _calculate_camera_parameters(self, robot_pos: tuple[float, float, float] | None = None) -> dict[str, Any]:
        """Calculate camera parameters for all camera modes.

        This unified method handles Fixed, Spherical, and Cartesian camera configurations
        and returns standardized parameters that simulators can use with their specific APIs.

        Parameters
        ----------
        robot_pos : tuple[float, float, float] | None, default=None
            Robot position (x, y, z). If None, will call _get_robot_position() for tracking modes.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'position': tuple[float, float, float] - Camera position in world coordinates
            - 'target': tuple[float, float, float] - Camera target in world coordinates
            - 'distance': float - Distance from camera to target
            - 'azimuth': float - Horizontal angle in degrees
            - 'elevation': float - Vertical angle in degrees
        """
        config: CameraConfig = self.config.camera  # Use union type, not FixedCameraConfig

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

            return {
                "position": position,
                "target": target,
                "distance": distance,
                "azimuth": azimuth,
                "elevation": elevation,
            }

        if isinstance(config, SphericalCameraConfig):
            # Spherical camera positioning relative to robot
            if self.robot_body_id is None:
                raise RuntimeError("Robot body not resolved for spherical camera mode")

            # Get robot position
            if robot_pos is None:
                robot_pos = self._get_robot_position()

            # Apply smoothing to target
            if self.smoothed_cam_target is None:
                self.smoothed_cam_target = robot_pos
            else:
                smooth_factor = self.config.camera_smoothing
                self.smoothed_cam_target = (
                    smooth_factor * self.smoothed_cam_target[0] + (1 - smooth_factor) * robot_pos[0],
                    smooth_factor * self.smoothed_cam_target[1] + (1 - smooth_factor) * robot_pos[1],
                    smooth_factor * self.smoothed_cam_target[2] + (1 - smooth_factor) * robot_pos[2],
                )

            # Calculate position from spherical coordinates
            position = self._spherical_to_cartesian(
                config.distance, config.azimuth, config.elevation, self.smoothed_cam_target
            )

            return {
                "position": position,
                "target": self.smoothed_cam_target,
                "distance": config.distance,
                "azimuth": config.azimuth,
                "elevation": config.elevation,
            }

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

            # Apply smoothing using shared method
            smoothed_cam_pos, smoothed_cam_target = self._apply_camera_smoothing(target_cam_pos, target_cam_target)

            # Convert to spherical for simulators that need it
            distance, azimuth, elevation = self._cartesian_to_spherical(smoothed_cam_pos, smoothed_cam_target)

            return {
                "position": smoothed_cam_pos,
                "target": smoothed_cam_target,
                "distance": distance,
                "azimuth": azimuth,
                "elevation": elevation,
            }

        raise ValueError(f"Unsupported camera config type: {type(self.config.camera)}")

    def _apply_camera_smoothing(
        self, target_cam_pos: tuple[float, float, float], target_cam_target: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Apply exponential moving average smoothing to camera positions.

        Parameters
        ----------
        target_cam_pos : tuple[float, float, float]
            Target camera position (x, y, z).
        target_cam_target : tuple[float, float, float]
            Target camera target (x, y, z).

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
            smooth_factor = self.config.camera_smoothing

            self.smoothed_cam_pos = (
                smooth_factor * self.smoothed_cam_pos[0] + (1 - smooth_factor) * target_cam_pos[0],
                smooth_factor * self.smoothed_cam_pos[1] + (1 - smooth_factor) * target_cam_pos[1],
                smooth_factor * self.smoothed_cam_pos[2] + (1 - smooth_factor) * target_cam_pos[2],
            )

            # Smooth camera target
            self.smoothed_cam_target = (
                smooth_factor * self.smoothed_cam_target[0] + (1 - smooth_factor) * target_cam_target[0],
                smooth_factor * self.smoothed_cam_target[1] + (1 - smooth_factor) * target_cam_target[1],
                smooth_factor * self.smoothed_cam_target[2] + (1 - smooth_factor) * target_cam_target[2],
            )

        return self.smoothed_cam_pos, self.smoothed_cam_target

    def _reset_camera_smoothing(self) -> None:
        """Reset camera smoothing state for a new episode."""
        self.smoothed_cam_pos = None
        self.smoothed_cam_target = None

    # ===== Shared Command Overlay Logic =====

    def _apply_command_overlay(self, image_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Apply command overlay to the image if enabled.

        Parameters
        ----------
        image_rgb : npt.NDArray[np.uint8]
            Input RGB image, shape (H, W, 3).

        Returns
        -------
        npt.NDArray[np.uint8]
            Image with command overlay applied if enabled, same shape as input.
        """
        if not self.config.show_command_overlay:
            return image_rgb

        record_env_id = self.config.record_env_id
        command_text = format_command_labels(getattr(self.simulator, "commands", None), env_id=record_env_id)
        return overlay_text_on_image(image_rgb.copy(), command_text, position=(50, 50), font_scale=0.8)

    # ===== Impl Functions (can be overridden) =====

    def _start_recording(self, episode_id: int) -> None:
        """Shared logic for starting recording.

        This method handles the common setup that all simulators need
        when starting a new recording episode.

        Parameters
        ----------
        episode_id : int
            The ID of the episode being recorded.
        """
        # Initialize frame buffer for new episode
        self._clear_frame_buffer()

        # Reset frame counter for decimation tracking
        self._frame_counter = 0

        # Reset camera tracking to start from current robot position
        self._reset_camera_smoothing()

        # Reset frame timing statistics
        self._frame_times.clear()

    def _stop_recording(self) -> None:
        """Shared logic for stopping recording."""
        if self.config.use_recording_thread:
            # Wait for thread to finish processing current frames
            self._wait_for_recording_thread_idle()

        # Print frame capture statistics if we captured any frames
        if self._frame_times:
            frame_times_array = np.array(self._frame_times)
            logger.debug(
                f"Frame capture stats: n={len(self._frame_times)} "
                f"mean={frame_times_array.mean():.2f}ms "
                f"std={frame_times_array.std():.2f}ms "
                f"min={frame_times_array.min():.2f}ms "
                f"max={frame_times_array.max():.2f}ms"
            )

        # Use shared encoding logic
        self._encode_and_save_video()

    def _wait_for_recording_thread_idle(self) -> None:
        """Wait for recording thread to finish processing current episode frames."""
        if not hasattr(self, "episode_complete_event"):
            return

        # Wait for thread to signal episode completion, needs to write and encode files...
        logger.warning("Waiting: episode_complete_event")
        if not self.episode_complete_event.wait(timeout=30.0):
            # ideally we make this async with a buffer and not wait at all, just keeping it the same as
            # holosoma for now...
            logger.warning("Recording thread did not complete writing within timeout, likely incomplete video!")

        # Reset event for next episode
        self.episode_complete_event.clear()

    def _cleanup_persistent_thread(self) -> None:
        """Clean up persistent thread during simulator shutdown."""
        if self.config.use_recording_thread and self.thread_active:
            # Signal thread to stop
            if self.stop_recording_event:
                self.stop_recording_event.set()

            # Wait for thread to finish
            if self.recording_thread:
                self.recording_thread.join(timeout=30.0)
                if self.recording_thread.is_alive():
                    logger.warning("Recording thread did not terminate cleanly, potentially incomplete final video!")
            self.thread_active = False
