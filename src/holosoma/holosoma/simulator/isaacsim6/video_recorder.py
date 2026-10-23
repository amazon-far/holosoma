"""IsaacSim-specific video recording implementation.

This module provides the IsaacSim implementation of the video recording interface,
using omni.replicator.core for efficient rendering and supporting both synchronous
and asynchronous recording modes with full feature parity to IsaacGym and MuJoCo.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

import torch
from loguru import logger


from isaaclab.utils.math import (
    create_rotation_matrix_from_view,
    quat_from_matrix,
)

from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

if TYPE_CHECKING:
    from holosoma.config_types.video import VideoConfig
    from holosoma.simulator.isaacsim6.isaacsim import IsaacSim


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

    def __init__(self, config: VideoConfig, simulator: IsaacSim) -> None:
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
        self._render_product: Any = None
        self._rgb_annotator: Any = None
        self._camera_prim_path: str | None = None
        # Declared here, not only where they are assigned: `setup_recording` raises on this lane
        # (video recording is unported), so the sole remaining assignment to either is the
        # ``= None`` in cleanup. mypy then infers their type as exactly ``None`` and rejects both
        # the tracking write and the cleanup test's fixtures. `Any` matches the two attributes
        # above and keeps the port's shape visible.
        self.camera_prim: Any = None
        self._view: Any = None

        logger.info(
            f"IsaacSim video recorder initialized - {'threaded' if config.use_recording_thread else 'synchronous'} mode"
        )
        logger.info(f"Camera config: {config.camera.type}")
        logger.info(f"Video dimensions: {config.width}x{config.height}")

    def setup_recording(self) -> None:
        """Unsupported on this lane: raises before touching the stage.

        Isaac Sim 6.1 removed ``isaacsim.core.prims``, which the recorder used to build an
        ``XFormPrim`` over its camera prim and drive robot-follow tracking. Unlike
        ``isaacsim.util.debug_draw``, which merely needs enabling, that module does not exist and
        no extension brings it back. Isaac Lab 3.0's replacement is
        ``isaaclab.sim.views.XformPrimView``, a factory-dispatched ``FrameView`` whose
        ``set_world_poses`` takes warp arrays rather than torch tensors and which has no
        ``initialize()``, so both the construction and the per-frame tracking write need porting.

        Raised rather than half-ported: a recorder that silently stops following the robot
        produces video of the wrong thing, which is worse than no video.

        Raised **first**, before any of the setup, which is the fix for a real defect -- the
        previous version authored the ``VideoCamera`` USD prim, set its focal length, aperture and
        clipping range, and only then raised. That left a camera prim on the stage after a failed
        setup, and the ~30 lines between the authoring and the raise were statically unreachable.
        The 5.1 lane (``simulator/isaacsim/video_recorder.py``) holds the working implementation
        and is the reference for the port.

        Raises
        ------
        NotImplementedError
            Always, naming the capability and the supported alternative.
        """
        raise NotImplementedError(
            "Video recording is not yet supported on the Isaac Sim 6.1 backend: Isaac Sim 6.1 "
            "removed `isaacsim.core.prims`, and the replacement "
            "(`isaaclab.sim.views.XformPrimView`) takes warp arrays, so the camera-tracking "
            "write needs porting. Use the Isaac Sim 5.1 backend (simulator:isaacsim) to record, "
            "or disable recording. See isaacsim6/capabilities.py."
        )

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
            rgb_array: npt.NDArray[np.uint8]
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

            # Use camera controller via shared helper method
            camera_params = self._get_camera_parameters()

            # Extract position and target from camera parameters
            position = torch.tensor([camera_params.position], device=self.simulator.device, dtype=torch.float32)
            target = torch.tensor([camera_params.target], device=self.simulator.device, dtype=torch.float32)

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
        failures: list[Exception] = []
        try:
            super().cleanup()
        except Exception as exc:
            failures.append(exc)

        if self.worker_stopped:
            failures.extend(self._cleanup_backend_resources())

        if failures:
            details = "; ".join(repr(exc) for exc in failures)
            raise RuntimeError(f"IsaacSim video recorder cleanup failed: {details}")
        logger.debug("IsaacSim video recorder cleanup completed")

    def _cleanup_backend_resources(self) -> list[Exception]:
        """Destroy Replicator and USD resources after the worker exits."""
        failures: list[Exception] = []
        if self._rgb_annotator is not None:
            try:
                if self._render_product is not None:
                    self._rgb_annotator.detach([self._render_product])
            except Exception as exc:
                failures.append(exc)
            else:
                self._rgb_annotator = None

        if self._rgb_annotator is None and self._render_product is not None:
            try:
                self._render_product.destroy()
            except Exception as exc:
                failures.append(exc)
            else:
                self._render_product = None

        if self._rgb_annotator is None and self._render_product is None and self._camera_prim_path is not None:
            camera_prim_path = self._camera_prim_path
            self.camera_prim = None
            self._view = None
            try:
                import omni.usd

                removed = omni.usd.get_context().get_stage().RemovePrim(camera_prim_path)
                if removed is False:
                    raise RuntimeError(f"Failed to remove camera prim {camera_prim_path!r}")
            except Exception as exc:
                failures.append(exc)
            else:
                self._camera_prim_path = None

        return failures

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
