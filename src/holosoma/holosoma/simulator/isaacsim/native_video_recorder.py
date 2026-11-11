"""Logger movies through ordinary Lab 3 cameras in the selected renderer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

if TYPE_CHECKING:
    from holosoma.config_types.video import VideoConfig
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim

_VIDEO_SENSOR = "holosoma_video_camera"


def build_video_camera(sim: IsaacSim) -> None:
    """Declare the movie camera before cloning and renderer stage preparation."""
    from isaaclab.sim import PinholeCameraCfg

    config = sim.video_config
    if _VIDEO_SENSOR in sim.scene.sensors:
        raise ValueError(f"Sensor name {_VIDEO_SENSOR!r} is reserved for logger movie recording.")
    aperture = 24.0
    cfg = sim.backend.create_camera_cfg(
        prim_path=f"{sim.backend.env_regex_ns}/{_VIDEO_SENSOR}",
        offset_position=(0.0, 0.0, 0.0),
        offset_orientation=(1.0, 0.0, 0.0, 0.0),
        spawn=PinholeCameraCfg(
            focal_length=aperture / (2.0 * math.tan(math.radians(config.vertical_fov) / 2.0)),
            vertical_aperture=aperture,
            horizontal_aperture=aperture * config.get_aspect_ratio(),
            clipping_range=(0.1, 1000.0),
        ),
        width=config.width,
        height=config.height,
        data_types=["rgb"],
    )
    camera = sim.backend.create_camera(cfg)
    sim.scene.sensors[_VIDEO_SENSOR] = camera


class NativeVideoRecorder(VideoRecorderInterface):
    """Use the selected native renderer with the shared tracking and encoding."""

    def __init__(self, config: VideoConfig, simulator: IsaacSim):
        super().__init__(config, simulator)
        self.simulator: IsaacSim = simulator
        self._camera = simulator.scene.sensors[_VIDEO_SENSOR]
        self._capture_step: int | None = None

    @property
    def camera(self) -> Any:
        """The scene-owned SDK camera included in frame-end captures."""
        return self._camera

    def prepare_frame(self) -> None:
        """Publish the movie pose before the SDK prepares the shared camera batch."""
        self._update_camera_position()

    def mark_frame_ready(self) -> None:
        """Remember a completed batch so encoding does not submit a second capture."""
        self._capture_step = self.simulator.sim.get_physics_step_count()

    def _capture_frame_impl(self) -> None:
        if self._capture_step != self.simulator.sim.get_physics_step_count():
            # Direct movie capture also works without the frame-end sensor hook.
            self.prepare_frame()
            cameras = (
                list(self.simulator.tiled_cameras.values()) if self.simulator.backend.shared_camera_rendering else []
            )
            self.simulator.backend.refresh_cameras([*cameras, self._camera])
            self.mark_frame_ready()
        rgb = self._camera.data.output["rgb"].torch[self.config.record_env_id, ..., :3]
        frame = np.asarray(rgb.cpu(), dtype=np.uint8).copy()
        self._add_frame(self._apply_command_overlay(frame))
        self._capture_step = None

    def _update_camera_position(self) -> None:
        parameters = self._get_camera_parameters()
        device = self.simulator.device
        eyes = torch.tensor([parameters.position], dtype=torch.float32, device=device)
        targets = torch.tensor([parameters.target], dtype=torch.float32, device=device)
        self._camera.set_world_poses_from_view(eyes, targets, env_ids=[self.config.record_env_id])

    def cleanup(self) -> None:
        super().cleanup()
        if self.worker_stopped:
            # The scene owns and closes the sensor through SimulationContext.
            self._camera = None
