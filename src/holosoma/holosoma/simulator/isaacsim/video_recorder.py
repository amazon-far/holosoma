"""Logger movies through the same Isaac Lab cameras as perception sensors."""

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


def build_video_camera(sim: IsaacSim) -> Any:
    """Declare the SDK movie camera before reset and renderer stage preparation.

    Lab 2 keeps one global movie view, avoiding a tiled image across every
    training environment. Lab 3 includes the cloned movie product in its
    shared sensor batch. Both sensors initialize through ordinary SDK callbacks.
    """
    from isaaclab.sim import PinholeCameraCfg

    config = sim.video_config
    if _VIDEO_SENSOR in sim.scene.sensors:
        raise ValueError(f"Sensor name {_VIDEO_SENSOR!r} is reserved for logger movie recording.")
    aperture = 24.0
    cfg = sim.backend.create_camera_cfg(
        prim_path=f"{sim.backend.env_regex_ns}/{_VIDEO_SENSOR}"
        if sim.backend.is_isaacsim6
        else f"/World/{_VIDEO_SENSOR}",
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
    if sim.backend.is_isaacsim6:
        sim.scene.sensors[_VIDEO_SENSOR] = camera
    return camera


class IsaacSimVideoRecorder(VideoRecorderInterface):
    """Record an SDK camera with shared tracking and encoding."""

    def __init__(self, config: VideoConfig, simulator: IsaacSim):
        super().__init__(config, simulator)
        self.simulator: IsaacSim = simulator
        self._camera = simulator.video_camera
        # SDK camera rows differ from the logical environment selected by the
        # logger: Lab 2 has one movie view, while Lab 3 batches the cloned views.
        self.camera_index = config.record_env_id if simulator.backend.is_isaacsim6 else 0
        self._capture_step: int | None = None
        self._usd_view: Any | None = None
        if simulator.backend.uses_kit and not simulator.backend.uses_kit_physx:
            from isaaclab.sim.views import UsdFrameView

            # Newton's camera view updates its sites, but the pinned SDK does
            # not publish this moving camera into Kit's USD/Fabric scene.
            path = f"{simulator.scene.env_prim_paths[config.record_env_id]}/{_VIDEO_SENSOR}"
            self._usd_view = UsdFrameView(path, device=simulator.device, stage=simulator.sim.stage)

    @property
    def camera(self) -> Any:
        """The SDK camera included in frame-end captures."""
        return self._camera

    def prepare_frame(self) -> None:
        """Publish the movie pose before the SDK prepares the shared camera batch."""
        self._update_camera_position()

    def mark_frame_ready(self) -> None:
        """Remember a completed batch so encoding does not submit a second capture."""
        self._capture_step = self.simulator._sim_step_counter

    def _capture_frame_impl(self) -> None:
        if self._capture_step != self.simulator._sim_step_counter:
            # Direct movie capture also works without the frame-end sensor hook.
            self.prepare_frame()
            cameras = (
                list(self.simulator.tiled_cameras.values()) if self.simulator.backend.shared_camera_rendering else []
            )
            self.simulator.render_for_sensors()
            self.simulator.backend.refresh_cameras([*cameras, self._camera])
            self.mark_frame_ready()
        rgb = self.simulator.backend.tensor(self._camera.data.output["rgb"])[self.camera_index, ..., :3]
        frame = np.asarray(rgb.cpu(), dtype=np.uint8).copy()
        self._add_frame(self._apply_command_overlay(frame))
        self._capture_step = None

    def _update_camera_position(self) -> None:
        parameters = self._get_camera_parameters()
        device = self.simulator.device
        eyes = torch.tensor([parameters.position], dtype=torch.float32, device=device)
        targets = torch.tensor([parameters.target], dtype=torch.float32, device=device)
        self._camera.set_world_poses_from_view(eyes, targets, env_ids=[self.camera_index])
        if self._usd_view is not None:
            import warp as wp
            from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
            from pxr import UsdGeom

            rotation = create_rotation_matrix_from_view(
                eyes, targets, UsdGeom.GetStageUpAxis(self.simulator.sim.stage), device=device
            )
            with self._usd_view.xform_world_space_writer() as writer:
                writer.set_poses(
                    wp.from_torch(eyes, dtype=wp.vec3f),
                    wp.from_torch(quat_from_matrix(rotation).contiguous(), dtype=wp.vec4f),
                )

    def cleanup(self) -> None:
        super().cleanup()
        if self.worker_stopped:
            # Lab 3's scene retains its sensor until SimulationContext closes.
            # Lab 2's standalone sensor releases through the SDK destructor.
            self.simulator.video_camera = None
            self._camera = None
            if self._usd_view is not None:
                self._usd_view.close()
                self._usd_view = None
