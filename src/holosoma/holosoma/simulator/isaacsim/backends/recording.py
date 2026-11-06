"""Native camera capture using Holosoma's shared video lifecycle and encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

from . import as_tensor

if TYPE_CHECKING:
    from holosoma.config_types.video import VideoConfig


class NativeVideoRecorder(VideoRecorderInterface):
    def __init__(self, config: VideoConfig, simulator: Any) -> None:
        super().__init__(config, simulator)
        self.camera = simulator.recording_camera
        self.device = simulator.device

    def _capture_frame_impl(self) -> None:
        camera = self.camera
        parameters = self._get_camera_parameters()
        device = self.device
        camera.set_world_poses_from_view(
            torch.tensor([parameters.position], dtype=torch.float32, device=device),
            torch.tensor([parameters.target], dtype=torch.float32, device=device),
            env_ids=[self.config.record_env_id],
        )
        camera.update(0.0, force_recompute=True)
        frame = (
            as_tensor(camera.data.output["rgb"])[self.config.record_env_id, ..., :3]
            .to(torch.uint8)
            .cpu()
            .numpy()
            .copy()
        )
        self._add_frame(self._apply_command_overlay(frame))
