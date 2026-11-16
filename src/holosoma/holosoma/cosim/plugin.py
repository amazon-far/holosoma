"""Simulator plugin that makes a simulation task a co-simulation source."""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.cosim.bridge import Publisher, camera_requests, image_name
from holosoma.cosim.cuda import CacheAllocator
from holosoma.simulator.base_simulator.hooks import Phase

if TYPE_CHECKING:
    import torch

    from holosoma.config_types.plugin import CosimPublisherPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.shared.sensor_manager import CameraRecord


class CosimPublisherPlugin:
    """Share ``simulator``'s state with a renderer and publish a frame whenever a camera is due.

    Remote cameras take the renderer's images: this plugin is their remote renderer. Without remote
    cameras, state is published after local cameras render. Installs the shared cache allocator, so it
    must run before anything reads simulator state; construction blocks until the renderer accepts.
    """

    def __init__(self, cfg: CosimPublisherPluginConfig, simulator: BaseSimulator) -> None:
        if not cfg.endpoint:
            raise ValueError("Set the co-simulation renderer endpoint with --plugin.<key>.endpoint")
        manager = simulator.sensor_manager
        if not manager.cameras:
            raise ValueError("The source publishes when its cameras are due; configure at least one camera")
        self.simulator = simulator
        state = simulator.state
        fields = tuple(state.writable_names() if cfg.fields is None else cfg.fields)
        self.allocator = CacheAllocator(fields)
        state.set_buffer_allocator(self.allocator)
        published = tuple(name for name in fields if state.get(name).numel())
        self.remote = manager.remote_cameras
        requests = [request for record in self.remote for request in camera_requests(record, manager.num_envs)]
        self.publisher = Publisher(
            simulator, self.allocator, endpoint=cfg.endpoint, fields=published, requests=requests
        )
        self.physics_step = 0
        hooks = simulator.hooks
        self.step_hook = hooks.add(Phase.POST_STEP, self._stepped, name="cosim.physics_step")
        self.capture_hook = None
        if self.remote:
            manager.set_remote_renderer(self)
        else:
            self.capture_hook = hooks.add(
                Phase.FRAME_END, self._capture, name="cosim.publish", after=simulator.sensor_render_hook
            )
        self.close_hook = hooks.add(Phase.CLOSE, self.close, name="cosim.close")

    def _stepped(self) -> None:
        self.physics_step += 1

    def _capture(self) -> None:
        if self.simulator.sensor_manager.last_due:
            self.publisher.publish(self.physics_step)

    def render_remote(self, cameras: list[CameraRecord]) -> None:
        """Publish a frame and set ``cameras``' buffers to copies of the renderer's images."""
        self.publisher.publish(self.physics_step)
        images = self.publisher.imports
        for record in cameras:
            for data_type in record.config.data_types:
                record.set_buffer(data_type, images[image_name(record.name, data_type)].clone())

    def rerender_remote(self, cameras: list[CameraRecord], env_ids: torch.Tensor) -> None:
        """Have the renderer render current state at the current time, and take its ``env_ids`` rows."""
        self.publisher.rerender()
        images = self.publisher.imports
        for record in cameras:
            for data_type in record.config.data_types:
                image = images[image_name(record.name, data_type)]
                if data_type in record.buffers:
                    record.set_buffer_rows(data_type, env_ids, image[env_ids])
                else:
                    record.set_buffer(data_type, image.clone())

    def close(self) -> None:
        if self.publisher.closed:
            return
        self.step_hook.remove()
        if self.capture_hook is None:
            self.simulator.sensor_manager.set_remote_renderer(None)
        else:
            self.capture_hook.remove()
        self.publisher.close()
