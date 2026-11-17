"""Co-simulation rendering, source side: the plugin and the layout a renderer must match.

The source runs any task with ``plugin.<key>:cosim-render``. :mod:`holosoma.cosim.render.renderer` is the
Holosoma renderer; any process that serves this plugin through :class:`holosoma.cosim.base.BufferHandler`
can render instead (see "Renderers outside Holosoma" in ``docs/source-clock-rendering.md``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from holosoma.cosim.base import CosimPlugin
from holosoma.cosim.bridge import BufferRequest, as_received
from holosoma.simulator.base_simulator.hooks import Phase

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CosimRenderPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.base_simulator.hooks import HookHandle, HookRegistry
    from holosoma.simulator.shared.sensor_manager import CameraRecord


def scene_layout(sim: BaseSimulator) -> dict[str, Any]:
    """The handshake ``meta``: joint, body and object order of the state fields, and environment origins."""
    layout: dict[str, Any] = as_received(
        {
            "dof_names": list(sim.dof_names),
            "body_names": list(sim.body_names),
            "object_names": list(sim.object_state_names),
            "env_origins": sim.scene.env_origins.cpu().tolist(),
        }
    )
    return layout


def image_name(camera: str, data_type: str) -> str:
    """The request name of ``camera``'s ``data_type`` output."""
    return f"{camera}/{data_type}"


def camera_output(name: str) -> tuple[str, str]:
    """The camera and data type a request ``name`` from :func:`image_name` refers to."""
    camera, _, data_type = name.rpartition("/")
    return camera, data_type


def camera_requests(record: CameraRecord, num_envs: int) -> list[BufferRequest]:
    """One request per output of ``record``: rgb ``uint8 [E, H, W, 3]``, depth ``float32 [E, H, W, 1]``."""
    config = record.config
    meta = as_received(
        {
            "mount": asdict(config.mount),
            "vertical_fov": config.vertical_fov,
            "near": config.near,
            "far": config.far,
            "depth_clipping_behavior": config.depth_clipping_behavior,
        }
    )
    return [
        BufferRequest(
            image_name(record.name, data_type),
            (num_envs, config.height, config.width, 3 if data_type == "rgb" else 1),
            "uint8" if data_type == "rgb" else "float32",
            meta,
        )
        for data_type in config.data_types
    ]


class CosimRenderPlugin(CosimPlugin):
    """Share ``simulator``'s state with a renderer and publish a frame whenever a camera is due.

    Remote cameras take the renderer's images: each frame they are filled after local sensors render
    and before consumers read them. Without remote cameras, state is published after local cameras render.
    """

    def __init__(self, cfg: CosimRenderPluginConfig, simulator: BaseSimulator) -> None:
        manager = simulator.sensor_manager
        if not manager.cameras:
            raise ValueError("The source publishes when its cameras are due; configure at least one camera")
        slot = simulator.hooks.installing_slot
        self.remote = [record for record in manager.cameras if slot is not None and record.config.remote == slot]
        super().__init__(cfg, simulator)

    def requests(self) -> list[BufferRequest]:
        num_envs = self.simulator.sensor_manager.num_envs
        return [request for record in self.remote for request in camera_requests(record, num_envs)]

    def meta(self) -> dict[str, Any]:
        return scene_layout(self.simulator)

    def install(self, hooks: HookRegistry) -> Iterable[HookHandle]:
        yield hooks.add(
            Phase.FRAME_END,
            self._frame,
            name="cosim.frame",
            after=self.simulator.sensor_capture_hook,
            before=self.simulator.sensor_render_hook,
        )

    def _frame(self) -> None:
        """Publish when a camera is due, and fill the due remote cameras from the renderer's images."""
        due = self.simulator.sensor_manager.last_due
        if self.remote:
            cameras = [record for record in self.remote if record.name in due]
            if not cameras:
                return
            self.publisher.publish()
            images = self.publisher.imports
            for record in cameras:
                for data_type in record.config.data_types:
                    record.set_buffer(data_type, images[image_name(record.name, data_type)].clone())
        elif due:
            self.publisher.publish()
