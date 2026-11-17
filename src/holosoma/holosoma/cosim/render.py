"""Co-simulation rendering: the source's plugin, and the renderer that renders its state.

The source runs any task with ``plugin.<key>:cosim-render``. The renderer runs:

usage: python -m holosoma.cosim.render --endpoint ipc://... <run_sim arguments>

The run_sim arguments describe the same scene as the source's, with the source's remote cameras configured
here as ordinary cameras. Start the renderer before the source; it exits when the source closes.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.cosim.base import BufferHandler, CosimPlugin
from holosoma.cosim.bridge import BufferRequest, Frame, Receiver, as_received
from holosoma.cosim.cameras import camera_requests, image_name
from holosoma.cosim.cuda import BufferSpec
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.utils.config_registry import parse_config

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CosimRenderPluginConfig
    from holosoma.envs.render_task import RenderTask
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.base_simulator.hooks import HookHandle, HookRegistry


def scene_layout(sim: BaseSimulator) -> dict[str, Any]:
    """The joint, object and environment layout both simulators' state must share."""
    layout: dict[str, Any] = as_received(
        {
            "dof_names": list(sim.dof_names),
            "object_names": list(sim.object_state_names),
            "env_origins": sim.scene.env_origins.cpu().tolist(),
        }
    )
    return layout


class CosimRenderHook(Enum):
    """Hooks :class:`CosimRenderPlugin` publishes, for ordering with ``HookRef(<slot>, member)``."""

    RERENDER = "rerender"
    """The ``EPISODE_START`` re-render of reset environments; order state writers before it."""


class CosimRenderPlugin(CosimPlugin):
    """Share ``simulator``'s state with a renderer and publish a frame whenever a camera is due.

    Remote cameras take the renderer's images: each frame they are filled after local sensors render
    and before consumers read them, and after resets the renderer re-renders the reset envs. Without
    remote cameras, state is published after local cameras render.
    """

    def __init__(self, cfg: CosimRenderPluginConfig, simulator: BaseSimulator) -> None:
        manager = simulator.sensor_manager
        if not manager.cameras:
            raise ValueError("The source publishes when its cameras are due; configure at least one camera")
        self.remote = [record for record in manager.cameras if record.config.remote]
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
        if self.remote:
            yield hooks.add(Phase.EPISODE_START, self._rerender, name="cosim.rerender", key=CosimRenderHook.RERENDER)

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

    def _rerender(self, env_ids: torch.Tensor) -> None:
        """Have the renderer render the reset envs' state at the current time, and take their rows."""
        self.publisher.publish()
        images = self.publisher.imports
        for record in self.remote:
            for data_type in record.config.data_types:
                image = images[image_name(record.name, data_type)]
                if data_type in record.buffers:
                    record.set_buffer_rows(data_type, env_ids, image[env_ids])
                else:
                    record.set_buffer(data_type, image.clone())


class RenderHandler(BufferHandler):
    """Mirror the source's state into ``task``, render it, and export the camera outputs the source requests.

    Adopts the source's environment origins, rejects any other scene difference, and reads the source fields
    writable on ``task``'s simulator, applying them and the source's time before each render.
    """

    def __init__(self, task: RenderTask) -> None:
        super().__init__(task.simulator.sim_device)
        self.task = task
        self.simulator = task.simulator

    def accept(self, meta: Mapping[str, Any]) -> None:
        origins = self.simulator.scene.env_origins
        self.simulator.match_env_origins(torch.tensor(meta["env_origins"], dtype=torch.float32, device=origins.device))
        own = scene_layout(self.simulator)
        if differing := sorted(key for key in {*meta, *own} if meta.get(key) != own.get(key)):
            raise ValueError(f"Scene {differing} differs between simulators")

    def reads(self, exports: Mapping[str, BufferSpec]) -> tuple[str, ...]:
        state = self.simulator.state
        wanted = {}
        for name in state.writable_names():
            own = state.get(name)
            wanted[name] = (tuple(own.shape), str(own.dtype).removeprefix("torch."))
        return self.matching(exports, wanted)

    def export_source(self, request: BufferRequest) -> Callable[[], torch.Tensor]:
        manager = self.simulator.sensor_manager
        camera, _, data_type = request.name.rpartition("/")
        record = manager.get(camera) if manager.has_camera(camera) else None
        own = {} if record is None else {output.name: output for output in camera_requests(record, self.task.num_envs)}
        if record is None or request.name not in own:
            raise ValueError(f"Receiver has no camera output {request.name!r} for the source's request")
        if request != own[request.name]:
            raise ValueError(
                f"Camera output {request.name!r} differs between simulators: "
                f"source {request}, receiver {own[request.name]}"
            )
        return lambda: record.buffers[data_type]

    def step(self, frame: Frame, state: Mapping[str, torch.Tensor]) -> None:
        self.simulator.set_kinematic_time(frame.sim_time)
        self.simulator.apply_state(state)
        self.task.render_frame()


def split_endpoint(argv: list[str]) -> tuple[str, list[str]]:
    """Return the ``--endpoint`` value and the remaining arguments."""
    for index, arg in enumerate(argv):
        if arg.startswith("--endpoint="):
            return arg.partition("=")[2], argv[:index] + argv[index + 1 :]
        if arg == "--endpoint" and index + 1 < len(argv):
            return argv[index + 1], argv[:index] + argv[index + 2 :]
    raise SystemExit("usage: python -m holosoma.cosim.render --endpoint ipc://... <run_sim arguments>")


def main(argv: list[str] | None = None) -> None:
    endpoint, rest = split_endpoint(list(sys.argv[1:] if argv is None else argv))
    config = parse_config(RunSimConfig, args=rest)
    with config.create_render_task() as task:
        Receiver(RenderHandler(task), endpoint=endpoint).run()
        logger.info(f"Co-simulation source closed after {task.frames_rendered} frames")


if __name__ == "__main__":
    main()
