"""Co-simulation rendering: the source's plugin, and the renderer that renders its state.

The source runs any task with ``plugin.<key>:cosim-render``. The renderer runs:

usage: python -m holosoma.cosim.render --endpoint ipc://... <run_sim arguments>

The run_sim arguments describe the same scene as the source's, with the source's remote cameras configured
here as ordinary cameras. Start the renderer before the source; it exits when the source closes.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.cosim.bridge import BufferRequest, Publisher, Receiver
from holosoma.cosim.cameras import camera_requests, image_name
from holosoma.cosim.cuda import BufferSpec, allocate_shared, shared_state
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.utils.config_registry import parse_config

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CosimRenderPluginConfig
    from holosoma.envs.render_task import RenderTask
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.shared.sensor_manager import CameraRecord


class CosimRenderPlugin:
    """Share ``simulator``'s state with a renderer and publish a frame whenever a camera is due.

    Remote cameras take the renderer's images: each frame they are filled after local sensors render
    and before consumers read them, and after resets the renderer re-renders the reset envs. Without
    remote cameras, state is published after local cameras render. Installs the shared cache
    allocator, so it must run before anything reads simulator state; construction blocks until the
    renderer accepts.
    """

    def __init__(self, cfg: CosimRenderPluginConfig, simulator: BaseSimulator) -> None:
        if not cfg.endpoint:
            raise ValueError("Set the co-simulation renderer endpoint with --plugin.<key>.endpoint")
        if torch.device(simulator.sim_device).type != "cuda":
            raise ValueError("Co-simulation shares state through CUDA IPC; run the source on a CUDA device")
        manager = simulator.sensor_manager
        if not manager.cameras:
            raise ValueError("The source publishes when its cameras are due; configure at least one camera")
        self.simulator = simulator
        self.remote = [record for record in manager.cameras if record.config.remote]
        requests = [request for record in self.remote for request in camera_requests(record, manager.num_envs)]
        self.publisher = Publisher(simulator, shared_state(simulator), endpoint=cfg.endpoint, requests=requests)
        hooks = simulator.hooks
        self.frame_hook = hooks.add(
            Phase.FRAME_END,
            self._frame,
            name="cosim.frame",
            after=simulator.sensor_capture_hook,
            before=simulator.sensor_render_hook,
        )
        self.episode_hook = (
            hooks.add(Phase.EPISODE_START, self._rerender, name="cosim.rerender") if self.remote else None
        )
        hooks.add(Phase.CLOSE, self.close, name="cosim.close")

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
        self.publisher.repeat()
        images = self.publisher.imports
        for record in self.remote:
            for data_type in record.config.data_types:
                image = images[image_name(record.name, data_type)]
                if data_type in record.buffers:
                    record.set_buffer_rows(data_type, env_ids, image[env_ids])
                else:
                    record.set_buffer(data_type, image.clone())

    def close(self) -> None:
        if self.publisher.closed:
            return
        self.frame_hook.remove()
        if self.episode_hook is not None:
            self.episode_hook.remove()
        self.publisher.close()


@dataclass(frozen=True)
class _Image:
    """One exported camera output, a view of its shared allocation."""

    record: CameraRecord
    data_type: str
    tensor: torch.Tensor


class RenderHandler:
    """Mirror the source's state into ``task``, render it, and export the camera outputs the source requests.

    Reads the source fields writable on ``task``'s simulator and applies them before each render.
    """

    def __init__(self, task: RenderTask) -> None:
        self.task = task
        self.simulator = task.simulator
        self._images: list[_Image] = []

    def reads(self, exports: Sequence[BufferSpec]) -> tuple[str, ...]:
        """The exported fields this task's simulator can write, each checked against its own field's layout."""
        state = self.simulator.state
        writable = set(state.writable_names())
        reads = [spec for spec in exports if spec.name in writable]
        for spec in reads:
            own = state.get(spec.name)
            if (spec.shape, spec.dtype) != (tuple(own.shape), str(own.dtype).removeprefix("torch.")):
                raise ValueError(
                    f"State field {spec.name!r} differs between simulators: "
                    f"source {spec.dtype} {spec.shape}, receiver {own.dtype} {tuple(own.shape)}"
                )
        return tuple(spec.name for spec in reads)

    def allocate(self, requests: list[BufferRequest]) -> dict[str, Any]:
        """Check each request against this task's camera output and allocate its export."""
        import warp as wp

        manager = self.simulator.sensor_manager
        owners = {}
        for request in requests:
            camera, _, data_type = request.name.rpartition("/")
            record = manager.get(camera) if manager.has_camera(camera) else None
            own = (
                {}
                if record is None
                else {output.name: output for output in camera_requests(record, self.task.num_envs)}
            )
            if record is None or request.name not in own:
                raise ValueError(f"Receiver has no camera output {request.name!r} for the source's request")
            if request != own[request.name]:
                raise ValueError(
                    f"Camera output {request.name!r} differs between simulators: "
                    f"source {request}, receiver {own[request.name]}"
                )
            owners[request.name] = allocate_shared(request.shape, request.dtype, self.simulator.sim_device)
            self._images.append(_Image(record, data_type, wp.to_torch(owners[request.name])))
        return owners

    def process(self, state: Mapping[str, torch.Tensor], sim_time: float, *, repeat: bool, phase: str | None) -> None:
        """Apply ``state``, render it, and copy each requested camera output into its export."""
        self.simulator.apply_state(state)
        self.task.render_frame(sim_time=sim_time, repeat=repeat)
        for image in self._images:
            image.tensor.copy_(image.record.buffers[image.data_type])


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
