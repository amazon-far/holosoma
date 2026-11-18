"""Co-simulation rendering, renderer side: render a source's state and export the cameras it requests.

usage: python -m holosoma.cosim.render --endpoint ipc://... <run_sim arguments>

The run_sim arguments describe the same scene as the source's, with the source's remote cameras configured
here as ordinary cameras. Start the renderer before the source; it exits when the source closes.

:class:`RenderTask` also renders supplied state without the bridge::

    with create_render_task(run_sim_config) as task:
        task.simulator.set_kinematic_time(time)  # what simulator.time() reports
        task.simulator.dof_pos[:] = dof_pos       # write writable state in place
        task.render_frame()
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import Any

import torch
from loguru import logger

from holosoma.config_types.env import EnvConfig, get_run_sim_env_config
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.cosim.base import BufferHandler
from holosoma.cosim.bridge import BufferRequest, Frame, Receiver
from holosoma.cosim.cuda import BufferSpec
from holosoma.cosim.render.plugin import camera_output, camera_requests, scene_layout
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.simulation_task import simulation_env_origins
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.simulator.session import task_session
from holosoma.utils.config_registry import parse_config


class RenderTask(BaseTask):
    """BaseTask specialization that renders externally applied state without physics.

    Startup runs with ``sim.kinematic_playback`` and without the terrain height scanner. Environments go to the
    backend's native placement where it has one, otherwise where a :class:`SimulationTask` would place them.
    Drive each frame by writing writable simulator state in place, then :meth:`render_frame`; every sensor
    renders each frame. Sensors and consumers that read ``simulator.time()`` see whatever
    ``simulator.set_kinematic_time()`` set.
    """

    def __init__(self, tyro_config: EnvConfig) -> None:
        simulator = tyro_config.simulator
        sim = replace(simulator.config.sim, kinematic_playback=True)
        terrain = tyro_config.terrain
        super().__init__(
            replace(
                tyro_config,
                simulator=replace(simulator, config=replace(simulator.config, sim=sim)),
                terrain=replace(terrain, terrain_term=replace(terrain.terrain_term, height_scanner=False)),
            )
        )
        self.frames_rendered = 0
        try:
            manager = self.simulator.sensor_manager
            if not manager.cameras:
                raise ValueError("RenderTask requires configured cameras")
            if any(record.effective_decimation != 1 for record in [*manager.cameras, *manager.lidars]):
                raise ValueError("RenderTask renders every sensor each frame; sensor decimation must be one")
            self.simulator.prepare_render_only()
        except BaseException:
            self.close()
            raise

    def _get_env_origins(self) -> torch.Tensor:
        origins = self.simulator.native_env_origins()
        return simulation_env_origins(self) if origins is None else origins

    def render_frame(self) -> None:
        """Render the applied state and publish sensors, without physics.

        Returns after GPU work finishes, so applied inputs can be released or reused.
        """
        self.hooks.emit(Phase.FRAME_BEGIN)
        self.simulator.render_applied_state()
        self.hooks.emit(Phase.FRAME_END)
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        self.frames_rendered += 1

    def step(self) -> Any:
        raise RuntimeError("RenderTask has no physics step; use render_frame()")


def create_render_task(run: RunSimConfig) -> AbstractContextManager[RenderTask]:
    """Return a context that builds a :class:`RenderTask` for ``run``'s scene and owns its resources."""
    config = replace(get_run_sim_env_config(run), env_class="holosoma.cosim.render.renderer.RenderTask")
    return task_session(config, RenderTask)


class RenderHandler(BufferHandler):
    """Mirror the source's state into ``task``, render it, and export the camera outputs the source requests.

    Adopts the source's environment origins, rejects any other scene difference, and links the source fields
    writable on ``task``'s simulator to its state, so the source's changes reach it at each render.
    """

    def __init__(self, task: RenderTask) -> None:
        super().__init__(task.simulator.sim_device)
        self.task = task
        self.simulator = task.simulator
        self._linked: tuple[str, ...] = ()

    def accept(self, meta: Mapping[str, Any]) -> None:
        """Adopt the source's origins and reject a different joint or object order; body state is not applied."""
        origins = self.simulator.scene.env_origins
        self.simulator.match_env_origins(torch.tensor(meta["env_origins"], dtype=torch.float32, device=origins.device))
        own = scene_layout(self.simulator)
        if differing := [key for key in ("dof_names", "object_names", "env_origins") if meta.get(key) != own[key]]:
            raise ValueError(f"Scene {differing} differs between simulators")

    def reads(self, exports: Mapping[str, BufferSpec]) -> tuple[str, ...]:
        state = self.simulator.state
        wanted = {}
        for name in state.writable_names():
            own = state.get(name)
            wanted[name] = (tuple(own.shape), str(own.dtype).removeprefix("torch."))
        return self.matching(exports, wanted)

    def targets(self, reads: Sequence[str]) -> Mapping[str, torch.Tensor]:
        self._linked = tuple(reads)
        return self.simulator.state.link(reads)

    def close(self) -> None:
        self.simulator.state.unlink(self._linked)
        self._linked = ()

    def export_source(self, request: BufferRequest) -> Callable[[], torch.Tensor]:
        manager = self.simulator.sensor_manager
        camera, data_type = camera_output(request.name)
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
    with create_render_task(config) as task:
        Receiver(RenderHandler(task), endpoint=endpoint).run()
        logger.info(f"Co-simulation source closed after {task.frames_rendered} frames")
