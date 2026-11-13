"""Task that renders externally supplied state at an external simulation clock."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch

from holosoma.config_types.env import EnvConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.simulation_task import simulation_env_origins
from holosoma.simulator.base_simulator.hooks import Phase


class RenderTask(BaseTask):
    """BaseTask specialization that renders externally applied state at an external clock, without physics.

    Startup runs with ``sim.kinematic_playback`` and without the terrain height scanner. Environments go to the
    backend's native placement where it has one, otherwise where a :class:`SimulationTask` would place them.
    Drive each frame with ``simulator.apply_state()`` and then :meth:`render_frame`; every sensor renders each
    frame.
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
        self._source_time: float | None = None
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

    def render_frame(self, *, sim_time: float) -> None:
        """Render the applied state at external time ``sim_time`` and publish sensors, without physics.

        ``sim_time`` must be finite, nonnegative and increase every frame. Returns after GPU work finishes, so
        applied inputs can be released or reused.
        """
        if not math.isfinite(sim_time) or sim_time < 0:
            raise ValueError("Source simulation time must be finite and nonnegative")
        if self._source_time is not None and sim_time <= self._source_time:
            raise ValueError("Source simulation time must advance for each frame")
        dt = sim_time if self._source_time is None else sim_time - self._source_time
        self.simulator.set_kinematic_time(sim_time)
        self.hooks.emit(Phase.FRAME_BEGIN)
        self.simulator.render_applied_state(dt)
        self.hooks.emit(Phase.FRAME_END)
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        self._source_time = sim_time
        self.frames_rendered += 1

    def step(self) -> Any:
        raise RuntimeError("RenderTask has no physics step; use render_frame()")
