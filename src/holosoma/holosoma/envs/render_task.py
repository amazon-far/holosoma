"""Task that renders externally supplied state without physics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from holosoma.config_types.env import EnvConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.simulation_task import simulation_env_origins
from holosoma.simulator.base_simulator.hooks import Phase


class RenderTask(BaseTask):
    """BaseTask specialization that renders externally applied state without physics.

    Startup runs with ``sim.kinematic_playback`` and without the terrain height scanner. Environments go to the
    backend's native placement where it has one, otherwise where a :class:`SimulationTask` would place them.
    Drive each frame with ``simulator.apply_state()`` and then :meth:`render_frame`; every sensor renders each
    frame. Sensors and consumers that read ``simulator.time()`` see whatever ``simulator.set_kinematic_time()`` set.
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
