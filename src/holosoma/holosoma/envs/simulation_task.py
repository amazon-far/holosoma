"""Manager-based task for simulation-only workloads."""

from __future__ import annotations

import time

import torch
from loguru import logger

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.utils.sim_utils import format_simulation_performance

_PERFORMANCE_LOG_INTERVAL_STEPS = 1000


class _PerformanceLog:
    """Log real-time performance every ``_PERFORMANCE_LOG_INTERVAL_STEPS`` physics steps, and the
    run's average on close."""

    def __init__(self, hooks: HookRegistry, sim_dt: float, control_decimation: int) -> None:
        self._sim_dt = sim_dt
        self._decimation = control_decimation
        self._control_steps = 0
        self._window_control_steps = 0
        self._start = self._window_start = time.perf_counter()
        hooks.add(Phase.FRAME_END, self._on_frame_end, name="simulation.performance")
        hooks.add(Phase.CLOSE, self._on_close, name="simulation.average_performance")

    def _on_frame_end(self) -> None:
        self._control_steps += 1
        self._window_control_steps += 1
        if self._window_control_steps * self._decimation < _PERFORMANCE_LOG_INTERVAL_STEPS:
            return
        now = time.perf_counter()
        logger.info(self._format(self._window_control_steps, now - self._window_start))
        self._window_start = now
        self._window_control_steps = 0

    def _on_close(self) -> None:
        elapsed = time.perf_counter() - self._start
        logger.info(self._format(self._control_steps, elapsed, "Average simulation performance"))

    def _format(self, control_steps: int, elapsed: float, prefix: str = "Simulation performance") -> str:
        return format_simulation_performance(
            control_steps * self._decimation, control_steps, self._sim_dt, elapsed, prefix=prefix
        )


class SimulationTask(BaseTask):
    """BaseTask specialization that advances the simulation until the simulator requests shutdown.

    It uses the ordinary task lifecycle, including the default robot reset, reset events, and the
    randomization and termination terms configured on ``RunSimConfig``, but owns no policy managers.
    """

    def _get_env_origins(self) -> torch.Tensor:
        """Place a single environment at the world origin, so its poses are world coordinates.

        Several environments keep the terrain's origins so they do not overlap.
        """
        if self.num_envs == 1:
            return torch.zeros(1, 3, device=self.device)
        return super()._get_env_origins()

    def run(self) -> None:
        """Start episodes and advance control frames until the simulator requests shutdown."""
        logger.info("Starting Holosoma simulation task...")
        logger.info(f"Robot: {self.config.robot.asset.robot_type}")
        logger.info(f"Simulator: {self.config.simulator._target_}")
        logger.info(f"Terrain: {self.config.terrain.terrain_term.mesh_type} ({self.config.terrain.terrain_term.func})")
        self.reset_envs_idx()

        # A simulation run is one open-ended episode; record it regardless of the episode interval.
        video_recorder = self.simulator.video_recorder
        if video_recorder is not None and video_recorder.enabled and not video_recorder.is_recording:
            video_recorder.start_recording(episode_id=0)

        sim_frequency = self.config.simulator.config.sim.fps
        control_decimation = self.config.simulator.config.sim.control_decimation_steps
        logger.info(
            f"Starting simulation at {sim_frequency:g} Hz physics and {sim_frequency / control_decimation:g} Hz control"
        )
        logger.info("Press Ctrl+C to stop simulation")

        _PerformanceLog(self.hooks, self.sim_dt, control_decimation)
        while not self.simulator.shutdown_requested:
            self.step()
        logger.info("Simulation loop exiting: shutdown requested.")
