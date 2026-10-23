"""Manager-based task for simulation-only workloads."""

from __future__ import annotations

import time

from loguru import logger

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.sim_utils import format_simulation_performance

_PERFORMANCE_LOG_INTERVAL_STEPS = 1000


class SimulationTask(BaseTask):
    """BaseTask specialization that advances the simulation until shutdown is requested.

    It uses the ordinary task lifecycle, including the default robot reset, reset events,
    and setup/reset/step randomization terms, but owns no policy managers.
    """

    def run(self) -> None:
        """Start episodes and advance control frames until the task requests shutdown."""
        logger.info("Starting Holosoma simulation task...")
        logger.info(f"Robot: {self.config.robot.asset.robot_type}")
        logger.info(f"Simulator: {self.config.simulator._target_}")
        logger.info(f"Terrain: {self.config.terrain.terrain_term.mesh_type} ({self.config.terrain.terrain_term.func})")
        self.reset_all()

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

        control_steps = 0
        window_control_steps = 0
        window_start = time.perf_counter()
        start_time = window_start
        while not self.shutdown_requested:
            self.step({})
            control_steps += 1
            window_control_steps += 1

            if window_control_steps * control_decimation >= _PERFORMANCE_LOG_INTERVAL_STEPS:
                now = time.perf_counter()
                logger.info(
                    format_simulation_performance(
                        window_control_steps * control_decimation,
                        window_control_steps,
                        1.0 / sim_frequency,
                        now - window_start,
                    )
                )
                window_start = now
                window_control_steps = 0

        logger.info("Simulation loop exiting: shutdown requested.")
        logger.info(
            format_simulation_performance(
                control_steps * control_decimation,
                control_steps,
                1.0 / sim_frequency,
                time.perf_counter() - start_time,
                prefix="Average simulation performance",
            )
        )
