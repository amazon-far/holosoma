"""Manager-based task for simulation-only workloads."""

from __future__ import annotations

import os
import sys
import time
import traceback

import torch
from loguru import logger

from holosoma.config_types.experiment import ExperimentConfig, ReplayExperimentConfig, SimulationExperimentConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.utils.sim_utils import format_simulation_performance

_PERFORMANCE_LOG_INTERVAL_STEPS = 1000


_THREAD_LIMIT_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def limit_cpu_math_threads(config: ExperimentConfig, device: str | None = None) -> None:
    """Serialize Torch's CPU intra-op pool for classic MuJoCo on cpu, unless the operator set a limit.

    Classic MuJoCo runs one env on the host, where every tensor is small enough that a machine-sized
    pool costs more per elementwise op in fork/join than the op itself. A no-op for Warp and
    IsaacSim, which keep this work on the GPU. ``device`` is the resolved simulation device; it
    defaults to ``config.run.device``.
    """
    from holosoma.config_types.simulator import MujocoBackend

    # Both halves are needed: ``mujoco_backend`` lives on the shared SimulatorInitConfig and
    # defaults to CLASSIC, so every backend's preset reports CLASSIC unless it overrides it. The
    # name is the same discriminator ``resolve_simulation_device`` uses.
    sim_config = config.simulator.config
    is_classic = sim_config.name == "mujoco" and getattr(sim_config, "mujoco_backend", None) == MujocoBackend.CLASSIC
    if device is None:
        device = config.run.device
    if not is_classic or device is None or not device.startswith("cpu"):
        return
    # Truthiness, not presence: OpenMP and MKL ignore an empty value, so an empty one is not an
    # operator's choice and must not suppress the cap.
    preset = [name for name in _THREAD_LIMIT_VARIABLES if os.environ.get(name)]
    if preset:
        logger.info(f"{', '.join(preset)} already set - leaving Torch at {torch.get_num_threads()} CPU thread(s)")
        return
    torch.set_num_threads(1)
    logger.info("Classic MuJoCo on cpu - limiting Torch to 1 CPU thread")


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
    randomization and termination terms configured on the environment, but owns no policy managers.
    """

    @classmethod
    def simulate(cls, config: SimulationExperimentConfig) -> None:
        """Build ``config``'s simulation task and run it until the simulator requests shutdown."""
        from holosoma.simulator.session import get_local_rank, resolve_simulation_device, task_session

        env_config = config.env_config()
        # Resolved as the task session will, so the CPU thread cap applies before the task is built.
        limit_cpu_math_threads(config, resolve_simulation_device(env_config, get_local_rank()))
        try:
            with task_session(env_config, cls) as task:
                task.run()
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            traceback.print_exc()
            sys.exit(1)

    @classmethod
    def replay(cls, config: ReplayExperimentConfig) -> None:
        """Play ``config.motion`` back on the robot with the motion-playback plugin, then stop."""
        from holosoma.config_types.environment import EnvironmentConfig
        from holosoma.config_types.experiment import part_fields
        from holosoma.config_types.plugin import MotionPlaybackPluginConfig

        motion = config.motion
        if motion is None or not (motion.motion_file or motion.motion_dir):
            raise SystemExit("Replay needs a motion: --motion.motion_file=<npz>, or an exp: preset that tracks one.")
        files = [f"{motion.motion_dir.rstrip('/')}/*.npz"] if motion.motion_dir else [motion.motion_file]
        objects = list(config.scene.rigid_objects)[:1]
        playback = MotionPlaybackPluginConfig(motion_files=files, objects=objects)
        env = {**part_fields(EnvironmentConfig, config), "plugin": {**config.plugin, "motion_playback": playback}}
        cls.simulate(SimulationExperimentConfig(**env, task=config.task, run=config.run))

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
