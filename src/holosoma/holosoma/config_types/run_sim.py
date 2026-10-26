"""
Configuration types for holosoma run_sim.py script.

This module provides a minimal configuration structure for direct simulation,
following the same pattern as ExperimentConfig. Direct simulations are used for
development and running sim2sim inference.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

import holosoma.config_values.plugin
import holosoma.config_values.robot
import holosoma.config_values.run_sim
import holosoma.config_values.scene
import holosoma.config_values.sensor
import holosoma.config_values.terrain
from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig, LoggerConfig
from holosoma.config_types.plugin import PluginConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.sensor import SensorConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.config_types.video import VideoConfig
from holosoma.utils.config_registry import UseRegistry

if TYPE_CHECKING:
    from holosoma.envs.simulation_task import SimulationTask


def default_training_config() -> TrainingConfig:
    """Create minimal training config for direct simulation."""
    return TrainingConfig(num_envs=1, headless=False, seed=42, torch_deterministic=False)


def default_logger_config() -> LoggerConfig:
    """Create minimal logger config for direct simulation."""
    return DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs")


# Use sim2sim-optimized configs from config_values.run_sim
SIMULATOR_DEFAULTS = holosoma.config_values.run_sim.RUN_SIM_REGISTRY


@dataclass(frozen=True)
class RunSimConfig:
    """
    Minimal configuration for direct simulation via run_sim.py.

    Usage Examples:
        python -m holosoma.run_sim simulator:mujoco robot:t1 terrain:terrain-locomotion-plane
        python -m holosoma.run_sim simulator:isaacsim robot:g1 terrain:terrain-locomotion-mix
    """

    # Core components for simulation - using Annotated subcommands like ExperimentConfig
    simulator: Annotated[SimulatorConfig, UseRegistry(SIMULATOR_DEFAULTS)] = holosoma.config_values.run_sim.mujoco

    robot: Annotated[RobotConfig, UseRegistry(holosoma.config_values.robot.ROBOT_REGISTRY)] = (
        holosoma.config_values.robot.g1_29dof
    )

    terrain: Annotated[TerrainManagerCfg, UseRegistry(holosoma.config_values.terrain.TERRAIN_REGISTRY)] = (
        holosoma.config_values.terrain.terrain_locomotion_plane
    )

    scene: Annotated[SceneConfig, UseRegistry(holosoma.config_values.scene.SCENE_REGISTRY)] = (
        holosoma.config_values.scene.empty
    )

    # Plugins: declare on the CLI as ``plugin.<key>:<variant>`` (resolved from
    # PLUGIN_REGISTRY), optionally with per-key leaf overrides
    # (e.g. ``--plugin.<key>.<field>=<value>``). Passed through to ``FullSimConfig.plugin`` and
    # instantiated against the live simulator after backend setup. Empty by default.
    plugin: dict[str, Annotated[PluginConfig, UseRegistry(holosoma.config_values.plugin.PLUGIN_REGISTRY)]] = field(
        default_factory=dict
    )

    # Mounted sensors, declared per-key on the CLI as ``sensor.<name>:<variant>`` (resolved from
    # SENSOR_REGISTRY), optionally with per-key field overrides. The dict key becomes the sensor name.
    sensor: dict[str, Annotated[SensorConfig, UseRegistry(holosoma.config_values.sensor.SENSOR_REGISTRY)]] = field(
        default_factory=dict
    )

    # Minimal configs needed for FullSimConfig
    training: TrainingConfig = field(default_factory=default_training_config)
    logger: LoggerConfig = field(default_factory=default_logger_config)

    env_class: str = "holosoma.envs.simulation_task.SimulationTask"
    """Task class to construct. Must be a ``SimulationTask`` subclass."""

    # Direct simulation timing control
    viewer_dt: float = 1 / 60.0
    """Sim-time period between viewer renders, in seconds (60 FPS default)."""

    time_scale: float = 1.0
    """Sim-time to real-time ratio for the simulation loop: 1 paces sim time to match real
    time, 2 runs twice as fast, 0 disables pacing entirely (run as fast as possible)."""

    device: str | None = None
    """Device to use for simulation. None (the default) auto-detects based on the simulator
    backend: cuda:0 for MuJoCo Warp, cuda:0 for IsaacSim when CUDA is available (else cpu), and
    cpu for classic MuJoCo. Pass an explicit value (e.g. "cpu" or "cuda:1") to override.
    """

    def __post_init__(self) -> None:
        if self.time_scale < 0:
            raise ValueError(f"RunSimConfig.time_scale must be >= 0, got {self.time_scale}.")
        if self.viewer_dt <= 0:
            raise ValueError(f"RunSimConfig.viewer_dt must be > 0, got {self.viewer_dt}.")

    def create_task(self) -> AbstractContextManager[SimulationTask]:
        """Return a context that builds the configured simulation task and owns its resources."""
        from holosoma.config_types.env import get_run_sim_env_config
        from holosoma.envs.simulation_task import SimulationTask
        from holosoma.simulator.session import task_session

        return task_session(get_run_sim_env_config(self), SimulationTask)
