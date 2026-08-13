"""
Configuration types for holosoma run_sim.py script.

This module provides a minimal configuration structure for direct simulation,
following the same pattern as ExperimentConfig. Direct simulations are used for
development and running sim2sim inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Annotated

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
from holosoma.config_types.sensor import CameraSensorConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.config_types.video import VideoConfig
from holosoma.utils.config_registry import UseRegistry


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
        python -m holosoma.run_sim simulator:isaacgym robot:g1 terrain:terrain-locomotion-mix
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
    # instantiated against the live simulator in ``BaseSimulator.__init__``. Empty by default.
    plugin: dict[str, Annotated[PluginConfig, UseRegistry(holosoma.config_values.plugin.PLUGIN_REGISTRY)]] = field(
        default_factory=dict
    )

    # Mounted cameras, declared per-key on the CLI as ``--sensor.<name>:<variant>`` (resolved from
    # CAMERA_REGISTRY), optionally with per-key field overrides (e.g. ``--sensor.<name>.width 224``).
    # The dict key becomes the sensor name. Empty by default (no cameras).
    sensor: dict[str, Annotated[CameraSensorConfig, UseRegistry(holosoma.config_values.sensor.CAMERA_REGISTRY)]] = (
        field(default_factory=dict)
    )

    # Minimal configs needed for FullSimConfig
    training: TrainingConfig = field(default_factory=default_training_config)
    logger: LoggerConfig = field(default_factory=default_logger_config)

    # Optional environment wrapper (only if needed for compatibility)
    env_class: str | None = None

    # Direct simulation timing control
    viewer_dt: float = 1 / 60.0
    """Viewer refresh rate in seconds (60 FPS default).

    Only used by run_sim.py for real-time display synchronization.
    """

    device: str | None = None
    """Device to use for simulation. None (the default) auto-detects based on the simulator
    backend: cuda:0 for MuJoCo Warp, cuda:0 for IsaacSim/IsaacGym when CUDA is available (else
    cpu), and cpu for classic MuJoCo. Pass an explicit value (e.g. "cpu" or "cuda:1") to override.
    """
