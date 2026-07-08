"""
Configuration types for holosoma run_sim.py script.

This module provides a minimal configuration structure for direct simulation,
following the same pattern as ExperimentConfig. Direct simulations are used for
development and running sim2sim inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tyro
from typing_extensions import Annotated

import holosoma.config_values.robot
import holosoma.config_values.run_sim
import holosoma.config_values.scene
import holosoma.config_values.terrain
from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig, LoggerConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.config_types.video import VideoConfig


def default_training_config() -> TrainingConfig:
    """Create minimal training config for direct simulation."""
    return TrainingConfig(num_envs=1, headless=False, seed=42, torch_deterministic=False)


def default_logger_config() -> LoggerConfig:
    """Create minimal logger config for direct simulation."""
    return DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs")


# Use sim2sim-optimized configs from config_values.run_sim
SIMULATOR_DEFAULTS = holosoma.config_values.run_sim.DEFAULTS


@dataclass(frozen=True)
class RunSimConfig:
    """
    Minimal configuration for direct simulation via run_sim.py.

    Usage Examples:
        python -m holosoma.run_sim simulator:mujoco robot:t1 terrain:terrain-locomotion-plane
        python -m holosoma.run_sim simulator:isaacgym robot:g1 terrain:terrain-locomotion-mix
    """

    # Core components for simulation - using Annotated subcommands like ExperimentConfig
    simulator: Annotated[
        SimulatorConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(SIMULATOR_DEFAULTS)),
    ] = holosoma.config_values.run_sim.mujoco

    robot: Annotated[
        RobotConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.robot.DEFAULTS)),
    ] = holosoma.config_values.robot.g1_29dof

    terrain: Annotated[
        TerrainManagerCfg,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.terrain.DEFAULTS)),
    ] = holosoma.config_values.terrain.terrain_locomotion_plane

    scene: Annotated[
        SceneConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.scene.DEFAULTS)),
    ] = holosoma.config_values.scene.empty

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
