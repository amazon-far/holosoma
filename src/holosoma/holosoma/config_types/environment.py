"""The fields describing the simulated world every task builds; every config type inherits them."""

from __future__ import annotations

import dataclasses
from typing import Annotated

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import holosoma.config_values.plugin
import holosoma.config_values.randomization
import holosoma.config_values.robot
import holosoma.config_values.scene
import holosoma.config_values.sensor
import holosoma.config_values.simulator
import holosoma.config_values.termination
import holosoma.config_values.terrain
from holosoma.config_types.plugin import PluginConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.sensor import SensorConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.utils.config_registry import UseRegistry


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class EnvironmentConfig:
    """The robot, the world it is simulated in, and the terms ``BaseTask`` runs for every task."""

    robot: Annotated[RobotConfig, UseRegistry(holosoma.config_values.robot.ROBOT_REGISTRY)] = (
        holosoma.config_values.robot.g1_29dof
    )
    simulator: Annotated[SimulatorConfig, UseRegistry(holosoma.config_values.simulator.SIMULATOR_REGISTRY)] = (
        holosoma.config_values.simulator.isaacsim
    )
    terrain: Annotated[TerrainManagerCfg, UseRegistry(holosoma.config_values.terrain.TERRAIN_REGISTRY)] = (
        holosoma.config_values.terrain.terrain_locomotion_plane
    )
    scene: Annotated[SceneConfig, UseRegistry(holosoma.config_values.scene.SCENE_REGISTRY)] = (
        holosoma.config_values.scene.empty
    )
    # Mounted sensors, declared per-key on the CLI as ``sensor.<name>:<variant>`` (resolved from
    # SENSOR_REGISTRY), optionally with per-key field overrides. The dict key becomes the sensor name.
    sensor: dict[str, Annotated[SensorConfig, UseRegistry(holosoma.config_values.sensor.SENSOR_REGISTRY)]] = (
        dataclasses.field(default_factory=dict)
    )
    # Plugins, declared per-key as ``plugin.<key>:<variant>`` (resolved from PLUGIN_REGISTRY),
    # optionally with per-key field overrides. Includes the camera-frame egress sinks (ROS2 publish,
    # viz window, video record) and any custom plugin. Installed after simulator setup.
    plugin: dict[str, Annotated[PluginConfig, UseRegistry(holosoma.config_values.plugin.PLUGIN_REGISTRY)]] = (
        dataclasses.field(default_factory=dict)
    )
    randomization: Annotated[
        RandomizationManagerCfg | None, UseRegistry(holosoma.config_values.randomization.RANDOMIZATION_REGISTRY)
    ] = holosoma.config_values.randomization.none
    """Randomization terms run at setup, on reset and every step. Terms that act on policy actions or
    policy-task state (actuator gains, action delay, pushes) need a policy task."""
    termination: Annotated[
        TerminationManagerCfg | None, UseRegistry(holosoma.config_values.termination.TERMINATION_REGISTRY)
    ] = holosoma.config_values.termination.none
    """Termination terms that end an episode and reset its environment. The default ends none."""
