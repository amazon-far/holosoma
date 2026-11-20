from __future__ import annotations

import dataclasses
import datetime
import json
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Annotated, Any, cast

import yaml
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import holosoma.config_values.action
import holosoma.config_values.algo
import holosoma.config_values.command
import holosoma.config_values.curriculum
import holosoma.config_values.logger
import holosoma.config_values.observation
import holosoma.config_values.plugin
import holosoma.config_values.randomization
import holosoma.config_values.reward
import holosoma.config_values.robot
import holosoma.config_values.run
import holosoma.config_values.scene
import holosoma.config_values.sensor
import holosoma.config_values.simulator
import holosoma.config_values.task
import holosoma.config_values.termination
import holosoma.config_values.terrain
from holosoma.config_types.action import ActionManagerCfg
from holosoma.config_types.algo import AlgoConfig
from holosoma.config_types.command import CommandManagerCfg
from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_types.logger import LoggerConfig
from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_types.plugin import PluginConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.run import RunConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.sensor import SensorConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.task import TaskConfigType
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.utils.config_registry import UseRegistry

if TYPE_CHECKING:
    from holosoma.envs.base_task.base_task import BaseTask


def now_timestamp() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class NightlyConfig:
    iterations: int
    metrics: dict[str, list[float | str]]


# Unknown fields are rejected, so code still passing a moved or removed field fails loudly.
@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class HolosomaConfig:
    """Top-level configuration for every workflow; ``task`` selects the workflow."""

    task: Annotated[TaskConfigType, UseRegistry(holosoma.config_values.task.TASK_REGISTRY)] = (
        holosoma.config_values.task.train
    )
    """Workflow to run (``task:<name>``); ``env_class`` is the environment it builds."""

    env_class: str = "holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager"

    run: Annotated[RunConfig, UseRegistry(holosoma.config_values.run.RUN_REGISTRY)] = holosoma.config_values.run.batch
    """How the run executes (``run:<profile>``): environment count, viewer, seed, device, pacing."""

    project: str = "default_project"
    """Project the run is filed under: its log and checkpoint directory and its W&B project."""

    name: str = "run"
    """Name the run is filed under: its log and checkpoint directory, and the base of its W&B run name."""

    algo: Annotated[AlgoConfig, UseRegistry(holosoma.config_values.algo.ALGO_REGISTRY)] = (
        holosoma.config_values.algo.ppo
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
    observation: Annotated[
        ObservationManagerCfg | None, UseRegistry(holosoma.config_values.observation.OBSERVATION_REGISTRY)
    ] = holosoma.config_values.observation.none
    action: Annotated[ActionManagerCfg | None, UseRegistry(holosoma.config_values.action.ACTION_REGISTRY)] = (
        holosoma.config_values.action.none
    )
    reward: Annotated[RewardManagerCfg | None, UseRegistry(holosoma.config_values.reward.REWARD_REGISTRY)] = (
        holosoma.config_values.reward.none
    )
    termination: Annotated[
        TerminationManagerCfg | None, UseRegistry(holosoma.config_values.termination.TERMINATION_REGISTRY)
    ] = holosoma.config_values.termination.none
    randomization: Annotated[
        RandomizationManagerCfg | None, UseRegistry(holosoma.config_values.randomization.RANDOMIZATION_REGISTRY)
    ] = holosoma.config_values.randomization.none
    command: Annotated[CommandManagerCfg | None, UseRegistry(holosoma.config_values.command.COMMAND_REGISTRY)] = (
        holosoma.config_values.command.none
    )
    curriculum: Annotated[
        CurriculumManagerCfg | None, UseRegistry(holosoma.config_values.curriculum.CURRICULUM_REGISTRY)
    ] = holosoma.config_values.curriculum.none
    robot: Annotated[RobotConfig, UseRegistry(holosoma.config_values.robot.ROBOT_REGISTRY)] = (
        holosoma.config_values.robot.g1_29dof
    )
    logger: Annotated[LoggerConfig, UseRegistry(holosoma.config_values.logger.LOGGER_REGISTRY)] = (
        holosoma.config_values.logger.disabled
    )
    nightly: NightlyConfig | None = None

    def get_nightly_config(self) -> HolosomaConfig:
        if self.nightly is None:
            raise ValueError("nightly config is missing.")

        return dataclasses.replace(
            self,
            algo=dataclasses.replace(
                self.algo,
                config=dataclasses.replace(  # type: ignore[arg-type]
                    self.algo.config,
                    num_learning_iterations=self.nightly.iterations,
                ),
            ),
        )

    def create_task(self) -> AbstractContextManager[BaseTask]:
        """Return a context that builds the configured environment and owns its resources.

        ``env_class`` must be or subclass the selected task's ``env_base``.
        """
        from holosoma.config_types.env import get_tyro_env_config
        from holosoma.simulator.session import task_session
        from holosoma.utils.helpers import get_class

        return task_session(get_tyro_env_config(self), get_class(self.task.env_base))

    def save_config(self, path: str) -> None:
        with open(path, "w") as file:
            yaml.safe_dump(self.to_serializable_dict(), file)

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the config."""
        # Directly using yaml.safe_dump does not handle string enums properly,
        # so round-trip through JSON to coerce typing objects into primitives.
        return cast("dict[str, Any]", json.loads(json.dumps(dataclasses.asdict(self))))


ExperimentConfig = HolosomaConfig
"""Deprecated name for ``HolosomaConfig``."""
