from __future__ import annotations

from pydantic.dataclasses import dataclass

from holosoma.config_types.action import ActionManagerCfg
from holosoma.config_types.command import CommandManagerCfg
from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_types.environment import EnvironmentConfig
from holosoma.config_types.mdp import MDPConfig
from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_types.plugin import PluginConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.run import RunConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorConfig, validate_camera_dict
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_types.terrain import TerrainManagerCfg


@dataclass(frozen=True)
class EnvConfig:
    """Collection of configs needed for constructing env classes."""

    env_class: str

    simulator: SimulatorConfig
    scene: SceneConfig
    sensors: dict[str, SensorConfig]
    terrain: TerrainManagerCfg
    observation: ObservationManagerCfg | None
    action: ActionManagerCfg | None
    reward: RewardManagerCfg | None
    termination: TerminationManagerCfg | None
    randomization: RandomizationManagerCfg | None
    command: CommandManagerCfg | None
    curriculum: CurriculumManagerCfg | None
    robot: RobotConfig
    run: RunConfig
    plugin: dict[str, PluginConfig]
    """Plugins to install on the simulator (key -> resolved PluginConfig), including the
    ROS2/viz/video egress sinks. Installed after the concrete simulator backend is ready."""


def build_env_config(env_class: str, env: EnvironmentConfig, run: RunConfig, mdp: MDPConfig | None = None) -> EnvConfig:
    """Bundle a config's parts for environment construction; ``mdp`` is None for tasks without MDP managers."""
    validate_camera_dict(
        {name: sensor for name, sensor in env.sensor.items() if isinstance(sensor, CameraSensorConfig)}
    )
    return EnvConfig(
        env_class=env_class,
        simulator=env.simulator,
        scene=env.scene,
        # The CLI declares sensors per-key in the dynamic ``sensor`` dict (key = sensor name).
        sensors=dict(env.sensor),
        plugin=dict(env.plugin),
        terrain=env.terrain,
        termination=env.termination,
        randomization=env.randomization,
        robot=env.robot,
        observation=mdp.observation if mdp else None,
        action=mdp.action if mdp else None,
        reward=mdp.reward if mdp else None,
        command=mdp.command if mdp else None,
        curriculum=mdp.curriculum if mdp else None,
        run=run,
    )
