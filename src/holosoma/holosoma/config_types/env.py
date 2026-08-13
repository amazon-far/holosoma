from __future__ import annotations

from pydantic.dataclasses import dataclass

from holosoma.config_types.action import ActionManagerCfg
from holosoma.config_types.command import CommandManagerCfg
from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.logger import LoggerConfig
from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_types.plugin import PluginConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.sensor import CameraSensorConfig, validate_camera_dict
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_types.terrain import TerrainManagerCfg


@dataclass(frozen=True)
class EnvConfig:
    """Collection of configs needed for constructing env classes."""

    env_class: str

    simulator: SimulatorConfig
    scene: SceneConfig
    sensors: dict[str, CameraSensorConfig]
    terrain: TerrainManagerCfg
    observation: ObservationManagerCfg | None
    action: ActionManagerCfg | None
    reward: RewardManagerCfg | None
    termination: TerminationManagerCfg | None
    randomization: RandomizationManagerCfg | None
    command: CommandManagerCfg | None
    curriculum: CurriculumManagerCfg | None
    robot: RobotConfig
    training: TrainingConfig
    logger: LoggerConfig
    plugin: dict[str, PluginConfig]
    """Plugins to install on the simulator (key -> resolved PluginConfig), including the
    ROS2/viz/video egress sinks. Threaded into FullSimConfig; installed in ``BaseSimulator.__init__``."""


def get_tyro_env_config(tyro_config: ExperimentConfig) -> EnvConfig:
    """Convert ExperimentConfig to EnvConfig for environment construction.

    Parameters
    ----------
    tyro_config : ExperimentConfig
        The experiment configuration containing all settings.

    Returns
    -------
    EnvConfig
        Environment configuration with extracted fields.
    """
    # Cross-camera validation (Warp render-flag agreement) across the assembled camera dict.
    validate_camera_dict(tyro_config.sensor)
    return EnvConfig(
        env_class=tyro_config.env_class,
        training=tyro_config.training,
        simulator=tyro_config.simulator,
        scene=tyro_config.scene,
        # The CLI declares cameras per-key in the dynamic ``sensor`` dict (key = sensor name).
        sensors=dict(tyro_config.sensor),
        plugin=dict(tyro_config.plugin),
        terrain=tyro_config.terrain,
        observation=tyro_config.observation,
        action=tyro_config.action,
        reward=tyro_config.reward,
        termination=tyro_config.termination,
        randomization=tyro_config.randomization,
        command=tyro_config.command,
        curriculum=tyro_config.curriculum,
        robot=tyro_config.robot,
        logger=tyro_config.logger,
    )
