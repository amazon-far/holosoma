from __future__ import annotations

from typing import TYPE_CHECKING

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
from holosoma.config_types.sensor import CameraSensorConfig, SensorConfig, validate_camera_dict
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_types.terrain import TerrainManagerCfg

if TYPE_CHECKING:
    from holosoma.config_types.run_sim import RunSimConfig


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
    training: TrainingConfig
    logger: LoggerConfig
    plugin: dict[str, PluginConfig]
    """Plugins to install on the simulator (key -> resolved PluginConfig), including the
    ROS2/viz/video egress sinks. Installed after the concrete simulator backend is ready."""

    device: str | None = None
    """Requested device. ``None`` resolves from the simulator backend and process rank."""

    time_scale: float = 0.0
    """Sim-time to real-time ratio for physics-step pacing. ``0`` disables pacing."""

    viewer_dt: float = 1 / 60.0
    """Sim-time period between viewer renders, in seconds."""

    def __post_init__(self) -> None:
        if self.time_scale < 0:
            raise ValueError(f"EnvConfig.time_scale must be >= 0, got {self.time_scale}.")
        if self.viewer_dt <= 0:
            raise ValueError(f"EnvConfig.viewer_dt must be > 0, got {self.viewer_dt}.")


def _validate_cameras(sensors: dict[str, SensorConfig]) -> None:
    validate_camera_dict({name: sensor for name, sensor in sensors.items() if isinstance(sensor, CameraSensorConfig)})


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
    _validate_cameras(tyro_config.sensor)
    return EnvConfig(
        env_class=tyro_config.env_class,
        training=tyro_config.training,
        simulator=tyro_config.simulator,
        scene=tyro_config.scene,
        # The CLI declares sensors per-key in the dynamic ``sensor`` dict (key = sensor name).
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


def get_run_sim_env_config(config: RunSimConfig) -> EnvConfig:
    """Convert RunSimConfig to EnvConfig for a manager-free simulation task.

    Parameters
    ----------
    config : RunSimConfig
        The simulation configuration.

    Returns
    -------
    EnvConfig
        Environment configuration with no policy managers.
    """
    _validate_cameras(config.sensor)
    return EnvConfig(
        env_class=config.env_class,
        training=config.training,
        simulator=config.simulator,
        scene=config.scene,
        sensors=dict(config.sensor),
        plugin=dict(config.plugin),
        terrain=config.terrain,
        observation=None,
        action=None,
        reward=None,
        termination=None,
        randomization=None,
        command=None,
        curriculum=None,
        robot=config.robot,
        logger=config.logger,
        device=config.device,
        time_scale=config.time_scale,
        viewer_dt=config.viewer_dt,
    )
