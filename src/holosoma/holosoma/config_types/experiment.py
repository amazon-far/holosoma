"""Top-level configuration types, one per workflow, each built from the parts that workflow uses."""

from __future__ import annotations

import dataclasses
import json
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Annotated, Any, cast

import yaml
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import holosoma.config_values.algo
import holosoma.config_values.simulator
from holosoma.config_types.algo import AlgoConfig
from holosoma.config_types.environment import EnvironmentConfig
from holosoma.config_types.evaluation import EvalConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.mdp import MDPConfig
from holosoma.config_types.run import RunConfig
from holosoma.config_types.video import VideoConfig
from holosoma.utils.config_registry import UseRegistry

if TYPE_CHECKING:
    from typing_extensions import Self

    from holosoma.config_types.env import EnvConfig
    from holosoma.envs.base_task.base_task import BaseTask

SIMULATE = "holosoma.envs.simulation_task:SimulationTask.simulate"
LOCOMOTION_TRAIN = "holosoma.envs.locomotion.locomotion_manager:LeggedRobotLocomotionManager.train"
LOCOMOTION_EVALUATE = "holosoma.envs.locomotion.locomotion_manager:LeggedRobotLocomotionManager.evaluate"


def task_class_path(task: str) -> str:
    """The dotted import path of the class a ``module:Class.function`` task names."""
    module, sep, attribute = task.partition(":")
    cls, dot, function = attribute.rpartition(".")
    if not (module and sep and cls and dot and function):
        raise ValueError(f"task must be 'module:Class.function', got {task!r}")
    return f"{module}.{cls}"


# Unknown fields are rejected, so code still passing a moved or removed field fails loudly.
_STRICT = ConfigDict(extra="forbid")


@dataclass(frozen=True, config=_STRICT)
class ExperimentConfig:
    """What every workflow's config holds: the task that runs it, the world, and how the run executes."""

    task: str = SIMULATE
    """The task class and the classmethod that runs this config, as ``module:Class.function``."""

    env: EnvironmentConfig = EnvironmentConfig()
    """The robot, the simulator and the world around it."""

    run: RunConfig = RunConfig()
    """How the run executes, and how it is named and logged."""

    _task_base = "holosoma.envs.base_task.base_task.BaseTask"
    """The class ``task``'s class must be or subclass."""

    def __post_init__(self) -> None:
        task_class_path(self.task)

    def env_config(self) -> EnvConfig:
        """The bundle of configs the task class is constructed from."""
        from holosoma.config_types.env import build_env_config

        return build_env_config(task_class_path(self.task), self.env, self.run)

    def create_task(self) -> AbstractContextManager[BaseTask]:
        """Return a context that builds ``task``'s class and owns its resources."""
        from holosoma.simulator.session import task_session
        from holosoma.utils.helpers import get_class

        return task_session(self.env_config(), get_class(self._task_base))

    def save_config(self, path: str) -> None:
        with open(path, "w") as file:
            yaml.safe_dump(self.to_serializable_dict(), file)

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the config."""
        # Directly using yaml.safe_dump does not handle string enums properly,
        # so round-trip through JSON to coerce typing objects into primitives.
        return cast("dict[str, Any]", json.loads(json.dumps(dataclasses.asdict(self))))


@dataclass(frozen=True, config=_STRICT)
class SimulationExperimentConfig(ExperimentConfig):
    """Simulation with no policy: one environment on an interactive simulator, in the viewer at real time."""

    env: EnvironmentConfig = EnvironmentConfig(simulator=holosoma.config_values.simulator.sim_mujoco)
    run: RunConfig = RunConfig(
        num_envs=1,
        headless=False,
        time_scale=1.0,
        logger=DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs"),
    )

    _task_base = "holosoma.envs.simulation_task.SimulationTask"


@dataclass(frozen=True, config=_STRICT)
class TrainingExperimentConfig(ExperimentConfig):
    """Training a policy on an MDP with an algorithm."""

    task: str = LOCOMOTION_TRAIN
    mdp: MDPConfig = MDPConfig()
    """The managers the policy is trained on."""

    algo: Annotated[AlgoConfig, UseRegistry(holosoma.config_values.algo.ALGO_REGISTRY)] = (
        holosoma.config_values.algo.ppo
    )
    """The algorithm, its hyperparameters and its training options."""

    _task_base = "holosoma.envs.policy_task.PolicyTask"

    def env_config(self) -> EnvConfig:
        from holosoma.config_types.env import build_env_config

        return build_env_config(task_class_path(self.task), self.env, self.run, self.mdp)

    def get_nightly_config(self) -> Self:
        """This config with the nightly check's shortened iteration count."""
        if self.algo.nightly is None:
            raise ValueError("nightly config is missing.")
        algo_config = dataclasses.replace(self.algo.config, num_learning_iterations=self.algo.nightly.iterations)
        return dataclasses.replace(self, algo=dataclasses.replace(self.algo, config=algo_config))  # type: ignore[arg-type]


@dataclass(frozen=True, config=_STRICT)
class EvaluationExperimentConfig(ExperimentConfig):
    """Evaluating a trained checkpoint on the MDP it was trained on."""

    task: str = LOCOMOTION_EVALUATE
    mdp: MDPConfig = MDPConfig()
    """The managers the policy was trained on, from the checkpoint's saved config."""

    eval: EvalConfig = EvalConfig()
    """What to evaluate and what to do while evaluating it."""

    run: RunConfig = RunConfig(num_envs=1, headless=False)

    _task_base = "holosoma.envs.policy_task.PolicyTask"

    def env_config(self) -> EnvConfig:
        from holosoma.config_types.env import build_env_config

        return build_env_config(task_class_path(self.task), self.env, self.run, self.mdp)
