"""Top-level configuration types, one per workflow, each holding the fields and parts that workflow uses."""

from __future__ import annotations

import dataclasses
import json
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Annotated, Any, cast

import yaml
from loguru import logger
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import holosoma.config_values.algo
import holosoma.config_values.simulator
from holosoma.config_types.algo import AlgoConfig
from holosoma.config_types.command import MotionConfig
from holosoma.config_types.environment import EnvironmentConfig
from holosoma.config_types.evaluation import EvalConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.mdp import MDPConfig
from holosoma.config_types.run import RunConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.video import VideoConfig
from holosoma.utils.config_registry import UseRegistry

if TYPE_CHECKING:
    from typing_extensions import Self

    from holosoma.config_types.env import EnvConfig
    from holosoma.envs.base_task.base_task import BaseTask

SIMULATE = "holosoma.envs.simulation_task:SimulationTask.simulate"
REPLAY = "holosoma.envs.simulation_task:SimulationTask.replay"
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


def part_fields(part: type, config: Any) -> dict[str, Any]:
    """``config``'s values for the fields ``part`` declares."""
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(part)}


@dataclass(frozen=True, config=_STRICT)
class ExperimentConfig(EnvironmentConfig):
    """What every workflow's config holds: the world, the task that runs it, and how the run executes."""

    task: str = SIMULATE
    """The task class and the classmethod that runs this config, as ``module:Class.function``."""

    run: RunConfig = RunConfig()
    """How the run executes, and how it is named and logged."""

    _task_base = "holosoma.envs.base_task.base_task.BaseTask"
    """The class ``task``'s class must be or subclass."""

    def __post_init__(self) -> None:
        task_class_path(self.task)

    @classmethod
    def base(cls, first: Self, exp: TrainingExperimentConfig | None) -> Self:  # noqa: ARG003
        """The config the command line is applied to: this type's defaults.

        ``first`` is the command line parsed once against the defaults, for the few values a base
        depends on; ``exp`` is the ``exp:`` preset, if one was given.
        """
        return cls()

    def env_config(self) -> EnvConfig:
        """The bundle of configs the task class is constructed from."""
        from holosoma.config_types.env import build_env_config

        return build_env_config(task_class_path(self.task), self, self.run)

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
    """Simulation with no policy: one environment on a sim-to-sim simulator, in the viewer at real time."""

    simulator: Annotated[SimulatorConfig, UseRegistry(holosoma.config_values.simulator.SIMULATOR_REGISTRY)] = (
        holosoma.config_values.simulator.mujoco_sim2sim
    )
    run: RunConfig = RunConfig(
        num_envs=1,
        headless=False,
        time_scale=1.0,
        logger=DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs"),
    )

    _task_base = "holosoma.envs.simulation_task.SimulationTask"

    @classmethod
    def base(cls, first: Self, exp: TrainingExperimentConfig | None) -> Self:  # noqa: ARG003
        """The simulation defaults, or ``exp``'s world on the sim-to-sim variant of its simulator.

        The experiment's randomization and termination terms are dropped: they act on the policy's
        managers, which a simulation has none of.
        """
        default = cls()
        if exp is None:
            return default
        simulator = holosoma.config_values.simulator.sim2sim(exp.simulator)
        env = {
            **part_fields(EnvironmentConfig, exp),
            "simulator": simulator,
            "randomization": None,
            "termination": None,
        }
        run = dataclasses.replace(default.run, project=exp.run.project, name=exp.run.name)
        return cls(**env, run=run)


@dataclass(frozen=True, config=_STRICT)
class ReplayExperimentConfig(ExperimentConfig):
    """Playing a whole-body-tracking motion back on the robot kinematically, in the viewer at real time."""

    task: str = REPLAY
    motion: MotionConfig | None = None
    """The motion to play: ``motion_file``, or every clip in ``motion_dir``."""

    run: RunConfig = SimulationExperimentConfig().run

    _task_base = "holosoma.envs.simulation_task.SimulationTask"

    @classmethod
    def base(cls, first: Self, exp: TrainingExperimentConfig | None) -> Self:  # noqa: ARG003
        """``exp``'s world with kinematic playback on, and the motion its motion command tracks.

        The experiment's randomization and termination terms are dropped: they act on dynamics and
        on the policy's managers, and playback is kinematic.
        """
        if exp is None:
            return cls()
        terms = exp.command.setup_terms if exp.command is not None else {}
        term = terms.get("motion_command")
        motion = term.params.get("motion_config") if term is not None else None
        if not isinstance(motion, MotionConfig):
            raise SystemExit(f"Replay needs an experiment with a motion command; {exp.task!r} has none.")
        simulator = exp.simulator
        sim = dataclasses.replace(simulator.config.sim, kinematic_playback=True)
        env = {
            **part_fields(EnvironmentConfig, exp),
            "simulator": dataclasses.replace(simulator, config=dataclasses.replace(simulator.config, sim=sim)),
            "randomization": None,
            "termination": None,
        }
        run = dataclasses.replace(cls().run, project=exp.run.project, name=exp.run.name)
        return cls(**env, motion=motion, run=run)


@dataclass(frozen=True, config=_STRICT)
class TrainingExperimentConfig(MDPConfig, ExperimentConfig):
    """Training a policy on an MDP with an algorithm."""

    task: str = LOCOMOTION_TRAIN

    algo: Annotated[AlgoConfig, UseRegistry(holosoma.config_values.algo.ALGO_REGISTRY)] = (
        holosoma.config_values.algo.ppo
    )
    """The algorithm, its hyperparameters and its training options."""

    _task_base = "holosoma.envs.policy_task.PolicyTask"

    def env_config(self) -> EnvConfig:
        from holosoma.config_types.env import build_env_config

        return build_env_config(task_class_path(self.task), self, self.run, self)

    @classmethod
    def base(cls, first: Self, exp: TrainingExperimentConfig | None) -> Self:  # noqa: ARG003
        """The ``exp:`` preset, or this type's defaults without one."""
        return exp if isinstance(exp, cls) else cls()

    def get_nightly_config(self) -> Self:
        """This config with the nightly check's shortened iteration count."""
        if self.algo.nightly is None:
            raise ValueError("nightly config is missing.")
        algo_config = dataclasses.replace(self.algo.config, num_learning_iterations=self.algo.nightly.iterations)
        return dataclasses.replace(self, algo=dataclasses.replace(self.algo, config=algo_config))  # type: ignore[arg-type]


@dataclass(frozen=True, config=_STRICT)
class EvaluationExperimentConfig(MDPConfig, ExperimentConfig):
    """Evaluating a trained checkpoint on the MDP it was trained on."""

    task: str = LOCOMOTION_EVALUATE

    eval: EvalConfig = EvalConfig()
    """What to evaluate and what to do while evaluating it."""

    run: RunConfig = RunConfig(num_envs=1, headless=False)

    _task_base = "holosoma.envs.policy_task.PolicyTask"

    @classmethod
    def base(cls, first: Self, exp: TrainingExperimentConfig | None) -> Self:  # noqa: ARG003
        """The evaluation config for ``first.eval.checkpoint``'s saved training config (see ``from_training``)."""
        from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config

        if first.eval.checkpoint is None:
            raise SystemExit("Evaluation needs --eval.checkpoint=<path or wandb:// URI>.")
        saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=first.eval.checkpoint))
        return cls.from_training(saved, first)

    @classmethod
    def from_training(cls, saved: TrainingExperimentConfig, first: Self) -> Self:
        """``saved``'s world and MDP with evaluation's defaults, keeping ``first``'s task and eval part.

        One env in the viewer, logger off, a deterministic spawn and no episode time limit; the saved
        seed and run naming are kept. ``first.task`` must name the class ``saved`` trained. A saved
        Isaac Gym simulator, which is retired, is replaced by ``first``'s simulator.
        """
        import holosoma.config_values.logger

        if task_class_path(first.task) != task_class_path(saved.task):
            raise SystemExit(
                f"task {first.task!r} does not match the checkpoint's class {task_class_path(saved.task)!r}."
            )
        simulator = saved.simulator
        if simulator.config.name == "isaacgym":
            logger.warning(f"The checkpoint ran on Isaac Gym, which is retired; using {first.simulator.config.name}.")
            simulator = first.simulator
        terrain = saved.terrain
        spawn = dataclasses.replace(terrain.terrain_term.spawn, randomize_tiles=False, xy_offset_range=0.0)
        sim = dataclasses.replace(simulator.config.sim, max_episode_length_s=100000.0)
        env = {
            **part_fields(EnvironmentConfig, saved),
            "terrain": dataclasses.replace(
                terrain, terrain_term=dataclasses.replace(terrain.terrain_term, spawn=spawn)
            ),
            "simulator": dataclasses.replace(simulator, config=dataclasses.replace(simulator.config, sim=sim)),
        }
        run = dataclasses.replace(
            saved.run, device=None, num_envs=1, headless=False, logger=holosoma.config_values.logger.disabled
        )
        return cls(**env, **part_fields(MDPConfig, saved), task=first.task, eval=first.eval, run=run)

    def env_config(self) -> EnvConfig:
        from holosoma.config_types.env import build_env_config

        return build_env_config(task_class_path(self.task), self, self.run, self)
