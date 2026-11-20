from __future__ import annotations

from typing import ClassVar

import pytest

from holosoma import cli
from holosoma.config_types.experiment import (
    LOCOMOTION_EVALUATE,
    LOCOMOTION_TRAIN,
    SIMULATE,
    EvaluationExperimentConfig,
    SimulationExperimentConfig,
    TrainingExperimentConfig,
)

pytestmark = pytest.mark.no_sim


class _RecordingTask:
    """A task class whose entry function records the config it is given instead of running."""

    seen: ClassVar[list[SimulationExperimentConfig]] = []

    @classmethod
    def simulate(cls, config: SimulationExperimentConfig) -> None:
        cls.seen.append(config)


@pytest.mark.parametrize(
    ("task", "config_type"),
    [
        (SIMULATE, SimulationExperimentConfig),
        (LOCOMOTION_TRAIN, TrainingExperimentConfig),
        (LOCOMOTION_EVALUATE, EvaluationExperimentConfig),
    ],
)
def test_resolve_task_reads_the_config_type_from_the_annotation(task: str, config_type: type) -> None:
    assert cli.resolve_task(task)[2] is config_type


def test_resolve_task_rejects_a_method_that_is_not_a_task_function() -> None:
    with pytest.raises(SystemExit, match="not a runnable task function"):
        cli.resolve_task(
            "holosoma.envs.locomotion.locomotion_manager:LeggedRobotLocomotionManager.load_checkpoint_state"
        )


def test_run_builds_the_named_config_and_calls_its_function() -> None:
    _RecordingTask.seen.clear()
    task = f"{__name__}:_RecordingTask.simulate"

    cli.run([f"--task={task}", "env.simulator:sim-mjwarp", "--run.num_envs=3"])

    (config,) = _RecordingTask.seen
    assert (config.task, config.env.simulator.config.sim.fps, config.run.num_envs) == (task, 400, 3)


def test_run_rejects_a_task_class_that_does_not_match_the_experiment() -> None:
    wbt_train = "holosoma.envs.wbt.wbt_manager:WholeBodyTrackingManager.train"
    with pytest.raises(SystemExit, match="does not match the experiment"):
        cli.run(["exp:g1-29dof", f"--task={wbt_train}"])


def test_replay_base_plays_the_experiment_motion_kinematically() -> None:
    from holosoma.config_types.experiment import ReplayExperimentConfig
    from holosoma.config_values import experiment

    first = ReplayExperimentConfig()
    base = ReplayExperimentConfig.base(first, experiment.g1_29dof_wbt)

    assert base.motion is not None
    assert base.motion.motion_file.endswith(".npz")
    assert base.env.simulator.config.sim.kinematic_playback
    assert (base.env.randomization, base.env.termination) == (None, None)
    with pytest.raises(SystemExit, match="motion command"):
        ReplayExperimentConfig.base(first, experiment.g1_29dof)


def test_simulation_base_puts_the_experiment_world_on_its_interactive_simulator() -> None:
    from holosoma.config_values import experiment

    base = SimulationExperimentConfig.base(SimulationExperimentConfig(), experiment.g1_29dof)

    assert base.env.robot == experiment.g1_29dof.env.robot
    assert base.env.simulator.config.bridge.enabled and base.env.simulator.config.virtual_gantry.enabled
    assert (base.env.randomization, base.env.termination) == (None, None)
    assert (base.run.num_envs, base.run.project) == (1, experiment.g1_29dof.run.project)
