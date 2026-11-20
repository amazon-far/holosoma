"""``RunConfig.device`` overrides the backend default for training and is not carried into eval."""

from __future__ import annotations

import dataclasses

import pytest

from holosoma.config_types.evaluation import EvalConfig
from holosoma.config_types.experiment import EvaluationExperimentConfig, TrainingExperimentConfig
from holosoma.config_values import experiment
from holosoma.config_values import simulator as simulator_values
from holosoma.simulator.session import resolve_simulation_device

pytestmark = pytest.mark.no_sim

_CLASSIC = dataclasses.replace(experiment.g1_29dof, simulator=simulator_values.mujoco)


def _with_device(device: str | None) -> TrainingExperimentConfig:
    return dataclasses.replace(_CLASSIC, run=dataclasses.replace(_CLASSIC.run, device=device))


@pytest.mark.parametrize(("device", "expected"), [(None, "cpu"), ("cuda:0", "cuda:0")])
def test_classic_mujoco_training_device(device: str | None, expected: str) -> None:
    config = _with_device(device)

    assert resolve_simulation_device(config.env_config(), local_rank=None) == expected


def test_eval_resolves_its_own_device() -> None:
    saved = _with_device("cuda:3")
    first = EvaluationExperimentConfig(task=f"{saved.task.rpartition('.')[0]}.evaluate", eval=EvalConfig())
    assert EvaluationExperimentConfig.from_training(saved, first).run.device is None
