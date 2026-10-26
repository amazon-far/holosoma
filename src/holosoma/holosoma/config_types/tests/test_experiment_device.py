"""``ExperimentConfig.device`` overrides the backend default for training and is not carried into eval."""

from __future__ import annotations

import dataclasses

import pytest

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_values import experiment
from holosoma.config_values import simulator as simulator_values
from holosoma.simulator.session import resolve_simulation_device

pytestmark = pytest.mark.no_sim

_CLASSIC = dataclasses.replace(experiment.g1_29dof, simulator=simulator_values.mujoco)


@pytest.mark.parametrize(("device", "expected"), [(None, "cpu"), ("cuda:0", "cuda:0")])
def test_classic_mujoco_training_device(device: str | None, expected: str) -> None:
    config = dataclasses.replace(_CLASSIC, device=device)

    assert resolve_simulation_device(get_tyro_env_config(config), local_rank=None) == expected


def test_eval_resolves_its_own_device() -> None:
    assert dataclasses.replace(_CLASSIC, device="cuda:3").get_eval_config().device is None
