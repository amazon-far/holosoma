"""A full reset keeps setup-time randomization on the locomotion task (MuJoCo ClassicBackend)."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_classic

from holosoma.config_values import experiment  # noqa: E402
from holosoma.config_values.simulator import SIMULATOR_REGISTRY  # noqa: E402


def test_full_reset_keeps_the_setup_joint_position_bias():
    preset = experiment.g1_29dof
    terms = dict(preset.randomization.setup_terms)
    bias_term = terms["setup_dof_pos_bias"]
    terms["setup_dof_pos_bias"] = dataclasses.replace(bias_term, params={**bias_term.params, "enabled": True})
    config = dataclasses.replace(
        preset,
        simulator=SIMULATOR_REGISTRY["mujoco"],
        randomization=dataclasses.replace(preset.randomization, setup_terms=terms),
        training=dataclasses.replace(preset.training, num_envs=1, headless=True),
    )

    with config.create_task() as env:
        biased = env.default_dof_pos.clone()
        assert not bool((biased == env.default_dof_pos_base).all())

        env.reset_envs_idx()
        env.compute_reset_observations()

        assert bool((env.default_dof_pos == biased).all())
