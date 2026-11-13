"""Termination and randomization terms configured on RunSimConfig run on SimulationTask (MuJoCo Classic)."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_classic

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg  # noqa: E402
from holosoma.config_types.run_sim import RunSimConfig  # noqa: E402
from holosoma.config_values import termination  # noqa: E402
from holosoma.simulator.base_simulator.hooks import Phase  # noqa: E402

EPISODE_S = 0.1


def test_a_timeout_term_ends_and_restarts_the_episode():
    base = RunSimConfig()
    sim_cfg = base.simulator.config
    config = dataclasses.replace(
        base,
        simulator=dataclasses.replace(
            base.simulator,
            config=dataclasses.replace(sim_cfg, sim=dataclasses.replace(sim_cfg.sim, max_episode_length_s=EPISODE_S)),
        ),
        termination=termination.TERMINATION_REGISTRY.get("g1_29dof"),
        training=dataclasses.replace(base.training, headless=True),
        time_scale=0.0,
    )

    with config.create_task() as task:
        ends: list[int] = []
        task.hooks.add(Phase.EPISODE_END, ends.append, name="test.episode_end")
        task.reset_envs_idx()
        for _ in range(int(2.5 * EPISODE_S / task.dt)):
            task.step()

        assert len(ends) == 2
        assert task.episode_count.tolist() == [3]


def test_a_reset_randomization_term_sets_the_joint_state():
    base = RunSimConfig()
    scale_twice = RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
        params={
            "joint_pos_scale_range": [2.0, 2.0],
            "joint_pos_bias_range": [0.0, 0.0],
            "joint_vel_range": [0.0, 0.0],
            "randomize_dof_pos_bias": False,
        },
    )
    config = dataclasses.replace(
        base,
        randomization=RandomizationManagerCfg(reset_terms={"randomize_dof_state": scale_twice}),
        training=dataclasses.replace(base.training, headless=True),
        time_scale=0.0,
    )

    with config.create_task() as task:
        task.reset_envs_idx()

        assert bool((task.simulator.dof_pos == 2.0 * task.default_dof_pos).all())
