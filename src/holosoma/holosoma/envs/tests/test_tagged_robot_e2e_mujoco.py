"""CPU (MuJoCo classic) env-level runs of the TAGGED shipped g1 presets: locomotion and whole-body tracking.

The refactor rewired the locomotion/WBT consumers from ``env.feet_indices`` / ``termination_contact_indices`` to
``#tag`` specs resolved through ``BaseTask.resolve_*``; the only other CPU e2e (``test_fixed_base_arm_e2e_mujoco``)
runs an untagged 3-DOF arm. ``g1_29dof`` (loco) and ``g1_29dof_wbt`` are built on the MuJoCo ClassicBackend with
one env and checked against the preset: feet-height raycasts on the two ``#feet_height`` bodies feeding
``feet_phase``, the ``contact`` termination bodies, finite per-term rewards, and ``MotionCommand`` setting up on
the floating base (its fixed-base guard must not fire) and tracking the clip's bodies.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_values import experiment, simulator
from holosoma.train_agent import get_tyro_env_config, training_context
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.name_tags import expand_name_spec
from holosoma.utils.safe_torch_import import torch

pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU); outside tests/simulators/ so the run-target comes from this sub-tag.
pytestmark = pytest.mark.mujoco_classic

_NUM_ENVS = 1  # ClassicBackend is single-env
_STEPS = 20


def _on_mujoco(base: ExperimentConfig, log_root: Path) -> ExperimentConfig:
    """``base`` on the MuJoCo classic preset with one env; randomizers the backend lacks are skipped."""
    assert base.randomization is not None
    return dataclasses.replace(
        base,
        simulator=simulator.mujoco,
        training=dataclasses.replace(base.training, num_envs=_NUM_ENVS, headless=True, seed=0),
        randomization=dataclasses.replace(base.randomization, ignore_unsupported=True),
        logger=DisabledLoggerConfig(base_dir=str(log_root)),
    )


def _step_zero_actions(env: Any, steps: int) -> None:
    actions = torch.zeros(_NUM_ENVS, env.dim_actions)
    for _ in range(steps):
        obs_dict, rewards, dones, infos = env.step({"actions": actions})
        for group in ("actor_obs", "critic_obs"):
            assert torch.isfinite(obs_dict[group]).all()
        assert torch.isfinite(rewards).all() and len(dones) == _NUM_ENVS
        assert not torch.isnan(infos["to_log"]["average_episode_length"])


def _body_indices(env: Any, names: list[str]) -> list[int]:
    return [env.simulator.find_rigid_body_indice(name) for name in names]


def test_g1_locomotion_resolves_its_tags_and_steps(tmp_path: Path) -> None:
    seeding(0)
    tyro_config = _on_mujoco(experiment.g1_29dof, tmp_path / "logs")
    robot = tyro_config.robot
    tyro_env_config = get_tyro_env_config(tyro_config)
    with training_context(tyro_config) as context, context.simulation_session(
        get_class(tyro_config.env_class)(tyro_env_config, device="cpu")
    ) as env:
        assert not env.robot_config.asset.fix_base_link
        assert env.body_names == list(robot.body_names)
        env.reset_all()

        # Terrain: feet-height raycasts on the two #feet_height bodies, in the tag's (left, right) order.
        terrain_state = env.terrain_manager.get_state("locomotion_terrain")
        assert terrain_state._compute_feet_heights
        feet_height_bodies = robot.body_tags["#feet_height"]
        assert feet_height_bodies == ["left_foot_contact_point", "right_foot_contact_point"]
        assert terrain_state._feet_height_indices.tolist() == _body_indices(env, feet_height_bodies)
        assert terrain_state.feet_heights.shape == (_NUM_ENVS, 2)

        # Termination: the contact term's spec expands to the preset's bodies through the env resolver.
        assert tyro_config.termination is not None
        contact = tyro_config.termination.terms["contact"]
        expected_bodies = expand_name_spec(contact.params["bodies"], robot.body_tags, robot.body_names, kind="body")
        assert expected_bodies[0] == "pelvis" and len(expected_bodies) == 13
        assert env.robot_names.resolve_body_names(contact.params["bodies"]) == expected_bodies
        assert env.robot_names.resolve_body_indices(contact.params["bodies"]).tolist() == _body_indices(
            env, expected_bodies
        )

        _step_zero_actions(env, _STEPS)

        # Standing on flat ground: the contact points read at ground level; every term ran (feet_phase reads
        # feet_heights, penalty_feet_ori / penalty_close_feet_xy read #feet) and no contact body touched down.
        assert (terrain_state.feet_heights.abs() < 0.05).all()
        sums = env.reward_manager.episode_sums
        assert {"feet_phase", "penalty_feet_ori", "penalty_close_feet_xy"} <= set(sums)
        assert all(torch.isfinite(value).all() for value in sums.values())
        assert float(sums["feet_phase"]) != 0.0
        assert not env.termination_manager.terminated.any()


def test_g1_wbt_motion_command_sets_up_on_the_floating_base_and_steps(tmp_path: Path) -> None:
    seeding(0)
    tyro_config = _on_mujoco(experiment.g1_29dof_wbt, tmp_path / "logs")
    assert tyro_config.command is not None
    motion_cfg = tyro_config.command.setup_terms["motion_command"].params["motion_config"]
    tyro_env_config = get_tyro_env_config(tyro_config)
    with training_context(tyro_config) as context, context.simulation_session(
        get_class(tyro_config.env_class)(tyro_env_config, device="cpu")
    ) as env:
        assert not env.robot_config.asset.fix_base_link
        motion_command = env.command_manager.get_state("motion_command")
        assert motion_command is not None
        assert motion_command.ref_body_index == env.simulator.find_rigid_body_indice(motion_cfg.body_name_ref[0])
        assert motion_command.tracked_body_indexes.tolist() == _body_indices(env, motion_cfg.body_names_to_track)
        assert motion_command.motion.time_step_total > 0

        env.reset_all()
        # The reset wrote the clip's root pose (plus noise) into the floating root, not the preset's spawn pose.
        spawn = torch.tensor(tyro_config.robot.init_state.position)
        assert not torch.allclose(env.simulator.robot_root_states[0, :3], spawn, atol=1e-3)

        _step_zero_actions(env, _STEPS)

        sums = env.reward_manager.episode_sums
        assert "motion_global_ref_position_error_exp" in sums and "undesired_contacts" in sums
        assert all(torch.isfinite(value).all() for value in sums.values())
