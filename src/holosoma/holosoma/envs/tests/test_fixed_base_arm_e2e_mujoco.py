"""Full-pipeline CPU test: a FIXED-BASE robot through the manager env and PPO on MuJoCo classic.

The ``test_arm_fixed`` preset (3-DOF arm, ``asset.fix_base_link=True``) is run through
``LeggedRobotLocomotionManager`` with the shared fixed-base arm experiment
(``tests/simulators/_fixed_base_env.py``, also used by the cross-backend ``fixed_base_env_assert.py``
harness). The welded base must keep ``robot_root_states`` at the spawn pose across steps and resets (every
reset flushes the spawn pose, a same-pose no-op for the simulator) while the PD-held joints droop under
gravity, the terrain manager must run without feet-height queries (the robot declares no ``feet_height``
tag), and PPO must train for two iterations on CPU. One in-process root write off the spawn pose exercises the
other half of the fixed-base rule (a kinematic teleport, velocity ignored) and turns the following reset into a
real write back to spawn. The isaacgym/cuda counterpart for a floating robot is test_e2e.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holosoma.simulator.shared.root_states import root_pose_changed
from holosoma.train_agent import get_tyro_env_config, training_context
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from tests.simulators._fixed_base_env import LEARNING_ITERATIONS, OBS_DIM, fixed_base_arm_experiment
from tests.simulators._robot_presets import test_arm_fixed

pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU); outside tests/simulators/ so the run-target comes from this sub-tag.
pytestmark = pytest.mark.mujoco_classic

_NUM_ENVS = 1  # ClassicBackend is single-env
_ZERO_ACTION_STEPS = 50
_TELEPORT_DZ = 0.5


def test_fixed_base_arm_steps_and_learns(tmp_path: Path) -> None:
    seeding(0)
    tyro_config = fixed_base_arm_experiment("mujoco", _NUM_ENVS, tmp_path / "logs")
    tyro_env_config = get_tyro_env_config(tyro_config)
    with training_context(tyro_config) as context, context.simulation_session(
        get_class(tyro_config.env_class)(tyro_env_config, device="cpu")
    ) as env:
        assert env.robot_config.asset.fix_base_link
        assert env.dim_actions == len(test_arm_fixed.dof_names)
        assert env.dof_names == list(test_arm_fixed.dof_names)
        assert env.body_names == list(test_arm_fixed.body_names)
        # No feet_height tag on this robot: the terrain manager samples no feet heights.
        assert not env.terrain_manager.get_state("locomotion_terrain")._compute_feet_heights

        obs_dict = env.reset_all()
        sim = env.simulator
        assert not hasattr(sim, "hard_dof_pos_limits")  # the limits live under dof_pos_limits only
        init = test_arm_fixed.init_state
        expected = torch.zeros(_NUM_ENVS, 13)
        expected[:, :3] = env._get_env_origins() + torch.tensor(init.position)
        expected[:, 3:7] = torch.tensor(init.orientation_xyzw)
        spawn = sim.robot_root_states.clone()
        assert torch.allclose(spawn, expected, atol=1e-5), f"spawn {spawn} != {expected}"
        base_idx, tip_idx = sim.find_rigid_body_indice("base_link"), sim.find_rigid_body_indice("slider")

        def base_body_pose() -> torch.Tensor:
            # robot_root_states is a mirror on a fixed base; the base body's physics pose is the real check.
            return torch.cat([sim.rigid_body_pos_w[:, base_idx], sim.rigid_body_quat_w[:, base_idx]], dim=1)

        def base_body_at_spawn() -> bool:
            return not root_pose_changed(base_body_pose(), spawn[:, :7])

        assert base_body_at_spawn()

        actions = torch.zeros(_NUM_ENVS, env.dim_actions)
        for _ in range(_ZERO_ACTION_STEPS):
            obs_dict, rewards, dones, infos = env.step({"actions": actions})
            for group in ("actor_obs", "critic_obs"):
                assert tuple(obs_dict[group].shape) == (_NUM_ENVS, OBS_DIM)
                assert torch.isfinite(obs_dict[group]).all()
            assert torch.isfinite(rewards).all()
            assert len(dones) == _NUM_ENVS
            assert not torch.isnan(infos["to_log"]["average_episode_length"])
            # The welded base never moves: constant spawn pose, zero velocity.
            assert torch.allclose(sim.robot_root_states, spawn, atol=1e-6)
        assert base_body_at_spawn()
        # The joints are live: gravity droops the PD-held elbow away from the zero default pose.
        assert float((sim.dof_pos - env.default_dof_pos).abs().max()) > 1e-3

        # A pose change with a velocity is a kinematic teleport of the welded base: the physics pose and the tip
        # follow, the joints are untouched and the velocity columns are dropped.
        target = spawn.clone()
        target[:, 2] += _TELEPORT_DZ
        target[:, 7:] = 1.0
        dof_before, tip_before = sim.dof_pos.clone(), sim.rigid_body_pos_w[:, tip_idx].clone()
        sim.set_actor_root_state_tensor_robots(torch.tensor([0]), target)
        env.step({"actions": actions})
        assert not root_pose_changed(base_body_pose(), target[:, :7])
        assert torch.allclose(sim.robot_root_states[:, :7], target[:, :7], atol=1e-6)
        assert torch.equal(sim.robot_root_states[:, 7:], torch.zeros(_NUM_ENVS, 6))
        assert torch.allclose(
            sim.rigid_body_pos_w[:, tip_idx, 2] - tip_before[:, 2], torch.full((_NUM_ENVS,), _TELEPORT_DZ), atol=2e-3
        )
        assert torch.allclose(sim.dof_pos, dof_before, atol=5e-2)

        # A subset reset writes the spawn pose (plus a random velocity the welded base ignores) and flushes it
        # on the next step: after the teleport a real write back to spawn, the mirror reads zero velocity again.
        env.reset_envs_idx(torch.tensor([0]))
        env.step({"actions": actions})
        assert torch.allclose(sim.robot_root_states, spawn, atol=1e-6)
        assert base_body_at_spawn()

        algo_cls = get_class(tyro_config.algo._target_)
        agent = algo_cls(env=env, config=tyro_config.algo.config, log_dir=str(tmp_path / "ppo"), device="cpu")
        agent.setup()
        agent.attach_checkpoint_metadata(tyro_config, None)  # save() embeds the experiment config
        agent.learn()
        assert agent.current_learning_iteration == LEARNING_ITERATIONS - 1
        assert (tmp_path / "ppo" / f"model_{LEARNING_ITERATIONS - 1:05d}.pt").exists()
        # Resets during learning (init_at_random_ep_len timeouts) leave the welded base at spawn.
        assert torch.allclose(sim.robot_root_states, spawn, atol=1e-6)
        assert base_body_at_spawn()
