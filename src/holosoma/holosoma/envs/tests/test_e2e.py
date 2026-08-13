import dataclasses

import pytest

from holosoma.config_values import experiment
from holosoma.train_agent import get_tyro_env_config, training_context
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch

# Builds the default (isaacgym) sim on CUDA; outside tests/simulators/ so set umbrella explicitly.
pytestmark = pytest.mark.isaacgym


def test_e2e_step():
    seeding(0)
    num_envs = 16
    device = "cuda"

    tyro_config = dataclasses.replace(
        experiment.g1_29dof, training=dataclasses.replace(experiment.g1_29dof.training, num_envs=num_envs)
    )

    with training_context(tyro_config):
        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(tyro_config.env_class)(tyro_env_config, device=device)
        obs_dict = env.reset_all()
        assert len(obs_dict["actor_obs"]) == num_envs
        assert len(obs_dict["critic_obs"]) == num_envs

        actions_dim = tyro_config.robot.actions_dim
        actions = torch.zeros(num_envs, actions_dim, device=device)

        n_dones = 0
        for _ in range(100):
            obs_dict, rewards, dones, infos = env.step({"actions": actions})
            assert len(obs_dict["actor_obs"]) == num_envs
            assert len(obs_dict["critic_obs"]) == num_envs
            assert len(rewards) == num_envs
            assert len(dones) == num_envs
            assert len(infos["episode"]["rew_tracking_lin_vel"]) == dones.sum()
            assert len(infos["episode_all"]["rew_tracking_lin_vel"]) == num_envs
            n_dones += dones.sum()
            assert not torch.isnan(infos["to_log"]["average_episode_length"])

        assert n_dones > 0

        env.reset_all()
        assert env.average_episode_length == 0.0
