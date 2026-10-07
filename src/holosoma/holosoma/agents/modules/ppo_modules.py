from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from holosoma.config_types.algo import ModuleConfig

from .modules import BaseModule

MIN_ACTION_STD = 1.0e-6


class PPOActor(nn.Module):
    def __init__(
        self,
        obs_dim_dict,
        module_config_dict: ModuleConfig,
        num_actions,
        init_noise_std,
        history_length: dict[str, int],
    ):
        super().__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict, history_length)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_noise_std = module_config_dict.min_noise_std
        self.min_mean_noise_std = module_config_dict.min_mean_noise_std
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        self.project_action_std()
        print(f"Actor Module: {self.actor_module.module}")

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict.output_dim):
            if output_dim == "robot_action_dim":
                module_config_dict.output_dim[idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        self.project_action_std()
        mean = self.actor(actor_obs)
        if self.min_noise_std:
            clamped_std = torch.clamp(self.std, min=self.min_noise_std)
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        elif self.min_mean_noise_std:
            current_mean = self.std.mean()
            if current_mean < self.min_mean_noise_std:
                scale_up = self.min_mean_noise_std / (current_mean + 1e-6)
                clamped_std = self.std * scale_up
            else:
                clamped_std = self.std
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        else:
            self.distribution = Normal(mean, mean * 0.0 + self.std)

    @torch.no_grad()
    def project_action_std(self) -> None:
        """Keep the directly optimized Gaussian scale finite and positive."""
        if not torch.isfinite(self.std).all():
            raise FloatingPointError("PPO action standard deviation became non-finite.")
        self.std.clamp_(min=MIN_ACTION_STD)

    def act(self, policy_state_dict):
        self.update_distribution(policy_state_dict["actor_obs"])
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict):
        return self.actor(policy_state_dict["actor_obs"])

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")


class PPOCritic(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, history_length: dict[str, int]):
        super().__init__()
        self.critic_module = BaseModule(obs_dim_dict, module_config_dict, history_length)
        print(f"Critic Module: {self.critic_module.module}")

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        return self.critic(critic_obs)

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states):
        pass


class PPOActorEncoder(PPOActor):
    def __init__(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std):
        super().__init__(obs_dim_dict, module_config_dict, num_actions, init_noise_std)
        self.module_input_name = module_config_dict.layer_config.module_input_name
        self.encoder_input_name = module_config_dict.layer_config.encoder_input_name

    def _get_input(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if actor_obs.shape[-1] != self.actor_module.input_dim:
            raise ValueError(f"Actor Obs must be {self.actor_module.input_dim}, got {actor_obs.shape[-1]}")
        self.encoder_obs = actor_obs[..., self.actor_module.input_indices_dict[self.encoder_input_name]]
        self.actor_encoder_obs = (
            self.actor_module.encoder(self.encoder_obs) if self.actor_module.encoder is not None else self.encoder_obs
        )
        self.actor_state_obs = torch.cat(
            [
                actor_obs[..., self.actor_module.input_indices_dict[actor_input_name]]
                for actor_input_name in self.module_input_name
            ],
            -1,
        )
        return torch.cat((self.actor_encoder_obs, self.actor_state_obs), dim=-1)

    def act(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        input_actor = self._get_input(actor_obs)
        return super().act({"actor_obs": input_actor})

    def act_inference(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        input_actor = self._get_input(actor_obs)
        return super().act_inference({"actor_obs": input_actor})


class PPOCriticEncoder(PPOCritic):
    def __init__(self, obs_dim_dict, module_config_dict):
        super().__init__(obs_dim_dict, module_config_dict)
        self.module_input_name = module_config_dict.layer_config.module_input_name
        self.encoder_input_name = module_config_dict.layer_config.encoder_input_name

    def _get_input(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.shape[-1] != self.critic_module.input_dim:
            raise ValueError(f"Critic Obs must be {self.critic_module.input_dim}, got {critic_obs.shape[-1]}")
        self.encoder_obs = critic_obs[..., self.critic_module.input_indices_dict[self.encoder_input_name]]
        self.critic_encoder_obs = (
            self.critic_module.encoder(self.encoder_obs) if self.critic_module.encoder is not None else self.encoder_obs
        )
        self.critic_state_obs = torch.cat(
            [
                critic_obs[..., self.critic_module.input_indices_dict[critic_input_name]]
                for critic_input_name in self.module_input_name
            ],
            -1,
        )
        return torch.cat((self.critic_encoder_obs, self.critic_state_obs), dim=-1)

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        input_critic = self._get_input(critic_obs)
        return super().evaluate({"critic_obs": input_critic})
