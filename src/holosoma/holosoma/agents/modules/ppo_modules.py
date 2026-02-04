from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from holosoma.config_types.algo import ModuleConfig

from .modules import BaseModule


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


class PPOActorWithMotionEncoder(nn.Module):
    """PPO Actor with motion encoder for future motion targets.
    
    Takes actor_obs and future_motion_targets as input.
    The future_motion_targets are encoded by a Conv1D encoder and concatenated with actor_obs.
    """
    
    def __init__(
        self,
        obs_dim_dict: dict,
        module_config_dict: ModuleConfig,
        num_actions: int,
        init_noise_std: float,
        history_length: dict[str, int],
        motion_encoder_config: dict,
    ):
        super().__init__()
        
        from .encoder_modules import Conv1DEncoder, ConvEncoderConfig
        
        # Build motion encoder
        motion_enc_cfg = ConvEncoderConfig(
            input_dim=motion_encoder_config["input_dim"],
            hidden_dim=motion_encoder_config.get("hidden_dim", 60),
            output_dim=motion_encoder_config.get("output_dim", 128),
            num_timesteps=motion_encoder_config.get("num_timesteps", 20),
            activation=motion_encoder_config.get("activation", "SiLU"),
        )
        self.motion_encoder = Conv1DEncoder(motion_enc_cfg)
        self.motion_encoder_output_dim = motion_enc_cfg.output_dim
        
        # Modify obs_dim_dict to account for motion encoder output
        modified_obs_dim_dict = obs_dim_dict.copy()
        # The actor_obs will now include motion encoder output
        # We need to adjust input_dim in module_config
        
        # Process module config
        module_config_dict = self._process_module_config(module_config_dict, num_actions)
        
        # Build actor module with adjusted input dim
        # Original actor_obs dim + motion_encoder_output_dim
        self.actor_obs_dim = obs_dim_dict.get("actor_obs", 0)
        combined_input_dim = self.actor_obs_dim + self.motion_encoder_output_dim
        
        # Create a simple MLP for actor
        from .modules import build_mlp_layer
        self.actor_module = build_mlp_layer(
            combined_input_dim,
            module_config_dict.layer_config.hidden_dims,
            num_actions,
            module_config_dict.layer_config,
        )
        
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_noise_std = module_config_dict.min_noise_std
        self.min_mean_noise_std = module_config_dict.min_mean_noise_std
        self.distribution = None
        Normal.set_default_validate_args(False)
        
        print(f"Actor Module (with Motion Encoder): {self.actor_module}")
        print(f"Motion Encoder: input_dim={motion_enc_cfg.input_dim}, output_dim={motion_enc_cfg.output_dim}, timesteps={motion_enc_cfg.num_timesteps}")
    
    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict.output_dim):
            if output_dim == "robot_action_dim":
                module_config_dict.output_dim[idx] = num_actions
        return module_config_dict
    
    @property
    def actor(self):
        return self.actor_module
    
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
    
    def encode_motion(self, future_motion_targets: torch.Tensor) -> torch.Tensor:
        """Encode future motion targets."""
        return self.motion_encoder(future_motion_targets)
    
    def update_distribution(self, actor_obs: torch.Tensor, motion_latent: torch.Tensor):
        """Update action distribution with combined input."""
        combined_input = torch.cat([actor_obs, motion_latent], dim=-1)
        mean = self.actor_module(combined_input)
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
    
    def act(self, policy_state_dict):
        """Sample action from distribution."""
        actor_obs = policy_state_dict["actor_obs"]
        future_motion = policy_state_dict["future_motion_targets"]
        motion_latent = self.encode_motion(future_motion)
        self.update_distribution(actor_obs, motion_latent)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, policy_state_dict):
        """Get deterministic action (mean)."""
        actor_obs = policy_state_dict["actor_obs"]
        future_motion = policy_state_dict["future_motion_targets"]
        motion_latent = self.encode_motion(future_motion)
        combined_input = torch.cat([actor_obs, motion_latent], dim=-1)
        return self.actor_module(combined_input)


class PPOCriticWithMotionEncoder(nn.Module):
    """PPO Critic with motion encoder for future motion targets.
    
    Takes critic_obs and future_motion_targets as input.
    The future_motion_targets are encoded by a Conv1D encoder and concatenated with critic_obs.
    """
    
    def __init__(
        self,
        obs_dim_dict: dict,
        module_config_dict: ModuleConfig,
        history_length: dict[str, int],
        motion_encoder_config: dict,
    ):
        super().__init__()
        
        from .encoder_modules import Conv1DEncoder, ConvEncoderConfig
        
        # Build motion encoder (can share weights with actor or have separate)
        motion_enc_cfg = ConvEncoderConfig(
            input_dim=motion_encoder_config["input_dim"],
            hidden_dim=motion_encoder_config.get("hidden_dim", 60),
            output_dim=motion_encoder_config.get("output_dim", 128),
            num_timesteps=motion_encoder_config.get("num_timesteps", 20),
            activation=motion_encoder_config.get("activation", "SiLU"),
        )
        self.motion_encoder = Conv1DEncoder(motion_enc_cfg)
        self.motion_encoder_output_dim = motion_enc_cfg.output_dim
        
        # Build critic module with adjusted input dim
        self.critic_obs_dim = obs_dim_dict.get("critic_obs", 0)
        combined_input_dim = self.critic_obs_dim + self.motion_encoder_output_dim
        
        from .modules import build_mlp_layer
        self.critic_module = build_mlp_layer(
            combined_input_dim,
            module_config_dict.layer_config.hidden_dims,
            1,  # Critic outputs a single value
            module_config_dict.layer_config,
        )
        
        print(f"Critic Module (with Motion Encoder): {self.critic_module}")
    
    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def encode_motion(self, future_motion_targets: torch.Tensor) -> torch.Tensor:
        """Encode future motion targets."""
        return self.motion_encoder(future_motion_targets)
    
    def evaluate(self, policy_state_dict):
        """Evaluate state value."""
        critic_obs = policy_state_dict["critic_obs"]
        future_motion = policy_state_dict["future_motion_targets"]
        motion_latent = self.encode_motion(future_motion)
        combined_input = torch.cat([critic_obs, motion_latent], dim=-1)
        return self.critic_module(combined_input)
    
    def get_hidden_states(self):
        return None
    
    def set_hidden_states(self, hidden_states):
        pass
