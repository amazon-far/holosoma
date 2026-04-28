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


class PPOActorWithHeightMap(nn.Module):
    """PPO Actor with height map encoder (CNN + CrossAttention).

    Optionally also supports a motion encoder for future motion targets.

    Input keys in policy_state_dict:
        - actor_obs: proprioceptive observations (used as query for attention)
        - height_map_obs: height map observation (used as key/value for attention)
        - future_motion_targets: (optional) future motion targets for Conv1D encoder
    """

    def __init__(
        self,
        obs_dim_dict: dict,
        module_config_dict: ModuleConfig,
        num_actions: int,
        init_noise_std: float,
        history_length: dict[str, int],
        height_map_encoder_config: dict,
        motion_encoder_config: dict | None = None,
    ):
        super().__init__()

        from .encoder_modules import HeightMapEncoder

        # Build height map encoder
        self.height_map_encoder = HeightMapEncoder(
            proprio_dim=obs_dim_dict.get("actor_obs", 0),
            latent_dim=height_map_encoder_config["latent_dim"],
            num_heads=height_map_encoder_config["num_heads"],
            map_height=height_map_encoder_config["map_height"],
            map_width=height_map_encoder_config["map_width"],
            num_channels=height_map_encoder_config.get("num_channels", 3),
            conv_hidden_channels=height_map_encoder_config.get("conv_hidden_channels", 16),
        )
        self.height_map_latent_dim = height_map_encoder_config["latent_dim"]

        # Optionally build motion encoder
        self.use_motion_encoder = motion_encoder_config is not None
        self.motion_encoder_output_dim = 0
        if self.use_motion_encoder:
            from .encoder_modules import Conv1DEncoder, ConvEncoderConfig
            motion_enc_cfg = ConvEncoderConfig(
                input_dim=motion_encoder_config["input_dim"],
                hidden_dim=motion_encoder_config.get("hidden_dim", 60),
                output_dim=motion_encoder_config.get("output_dim", 128),
                num_timesteps=motion_encoder_config.get("num_timesteps", 20),
                activation=motion_encoder_config.get("activation", "SiLU"),
            )
            self.motion_encoder = Conv1DEncoder(motion_enc_cfg)
            self.motion_encoder_output_dim = motion_enc_cfg.output_dim

        # Process module config
        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        # Build actor MLP: input = actor_obs + height_map_latent + (motion_latent)
        actor_obs_dim = obs_dim_dict.get("actor_obs", 0)
        combined_input_dim = actor_obs_dim + self.height_map_latent_dim + self.motion_encoder_output_dim

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

        print(f"Actor Module (with HeightMap): {self.actor_module}")
        print(f"HeightMap Encoder: latent_dim={self.height_map_latent_dim}")
        if self.use_motion_encoder:
            print(f"Motion Encoder: output_dim={self.motion_encoder_output_dim}")

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

    def _build_combined_input(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        height_map = policy_state_dict["height_map_obs"]

        # Height map cross-attention: actor_obs is query, height_map is key/value
        map_latent = self.height_map_encoder(height_map, actor_obs)

        parts = [actor_obs, map_latent]

        if self.use_motion_encoder:
            future_motion = policy_state_dict["future_motion_targets"]
            motion_latent = self.motion_encoder(future_motion)
            parts.append(motion_latent)

        return torch.cat(parts, dim=-1)

    def update_distribution(self, combined_input):
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
        combined = self._build_combined_input(policy_state_dict)
        self.update_distribution(combined)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict):
        combined = self._build_combined_input(policy_state_dict)
        return self.actor_module(combined)


class PPOCriticWithHeightMap(nn.Module):
    """PPO Critic with height map encoder (CNN + CrossAttention).

    Optionally also supports a motion encoder for future motion targets.
    """

    def __init__(
        self,
        obs_dim_dict: dict,
        module_config_dict: ModuleConfig,
        history_length: dict[str, int],
        height_map_encoder_config: dict,
        motion_encoder_config: dict | None = None,
    ):
        super().__init__()

        from .encoder_modules import HeightMapEncoder

        self.height_map_encoder = HeightMapEncoder(
            proprio_dim=obs_dim_dict.get("critic_obs", 0),
            latent_dim=height_map_encoder_config["latent_dim"],
            num_heads=height_map_encoder_config["num_heads"],
            map_height=height_map_encoder_config["map_height"],
            map_width=height_map_encoder_config["map_width"],
            num_channels=height_map_encoder_config.get("num_channels", 3),
            conv_hidden_channels=height_map_encoder_config.get("conv_hidden_channels", 16),
        )
        self.height_map_latent_dim = height_map_encoder_config["latent_dim"]

        self.use_motion_encoder = motion_encoder_config is not None
        self.motion_encoder_output_dim = 0
        if self.use_motion_encoder:
            from .encoder_modules import Conv1DEncoder, ConvEncoderConfig
            motion_enc_cfg = ConvEncoderConfig(
                input_dim=motion_encoder_config["input_dim"],
                hidden_dim=motion_encoder_config.get("hidden_dim", 60),
                output_dim=motion_encoder_config.get("output_dim", 128),
                num_timesteps=motion_encoder_config.get("num_timesteps", 20),
                activation=motion_encoder_config.get("activation", "SiLU"),
            )
            self.motion_encoder = Conv1DEncoder(motion_enc_cfg)
            self.motion_encoder_output_dim = motion_enc_cfg.output_dim

        critic_obs_dim = obs_dim_dict.get("critic_obs", 0)
        combined_input_dim = critic_obs_dim + self.height_map_latent_dim + self.motion_encoder_output_dim

        from .modules import build_mlp_layer
        self.critic_module = build_mlp_layer(
            combined_input_dim,
            module_config_dict.layer_config.hidden_dims,
            1,
            module_config_dict.layer_config,
        )

        print(f"Critic Module (with HeightMap): {self.critic_module}")

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        height_map = policy_state_dict["height_map_obs"]

        map_latent = self.height_map_encoder(height_map, critic_obs)
        parts = [critic_obs, map_latent]

        if self.use_motion_encoder:
            future_motion = policy_state_dict["future_motion_targets"]
            motion_latent = self.motion_encoder(future_motion)
            parts.append(motion_latent)

        combined = torch.cat(parts, dim=-1)
        return self.critic_module(combined)

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states):
        pass


class PPOActorWithHeightMapVideoMimic(nn.Module):
    """PPO Actor with VideoMimic-style height map encoder.

    Two combine modes (selected via ``height_map_encoder_config['combine_mode']``):
      * ``"concat"`` (default): heightmap latent is concatenated with actor_obs
        (and optional motion latent) before the MLP — VideoMimic's
        ``flatten_then_embed_with_attention``.
      * ``"add_to_hidden"``: heightmap latent is added to the activation right
        after the MLP's first Linear layer — VideoMimic's
        ``flatten_then_embed_with_attention_to_hidden``. Requires
        ``latent_dim == hidden_dims[0]``. Pretrained MLP weights remain
        shape-compatible since the MLP input dim does not change.
    """

    def __init__(
        self,
        obs_dim_dict: dict,
        module_config_dict: ModuleConfig,
        num_actions: int,
        init_noise_std: float,
        history_length: dict[str, int],
        height_map_encoder_config: dict,
        motion_encoder_config: dict | None = None,
    ):
        super().__init__()

        from .encoder_modules import HeightMapEncoderVideoMimic

        height_map_input_dim = obs_dim_dict.get("height_map_obs", 0)
        if height_map_input_dim == 0:
            raise ValueError("height_map_obs dimension missing from obs_dim_dict for VideoMimic heightmap encoder")
        self.height_map_encoder = HeightMapEncoderVideoMimic(
            input_dim=height_map_input_dim,
            latent_dim=height_map_encoder_config["latent_dim"],
        )
        self.height_map_latent_dim = height_map_encoder_config["latent_dim"]
        self.combine_mode = height_map_encoder_config.get("combine_mode", "concat")
        if self.combine_mode not in ("concat", "add_to_hidden"):
            raise ValueError(f"Unknown combine_mode {self.combine_mode!r} for VideoMimic heightmap")

        self.use_motion_encoder = motion_encoder_config is not None
        self.motion_encoder_output_dim = 0
        if self.use_motion_encoder:
            from .encoder_modules import Conv1DEncoder, ConvEncoderConfig
            motion_enc_cfg = ConvEncoderConfig(
                input_dim=motion_encoder_config["input_dim"],
                hidden_dim=motion_encoder_config.get("hidden_dim", 60),
                output_dim=motion_encoder_config.get("output_dim", 128),
                num_timesteps=motion_encoder_config.get("num_timesteps", 20),
                activation=motion_encoder_config.get("activation", "SiLU"),
            )
            self.motion_encoder = Conv1DEncoder(motion_enc_cfg)
            self.motion_encoder_output_dim = motion_enc_cfg.output_dim

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        actor_obs_dim = obs_dim_dict.get("actor_obs", 0)
        if self.combine_mode == "concat":
            combined_input_dim = actor_obs_dim + self.height_map_latent_dim + self.motion_encoder_output_dim
        else:
            first_hidden_dim = module_config_dict.layer_config.hidden_dims[0]
            if self.height_map_latent_dim != first_hidden_dim:
                raise ValueError(
                    f"combine_mode='add_to_hidden' requires heightmap latent_dim={self.height_map_latent_dim} "
                    f"to equal actor's first hidden_dim={first_hidden_dim}"
                )
            combined_input_dim = actor_obs_dim + self.motion_encoder_output_dim

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

        print(f"Actor Module (VideoMimic HeightMap): {self.actor_module}")
        print(f"HeightMap Encoder (VideoMimic): input_dim={height_map_input_dim}, latent_dim={self.height_map_latent_dim}")
        if self.use_motion_encoder:
            print(f"Motion Encoder: output_dim={self.motion_encoder_output_dim}")

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

    def _build_combined_input(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        height_map = policy_state_dict["height_map_obs"]
        map_latent = self.height_map_encoder(height_map)
        if self.combine_mode == "concat":
            parts = [actor_obs, map_latent]
            if self.use_motion_encoder:
                future_motion = policy_state_dict["future_motion_targets"]
                parts.append(self.motion_encoder(future_motion))
            return torch.cat(parts, dim=-1)
        # add_to_hidden: keep map_latent separate, do not concat into MLP input
        parts = [actor_obs]
        if self.use_motion_encoder:
            future_motion = policy_state_dict["future_motion_targets"]
            parts.append(self.motion_encoder(future_motion))
        mlp_input = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return (mlp_input, map_latent)

    def _run_actor_module(self, mlp_input):
        if self.combine_mode == "concat":
            return self.actor_module(mlp_input)
        # add_to_hidden: mlp_input is a (mlp_input, map_latent) tuple
        x, map_latent = mlp_input
        x = self.actor_module[0](x)
        x = x + map_latent
        for layer in self.actor_module[1:]:
            x = layer(x)
        return x

    def update_distribution(self, combined_input):
        mean = self._run_actor_module(combined_input)
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
        combined = self._build_combined_input(policy_state_dict)
        self.update_distribution(combined)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict):
        combined = self._build_combined_input(policy_state_dict)
        return self._run_actor_module(combined)


class PPOCriticWithHeightMapVideoMimic(nn.Module):
    """PPO Critic counterpart of :class:`PPOActorWithHeightMapVideoMimic`."""

    def __init__(
        self,
        obs_dim_dict: dict,
        module_config_dict: ModuleConfig,
        history_length: dict[str, int],
        height_map_encoder_config: dict,
        motion_encoder_config: dict | None = None,
    ):
        super().__init__()

        from .encoder_modules import HeightMapEncoderVideoMimic

        height_map_input_dim = obs_dim_dict.get("height_map_obs", 0)
        if height_map_input_dim == 0:
            raise ValueError("height_map_obs dimension missing from obs_dim_dict for VideoMimic heightmap encoder")
        self.height_map_encoder = HeightMapEncoderVideoMimic(
            input_dim=height_map_input_dim,
            latent_dim=height_map_encoder_config["latent_dim"],
        )
        self.height_map_latent_dim = height_map_encoder_config["latent_dim"]
        self.combine_mode = height_map_encoder_config.get("combine_mode", "concat")
        if self.combine_mode not in ("concat", "add_to_hidden"):
            raise ValueError(f"Unknown combine_mode {self.combine_mode!r} for VideoMimic heightmap")

        self.use_motion_encoder = motion_encoder_config is not None
        self.motion_encoder_output_dim = 0
        if self.use_motion_encoder:
            from .encoder_modules import Conv1DEncoder, ConvEncoderConfig
            motion_enc_cfg = ConvEncoderConfig(
                input_dim=motion_encoder_config["input_dim"],
                hidden_dim=motion_encoder_config.get("hidden_dim", 60),
                output_dim=motion_encoder_config.get("output_dim", 128),
                num_timesteps=motion_encoder_config.get("num_timesteps", 20),
                activation=motion_encoder_config.get("activation", "SiLU"),
            )
            self.motion_encoder = Conv1DEncoder(motion_enc_cfg)
            self.motion_encoder_output_dim = motion_enc_cfg.output_dim

        critic_obs_dim = obs_dim_dict.get("critic_obs", 0)
        if self.combine_mode == "concat":
            combined_input_dim = critic_obs_dim + self.height_map_latent_dim + self.motion_encoder_output_dim
        else:
            first_hidden_dim = module_config_dict.layer_config.hidden_dims[0]
            if self.height_map_latent_dim != first_hidden_dim:
                raise ValueError(
                    f"combine_mode='add_to_hidden' requires heightmap latent_dim={self.height_map_latent_dim} "
                    f"to equal critic's first hidden_dim={first_hidden_dim}"
                )
            combined_input_dim = critic_obs_dim + self.motion_encoder_output_dim

        from .modules import build_mlp_layer
        self.critic_module = build_mlp_layer(
            combined_input_dim,
            module_config_dict.layer_config.hidden_dims,
            1,
            module_config_dict.layer_config,
        )

        print(f"Critic Module (VideoMimic HeightMap): {self.critic_module}")
        print(f"HeightMap Encoder (VideoMimic): input_dim={height_map_input_dim}, latent_dim={self.height_map_latent_dim}")

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        height_map = policy_state_dict["height_map_obs"]
        map_latent = self.height_map_encoder(height_map)
        if self.combine_mode == "concat":
            parts = [critic_obs, map_latent]
            if self.use_motion_encoder:
                future_motion = policy_state_dict["future_motion_targets"]
                parts.append(self.motion_encoder(future_motion))
            return self.critic_module(torch.cat(parts, dim=-1))
        # add_to_hidden
        parts = [critic_obs]
        if self.use_motion_encoder:
            future_motion = policy_state_dict["future_motion_targets"]
            parts.append(self.motion_encoder(future_motion))
        mlp_input = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        x = self.critic_module[0](mlp_input)
        x = x + map_latent
        for layer in self.critic_module[1:]:
            x = layer(x)
        return x

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states):
        pass
