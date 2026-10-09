from __future__ import annotations

from typing import Any, cast

import torch

from holosoma.agents.modules.ppo_modules import PPOActor, PPOActorEncoder, PPOCritic, PPOCriticEncoder
from holosoma.config_types.algo import ModuleConfig


def setup_ppo_actor_module(
    obs_dim_dict: dict[str, Any],
    module_config: ModuleConfig,
    num_actions: int,
    init_noise_std: float,
    device: torch.device | str,
    history_length: dict[str, int],
) -> PPOActor | PPOActorEncoder:
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return cast(
            "PPOActorEncoder",
            PPOActorEncoder(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config,
                num_actions=num_actions,
                init_noise_std=init_noise_std,
                history_length=history_length,
            ).to(device),
        )
    if module_type == "MLP":
        return cast(
            "PPOActor",
            PPOActor(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config,
                num_actions=num_actions,
                init_noise_std=init_noise_std,
                history_length=history_length,
            ).to(device),
        )

    raise ValueError(f"Invalid actor type: {module_type}")


def setup_ppo_critic_module(
    obs_dim_dict: dict[str, Any],
    module_config: ModuleConfig,
    device: torch.device | str,
    history_length: dict[str, int],
) -> PPOCritic | PPOCriticEncoder:
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return cast(
            "PPOCriticEncoder",
            PPOCriticEncoder(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config,
                history_length=history_length,
            ).to(device),
        )
    if module_type == "MLP":
        return cast(
            "PPOCritic",
            PPOCritic(
                obs_dim_dict=obs_dim_dict,
                module_config_dict=module_config,
                history_length=history_length,
            ).to(device),
        )
    raise ValueError(f"Invalid critic type: {module_type}")
