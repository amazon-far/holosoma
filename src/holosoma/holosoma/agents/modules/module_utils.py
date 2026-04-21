from __future__ import annotations

from holosoma.agents.modules.ppo_modules import (
    PPOActor,
    PPOActorEncoder,
    PPOActorWithHeightMap,
    PPOActorWithHeightMapVideoMimic,
    PPOActorWithMotionEncoder,
    PPOCritic,
    PPOCriticEncoder,
    PPOCriticWithHeightMap,
    PPOCriticWithHeightMapVideoMimic,
    PPOCriticWithMotionEncoder,
)


def setup_ppo_actor_module(
    obs_dim_dict,
    module_config,
    num_actions,
    init_noise_std,
    device,
    history_length: dict[str, int],
    motion_encoder_config: dict | None = None,
    height_map_encoder_config: dict | None = None,
):
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOActorEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
        ).to(device)
    if module_type == "MLP":
        return PPOActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        ).to(device)
    if module_type == "MLPWithMotionEncoder":
        if motion_encoder_config is None:
            raise ValueError("motion_encoder_config is required for MLPWithMotionEncoder")
        return PPOActorWithMotionEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
            motion_encoder_config=motion_encoder_config,
        ).to(device)
    if module_type == "MLPWithHeightMap":
        if height_map_encoder_config is None:
            raise ValueError("height_map_encoder_config is required for MLPWithHeightMap")
        return PPOActorWithHeightMap(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
            height_map_encoder_config=height_map_encoder_config,
            motion_encoder_config=motion_encoder_config,
        ).to(device)
    if module_type == "MLPWithHeightMapVideoMimic":
        if height_map_encoder_config is None:
            raise ValueError("height_map_encoder_config is required for MLPWithHeightMapVideoMimic")
        return PPOActorWithHeightMapVideoMimic(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
            height_map_encoder_config=height_map_encoder_config,
            motion_encoder_config=motion_encoder_config,
        ).to(device)

    raise ValueError(f"Invalid actor type: {module_type}")


def setup_ppo_critic_module(
    obs_dim_dict,
    module_config,
    device,
    history_length: dict[str, int],
    motion_encoder_config: dict | None = None,
    height_map_encoder_config: dict | None = None,
):
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOCriticEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
        ).to(device)
    if module_type == "MLP":
        return PPOCritic(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
        ).to(device)
    if module_type == "MLPWithMotionEncoder":
        if motion_encoder_config is None:
            raise ValueError("motion_encoder_config is required for MLPWithMotionEncoder")
        return PPOCriticWithMotionEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
            motion_encoder_config=motion_encoder_config,
        ).to(device)
    if module_type == "MLPWithHeightMap":
        if height_map_encoder_config is None:
            raise ValueError("height_map_encoder_config is required for MLPWithHeightMap")
        return PPOCriticWithHeightMap(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
            height_map_encoder_config=height_map_encoder_config,
            motion_encoder_config=motion_encoder_config,
        ).to(device)
    if module_type == "MLPWithHeightMapVideoMimic":
        if height_map_encoder_config is None:
            raise ValueError("height_map_encoder_config is required for MLPWithHeightMapVideoMimic")
        return PPOCriticWithHeightMapVideoMimic(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
            height_map_encoder_config=height_map_encoder_config,
            motion_encoder_config=motion_encoder_config,
        ).to(device)
    raise ValueError(f"Invalid critic type: {module_type}")
