"""Shared fixtures for the agent module tests."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import pytest

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.agents.ppo.ppo import EmpiricalNormalization
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.utils.safe_torch_import import torch


def ppo_teacher_checkpoint(
    num_obs: int,
    num_actions: int,
    hidden_dims: list[int],
    *,
    normalizer: EmpiricalNormalization | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], Any]:
    """A holosoma PPO training checkpoint of an MLP actor reading ``actor_obs``, and that actor."""
    torch.manual_seed(seed)
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=["robot_action_dim"],
        layer_config=LayerConfig(hidden_dims=list(hidden_dims), activation="ELU"),
    )
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": num_obs},
        module_config=config,
        num_actions=num_actions,
        init_noise_std=1.0,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    checkpoint = {
        "actor_model_state_dict": actor.state_dict(),
        "actor_obs_normalizer_state_dict": normalizer.state_dict() if normalizer is not None else None,
        "experiment_config": {
            "algo": {
                "_target_": "holosoma.agents.ppo.ppo.PPO",
                "config": {
                    "module_dict": {"actor": dataclasses.asdict(config)},
                    "empirical_normalization": normalizer is not None,
                },
            },
            "observation": {"groups": {"actor_obs": {"history_length": 1}}},
        },
        "iter": 100,
    }
    return checkpoint, actor


def fitted_normalizer(num_obs: int, seed: int = 0) -> EmpiricalNormalization:
    """A PPO observation normalizer fitted to shifted, scaled noise."""
    generator = torch.Generator().manual_seed(seed)
    normalizer = EmpiricalNormalization(shape=num_obs, device="cpu")
    scale = torch.linspace(0.5, 4.0, num_obs)
    normalizer(torch.randn(64, num_obs, generator=generator) * scale + torch.linspace(1.0, -3.0, num_obs))
    normalizer.eval()
    return normalizer


@pytest.fixture
def make_ppo_teacher_checkpoint() -> Callable[..., tuple[dict[str, Any], Any]]:
    return ppo_teacher_checkpoint


@pytest.fixture
def make_fitted_normalizer() -> Callable[..., EmpiricalNormalization]:
    return fitted_normalizer
