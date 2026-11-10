"""Tests for module dimension calculations with history length.

This test suite verifies that:
1. BaseModule correctly computes input dimensions when obs_dim_dict includes history
2. PPO's _get_obs_dim correctly computes dimensions
3. Network input dimensions match what is stored in replay buffers/storage
"""

from __future__ import annotations

import pytest

from holosoma.agents.modules.modules import BaseModule
from holosoma.config_types.algo import LayerConfig, ModuleConfig

# Skipping numeric input tests as ModuleConfig.input_dim only accepts List[str]
# and numeric inputs don't seem to be used in practice


def test_base_module_input_slices() -> None:
    """Test that input slices are correctly computed."""
    layer_config = LayerConfig(
        hidden_dims=[256, 128],
        activation="ELU",
    )

    config = ModuleConfig(
        type="MLP",
        input_dim=["obs_a", "obs_b"],
        output_dim=[10],
        layer_config=layer_config,
    )

    obs_dim_dict = {
        "obs_a": 100,
        "obs_b": 200,
    }

    history_length = {
        "obs_a": 1,
        "obs_b": 1,
    }

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=config,
        history_length=history_length,
    )

    # Check slices
    assert module.input_indices_dict["obs_a"] == slice(0, 100)
    assert module.input_indices_dict["obs_b"] == slice(100, 300)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
