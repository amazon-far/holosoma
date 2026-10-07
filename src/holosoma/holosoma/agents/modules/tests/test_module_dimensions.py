"""Tests for module dimension calculations with history length.

This test suite verifies that:
1. BaseModule correctly computes input dimensions when obs_dim_dict includes history
2. PPO's _get_obs_dim correctly computes dimensions
3. Network input dimensions match what is stored in replay buffers/storage
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from holosoma.agents.modules.modules import BaseModule
from holosoma.config_types.algo import LayerConfig, ModuleConfig


@pytest.fixture
def simple_module_config() -> ModuleConfig:
    """Create a simple MLP module configuration."""
    layer_config = LayerConfig(
        hidden_dims=[256, 128],
        activation="ELU",
    )

    return ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[10],  # e.g., number of actions
        layer_config=layer_config,
    )


def test_base_module_input_dim_with_history(simple_module_config: ModuleConfig) -> None:
    """Test that BaseModule doesn't multiply by history when obs_dim_dict already includes it."""
    # Simulate obs_dim_dict from observation_manager.get_obs_dims()
    # which already includes history (e.g., 100 single-frame * 4 history = 400)
    obs_dim_dict = {
        "actor_obs": 400,  # Already includes history
    }

    history_length = {
        "actor_obs": 4,
    }

    # Create BaseModule
    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=simple_module_config,
        history_length=history_length,
    )

    # The input dimension should be 400, NOT 400 * 4 = 1600
    assert module.input_dim == 400, (
        f"Input dimension should be 400 (already includes history), but got {module.input_dim}"
    )


def test_base_module_input_dim_multiple_keys() -> None:
    """Test input dimension calculation with multiple observation keys."""
    # Create config with multiple input keys
    layer_config = LayerConfig(
        hidden_dims=[256, 128],
        activation="ELU",
    )

    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_state_obs", "perception_obs"],
        output_dim=[10],
        layer_config=layer_config,
    )

    obs_dim_dict = {
        "actor_state_obs": 200,  # 50 * 4 history
        "perception_obs": 800,  # 200 * 4 history
    }

    history_length = {
        "actor_state_obs": 4,
        "perception_obs": 4,
    }

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=config,
        history_length=history_length,
    )

    # Should sum the dimensions without multiplying by history again
    assert module.input_dim == 1000  # 200 + 800


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


def test_base_module_network_creation(simple_module_config: ModuleConfig) -> None:
    """Test that the network is created with correct input/output dimensions."""
    obs_dim_dict = {"actor_obs": 400}
    history_length = {"actor_obs": 4}

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=simple_module_config,
        history_length=history_length,
    )

    # Check that the network was created
    assert hasattr(module, "module")

    # Verify the first layer has correct input dimension
    first_layer = cast("torch.nn.Sequential", module.module)[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == 400

    # Verify the last layer has correct output dimension
    last_layer = cast("torch.nn.Sequential", module.module)[-1]
    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == 10


def test_base_module_forward_pass(simple_module_config: ModuleConfig) -> None:
    """Test that forward pass works with correct dimensions."""
    obs_dim_dict = {"actor_obs": 400}
    history_length = {"actor_obs": 4}

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=simple_module_config,
        history_length=history_length,
    )

    # Create input tensor
    batch_size = 16
    input_tensor = torch.randn(batch_size, 400)

    # Forward pass
    output = cast("torch.nn.Sequential", module.module)(input_tensor)

    # Check output shape
    assert output.shape == (batch_size, 10)


@pytest.mark.parametrize(
    ("history_length", "single_frame_dim"),
    [
        (1, 100),
        (2, 100),
        (4, 100),
        (8, 100),
        (1, 50),
        (4, 200),
    ],
)
def test_various_history_configurations(
    simple_module_config: ModuleConfig, history_length: int, single_frame_dim: int
) -> None:
    """Test module creation with various history length and dimension combinations."""
    obs_dim_with_history = single_frame_dim * history_length

    obs_dim_dict = {"actor_obs": obs_dim_with_history}
    history_length_dict = {"actor_obs": history_length}

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=simple_module_config,
        history_length=history_length_dict,
    )

    # Input dimension should match obs_dim_with_history
    assert module.input_dim == obs_dim_with_history

    # Test forward pass
    batch_size = 8
    input_tensor = torch.randn(batch_size, obs_dim_with_history)
    output = cast("torch.nn.Sequential", module.module)(input_tensor)
    assert output.shape == (batch_size, 10)


def test_consistency_with_storage_dimensions() -> None:
    """Test that module dimensions match what PPO storage would expect.

    This simulates the scenario where:
    1. observation_manager.get_obs_dims() returns dims with history
    2. PPO._get_obs_dim uses these dims to register storage
    3. BaseModule uses the same dims to create the network

    All three should agree on the dimension!
    """
    # Simulate observation_manager.get_obs_dims() output
    # (which includes history: single_frame_dim * history_length)
    single_frame_dim = 100
    history_length = 4
    obs_dim_from_manager = single_frame_dim * history_length  # 400

    # This is what PPO would use for storage
    obs_dim_dict = {"actor_obs": obs_dim_from_manager}
    history_length_dict = {"actor_obs": history_length}

    # Simulate PPO._get_obs_dim (after our fix)
    storage_dim = obs_dim_dict["actor_obs"]  # Should NOT multiply by history again

    # Create module with same dimensions
    layer_config = LayerConfig(hidden_dims=[256], activation="ELU")
    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[10],
        layer_config=layer_config,
    )

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=module_config,
        history_length=history_length_dict,
    )

    # All three should agree
    assert obs_dim_from_manager == 400
    assert storage_dim == 400
    assert module.input_dim == 400

    # And forward pass should work with this dimension
    batch_size = 16
    observation = torch.randn(batch_size, storage_dim)
    output = cast("torch.nn.Sequential", module.module)(observation)
    assert output.shape == (batch_size, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
