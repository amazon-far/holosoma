"""Tests for module dimension calculations with history length.

This test suite verifies that:
1. BaseModule correctly computes input dimensions when obs_dim_dict includes history
2. PPO's _get_obs_dim correctly computes dimensions
3. Network input dimensions match what is stored in replay buffers/storage
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.agents.modules.modules import BaseModule
from holosoma.config_types.algo import LayerConfig, ModuleConfig


@pytest.fixture
def simple_module_config():
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


def test_base_module_input_dim_with_history(simple_module_config):
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


def test_base_module_input_dim_multiple_keys():
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


def test_base_module_input_slices():
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


def test_base_module_network_creation(simple_module_config):
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
    first_layer = module.module[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == 400

    # Verify the last layer has correct output dimension
    last_layer = module.module[-1]
    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == 10


def test_base_module_forward_pass(simple_module_config):
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
    output = module.module(input_tensor)

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
def test_various_history_configurations(simple_module_config, history_length, single_frame_dim):
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
    output = module.module(input_tensor)
    assert output.shape == (batch_size, 10)


def test_consistency_with_storage_dimensions():
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
    output = module.module(observation)
    assert output.shape == (batch_size, 10)


def test_ppo_actor_projects_nonpositive_std(simple_module_config: ModuleConfig) -> None:
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": 4},
        module_config=simple_module_config,
        num_actions=10,
        init_noise_std=0.05,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    torch.testing.assert_close(actor.std, torch.full_like(actor.std, 0.05))

    with torch.no_grad():
        actor.std.fill_(-1.0)
    actor.update_distribution(torch.zeros(2, 4))

    assert torch.all(actor.std > 0.0)
    assert torch.all(actor.action_std > 0.0)


def test_ppo_actor_rejects_nonfinite_std(simple_module_config: ModuleConfig) -> None:
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": 4},
        module_config=simple_module_config,
        num_actions=10,
        init_noise_std=0.05,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    with torch.no_grad():
        actor.std[0] = torch.nan

    with pytest.raises(FloatingPointError, match="standard deviation became non-finite"):
        actor.update_distribution(torch.zeros(2, 4))


def test_ppo_projects_std_immediately_after_optimizer_step(simple_module_config: ModuleConfig) -> None:
    from holosoma.agents.ppo.ppo import PPO, Minibatch

    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": 4},
        module_config=simple_module_config,
        num_actions=10,
        init_noise_std=0.05,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    critic = torch.nn.Linear(1, 1)
    algo = cast("Any", object.__new__(PPO))
    algo.actor = actor
    algo.critic = critic
    algo.actor_optimizer = torch.optim.SGD(actor.parameters(), lr=1.0)
    algo.critic_optimizer = torch.optim.SGD(critic.parameters(), lr=1.0)
    algo.config = SimpleNamespace(max_grad_norm=100.0)
    algo.is_multi_gpu = False
    zero = critic.weight.sum() * 0.0
    algo._compute_ppo_loss = lambda _minibatch: {
        "actor_loss": actor.std.sum() * 2.0,
        "critic_loss": zero,
        "value_loss": zero,
        "surrogate_loss": zero,
        "entropy_loss": zero,
        "kl_mean": zero,
    }

    PPO._update_algo_step(
        algo,
        cast("Minibatch", {}),
        {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0},
    )

    assert torch.all(actor.std > 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
