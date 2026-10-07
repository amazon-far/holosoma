"""Tests for symmetry augmentation utilities with history length support.

This test suite verifies that:
1. Observation dimensions are correctly computed with history
2. Symmetry transformations work correctly with history_length > 1
3. The mirror_xz_plane function properly handles observations with history
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from holosoma.agents.modules.augmentation_utils import SymmetryUtils


@pytest.fixture
def mock_env_with_history() -> MagicMock:
    """Create a mock environment with history length > 1."""
    env = MagicMock()
    env.device = "cpu"

    # Mock robot config with symmetry mappings
    robot_config = MagicMock()
    robot_config.dof_names = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ]
    robot_config.symmetry_joint_names = {
        "left_hip_yaw": "right_hip_yaw",
        "left_hip_roll": "right_hip_roll",
        "left_hip_pitch": "right_hip_pitch",
        "left_knee": "right_knee",
        "left_ankle": "right_ankle",
        "right_hip_yaw": "left_hip_yaw",
        "right_hip_roll": "left_hip_roll",
        "right_hip_pitch": "left_hip_pitch",
        "right_knee": "left_knee",
        "right_ankle": "left_ankle",
    }
    robot_config.flip_sign_joint_names = [
        "left_hip_yaw",
        "right_hip_yaw",  # yaw joints get sign flipped
    ]
    env.robot_config = robot_config

    # Mock observation manager with history
    obs_manager = MagicMock()
    obs_manager.cfg.groups = {
        "actor_obs": MagicMock(
            history_length=4,
            terms={
                "base_ang_vel": MagicMock(),
                "projected_gravity": MagicMock(),
                "dof_pos": MagicMock(),
                "dof_vel": MagicMock(),
            },
        ),
        "critic_obs": MagicMock(
            history_length=1,
            terms={
                "base_lin_vel": MagicMock(),
            },
        ),
    }

    # Mock _compute_term to return tensors with expected dimensions
    def mock_compute_term(group_name: str, term_name: str, term_cfg: MagicMock) -> torch.Tensor:
        num_envs = 2
        dims = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "dof_pos": 10,
            "dof_vel": 10,
            "base_lin_vel": 3,
        }
        return torch.randn(num_envs, dims[term_name])

    obs_manager._compute_term = mock_compute_term
    env.observation_manager = obs_manager

    # Mock history_length dict
    env.history_length = {
        "actor_obs": 4,
        "critic_obs": 1,
    }

    return env


@pytest.fixture
def mock_env_direct_config() -> MagicMock:
    """Create a mock environment using direct config (not observation manager)."""
    env = MagicMock()
    env.device = "cpu"

    # Mock robot config
    robot_config = MagicMock()
    robot_config.dof_names = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
    ]
    robot_config.symmetry_joint_names = {
        "left_hip_yaw": "right_hip_yaw",
        "left_hip_roll": "right_hip_roll",
        "left_hip_pitch": "right_hip_pitch",
        "right_hip_yaw": "left_hip_yaw",
        "right_hip_roll": "left_hip_roll",
        "right_hip_pitch": "left_hip_pitch",
    }
    robot_config.flip_sign_joint_names = ["left_hip_yaw", "right_hip_yaw"]
    env.robot_config = robot_config

    # No observation manager (direct config mode)
    env.observation_manager = None

    # Mock config for direct observation system
    config = MagicMock()
    config.obs.obs_dims = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "dof_pos": 6,
        "dof_vel": 6,
    }
    config.obs.obs_dict = {
        "actor_obs": ["base_ang_vel", "projected_gravity", "dof_pos", "dof_vel"],
    }
    env.config = config

    # Dimensions already include history
    env.dim_obs = {
        "actor_obs": 18 * 4,  # (3+3+6+6) * 4 = 72
    }

    # History length dict
    env.history_length = {
        "actor_obs": 4,
    }

    return env


def test_symmetry_utils_initialization_with_history(mock_env_with_history: MagicMock) -> None:
    """Test that SymmetryUtils correctly initializes with history length > 1."""
    symmetry_utils = SymmetryUtils(mock_env_with_history)

    # Check that history lengths were stored correctly
    assert symmetry_utils.history_lengths["actor_obs"] == 4
    assert symmetry_utils.history_lengths["critic_obs"] == 1

    # Check that observation dimensions include history
    # actor_obs: (3 + 3 + 10 + 10) * 4 = 104
    assert symmetry_utils.observation_dims["actor_obs"] == 104
    # critic_obs: 3 * 1 = 3
    assert symmetry_utils.observation_dims["critic_obs"] == 3

    # Check that single-frame dimensions are correct
    assert symmetry_utils.observation_dims_single_frame["actor_obs"] == 26  # 3+3+10+10
    assert symmetry_utils.observation_dims_single_frame["critic_obs"] == 3

    # Check that sub-observation indices were created for both full and single-frame
    assert "base_ang_vel" in symmetry_utils.sub_observation_indices["actor_obs"]
    assert "base_ang_vel" in symmetry_utils.sub_observation_indices_single_frame["actor_obs"]


def test_symmetry_utils_initialization_direct_config(mock_env_direct_config: MagicMock) -> None:
    """Test that SymmetryUtils works with direct config (no observation manager)."""
    symmetry_utils = SymmetryUtils(mock_env_direct_config)

    # Check dimensions
    assert symmetry_utils.observation_dims["actor_obs"] == 72  # 18 * 4
    assert symmetry_utils.observation_dims_single_frame["actor_obs"] == 18
    assert symmetry_utils.history_lengths["actor_obs"] == 4


def test_mirror_xz_plane_with_history(mock_env_with_history: MagicMock) -> None:
    """Test that mirror_xz_plane correctly handles observations with history."""
    symmetry_utils = SymmetryUtils(mock_env_with_history)

    batch_size = 16
    obs_dim_with_history = symmetry_utils.observation_dims["actor_obs"]

    # Create a test observation tensor
    observation = torch.randn(batch_size, obs_dim_with_history)

    # Apply mirroring
    mirrored_obs = symmetry_utils.mirror_xz_plane(
        observation=observation, env=mock_env_with_history, obs_list=["actor_obs"]
    )

    # Check that output shape matches input
    assert mirrored_obs.shape == observation.shape
    assert mirrored_obs.shape == (batch_size, obs_dim_with_history)

    # Check that the operation doesn't raise any dimension mismatch errors
    # (The actual bug we fixed would cause a RuntimeError here)


def test_augment_observations_with_history(mock_env_with_history: MagicMock) -> None:
    """Test that augment_observations doubles the batch size correctly with history."""
    symmetry_utils = SymmetryUtils(mock_env_with_history)

    batch_size = 32
    obs_dim_with_history = symmetry_utils.observation_dims["actor_obs"]

    # Create a test observation tensor
    observation = torch.randn(batch_size, obs_dim_with_history)

    # Apply augmentation (should concatenate original + mirrored)
    augmented_obs = symmetry_utils.augment_observations(
        obs=observation, env=mock_env_with_history, obs_list=["actor_obs"]
    )

    # Check that batch size doubled
    assert augmented_obs.shape == (batch_size * 2, obs_dim_with_history)

    # Check that first half matches original
    assert torch.allclose(augmented_obs[:batch_size], observation)


def test_augment_actions(mock_env_with_history: MagicMock) -> None:
    """Test that action augmentation works correctly."""
    symmetry_utils = SymmetryUtils(mock_env_with_history)

    batch_size = 32
    action_dim = 10  # Number of joints

    # Create test actions
    actions = torch.randn(batch_size, action_dim)

    # Apply augmentation
    augmented_actions = symmetry_utils.augment_actions(actions)

    # Check that batch size doubled
    assert augmented_actions.shape == (batch_size * 2, action_dim)

    # Check that first half matches original
    assert torch.allclose(augmented_actions[:batch_size], actions)


def test_consistency_between_observation_dims_and_indices(mock_env_with_history: MagicMock) -> None:
    """Test that observation dimensions and indices are consistent."""
    symmetry_utils = SymmetryUtils(mock_env_with_history)

    # For each observation group, verify that indices span the full dimension
    for obs_key in ["actor_obs"]:
        total_dim_from_indices = 0

        # Sum up dimensions from single-frame indices
        for sub_obs_key in symmetry_utils.sub_observation_keys[obs_key]:
            indices = symmetry_utils.sub_observation_indices_single_frame[obs_key][sub_obs_key]
            total_dim_from_indices += len(indices)

        # Should match single-frame dimension
        assert total_dim_from_indices == symmetry_utils.observation_dims_single_frame[obs_key]

        # Full dimension should be single-frame * history
        expected_full_dim = (
            symmetry_utils.observation_dims_single_frame[obs_key] * symmetry_utils.history_lengths[obs_key]
        )
        assert symmetry_utils.observation_dims[obs_key] == expected_full_dim


def test_reshape_with_history_works(mock_env_with_history: MagicMock) -> None:
    """Test that reshaping to [batch, history, single_frame_dim] works correctly."""
    symmetry_utils = SymmetryUtils(mock_env_with_history)

    batch_size = 8
    obs_dim_with_history = symmetry_utils.observation_dims["actor_obs"]
    history_length = symmetry_utils.history_lengths["actor_obs"]
    single_frame_dim = symmetry_utils.observation_dims_single_frame["actor_obs"]

    # Create observation
    observation = torch.randn(batch_size, obs_dim_with_history)

    # Reshape as done in mirror_xz_plane
    reshaped = observation.reshape(batch_size, history_length, single_frame_dim)

    # Verify shape
    assert reshaped.shape == (batch_size, history_length, single_frame_dim)

    # Verify we can flatten back
    flattened = reshaped.reshape(batch_size, obs_dim_with_history)
    assert torch.allclose(flattened, observation)


def test_multiple_observation_keys(mock_env_with_history: MagicMock) -> None:
    """Test mirroring with multiple observation keys in the list."""
    symmetry_utils = SymmetryUtils(mock_env_with_history)

    batch_size = 16
    actor_obs_dim = symmetry_utils.observation_dims["actor_obs"]
    critic_obs_dim = symmetry_utils.observation_dims["critic_obs"]
    total_dim = actor_obs_dim + critic_obs_dim

    # Create concatenated observation
    observation = torch.randn(batch_size, total_dim)

    # Apply mirroring with both keys
    mirrored_obs = symmetry_utils.mirror_xz_plane(
        observation=observation, env=mock_env_with_history, obs_list=["actor_obs", "critic_obs"]
    )

    # Check output shape
    assert mirrored_obs.shape == (batch_size, total_dim)


@pytest.mark.parametrize("history_length", [1, 2, 4, 8])
def test_different_history_lengths(history_length: int) -> None:
    """Test that the system works with various history lengths."""
    # Create a simple mock environment
    env = MagicMock()
    env.device = "cpu"

    # Mock robot config
    robot_config = MagicMock()
    robot_config.dof_names = ["left_joint", "right_joint"]
    robot_config.symmetry_joint_names = {
        "left_joint": "right_joint",
        "right_joint": "left_joint",
    }
    robot_config.flip_sign_joint_names = ["left_joint"]  # Flip left joint for testing
    env.robot_config = robot_config

    # Mock observation manager
    obs_manager = MagicMock()
    obs_manager.cfg.groups = {
        "actor_obs": MagicMock(history_length=history_length, terms={"dof_pos": MagicMock()}),
    }

    def mock_compute_term(group_name: str, term_name: str, term_cfg: MagicMock) -> torch.Tensor:
        return torch.randn(2, 2)  # 2 dofs

    obs_manager._compute_term = mock_compute_term
    env.observation_manager = obs_manager
    env.history_length = {"actor_obs": history_length}

    # Initialize and test
    symmetry_utils = SymmetryUtils(env)

    # Check dimensions
    assert symmetry_utils.observation_dims["actor_obs"] == 2 * history_length
    assert symmetry_utils.observation_dims_single_frame["actor_obs"] == 2

    # Test mirroring
    batch_size = 4
    observation = torch.randn(batch_size, 2 * history_length)
    mirrored = symmetry_utils.mirror_xz_plane(observation=observation, env=env, obs_list=["actor_obs"])

    assert mirrored.shape == observation.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
