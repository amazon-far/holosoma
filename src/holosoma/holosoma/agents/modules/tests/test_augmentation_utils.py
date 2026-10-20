"""Tests for symmetry augmentation utilities with history length support.

This test suite verifies that:
1. Observation dimensions are correctly computed with history
2. Symmetry transformations work correctly with history_length > 1
3. The mirror_xz_plane function properly handles observations with history
4. The joint index map / sign-flip mask come from ``JointSymmetryConfig`` via the env resolver
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch

from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.config_types.algo import JointSymmetryConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.observation.terms.locomotion import base_lin_vel
from holosoma.utils.name_tags import expand_name_spec

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager

pytestmark = pytest.mark.no_sim


def _install_joint_resolver(env: MagicMock, dof_names: list[str], joint_tags: dict[str, list[str]]) -> None:
    """Give a mock env the ``dof_names`` / ``resolve_joint_indices`` surface SymmetryUtils reads."""
    env.dof_names = dof_names
    env.robot_config = SimpleNamespace(joint_tags=joint_tags)

    def resolve_joint_indices(spec: Sequence[str]) -> torch.Tensor:
        names = expand_name_spec(spec, joint_tags, dof_names, kind="joint")
        return torch.tensor([dof_names.index(name) for name in names], dtype=torch.long)

    env.robot_names.resolve_joint_indices = resolve_joint_indices


_LEG_DOF_NAMES = [
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
_LEG_JOINT_TAGS = {
    "#mirror_left": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"],
    "#mirror_right": ["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"],
    "#mirror_flip_sign": ["left_hip_yaw", "right_hip_yaw"],  # yaw joints get sign flipped
}


@pytest.fixture
def mock_env_with_history() -> MagicMock:
    """Create a mock environment with history length > 1."""
    env = MagicMock()
    env.device = "cpu"
    _install_joint_resolver(env, _LEG_DOF_NAMES, _LEG_JOINT_TAGS)

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


def test_symmetry_utils_initialization_with_history(mock_env_with_history: MagicMock) -> None:
    """Test that SymmetryUtils correctly initializes with history length > 1."""
    symmetry_utils = SymmetryUtils(mock_env_with_history, JointSymmetryConfig())

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


def test_joint_index_map_and_sign_flip_mask(mock_env_with_history: MagicMock) -> None:
    """Left/right pairs swap, everything else maps to itself; flip-sign joints get -1."""
    symmetry_utils = SymmetryUtils(mock_env_with_history, JointSymmetryConfig())

    assert symmetry_utils.joint_index_map.tolist() == [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    assert symmetry_utils.joint_index_map.dtype == torch.long
    assert symmetry_utils.sign_flip_mask.tolist() == [-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    assert symmetry_utils.sign_flip_mask.dtype == torch.float


def test_explicit_joint_names_and_unpaired_joints(mock_env_with_history: MagicMock) -> None:
    """A spec of exact names works without tags; unpaired joints keep their index."""
    symmetry = JointSymmetryConfig(left=["left_knee", "left_ankle"], right=["right_knee", "right_ankle"], flip_sign=[])
    symmetry_utils = SymmetryUtils(mock_env_with_history, symmetry)

    assert symmetry_utils.joint_index_map.tolist() == [0, 1, 2, 8, 9, 5, 6, 7, 3, 4]
    assert symmetry_utils.sign_flip_mask.tolist() == [1.0] * 10


def test_left_right_length_mismatch_raises(mock_env_with_history: MagicMock) -> None:
    symmetry = JointSymmetryConfig(left=["#mirror_left"], right=["right_knee"])
    with pytest.raises(ValueError, match=r"left \(5 joints\) and \.right \(1 joints\)"):
        SymmetryUtils(mock_env_with_history, symmetry)


def test_overlapping_left_right_is_rejected_as_non_involutive(mock_env_with_history: MagicMock) -> None:
    """left=[0,1,2] / right=[2,3,4]: index 4 maps to 2, which maps to 0, so the mirror is not its own inverse."""
    symmetry = JointSymmetryConfig(
        left=["left_hip_yaw", "left_hip_roll", "left_hip_pitch"],
        right=["left_hip_pitch", "left_knee", "left_ankle"],
        flip_sign=[],
    )
    with pytest.raises(ValueError, match=r"share joints \['left_hip_pitch'\]"):
        SymmetryUtils(mock_env_with_history, symmetry)


def test_unknown_tag_propagates_from_resolver(mock_env_with_history: MagicMock) -> None:
    with pytest.raises(ValueError, match="Unknown joint tag '#nope'"):
        SymmetryUtils(mock_env_with_history, JointSymmetryConfig(left=["#nope"]))


@pytest.mark.parametrize("side", ["left", "right"])
def test_runtime_rejects_repetitions_hidden_by_selection_deduplication(
    mock_env_with_history: MagicMock, side: str
) -> None:
    symmetry = JointSymmetryConfig()
    getattr(symmetry, side).append(f"{side}_knee")
    with pytest.raises(ValueError, match=rf"{side}.*{side}_knee.*(repeat|more than once)"):
        SymmetryUtils(mock_env_with_history, symmetry)


def test_runtime_rejects_repetition_within_one_nested_tag(mock_env_with_history: MagicMock) -> None:
    tags = {
        **_LEG_JOINT_TAGS,
        "#left_leg": _LEG_JOINT_TAGS["#mirror_left"],
        "#mirror_left": ["#left_leg", "left_knee"],
    }
    _install_joint_resolver(mock_env_with_history, _LEG_DOF_NAMES, tags)
    with pytest.raises(ValueError, match=r"left.*left_knee.*(repeat|more than once)"):
        SymmetryUtils(mock_env_with_history, JointSymmetryConfig())


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([], [], "non-empty"),
        (["left_knee"], ["left_knee"], "share"),
    ],
)
def test_runtime_rejects_invalid_pair_sets(
    mock_env_with_history: MagicMock, left: list[str], right: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SymmetryUtils(mock_env_with_history, JointSymmetryConfig(left=left, right=right, flip_sign=[]))


def test_runtime_rejects_a_sign_mask_that_breaks_double_mirroring(mock_env_with_history: MagicMock) -> None:
    with pytest.raises(ValueError, match=r"flip_sign.*left_hip_yaw.*right_hip_yaw"):
        SymmetryUtils(mock_env_with_history, JointSymmetryConfig(flip_sign=["left_hip_yaw"]))


def test_tagged_and_explicit_correspondences_preserve_pair_order(mock_env_with_history: MagicMock) -> None:
    left, right = ["left_knee", "left_ankle"], ["right_ankle", "right_knee"]
    tags = {**_LEG_JOINT_TAGS, "#ordered_left": left, "#ordered_right": right}
    _install_joint_resolver(mock_env_with_history, _LEG_DOF_NAMES, tags)
    tagged = JointSymmetryConfig(left=["#ordered_left"], right=["#ordered_right"], flip_sign=[])
    explicit = JointSymmetryConfig(left=left, right=right, flip_sign=[])
    actions = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    expected = torch.tensor([[0, 1, 2, 9, 8, 5, 6, 7, 4, 3]], dtype=torch.float32)
    for config in (tagged, explicit):
        symmetry = SymmetryUtils(mock_env_with_history, config)
        mirrored = symmetry.augment_actions(actions)[1:]
        torch.testing.assert_close(mirrored, expected)
        torch.testing.assert_close(symmetry.augment_actions(mirrored)[1:], actions)
    assert tagged.left == ["#ordered_left"]
    assert tagged.right == ["#ordered_right"]


def test_shipped_g1_mirror_tags_pair_left_with_right() -> None:
    """The g1 preset's mirror tags produce an involution pairing each left joint with its right twin."""
    from holosoma.config_values.robot import g1_29dof

    env = MagicMock()
    env.device = "cpu"
    env.history_length = {}
    env.observation_manager = MagicMock()
    env.observation_manager.cfg.groups = {}
    _install_joint_resolver(env, g1_29dof.dof_names, g1_29dof.joint_tags)

    symmetry_utils = SymmetryUtils(env, JointSymmetryConfig())
    joint_map = symmetry_utils.joint_index_map.tolist()
    dof = g1_29dof.dof_names

    assert len(joint_map) == len(dof) == 29
    assert [joint_map[j] for j in joint_map] == list(range(len(dof)))  # involution
    for i, name in enumerate(dof):
        if name.startswith("left_"):
            assert dof[joint_map[i]] == name.replace("left_", "right_", 1)
        elif name.startswith("right_"):
            assert dof[joint_map[i]] == name.replace("right_", "left_", 1)
        else:
            assert joint_map[i] == i, name  # waist joints map to themselves

    flipped = {dof[i] for i, sign in enumerate(symmetry_utils.sign_flip_mask.tolist()) if sign == -1.0}
    assert flipped == {name for name in dof if "_roll_" in name or "_yaw_" in name}
    assert len(flipped) == 16


def test_mirror_xz_plane_with_history(mock_env_with_history: MagicMock) -> None:
    """Test that mirror_xz_plane correctly handles observations with history."""
    symmetry_utils = SymmetryUtils(mock_env_with_history, JointSymmetryConfig())

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


def _manager_env(cfg: ObservationManagerCfg) -> SimpleNamespace:
    """Minimal tensor state consumed by the real manager and locomotion observation terms."""
    roots = torch.zeros(2, 13)
    roots[:, 6] = 1.0
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        dof_names=["left", "right"],
        robot_config=SimpleNamespace(
            joint_tags={
                "#mirror_left": ["left"],
                "#mirror_right": ["right"],
                "#mirror_flip_sign": ["left", "right"],
            }
        ),
        history_length={name: group.history_length for name, group in cfg.groups.items()},
        base_quat=roots[:, 3:7],
        default_dof_pos=torch.zeros(1, 2),
        simulator=SimpleNamespace(robot_root_states=roots, dof_pos=torch.zeros(2, 2)),
    )
    env.observation_manager = ObservationManager(cfg, env, env.device)
    return env


def _planar_base_lin_vel(env: SimpleNamespace) -> torch.Tensor:
    return base_lin_vel(cast("LeggedRobotLocomotionManager", env))[:, :2]


def test_mirror_xz_plane_history_is_term_major() -> None:
    """Pin manager layout and physical mirrors with two distinct frames, without symmetry-derived indices."""
    cfg = ObservationManagerCfg(
        groups={
            "actor_obs": ObsGroupCfg(
                history_length=2,
                # Deliberately reverse insertion order: the manager concatenates alphabetically.
                terms={
                    "dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:dof_pos"),
                    "base_lin_vel": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:base_lin_vel"),
                    "base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:base_ang_vel"),
                },
            )
        }
    )
    env = _manager_env(cfg)
    symmetry_utils = SymmetryUtils(env, JointSymmetryConfig())
    roots = env.simulator.robot_root_states
    roots[:, 10:13] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    roots[:, 7:10] = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    env.simulator.dof_pos[:] = torch.tensor([[13.0, 14.0], [15.0, 16.0]])
    env.observation_manager.compute_group("actor_obs")

    roots[:, 10:13] = torch.tensor([[21.0, 22.0, 23.0], [24.0, 25.0, 26.0]])
    roots[:, 7:10] = torch.tensor([[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]])
    env.simulator.dof_pos[:] = torch.tensor([[33.0, 34.0], [35.0, 36.0]])
    obs = env.observation_manager.compute_group("actor_obs")
    expected_input = torch.tensor(
        [
            [1, 2, 3, 21, 22, 23, 7, 8, 9, 27, 28, 29, 13, 14, 33, 34],
            [4, 5, 6, 24, 25, 26, 10, 11, 12, 30, 31, 32, 15, 16, 35, 36],
        ],
        dtype=torch.float,
    )
    # Angular velocity flips x/z, linear velocity flips y, and both joints swap and flip sign.
    expected_mirror = torch.tensor(
        [
            [-1, 2, -3, -21, 22, -23, 7, -8, 9, 27, -28, 29, -14, -13, -34, -33],
            [-4, 5, -6, -24, 25, -26, 10, -11, 12, 30, -31, 32, -16, -15, -36, -35],
        ],
        dtype=torch.float,
    )
    torch.testing.assert_close(obs, expected_input, rtol=0, atol=0)
    out = symmetry_utils.mirror_xz_plane(obs, env, ["actor_obs"])
    torch.testing.assert_close(out, expected_mirror, rtol=0, atol=0)
    torch.testing.assert_close(obs, expected_input, rtol=0, atol=0)


@pytest.mark.parametrize(("actor_history", "critic_history"), [(1, 1), (2, 3)])
@pytest.mark.parametrize("reverse_groups", [False, True], ids=["actor-first", "critic-first"])
def test_shared_term_name_uses_group_local_width(actor_history: int, critic_history: int, reverse_groups: bool) -> None:
    groups = {
        "actor_obs": ObsGroupCfg(
            terms={"base_lin_vel": ObsTermCfg(func=f"{__name__}:_planar_base_lin_vel")},
            history_length=actor_history,
        ),
        "critic_obs": ObsGroupCfg(
            terms={"base_lin_vel": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:base_lin_vel")},
            history_length=critic_history,
        ),
    }
    if reverse_groups:
        groups = dict(reversed(list(groups.items())))
    env = _manager_env(ObservationManagerCfg(groups=groups))
    symmetry_utils = SymmetryUtils(env, JointSymmetryConfig())
    obs: dict[str, torch.Tensor] = {}
    for velocities in (
        [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]],
        [[4.0, 5.0, 6.0], [14.0, 15.0, 16.0]],
        [[7.0, 8.0, 9.0], [17.0, 18.0, 19.0]],
    ):
        env.simulator.robot_root_states[:, 7:10] = torch.tensor(velocities)
        obs = env.observation_manager.compute()

    expected_actor = [[7, -8], [17, -18]] if actor_history == 1 else [[4, -5, 7, -8], [14, -15, 17, -18]]
    expected_critic = (
        [[7, -8, 9], [17, -18, 19]]
        if critic_history == 1
        else [[1, -2, 3, 4, -5, 6, 7, -8, 9], [11, -12, 13, 14, -15, 16, 17, -18, 19]]
    )
    expected_groups = {
        "actor_obs": torch.tensor(expected_actor, dtype=torch.float),
        "critic_obs": torch.tensor(expected_critic, dtype=torch.float),
    }
    observation = torch.cat([obs[name] for name in groups], dim=-1)
    expected = torch.cat([expected_groups[name] for name in groups], dim=-1)
    mirrored = symmetry_utils.mirror_xz_plane(observation, env, list(groups))
    torch.testing.assert_close(mirrored, expected, rtol=0, atol=0)


def test_augment_observations_with_history() -> None:
    """The public augmentation returns original frames followed by their physical mirror."""
    env = _manager_env(
        ObservationManagerCfg(
            groups={
                "actor_obs": ObsGroupCfg(
                    history_length=2,
                    terms={
                        "base_lin_vel": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:base_lin_vel"),
                        "dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:dof_pos"),
                    },
                )
            }
        )
    )
    symmetry_utils = SymmetryUtils(env, JointSymmetryConfig())
    env.simulator.robot_root_states[:, 7:10] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    env.simulator.dof_pos[:] = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    env.observation_manager.compute_group("actor_obs")
    env.simulator.robot_root_states[:, 7:10] = torch.tensor([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]])
    env.simulator.dof_pos[:] = torch.tensor([[17.0, 18.0], [19.0, 20.0]])
    observation = env.observation_manager.compute_group("actor_obs")

    expected_original = torch.tensor(
        [[1, 2, 3, 11, 12, 13, 7, 8, 17, 18], [4, 5, 6, 14, 15, 16, 9, 10, 19, 20]],
        dtype=torch.float,
    )
    # Linear velocity flips y; each frame's two joints swap and both negate.
    expected_mirror = torch.tensor(
        [[1, -2, 3, 11, -12, 13, -8, -7, -18, -17], [4, -5, 6, 14, -15, 16, -10, -9, -20, -19]],
        dtype=torch.float,
    )
    augmented_obs = symmetry_utils.augment_observations(observation, env, ["actor_obs"])

    torch.testing.assert_close(augmented_obs, torch.cat((expected_original, expected_mirror)), rtol=0, atol=0)
    torch.testing.assert_close(observation, expected_original, rtol=0, atol=0)


def test_augment_actions(mock_env_with_history: MagicMock) -> None:
    """Test that action augmentation works correctly."""
    symmetry_utils = SymmetryUtils(mock_env_with_history, JointSymmetryConfig())

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

    # Second half is the mirror: left/right swapped, yaw joints negated
    mirrored = augmented_actions[batch_size:]
    assert torch.allclose(mirrored[:, 5:], actions[:, :5] * torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(mirrored[:, :5], actions[:, 5:] * torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0]))


def test_consistency_between_observation_dims_and_indices(mock_env_with_history: MagicMock) -> None:
    """Test that observation dimensions and indices are consistent."""
    symmetry_utils = SymmetryUtils(mock_env_with_history, JointSymmetryConfig())

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


def test_multiple_observation_keys(mock_env_with_history: MagicMock) -> None:
    """Test mirroring with multiple observation keys in the list."""
    symmetry_utils = SymmetryUtils(mock_env_with_history, JointSymmetryConfig())

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
    _install_joint_resolver(
        env,
        ["left_joint", "right_joint"],
        {
            "#mirror_left": ["left_joint"],
            "#mirror_right": ["right_joint"],
            "#mirror_flip_sign": ["left_joint", "right_joint"],
        },
    )

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
    symmetry_utils = SymmetryUtils(env, JointSymmetryConfig())

    # Check dimensions
    assert symmetry_utils.observation_dims["actor_obs"] == 2 * history_length
    assert symmetry_utils.observation_dims_single_frame["actor_obs"] == 2
    assert symmetry_utils.joint_index_map.tolist() == [1, 0]
    assert symmetry_utils.sign_flip_mask.tolist() == [-1.0, -1.0]

    # Test mirroring
    batch_size = 4
    observation = torch.randn(batch_size, 2 * history_length)
    mirrored = symmetry_utils.mirror_xz_plane(observation=observation, env=env, obs_list=["actor_obs"])

    assert mirrored.shape == observation.shape
    restored = symmetry_utils.mirror_xz_plane(observation=mirrored, env=env, obs_list=["actor_obs"])
    torch.testing.assert_close(restored, observation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
