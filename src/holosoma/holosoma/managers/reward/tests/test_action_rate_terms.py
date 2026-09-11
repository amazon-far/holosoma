"""Tests for the action-rate penalty terms against a real ActionManager (no sim).

``penalty_action_rate`` reads the action history straight off the action manager, so driving the
real ``ActionManager`` with a no-op action term exercises the full term -> manager path without a
simulator backend.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.action.base import ActionTermBase
from holosoma.managers.action.manager import ActionManager
from holosoma.managers.reward.terms import locomotion, wbt
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_ACTION_DIM = 3

# Both copies of penalty_action_rate must behave identically; every test runs against both.
_TERMS = [wbt.penalty_action_rate, locomotion.penalty_action_rate]


class _NoOpActionTerm(ActionTermBase):
    """Action term that claims a slice of the action vector and does nothing with it.

    The tests only exercise the manager's action history, so the term needs no buffers of its own.
    """

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    def process_actions(self, actions: torch.Tensor) -> None:
        return

    def apply_actions(self) -> None:
        return


class _Env:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = "cpu"
        self.action_manager = ActionManager(
            ActionManagerCfg(terms={"noop": ActionTermCfg(func=f"{__name__}:_NoOpActionTerm")}),
            env=self,
            device="cpu",
        )


def _make_env(num_envs: int = 2) -> _Env:
    return _Env(num_envs)


@pytest.mark.parametrize("penalty_action_rate", _TERMS)
def test_default_penalizes_sampled_action_change(penalty_action_rate):
    env = _make_env()
    sampled = [torch.full((2, _ACTION_DIM), 0.25), torch.full((2, _ACTION_DIM), -0.75)]
    means = [torch.zeros(2, _ACTION_DIM), torch.zeros(2, _ACTION_DIM)]

    for action, mean in zip(sampled, means):
        env.action_manager.process_actions(action, mean)

    expected = torch.full((2,), _ACTION_DIM * (-0.75 - 0.25) ** 2)
    torch.testing.assert_close(penalty_action_rate(env), expected)


@pytest.mark.parametrize("penalty_action_rate", _TERMS)
def test_mean_variant_ignores_sampling_noise(penalty_action_rate):
    env = _make_env()

    # Same means, samples far away from them: the mean penalty must land on the mean delta only.
    for mean, offset in ((torch.full((2, _ACTION_DIM), 0.1), 10.0), (torch.full((2, _ACTION_DIM), 0.3), -7.0)):
        env.action_manager.process_actions(mean + offset, mean)

    expected = torch.full((2,), _ACTION_DIM * (0.3 - 0.1) ** 2)
    torch.testing.assert_close(penalty_action_rate(env, use_mean_action=True), expected)


@pytest.mark.parametrize("penalty_action_rate", _TERMS)
def test_mean_mirrors_sampled_action_when_not_supplied(penalty_action_rate):
    """Inference and replay call process_actions without a mean; both flavors must agree there."""
    env = _make_env()
    for action in (torch.full((2, _ACTION_DIM), 0.4), torch.full((2, _ACTION_DIM), -0.2)):
        env.action_manager.process_actions(action)

    expected = torch.full((2,), _ACTION_DIM * (-0.2 - 0.4) ** 2)
    torch.testing.assert_close(penalty_action_rate(env, use_mean_action=True), expected)
    torch.testing.assert_close(penalty_action_rate(env, use_mean_action=False), expected)


@pytest.mark.parametrize("reset_all", [True, False])
def test_reset_clears_mean_history(reset_all):
    env = _make_env(num_envs=3)
    manager = env.action_manager
    for mean in (torch.full((3, _ACTION_DIM), 0.5), torch.full((3, _ACTION_DIM), 1.5)):
        manager.process_actions(torch.zeros(3, _ACTION_DIM), mean)

    reset_ids = None if reset_all else torch.tensor([0, 2])
    manager.reset(env_ids=reset_ids)

    cleared = [0, 1, 2] if reset_all else [0, 2]
    assert torch.all(manager.mean_action[cleared] == 0.0)
    assert torch.all(manager.prev_mean_action[cleared] == 0.0)
    if not reset_all:
        # Environment 1 was not reset, so its history must survive untouched.
        torch.testing.assert_close(manager.mean_action[1], torch.full((_ACTION_DIM,), 1.5))
        torch.testing.assert_close(manager.prev_mean_action[1], torch.full((_ACTION_DIM,), 0.5))


def test_mismatched_mean_action_dim_is_rejected():
    env = _make_env()
    with pytest.raises(ValueError, match="Invalid mean action shape"):
        env.action_manager.process_actions(torch.zeros(2, _ACTION_DIM), torch.zeros(2, _ACTION_DIM + 1))


class _SteppableEnv(_Env):
    """Borrows BaseTask's step/_pre_physics_step to pin the actor_state plumbing without a sim."""

    step = BaseTask.step
    _pre_physics_step = BaseTask._pre_physics_step

    def __init__(self, num_envs: int = 2):
        super().__init__(num_envs)
        self._pending_mean_actions: torch.Tensor | None = None
        self.obs_buf_dict: dict[str, torch.Tensor] = {}
        self.extras: dict[str, torch.Tensor] = {}
        self.rew_buf = None
        self.reset_buf = None

    def _physics_step(self) -> None:
        return

    def _post_physics_step(self) -> None:
        return


def test_step_forwards_mean_actions_from_actor_state():
    env = _SteppableEnv()
    actions = torch.full((2, _ACTION_DIM), 0.9)
    mean_actions = torch.full((2, _ACTION_DIM), 0.4)

    env.step({"actions": actions, "mean_actions": mean_actions})

    torch.testing.assert_close(env.action_manager.action, actions)
    torch.testing.assert_close(env.action_manager.mean_action, mean_actions)


def test_step_without_mean_actions_mirrors_the_sampled_action():
    """A step that omits the key must not inherit the mean stashed by an earlier step."""
    env = _SteppableEnv()
    env.step({"actions": torch.full((2, _ACTION_DIM), 0.9), "mean_actions": torch.full((2, _ACTION_DIM), 0.4)})

    actions = torch.full((2, _ACTION_DIM), -0.5)
    env.step({"actions": actions})

    torch.testing.assert_close(env.action_manager.mean_action, actions)
