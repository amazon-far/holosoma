from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.action.base import ActionTermBase
from holosoma.managers.action.manager import ActionManager
from holosoma.managers.reward.manager import RewardManager
from holosoma.managers.reward.terms.wbt import penalty_action_rate

pytestmark = pytest.mark.no_sim


class _ClippingActionTerm(ActionTermBase):
    def __init__(self, cfg: ActionTermCfg, env: SimpleNamespace):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 2)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 2

    def process_actions(self, actions: torch.Tensor) -> None:
        assert self._raw_actions is not None
        assert self._processed_actions is not None
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0)

    def apply_actions(self) -> None:
        return None


class _NonFiniteActionTerm(_ClippingActionTerm):
    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        assert self._processed_actions is not None
        self._processed_actions[1, 0] = float("nan")


def _action_manager(monkeypatch: pytest.MonkeyPatch, term_type: type[ActionTermBase]) -> ActionManager:
    monkeypatch.setattr(ActionManager, "_resolve_class", lambda _self, _path: term_type)
    env = SimpleNamespace(num_envs=2, device="cpu")
    cfg = ActionManagerCfg(terms={"joint": ActionTermCfg(func="test:ActionTerm")})
    return ActionManager(cfg, env, "cpu")


def test_action_history_uses_clipped_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _action_manager(monkeypatch, _ClippingActionTerm)
    first_raw = torch.tensor([[2.0, -0.5], [-3.0, 4.0]])
    manager.process_actions(first_raw)

    assert torch.equal(manager.action, torch.tensor([[1.0, -0.5], [-1.0, 1.0]]))
    assert torch.equal(manager.prev_action, torch.zeros(2, 2))
    assert torch.equal(manager.get_term("joint").raw_actions, first_raw)

    manager.process_actions(torch.tensor([[0.25, -2.0], [0.5, 0.75]]))
    assert torch.equal(manager.prev_action, torch.tensor([[1.0, -0.5], [-1.0, 1.0]]))
    assert torch.equal(manager.action, torch.tensor([[0.25, -1.0], [0.5, 0.75]]))
    torch.testing.assert_close(
        penalty_action_rate(cast("Any", SimpleNamespace(action_manager=manager))),
        torch.tensor([0.8125, 2.3125]),
    )


def test_action_manager_rejects_non_finite_raw_actions_without_advancing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _action_manager(monkeypatch, _ClippingActionTerm)
    manager.process_actions(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    action_before = manager.action.clone()
    prev_before = manager.prev_action.clone()

    with pytest.raises(FloatingPointError, match=r"Raw actions.*env_ids=\[1\]"):
        manager.process_actions(torch.tensor([[0.5, 0.6], [float("nan"), 0.7]]))

    assert torch.equal(manager.action, action_before)
    assert torch.equal(manager.prev_action, prev_before)


def test_action_manager_rejects_non_finite_processed_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _action_manager(monkeypatch, _NonFiniteActionTerm)

    with pytest.raises(FloatingPointError, match=r"Processed actions from term 'joint'.*env_ids=\[1\]"):
        manager.process_actions(torch.zeros(2, 2))

    assert torch.equal(manager.action, torch.zeros(2, 2))
    assert torch.equal(manager.prev_action, torch.zeros(2, 2))


def _reward_manager(monkeypatch: pytest.MonkeyPatch, reward: torch.Tensor, *, weight: float = 1.0) -> RewardManager:
    monkeypatch.setattr(RewardManager, "_resolve_function", lambda _self, _func: lambda _env: reward)
    cfg = RewardManagerCfg(terms={"test": RewardTermCfg(func="test:reward", weight=weight)})
    env = SimpleNamespace(num_envs=2, max_episode_length_s=1.0)
    return RewardManager(cfg, env, "cpu")


def test_reward_manager_rejects_non_finite_raw_rewards(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _reward_manager(monkeypatch, torch.tensor([0.0, float("nan")]))

    with pytest.raises(FloatingPointError, match=r"Reward term 'test' raw output.*env_ids=\[1\]"):
        manager.compute(1.0)


def test_reward_manager_rejects_non_finite_scaled_rewards(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _reward_manager(monkeypatch, torch.tensor([1.0, torch.finfo(torch.float32).max]), weight=2.0)

    with pytest.raises(FloatingPointError, match=r"Reward term 'test' scaled output.*env_ids=\[1\]"):
        manager.compute(1.0)
