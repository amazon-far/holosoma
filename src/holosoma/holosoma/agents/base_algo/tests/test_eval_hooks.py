"""Tests for the evaluation callback hooks BaseAlgo shares with every algorithm."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.algo import AlgoInitConfig

pytestmark = pytest.mark.no_sim


class _HookLogger(RLEvalCallback):
    """Appends every hook it receives to the algorithm's event log."""

    def __init__(self, config: Any, training_loop: Any) -> None:
        super().__init__(config, training_loop)
        self.name = config.name
        # The push and payload callbacks scan this list during their own
        # construction to share state with an earlier callback, so record what
        # was visible at that moment.
        self.visible_at_construction = [cb.name for cb in training_loop.eval_callbacks]

    def _record(self, hook: str) -> None:
        self.training_loop.events.append(f"{self.name}.{hook}")

    def on_pre_evaluate_policy(self) -> None:
        self._record("pre_evaluate_policy")

    def on_pre_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        self._record("pre_eval_env_step")
        actor_state["pre_step_visits"] = actor_state.get("pre_step_visits", 0) + 1
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        self._record("post_eval_env_step")
        actor_state["post_step_visits"] = actor_state.get("post_step_visits", 0) + 1
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        self._record("post_evaluate_policy")


# Resolved against this module in sys.modules, so it works however pytest named it.
_HOOK_LOGGER_TARGET = f"{__name__}.{_HookLogger.__qualname__}"


def _algo_config(eval_callbacks: Any) -> AlgoInitConfig:
    """Stand in for a real algorithm config, which only eval_callbacks is read from."""
    return cast("AlgoInitConfig", SimpleNamespace(eval_callbacks=eval_callbacks))


class _FakeEnv:
    def __init__(self) -> None:
        self.evaluating = False
        self.step_count = 0

    def set_is_evaluating(self) -> None:
        self.evaluating = True

    def step(self, actor_state: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.step_count += 1
        return {"actor_obs": self.step_count}, 0.0, False, {}


class _FakeAlgo(BaseAlgo):
    """Minimal algorithm driving the shared hooks the way the real ones do."""

    def __init__(self, env: Any, eval_callbacks: Any) -> None:
        super().__init__(env, _algo_config(eval_callbacks), device="cpu")
        self.events: list[str] = []
        self.eval_mode_calls = 0

    def _eval_mode(self) -> None:
        self.eval_mode_calls += 1

    def _compute_eval_actions(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        actor_state["actions"] = actor_state["obs"]["actor_obs"]
        return actor_state

    def evaluate_policy(self, max_eval_steps: int | None = None) -> dict[str, Any]:
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state: dict[str, Any] = {"obs": {"actor_obs": 0}}
        for step in range(max_eval_steps or 0):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
        self._post_evaluate_policy()
        return actor_state


def _callback_config(name: str) -> SimpleNamespace:
    return SimpleNamespace(_target_=_HOOK_LOGGER_TARGET, config=SimpleNamespace(name=name))


def test_hooks_fire_in_order_once_per_step() -> None:
    algo = _FakeAlgo(_FakeEnv(), {"only": _callback_config("only")})

    actor_state = algo.evaluate_policy(max_eval_steps=3)

    assert algo.events == [
        "only.pre_evaluate_policy",
        *["only.pre_eval_env_step", "only.post_eval_env_step"] * 3,
        "only.post_evaluate_policy",
    ]
    assert actor_state["pre_step_visits"] == 3
    assert actor_state["post_step_visits"] == 3


def test_callbacks_are_built_in_config_order_and_see_earlier_ones() -> None:
    configs = {"first": _callback_config("first"), "second": _callback_config("second")}
    algo = _FakeAlgo(_FakeEnv(), configs)

    algo._create_eval_callbacks()

    assert [cb.name for cb in algo.eval_callbacks] == ["first", "second"]
    assert algo.eval_callbacks[0].visible_at_construction == []
    assert algo.eval_callbacks[1].visible_at_construction == ["first"]


def test_a_second_evaluation_does_not_stack_duplicate_callbacks() -> None:
    algo = _FakeAlgo(_FakeEnv(), {"only": _callback_config("only")})

    algo.evaluate_policy(max_eval_steps=1)
    algo.evaluate_policy(max_eval_steps=1)

    assert [cb.name for cb in algo.eval_callbacks] == ["only"]


def test_evaluation_runs_without_configured_callbacks() -> None:
    env = _FakeEnv()
    algo = _FakeAlgo(env, None)

    algo.evaluate_policy(max_eval_steps=2)

    assert algo.eval_callbacks == []
    assert algo.events == []
    assert env.step_count == 2


def test_pre_evaluate_policy_switches_mode_and_flags_the_env() -> None:
    env = _FakeEnv()
    algo = _FakeAlgo(env, None)

    algo._pre_evaluate_policy()

    assert algo.eval_mode_calls == 1
    assert env.evaluating is True


def test_env_step_folds_the_environment_result_into_actor_state() -> None:
    algo = _FakeAlgo(_FakeEnv(), None)

    actor_state = algo.env_step({"actions": 0})

    assert actor_state["obs"] == {"actor_obs": 1}
    assert set(actor_state) == {"actions", "obs", "rewards", "dones", "extras"}
