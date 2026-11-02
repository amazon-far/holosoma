"""Envs reset (or teleported) since the last step are re-rendered once, just before it reads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 5


class _RecordingTerm(ObservationTermBase):
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.env.events.append(("obs_reset", None if env_ids is None else env_ids.tolist()))

    def read(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        env.events.append(("read", None if env_ids is None else env_ids.tolist()))
        return torch.zeros(NUM_ENVS if env_ids is None else len(env_ids), 1)


def _manager(events: list[tuple[Any, ...]]) -> ObservationManager:
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device="cpu",
        events=events,
        simulator=SimpleNamespace(rerender_sensors=lambda env_ids: events.append(("rerender", env_ids.tolist()))),
    )
    term = ObsTermCfg(func="holosoma.managers.observation.tests.test_observation_rerender:_RecordingTerm")
    cfg = ObservationManagerCfg(groups={"policy": ObsGroupCfg(terms={"term": term})})
    manager = ObservationManager(cfg, env, device="cpu")
    events.clear()  # drop reads made while sizing the term
    return manager


def test_terminal_read_sees_the_state_before_the_reset() -> None:
    events: list[tuple[Any, ...]] = []
    manager = _manager(events)

    manager.new_frame()
    manager.step_terminal(torch.tensor([2]))
    manager.reset(torch.tensor([2]))
    manager.step()

    assert events == [("read", [2]), ("obs_reset", [2]), ("rerender", [2]), ("read", None)]


def test_resets_are_rerendered_once_before_the_next_step() -> None:
    events: list[tuple[Any, ...]] = []
    manager = _manager(events)

    manager.reset(torch.tensor([3, 1]))
    manager.reset(torch.tensor([3]))  # e.g. a clip-end teleport after an episode reset
    events.clear()
    manager.new_frame()
    manager.step()
    manager.new_frame()
    manager.step()

    assert events == [("rerender", [1, 3]), ("read", None), ("read", None)]


def test_full_reset_rerenders_every_env() -> None:
    events: list[tuple[Any, ...]] = []
    manager = _manager(events)

    manager.reset()
    events.clear()
    manager.new_frame()
    manager.step()

    assert events == [("rerender", list(range(NUM_ENVS))), ("read", None)]


def test_empty_reset_means_no_rerender() -> None:
    events: list[tuple[Any, ...]] = []
    manager = _manager(events)

    # BaseTask._post_physics_step resets an empty set of envs on most steps.
    manager.reset(torch.tensor([], dtype=torch.long))
    events.clear()
    manager.new_frame()
    manager.step()

    assert events == [("read", None)]
