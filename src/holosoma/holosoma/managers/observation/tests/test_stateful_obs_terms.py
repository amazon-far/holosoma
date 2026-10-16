"""How the observation manager reads stateful terms: step reads, peeks, and shape queries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 2
_HERE = "holosoma.managers.observation.tests.test_stateful_obs_terms"


class StepCounterTerm(ObservationTermBase):
    """Reports how many step reads it has seen; peeks report the count without advancing."""

    calls: list[str]

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        self.count = 0
        self.calls = []

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        self.calls.append("call")
        self.count += 1
        return torch.full((NUM_ENVS, 1), float(self.count))

    def peek(self, env: Any, **kwargs: Any) -> torch.Tensor:
        self.calls.append("peek")
        return torch.full((NUM_ENVS, 1), float(self.count))

    def output_shape(self, env: Any, **kwargs: Any) -> tuple[int, ...]:
        self.calls.append("shape")
        return (1,)


class ImageTerm(ObservationTermBase):
    """Stateless image term relying on the default peek and output_shape."""

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(NUM_ENVS, 4, 5, 1)


def counter_func(env: Any) -> torch.Tensor:
    return torch.ones(NUM_ENVS, 3)


def _manager(history_length: int = 1) -> ObservationManager:
    cfg = ObservationManagerCfg(
        groups={
            "policy": ObsGroupCfg(
                concatenate=True,
                enable_noise=False,
                history_length=history_length,
                terms={
                    "counter": ObsTermCfg(func=f"{_HERE}:StepCounterTerm"),
                    "plain": ObsTermCfg(func=f"{_HERE}:counter_func"),
                },
            ),
            "images": ObsGroupCfg(
                concatenate=False,
                enable_noise=False,
                history_length=1,
                terms={"image": ObsTermCfg(func=f"{_HERE}:ImageTerm")},
            ),
        }
    )
    return ObservationManager(cfg, SimpleNamespace(num_envs=NUM_ENVS, device="cpu"), device="cpu")


def _policy(obs: dict[str, Any]) -> torch.Tensor:
    group = obs["policy"]
    assert isinstance(group, torch.Tensor)
    return group


def _counter(manager: ObservationManager) -> StepCounterTerm:
    # The manager imports the term by dotted path, which can be a second copy of this module
    # under pytest, so compare by name rather than with isinstance.
    term = manager._term_instances["policy"]["counter"]
    assert type(term).__name__ == "StepCounterTerm"
    return cast("StepCounterTerm", term)


def test_step_compute_advances_stateful_terms_and_peek_does_not() -> None:
    manager = _manager()
    counter = _counter(manager)
    counter.calls.clear()

    first = manager.compute()
    peeked = manager.compute(modify_history=False)
    second = manager.compute()

    assert _policy(first)[:, 0].tolist() == [1.0, 1.0]
    assert _policy(peeked)[:, 0].tolist() == [1.0, 1.0]
    assert _policy(second)[:, 0].tolist() == [2.0, 2.0]
    assert counter.calls == ["call", "peek", "call"]


def test_obs_dims_come_from_output_shape_without_reading_the_term() -> None:
    manager = _manager(history_length=3)
    counter = _counter(manager)
    counter.calls.clear()

    dims = manager.get_obs_dims()

    assert dims == {"policy": (1 + 3) * 3, "images": {"image": (4, 5, 1)}}
    assert counter.calls == ["shape"]
    assert counter.count == 0


def test_default_peek_and_output_shape_use_call_for_side_effect_free_terms() -> None:
    term = ImageTerm(cfg=None, env=None)  # type: ignore[arg-type]

    assert term.peek(None).shape == (NUM_ENVS, 4, 5, 1)
    assert term.output_shape(None) == (4, 5, 1)


def test_peek_preserves_group_history() -> None:
    manager = _manager(history_length=2)

    manager.compute()
    manager.compute(modify_history=False)
    after = manager.compute()

    # Terms concatenate in sorted order, each as its flattened history: "counter" comes first.
    # Its history holds the two step reads (counts 1 and 2), not the peek in between.
    assert _policy(after)[0, :2].tolist() == [1.0, 2.0]
