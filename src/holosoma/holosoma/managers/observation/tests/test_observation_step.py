"""The observation manager's per-frame clock: one advancing step, and a row-subset terminal read."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 2
_HERE = "holosoma.managers.observation.tests.test_observation_step"


class CounterTerm(ObservationTermBase):
    """Per-env count of how many times each env was advanced."""

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        self.count = torch.zeros(env.num_envs)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.count[:] = 0
        else:
            self.count[env_ids] = 0

    def __call__(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        if env_ids is None:
            self.count += 1
            return self.count.unsqueeze(1)
        self.count[env_ids] += 1
        return self.count[env_ids].unsqueeze(1)


def env_value(env: Any) -> torch.Tensor:
    return env.value


def _env() -> SimpleNamespace:
    return SimpleNamespace(num_envs=NUM_ENVS, device="cpu", value=torch.zeros(NUM_ENVS, 1))


def _manager(env: SimpleNamespace, **group_kwargs: Any) -> ObservationManager:
    terms = group_kwargs.pop("terms", {"counter": ObsTermCfg(func=f"{_HERE}:CounterTerm")})
    cfg = ObservationManagerCfg(groups={"policy": ObsGroupCfg(terms=terms, **group_kwargs)})
    return ObservationManager(cfg, env, device="cpu")


def _policy(obs: dict[str, Any]) -> torch.Tensor:
    group = obs["policy"]
    assert isinstance(group, torch.Tensor)
    return group


def _step(manager: ObservationManager) -> torch.Tensor:
    manager.new_frame()
    return _policy(manager.step())


def test_step_advances_once_per_frame() -> None:
    manager = _manager(_env())

    assert _step(manager)[:, 0].tolist() == [1.0, 1.0]
    assert _step(manager)[:, 0].tolist() == [2.0, 2.0]


def test_second_step_in_a_frame_raises() -> None:
    manager = _manager(_env())
    _step(manager)

    with pytest.raises(RuntimeError, match="without an open frame"):
        manager.step()


def test_step_terminal_advances_only_its_rows() -> None:
    manager = _manager(_env())
    _step(manager)

    manager.new_frame()
    terminal = _policy(manager.step_terminal(torch.tensor([0])))
    assert terminal.tolist() == [[2.0]]

    manager.reset(torch.tensor([0]))
    # Row 1 was not advanced by the terminal read, so it advances once from 1 to 2.
    assert _policy(manager.step())[:, 0].tolist() == [1.0, 2.0]


def test_step_before_resetting_terminal_rows_raises() -> None:
    manager = _manager(_env())
    manager.new_frame()
    manager.step_terminal(torch.tensor([1]))

    with pytest.raises(RuntimeError, match="before resetting"):
        manager.step()


def test_step_terminal_after_step_or_twice_raises() -> None:
    manager = _manager(_env())
    _step(manager)
    with pytest.raises(RuntimeError, match="once per frame"):
        manager.step_terminal(torch.tensor([0]))

    manager.new_frame()
    manager.step_terminal(torch.tensor([0]))
    with pytest.raises(RuntimeError, match="once per frame"):
        manager.step_terminal(torch.tensor([0]))


def test_terminal_history_appends_current_frame_without_modifying_buffers() -> None:
    env = _env()
    manager = _manager(env, history_length=2, terms={"value": ObsTermCfg(func=f"{_HERE}:env_value")})
    env.value = torch.tensor([[1.0], [10.0]])
    _step(manager)

    env.value = torch.tensor([[2.0], [20.0]])
    manager.new_frame()
    assert _policy(manager.step_terminal(torch.tensor([1]))).tolist() == [[10.0, 20.0]]

    manager.reset(torch.tensor([1]))
    # Env 0's history holds one entry per step; env 1's was cleared by the reset.
    assert _policy(manager.step()).tolist() == [[1.0, 2.0], [0.0, 20.0]]


def test_noise_does_not_draw_from_the_global_rng() -> None:
    manager = _manager(_env(), enable_noise=True, terms={"value": ObsTermCfg(func=f"{_HERE}:env_value", noise=0.5)})
    rng_state = torch.get_rng_state()

    noisy = _step(manager)

    assert torch.equal(torch.get_rng_state(), rng_state)
    assert not torch.equal(noisy, torch.zeros_like(noisy))


def test_noise_follows_the_global_seed() -> None:
    def noise_after_seed(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        manager = _manager(_env(), enable_noise=True, terms={"value": ObsTermCfg(func=f"{_HERE}:env_value", noise=0.5)})
        return _step(manager)

    # The global seed carries the per-rank offset, so ranks get different noise.
    assert torch.equal(noise_after_seed(1), noise_after_seed(1))
    assert not torch.equal(noise_after_seed(1), noise_after_seed(2))


class LeadingViewTerm(CounterTerm):
    """Counter that serves the first ``len(env_ids)`` envs as a view of its buffer."""

    def __call__(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        if env_ids is None:
            return super().__call__(env)
        self.count[env_ids] += 1
        return self.count[: len(env_ids)].unsqueeze(1)


def test_terminal_obs_survive_reset_of_term_state() -> None:
    manager = _manager(_env(), terms={"counter": ObsTermCfg(func=f"{_HERE}:LeadingViewTerm")})
    _step(manager)

    manager.new_frame()
    terminal = _policy(manager.step_terminal(torch.tensor([0])))
    manager.reset(torch.tensor([0]))

    assert terminal.tolist() == [[2.0]]


class OldContractTerm(CounterTerm):
    """A term written before env_ids: swallows it and computes every env."""

    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        return super().__call__(env)


def test_old_contract_term_is_rejected_at_sizing() -> None:
    manager = _manager(_env(), terms={"counter": ObsTermCfg(func=f"{_HERE}:OldContractTerm")})

    with pytest.raises(ValueError, match="'counter' in group 'policy' returned 2 rows for 0 env_ids"):
        manager.get_obs_dims()


def test_get_obs_dims_does_not_advance_stateful_terms() -> None:
    manager = _manager(_env())

    assert manager.get_obs_dims() == {"policy": 1}
    assert _step(manager)[:, 0].tolist() == [1.0, 1.0]


def assert_advances_only_env_ids(make_term: Callable[[], ObservationTermBase], env: Any, steps: int = 3) -> None:
    """Check a stateful term's contract: advancing env 0 leaves env 1 as a fresh instance's."""
    term, fresh = make_term(), make_term()
    for _ in range(steps):
        term(env, env_ids=torch.tensor([0]))
    assert torch.equal(term(env)[1], fresh(env)[1])


def test_counter_term_honours_env_ids() -> None:
    env = _env()
    assert_advances_only_env_ids(lambda: CounterTerm(cfg=None, env=env), env)
