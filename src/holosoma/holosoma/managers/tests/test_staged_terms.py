"""StagedTerms: shared instances, stage filtering, call order and config errors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.managers.staged_terms import StagedTerms

pytestmark = pytest.mark.no_sim

CALLS: list[tuple[Any, ...]] = []


class _Base:
    def __init__(self, cfg: Any, env: Any) -> None:
        self.cfg = cfg

    def setup(self) -> None:
        CALLS.append(("class.setup", id(self)))

    def reset(self, env_ids: Any) -> None:
        CALLS.append(("class.reset", id(self), env_ids))

    def step(self) -> None:
        CALLS.append(("class.step", id(self)))


class _Term(_Base):
    pass


class _OtherTerm(_Base):
    pass


class _NotATerm:
    def __init__(self, cfg: Any, env: Any) -> None:
        pass


def _func(env: Any, *args: Any, **params: Any) -> None:
    CALLS.append(("func", *args, *sorted(params.items())))


def _term(func: Any, **params: Any) -> SimpleNamespace:
    return SimpleNamespace(func=func, params=params)


def _cfg(
    setup: dict[str, Any] | None = None, reset: dict[str, Any] | None = None, step: dict[str, Any] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(setup_terms=setup or {}, reset_terms=reset or {}, step_terms=step or {})


def _staged(cfg: SimpleNamespace, **kwargs: Any) -> StagedTerms:
    return StagedTerms(cfg, env="env", term_base=_Base, context="test term", owner="owner", **kwargs)


@pytest.fixture(autouse=True)
def _clear_calls() -> None:
    CALLS.clear()


def test_class_listed_in_several_stages_is_one_instance_that_runs_only_there_before_functions() -> None:
    staged = _staged(_cfg(reset={"t": _term(_Term), "f": _term(_func, k=1)}, step={"t": _term(_Term)}))
    instance = staged.instances["t"]

    staged.run("setup", "env")
    staged.run("reset", "env", "ids")
    staged.run("step", "env")

    assert list(staged.instances) == ["t"]
    assert instance.manager == "owner"
    assert [("class.reset", id(instance), "ids"), ("func", "ids", ("k", 1)), ("class.step", id(instance))] == CALLS


def test_run_every_stage_runs_a_class_listed_under_one_stage_in_all_three() -> None:
    staged = _staged(_cfg(setup={"t": _term(_Term)}), class_terms_run_every_stage=True)

    for stage in ("setup", "reset", "step"):
        staged.run(stage, "env", "ids")

    assert [call[0] for call in CALLS] == ["class.setup", "class.reset", "class.step"]


@pytest.mark.parametrize(
    ("cfg", "error"),
    [
        (_cfg(setup={"t": _term(_NotATerm)}), TypeError),
        (_cfg(setup={"t": _term(_Term)}, reset={"t": _term(_OtherTerm)}), ValueError),
        (_cfg(setup={"t": _term(_func)}, reset={"t": _term(_Term)}), ValueError),
        (_cfg(setup={"t": _term(_Term)}, reset={"t": _term(_func)}), ValueError),
    ],
    ids=["class-without-base", "conflicting-classes", "function-then-class", "class-then-function"],
)
def test_misconfigured_terms_are_rejected(cfg: SimpleNamespace, error: type[Exception]) -> None:
    with pytest.raises(error, match="'t'"):
        _staged(cfg)
