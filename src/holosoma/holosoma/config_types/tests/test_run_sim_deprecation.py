"""Tests for the RunSimConfig.env_class deprecation shim."""

from __future__ import annotations

import dataclasses
import datetime
import warnings

import pytest

import holosoma.config_types.run_sim as run_sim_module
from holosoma.config_types.run_sim import _ENV_CLASS_REMOVAL_DATE, RunSimConfig


def _future_warnings(record: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in record if issubclass(w.category, FutureWarning)]


def test_default_construction_is_silent() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cfg = RunSimConfig()
        _ = cfg.env_class  # None default: reads stay silent too
        dataclasses.asdict(cfg)
        dataclasses.replace(cfg, viewer_dt=0.02)
    assert _future_warnings(record) == []


def test_explicit_env_class_warns_on_init_and_read() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cfg = RunSimConfig(env_class="pkg.mod.Env")
    assert len(_future_warnings(record)) == 1
    assert "no effect" in str(_future_warnings(record)[0].message).lower()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert cfg.env_class == "pkg.mod.Env"  # tolerated, but warns
    assert len(_future_warnings(record)) == 1


def test_env_class_errors_after_removal_date(monkeypatch: pytest.MonkeyPatch) -> None:
    day_after = _ENV_CLASS_REMOVAL_DATE + datetime.timedelta(days=1)

    class _FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> _FrozenDateTime:
            return cls(day_after.year, day_after.month, day_after.day, tzinfo=tz)

    monkeypatch.setattr(run_sim_module.datetime, "datetime", _FrozenDateTime)
    with pytest.raises(RuntimeError, match="deprecated"):
        RunSimConfig(env_class="pkg.mod.Env")
