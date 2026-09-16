from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.train_agent import _apply_preprocess_hook

pytestmark = pytest.mark.no_sim


def _config(kwargs: str) -> ExperimentConfig:
    return cast(
        "ExperimentConfig",
        SimpleNamespace(
            training=SimpleNamespace(
                preprocess_hook="tests.fake:hook",
                preprocess_hook_kwargs=kwargs,
            )
        ),
    )


def test_preprocess_hook_receives_decoded_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config('{"value": 7}')

    def hook(received, *, value):
        assert received is config
        assert value == 7
        return received

    monkeypatch.setattr("holosoma.managers.utils.resolve_callable", lambda *_args, **_kwargs: hook)

    assert _apply_preprocess_hook(config) is config


@pytest.mark.parametrize("kwargs", ["[1, 2]", '"value"'])
def test_preprocess_hook_requires_a_json_object(kwargs: str) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _apply_preprocess_hook(_config(kwargs))


def test_preprocess_hook_reports_invalid_json() -> None:
    with pytest.raises(ValueError, match="Invalid JSON"):
        _apply_preprocess_hook(_config("{"))


def test_preprocess_hook_rejects_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("holosoma.managers.utils.resolve_callable", lambda *_args, **_kwargs: lambda *_a, **_k: None)

    with pytest.raises(TypeError, match="returned None"):
        _apply_preprocess_hook(_config("{}"))
