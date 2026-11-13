"""Minimal plugin owner for unit tests that drive plugins against a fake simulator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def plugin_task(simulator: Any, **attrs: Any) -> Any:
    """Return a stand-in task exposing ``simulator``, its ``hooks``, and any extra ``attrs``.

    ``num_envs`` defaults to the fake simulator's ``training_config.num_envs`` when it has one.
    """
    training_config = getattr(simulator, "training_config", None)
    if training_config is not None:
        attrs.setdefault("num_envs", training_config.num_envs)
    return SimpleNamespace(simulator=simulator, hooks=simulator.hooks, **attrs)
