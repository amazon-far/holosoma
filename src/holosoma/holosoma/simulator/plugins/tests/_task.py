"""Minimal plugin owner for unit tests that drive plugins against a fake simulator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def plugin_task(simulator: Any, **attrs: Any) -> Any:
    """Return a stand-in task exposing ``simulator``, its ``hooks``, and any extra ``attrs``."""
    return SimpleNamespace(simulator=simulator, hooks=simulator.hooks, **attrs)
