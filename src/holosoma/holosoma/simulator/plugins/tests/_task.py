"""Minimal plugin owner for unit tests that drive plugins against a fake simulator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def plugin_task(simulator: Any, **attrs: Any) -> Any:
    """Return a stand-in task exposing ``simulator`` and any extra ``attrs``.

    ``hooks`` defaults to the fake simulator's own registry, which the unit tests emit on directly.
    """
    attrs.setdefault("hooks", getattr(simulator, "hooks", None))
    return SimpleNamespace(simulator=simulator, **attrs)
