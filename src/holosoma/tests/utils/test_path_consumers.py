"""Regression tests for configuration consumers of the generic path resolver."""

from __future__ import annotations

import pytest

from holosoma.managers.command.terms import wbt

pytestmark = pytest.mark.no_sim


def test_empty_motion_directory_list_does_not_expand_root_glob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def resolve_paths(pattern: str):
        calls.append(pattern)
        return iter(())

    monkeypatch.setattr(wbt, "resolve_paths", resolve_paths)

    with pytest.raises(AssertionError, match=r"No \.npz files found"):
        wbt.MultiMotionLoader(" , ", [], [])

    assert calls == []
