"""Offscreen camera rendering must never implicitly request a viewer."""

from __future__ import annotations

import argparse
import gc
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
import pytest
from holosoma.simulator.isaacsim6.launcher import (
    LaunchedSimulation,
    active_simulation_cfg,
    apply_launcher_args,
)

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    "headless,rendering,requested,expected",
    [
        (True, False, (), []),
        (True, True, (), []),
        (False, False, (), ["kit"]),
        (False, True, (), ["kit"]),
        (True, True, ("rerun",), ["rerun"]),
        (False, True, ("viser", "rerun", "viser"), ["viser", "rerun"]),
        (False, True, ("kit", "rerun"), ["kit", "rerun"]),
    ],
)
def test_renderer_and_visualizers_are_independent(
    headless: bool, rendering: bool, requested: tuple[str, ...], expected: list[str]
) -> None:
    args = argparse.Namespace()
    apply_launcher_args(args, headless=headless, needs_rendering=rendering, visualizers=requested)
    assert args.visualizer == expected
    assert args.enable_cameras is rendering


def test_runtime_survives_a_discarded_app_handle_until_explicit_close() -> None:
    closed = []

    @contextmanager
    def context() -> Iterator[None]:
        try:
            yield
        finally:
            closed.append(True)

    cfg = object()
    runtime = LaunchedSimulation(context(), cfg)
    reference = weakref.ref(runtime)
    del runtime
    gc.collect()
    assert active_simulation_cfg() is cfg
    assert not closed
    retained = reference()
    assert retained is not None
    retained.close_with_failure(None)
    assert closed == [True]
    with pytest.raises(RuntimeError, match="Launch Isaac Lab"):
        active_simulation_cfg()
