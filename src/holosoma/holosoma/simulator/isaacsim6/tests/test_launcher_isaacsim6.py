"""The Isaac Sim 6.1 lane's launcher-argument mapping.

Why these are worth testing on the CPU lane
-------------------------------------------
``apply_launcher_args`` is pure namespace manipulation -- no Isaac Sim, no Kit -- but it decides
whether a run renders at all, and every way it can be wrong is silent. Isaac Lab 3.0 deleted
``--headless`` and ``--enable_cameras``, so a mapping that sets the old attribute names does
nothing at all: ``argparse`` never defines them, ``setattr`` succeeds, and the run comes up
headless while the configuration says otherwise.

The ``isaac_visualizers`` cases exist because exactly that happened: the field was validated by
``capabilities._validate_visualizers`` and then never read by anything, so every run resolved to
``kit`` or ``None`` and a configured ``rerun`` / ``viser`` / ``newton`` selection was discarded
without a word. Nothing failed, which is why no test caught it -- the rendering backend is
observable only in what the launcher is handed.
"""

from __future__ import annotations

import argparse

import pytest

from holosoma.simulator.isaacsim6.launcher import KIT_VISUALIZER, apply_launcher_args

pytestmark = pytest.mark.no_sim


def _args() -> argparse.Namespace:
    """A namespace with ``visualizer`` predefined, as ``add_launcher_args`` leaves it."""
    return argparse.Namespace(visualizer="SENTINEL")


def test_headless_without_rendering_requests_no_visualizer() -> None:
    """A true headless run is 3.0's default and the cheapest configuration."""
    args = _args()
    apply_launcher_args(args, headless=True, needs_rendering=False)
    assert args.visualizer is None


def test_non_headless_requests_kit() -> None:
    """A GUI viewport needs Kit; `headless=False` alone is silently ignored by 3.0."""
    args = _args()
    apply_launcher_args(args, headless=False, needs_rendering=False)
    assert args.visualizer == KIT_VISUALIZER


def test_headless_with_rendering_still_requests_kit() -> None:
    """Offscreen RTX sensors need Kit even with no window.

    This is the replacement for 2.3.2's ``--enable_cameras``: there is no separate flag in 3.0, so
    a headless run with a ``TiledCamera`` must still ask for the Kit renderer or the camera
    returns nothing.
    """
    args = _args()
    apply_launcher_args(args, headless=True, needs_rendering=True)
    assert args.visualizer == KIT_VISUALIZER


def test_explicit_visualizers_are_passed_through() -> None:
    """An explicit selection must actually reach the launcher.

    The regression test for the field being inert: before this, the result was ``kit`` regardless
    of what was configured.
    """
    args = _args()
    apply_launcher_args(args, headless=True, needs_rendering=False, visualizers=("rerun",))
    assert args.visualizer == "rerun"


def test_multiple_visualizers_become_a_csv() -> None:
    """Isaac Lab 3.0's ``--visualizer`` takes a comma-separated list of backend names."""
    args = _args()
    apply_launcher_args(args, headless=False, needs_rendering=False, visualizers=("rerun", "viser"))
    assert args.visualizer == "rerun,viser"


def test_rendering_adds_kit_to_an_explicit_selection() -> None:
    """A configured camera must keep working alongside another visualizer.

    Requesting ``rerun`` with a ``TiledCamera`` configured must not turn the camera off, so ``kit``
    is unioned in rather than the selection replacing it.
    """
    args = _args()
    apply_launcher_args(args, headless=False, needs_rendering=True, visualizers=("rerun",))
    assert args.visualizer == f"rerun,{KIT_VISUALIZER}"


def test_kit_is_not_duplicated_when_already_requested() -> None:
    """``kit,kit`` is not a name Isaac Lab accepts, so the union must dedupe."""
    args = _args()
    apply_launcher_args(args, headless=False, needs_rendering=True, visualizers=("kit", "rerun"))
    assert args.visualizer == "kit,rerun"


def test_duplicate_names_are_collapsed_in_order() -> None:
    """A repeated name is collapsed, keeping first-seen order so the CSV is deterministic."""
    args = _args()
    apply_launcher_args(args, headless=False, needs_rendering=False, visualizers=("viser", "rerun", "viser"))
    assert args.visualizer == "viser,rerun"


def test_the_old_2_3_2_flags_are_never_set() -> None:
    """Isaac Lab 3.0 has no ``headless`` / ``enable_cameras`` launcher arguments.

    Setting either would be a silent no-op rather than an error -- ``argparse`` does not define
    them in 3.0, so the assignment lands on the namespace and is never read. Asserting their
    absence pins that this lane maps rendering intent onto ``visualizer`` only.
    """
    args = _args()
    apply_launcher_args(args, headless=True, needs_rendering=True, visualizers=("kit",))
    assert not hasattr(args, "headless")
    assert not hasattr(args, "enable_cameras")
