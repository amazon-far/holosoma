"""Isaac Lab 3.x launcher arguments for this lane.

``utils/sim_utils.setup_isaaclab_launcher`` runs before any simulator object exists, so it cannot
ask a live instance which generation it is driving. Instead it looks for a ``launcher`` module in
the configured backend package and delegates the argument mapping to it. A lane without this
module keeps the historical 2.3.2 mapping, which is why the 5.1 lane needs no changes and no
version check appears in shared code.

Isaac Lab 3.0 replaced the launcher's rendering flags outright:

* ``--headless`` is **gone**. 3.0 is headless unless a visualizer is requested, so the boolean
  inverts into ``--visualizer``: a CSV of backends (``kit,newton,rerun,viser``) that defaults to
  ``None``. Leaving the old attribute set would be worse than useless -- ``argparse`` never
  defines it, so assigning ``args.headless = False`` silently does nothing and a run intended to
  show a window comes up headless, which is how a GUI request gets lost with no error.
* ``--enable_cameras`` is **gone** as well. Offscreen RTX rendering for ``TiledCamera`` needs the
  Kit visualizer running, so it maps to the same ``kit`` backend rather than a separate flag.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

KIT_VISUALIZER = "kit"
"""The Kit visualizer backend, needed for both a GUI viewport and offscreen RTX sensors."""


def apply_launcher_args(
    args: argparse.Namespace,
    *,
    headless: bool,
    needs_rendering: bool,
    visualizers: Sequence[str] = (),
) -> None:
    """Translate holosoma's rendering intent into Isaac Lab 3.x launcher arguments.

    Parameters
    ----------
    args
        The namespace produced by ``add_launcher_args`` / ``AppLauncher.add_app_launcher_args``,
        mutated in place.
    headless
        ``config.training.headless`` -- whether to run without a visible window.
    needs_rendering
        Whether video recording or a configured camera requires the RTX render pipeline, even
        when running without a window.
    visualizers
        ``simulator.config.isaac_visualizers`` -- the explicitly requested rendering backends.
        Empty means "decide from ``headless`` / ``needs_rendering``".

    Notes
    -----
    An explicit selection wins, and is passed through verbatim. It is additionally unioned with
    ``kit`` when ``needs_rendering`` is set, because an offscreen RTX sensor needs the Kit
    renderer regardless of which other visualizers were asked for -- requesting ``rerun`` with a
    ``TiledCamera`` configured must not turn the camera off.

    With no explicit selection, ``visualizer`` stays ``None`` for a true headless run (3.0's
    default and the cheapest configuration) and becomes ``kit`` when a window or an RTX sensor
    needs it.

    Honouring ``visualizers`` here is the whole point of the field. It was previously validated by
    ``capabilities._validate_visualizers`` and then never read, so every run resolved to ``kit``
    or ``None`` and a configured ``rerun`` / ``viser`` / ``newton`` selection was silently
    discarded -- the exact silent-drop this module's docstring describes for ``--headless``.
    """
    requested = list(dict.fromkeys(visualizers))
    if requested:
        if needs_rendering and KIT_VISUALIZER not in requested:
            requested.append(KIT_VISUALIZER)
        args.visualizer = ",".join(requested)
        return
    if headless and not needs_rendering:
        args.visualizer = None
        return
    args.visualizer = KIT_VISUALIZER
