"""Run several harness ``main()`` cells in one process and record each cell's outcome.

Usage: ``python _batch_harness.py SPEC.json OUT.json``. ``SPEC`` maps a cell id to
``[harness_module_stem, [argv...]]``; ``OUT`` is rewritten after every cell with
``{cell_id: {"returncode": int, "stdout": str}}``, so cells finished before a crash keep their result.
Each cell builds its own sim; only the interpreter and imports are shared. On Isaac Sim the Kit app
is also shared: it is launched by the first cell and kept open, and each cell gets a fresh stage and
``SimulationContext``.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)


def _share_isaacsim_app() -> None:
    """Make every cell's ``setup_simulation_environment`` reuse one Kit app and never close it."""
    from holosoma.utils import sim_utils

    launch, close = sim_utils.setup_isaaclab_launcher, sim_utils._close_simulation_app
    apps: list = []

    def launch_once(*args, **kwargs):
        if not apps:
            # Kit swaps sys.stdout during launch; keep the cell's capture buffer in place.
            stdout = sys.stdout
            apps.append(launch(*args, **kwargs))
            sys.stdout = stdout
        return apps[0]

    def close_unshared(simulation_app, failure):
        if simulation_app not in apps:
            close(simulation_app, failure)

    sim_utils.setup_isaaclab_launcher = launch_once
    sim_utils._close_simulation_app = close_unshared


def _reset_isaacsim_stage() -> None:
    """Drop the finished cell's ``SimulationContext`` and open a fresh stage, keeping the Kit app."""
    if "isaaclab.sim" not in sys.modules:
        return
    import isaaclab.sim as sim_utils
    import omni.timeline

    context = sim_utils.SimulationContext.instance()
    if context is not None:
        # clear_instance unsubscribes the stop handler, which would otherwise render until replay.
        context.clear_all_callbacks()
        context.clear_instance()
    omni.timeline.get_timeline_interface().stop()
    sim_utils.create_new_stage()
    # Kit's import stat cache keeps each failed stat's exception, whose traceback pins the frames
    # (and so the sim) of every lazy import made during the cell.
    from omni.ext._impl import stat_cache

    stat_cache._stat_cache.clear()
    gc.collect()
    from isaaclab.sensors import RayCaster

    # Ray casters share a class-level mesh cache that is only dropped when the last instance dies.
    if RayCaster._instance_count:
        raise RuntimeError(f"{RayCaster._instance_count} RayCaster(s) outlived their cell")


def main() -> int:
    spec_path, out_path = sys.argv[1:3]
    with open(spec_path) as f:
        cells = json.load(f)
    _share_isaacsim_app()
    results: dict[str, dict] = {}
    for cell_id, (module, args) in cells.items():
        buf = io.StringIO()
        sys.argv = [module, *args]
        with contextlib.redirect_stdout(buf):
            print(f"=== cell {cell_id}: {module} {' '.join(args)}", flush=True)
            try:
                rc = importlib.import_module(f"tests.simulators.{module}").main()
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
            except Exception:
                traceback.print_exc(file=buf)
                rc = 1
        gc.collect()
        results[cell_id] = {"returncode": rc, "stdout": buf.getvalue()}
        print(buf.getvalue(), end="", flush=True)
        tmp = Path(f"{out_path}.tmp")
        tmp.write_text(json.dumps(results))
        tmp.replace(out_path)
        _reset_isaacsim_stage()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
