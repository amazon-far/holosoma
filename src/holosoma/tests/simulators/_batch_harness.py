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
import sys
import traceback
import weakref
from pathlib import Path

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from tests.simulators._sim_harness import run_and_hard_exit

_live_cameras: weakref.WeakSet = weakref.WeakSet()


def _track_cameras() -> None:
    """Record every isaaclab ``Camera`` (TiledCamera included) built from here on in ``_live_cameras``."""
    from isaaclab.sensors import Camera

    init = Camera.__init__

    def tracked_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _live_cameras.add(self)

    Camera.__init__ = tracked_init


def _launch_settings(config, device) -> tuple:
    """The parts of a cell's config that ``setup_isaaclab_launcher`` turns into Kit launch arguments."""
    from holosoma.config_types.sensor import CameraSensorConfig

    rendering = (
        config.logger.video.enabled
        or config.logger.headless_recording
        or any(isinstance(s, CameraSensorConfig) for s in config.sensors.values())
    )
    launcher_args = sorted((k, repr(v)) for k, v in config.simulator.config.isaacsim.app_launcher_args().items())
    return config.training.headless, rendering, tuple(launcher_args), device


def _share_isaacsim_app() -> None:
    """Make every cell's ``launch_simulator`` reuse the Kit app the first cell launched.

    A cell whose launch settings differ from the first cell's raises: it would otherwise run on an app
    launched for someone else.
    """
    from holosoma.utils import sim_utils

    launch = sim_utils.setup_isaaclab_launcher
    apps: list = []
    first: list[tuple] = []

    def launch_once(config, device):
        settings = _launch_settings(config, device)
        if not apps:
            first.append(settings)
            # Kit swaps sys.stdout during launch; keep the cell's capture buffer in place.
            stdout = sys.stdout
            apps.append(launch(config, device))
            sys.stdout = stdout
            _track_cameras()
        elif settings != first[0]:
            raise RuntimeError(
                f"cell needs Kit launch settings {settings}, but the shared app was launched with {first[0]}"
            )
        return apps[0]

    sim_utils.setup_isaaclab_launcher = launch_once


def _reset_isaacsim_stage() -> None:
    """Drop the finished cell's ``SimulationContext`` and open a fresh stage, keeping the Kit app."""
    if "isaaclab.sim" not in sys.modules:
        return
    import isaaclab.sim as sim_utils
    import omni.timeline

    context = sim_utils.SimulationContext.instance()
    context_ref = None
    if context is not None:
        # clear_instance unsubscribes the stop handler, which would otherwise render until replay.
        context.clear_all_callbacks()
        context.clear_instance()
        # clear_instance leaves this observer subscribed, and its bound method keeps the context alive.
        context._on_post_physics_ready_callback = None
        context_ref = weakref.ref(context)
        del context
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
    # A camera's annotators are detached only when the camera is collected.
    if _live_cameras:
        raise RuntimeError(f"{len(_live_cameras)} camera(s) outlived their cell")
    if context_ref is not None and context_ref() is not None:
        raise RuntimeError("SimulationContext outlived its cell")


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
        # State the cell leaves behind fails that cell, and later cells don't run on top of it.
        leaked = False
        try:
            _reset_isaacsim_stage()
        except Exception:
            traceback.print_exc(file=buf)
            _fail_result_file(args)
            rc, leaked = 1, True
        results[cell_id] = {"returncode": rc, "stdout": buf.getvalue()}
        print(buf.getvalue(), end="", flush=True)
        tmp = Path(f"{out_path}.tmp")
        tmp.write_text(json.dumps(results))
        tmp.replace(out_path)
        if leaked:
            break
    return 0


def _fail_result_file(args: list[str]) -> None:
    """Overwrite the cell's ``--result-file`` so a cell that passed its checks still reads as failed."""
    if "--result-file" in args:
        Path(args[args.index("--result-file") + 1]).write_text("FAIL: cell left state behind")


if __name__ == "__main__":
    run_and_hard_exit(main)
