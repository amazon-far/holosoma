"""Shared simulation utilities for holosoma.

Provider launch, signal handling, and dependency-ordered teardown used by
:func:`holosoma.simulator.session.task_session`.
"""

from __future__ import annotations

import argparse
import functools
import os
import signal
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Any, TypeVar

from loguru import logger
from typing_extensions import ParamSpec

from holosoma.config_types.env import EnvConfig
from holosoma.config_types.sensor import CameraSensorConfig
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type

P = ParamSpec("P")
R = TypeVar("R")


@contextmanager
def exit_on_sigterm() -> Iterator[None]:
    """Turn SIGTERM into normal Python unwinding without changing SIGINT."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def graceful_simulation_signals(func: Callable[P, R]) -> Callable[P, R]:
    """Install simulation signal semantics around one public entrypoint."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with exit_on_sigterm():
            return func(*args, **kwargs)

    return wrapper


def format_simulation_performance(
    physics_steps: int,
    control_steps: int,
    sim_dt: float,
    wall_time: float,
    *,
    prefix: str = "Simulation performance",
) -> str:
    """Format simulation progress in terms of simulated and wall-clock time."""
    if wall_time <= 0:
        return f"{prefix}: unavailable (no wall time elapsed)"

    sim_time = physics_steps * sim_dt
    return (
        f"{prefix}: {sim_time / wall_time:.2f}x real time "
        f"({sim_time:.2f} sim s / {wall_time:.2f} wall s) | "
        f"{physics_steps / wall_time:.1f} physics steps/s | "
        f"{control_steps / wall_time:.1f} control steps/s"
    )


def setup_simulator_imports(config: EnvConfig) -> None:
    """Setup simulator-specific imports without side effects.

    Parameters
    ----------
    config : EnvConfig
        Configuration containing simulator settings.
    """
    set_simulator_type(config.simulator)
    simulator_type = get_simulator_type()

    if simulator_type == SimulatorType.MUJOCO:
        import mujoco

        assert mujoco is not None

    # IsaacSim imports handled in setup_isaaclab_launcher


def _ensure_render_pipeline_active() -> None:
    """Force IsaacLab's offscreen render pipeline on when cameras are enabled.

    ``AppLauncher`` only sets ``/isaaclab/render/offscreen`` when ``enable_cameras AND headless``.
    ``SimulationContext`` reads that flag at construction and, with no GUI and no offscreen render,
    latches ``render_mode = NO_GUI_OR_RENDERING`` — a terminal state in which ``sim.render()`` is a
    no-op, so camera sensors never receive a new frame. A non-headless run with cameras (the
    ``RunSimConfig`` default) lands exactly there: cameras are enabled but nothing renders.

    Setting the flag after the app exists but before ``SimulationContext`` is built puts the run in
    ``PARTIAL_RENDERING`` instead, which is what the camera path needs. A real GUI run is unaffected:
    ``_has_gui`` already selects ``FULL_RENDERING``.
    """
    import carb

    settings = carb.settings.get_settings()
    if not settings.get("/isaaclab/render/offscreen"):
        settings.set_bool("/isaaclab/render/offscreen", True)


def setup_isaaclab_launcher(config: EnvConfig, device: str) -> Any | None:
    """Handle IsaacSim-specific launcher setup.

    Parameters
    ----------
    config : EnvConfig
        Configuration of the task the app will host. ``training.num_envs`` is per process.
    device : str
        Resolved device string (e.g., 'cuda:0', 'cpu').

    Returns
    -------
    Any | None
        IsaacSim simulation app instance, or None for other simulators.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run simulation with IsaacSim.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--env_spacing", type=int, default=20, help="Distance between environments in simulator.")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to store the output.")
    AppLauncher.add_app_launcher_args(parser)

    # Parse known arguments to get argparse params
    args_cli, unknown_args = parser.parse_known_args()

    args_cli.num_envs = config.training.num_envs
    args_cli.seed = config.training.seed
    args_cli.env_spacing = config.scene.env_spacing
    args_cli.output_dir = config.logger.base_dir
    args_cli.headless = config.training.headless
    args_cli.device = device
    # Distribute the simulator across GPUs when using multi-GPU training.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        args_cli.distributed = True

    # LiDAR ray queries do not need RTX; only video and TiledCamera sensors enable the renderer.
    video_enabled = config.logger.video.enabled or config.logger.headless_recording
    cameras_enabled = any(isinstance(sensor, CameraSensorConfig) for sensor in config.sensors.values())
    needs_rendering = video_enabled or cameras_enabled
    if needs_rendering:
        args_cli.enable_cameras = True

    isaacsim_config = config.simulator.config.isaacsim
    for name, value in isaacsim_config.app_launcher_args().items():
        setattr(args_cli, name, value)

    simulation_app: Any = None
    try:
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        if needs_rendering:
            _ensure_render_pipeline_active()

        logger.info(f"IsaacSim args_cli: {args_cli}")
        logger.info(f"IsaacSim unknown_args: {unknown_args}")
        sys.argv = [sys.argv[0]] + unknown_args
        return simulation_app
    except BaseException:
        if simulation_app is not None:
            try:
                close_simulation_app(simulation_app)
            except Exception:
                logger.exception("Simulation app cleanup failed after launcher setup error")
        raise


def setup_keyboard_listener(env: Any) -> threading.Thread:
    """Setup keyboard listener thread for simulation control.

    Parameters
    ----------
    env
        Environment instance to control.

    Returns
    -------
    threading.Thread
        Keyboard listener thread (already started).
    """

    def on_press(key: Any, env: Any) -> None:
        """Handle keyboard input for simulation control."""
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "n":
                    if hasattr(env, "next_task"):
                        env.next_task()
                        logger.info("Moved to the next task.")
                # Force Control
                elif key.char == "1":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale /= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
                elif key.char == "2":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale *= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
        except AttributeError:
            pass

    def listen_for_keypress(env: Any) -> None:
        """Listen for keyboard input in a separate thread."""
        try:
            # Delay import so that one can run the rest of this script in headless mode.
            # Trying to import pynput in headless mode gives the following error:
            # ImportError: this platform is not supported:
            # ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
            from pynput import keyboard as pynput_keyboard

            logger.info("Keyboard controls:")
            logger.info("  n - Next task (if supported)")
            logger.info("  1/2 - Decrease/Increase force scale (if supported)")

            with pynput_keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
                listener.join()
        except ImportError:
            logger.warning("pynput not available - keyboard controls disabled")
        except Exception as e:
            logger.warning(f"Keyboard listener failed: {e}")

    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()
    return key_listener_thread


def _simulation_exit_code(failure: BaseException | None) -> int:
    """Match Python's exception and signal exit statuses."""
    if failure is None:
        return 0
    if isinstance(failure, KeyboardInterrupt):
        return 128 + signal.SIGINT
    if isinstance(failure, SystemExit):
        if failure.code is None:
            return 0
        if isinstance(failure.code, int):
            return int(failure.code)
    return 1


def _flush_simulation_logs() -> None:
    """Attempt both delivery steps without propagating diagnostic errors."""
    with suppress(Exception):
        logger.complete()
    with suppress(Exception):
        sys.stderr.flush()


def _report_simulation_failure(failure: BaseException | None) -> None:
    """Deliver the original failure without changing provider lifetime."""
    if failure is None:
        return
    try:
        if isinstance(failure, SystemExit):
            if failure.code is not None and not isinstance(failure.code, int):
                logger.error("Simulation exited: {}", failure.code)
        elif not isinstance(failure, KeyboardInterrupt):
            logger.opt(exception=failure).error("Simulation failed before Isaac Sim shutdown")
    except Exception:
        with suppress(Exception):
            stderr = sys.__stderr__
            if stderr is not None:
                if isinstance(failure, SystemExit):
                    print(failure.code, file=stderr)
                else:
                    traceback.print_exception(type(failure), failure, failure.__traceback__, file=stderr)
                stderr.flush()
    _flush_simulation_logs()


def close_simulation_app(simulation_app: Any) -> None:
    """Close simulation app with workarounds for known issues.

    For a non-None Isaac Sim app, report the active failure before closing.
    A nonzero failure status queues an uncancellable Kit quit request.

    Parameters
    ----------
    simulation_app : Any
        The simulation app instance returned by setup_isaaclab_launcher().
        Can be None for simulators that don't have an app (e.g., MuJoCo).
    """
    failure = sys.exc_info()[1]
    if simulation_app is not None and get_simulator_type() == SimulatorType.ISAACSIM:
        _report_simulation_failure(failure)
    _close_simulation_app(simulation_app, failure)


def _close_simulation_app(simulation_app: Any, failure: BaseException | None) -> None:
    """Shut down the provider after its owner has reported any active failure."""
    if simulation_app is not None and get_simulator_type() == SimulatorType.ISAACSIM:
        # Kit can terminate the process inside close(), before Python reaches the
        # caller's exception handler. Set its exit status before closing the app.
        return_code = _simulation_exit_code(failure)
        if return_code:
            try:
                simulation_app.app.post_uncancellable_quit(return_code)
            except Exception as exc:
                with suppress(Exception):
                    logger.warning("Could not set Isaac Sim exit status to {}: {}", return_code, exc)

        with suppress(Exception):
            logger.info("Shutting down simulation app...")
        try:
            # Work-around for IsaacLab hanging headless.
            # Patch the close_stage method to avoid hanging
            import omni.usd

            context = omni.usd.get_context()
            context_class = context.__class__

            # Replace with a no-op version
            def noop_close_stage(self: Any, *args: Any, **kwargs: Any) -> bool:
                with suppress(Exception):
                    logger.debug("Skipping close_stage() to avoid hanging")
                return True

            # Apply the patch
            context_class.close_stage = noop_close_stage
            with suppress(Exception):
                logger.debug("Successfully patched close_stage method")
        except Exception as e:
            with suppress(Exception):
                logger.warning(f"Could not patch close_stage method: {e}")

        try:
            # Work-around for IsaacLab SimulationContext._app_control_on_stop_handle_fn
            # hanging in an infinite render() loop on shutdown. When simulation_app.close()
            # triggers a timeline STOP event, the callback spins waiting for the timeline to
            # start playing again — which never happens. Disabling the callback prevents this.
            from isaaclab.sim import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context._disable_app_control_on_stop_handle = True
                with suppress(Exception):
                    logger.debug("Disabled SimulationContext app_control_on_stop_handle to prevent shutdown hang")
        except Exception as e:
            with suppress(Exception):
                logger.warning(f"Could not disable app_control_on_stop_handle: {e}")

        # Now close the app
        _flush_simulation_logs()
        simulation_app.close(wait_for_replicator=False)
        with suppress(Exception):
            logger.info("Simulation app closed.")


def _close_environment(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if close is not None:
        close()
        return
    simulator = getattr(env, "simulator", None)
    if simulator is not None:
        simulator.close()


def close_simulation_resources(env: Any, simulation_app: Any) -> None:
    """Close an environment before the provider application it depends on.

    For a non-None Isaac Sim app, report the active failure before teardown
    and request its nonzero exit status when proceeding to app shutdown.
    Failed environment teardown leaves the provider open without queuing quit.
    """
    failure = sys.exc_info()[1]
    native_app = False
    # An unset backend must not prevent environment cleanup.
    with suppress(RuntimeError):
        native_app = simulation_app is not None and get_simulator_type() == SimulatorType.ISAACSIM
    if native_app:
        # Teardown can block, so deliver the original failure before entering it.
        _report_simulation_failure(failure)
    try:
        _close_environment(env)
    except Exception as exc:
        # The original failure was reported above; report the cleanup error
        # only when there was no earlier failure.
        if native_app and failure is None:
            _report_simulation_failure(exc)
        raise RuntimeError("Environment cleanup failed; provider application remains open") from exc

    try:
        _close_simulation_app(simulation_app, failure)
    except Exception as exc:
        raise RuntimeError("Simulation app cleanup failed") from exc
