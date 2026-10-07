"""Shared simulation utilities for holosoma.

This module provides common functionality for setting up and running simulations,
shared between eval_agent.py and run_sim.py.
"""

from __future__ import annotations

import argparse
import functools
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager, suppress
from types import TracebackType
from typing import Any, Callable, Iterator, TypeVar

from loguru import logger
from typing_extensions import ParamSpec, Self

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.sensor import CameraSensorConfig, validate_camera_dict
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.rate import RateLimiter
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type
from holosoma.utils.torch_utils import to_torch

_PERFORMANCE_LOG_INTERVAL_STEPS = 1000

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


def _format_simulation_performance(
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


def setup_simulator_imports(config: ExperimentConfig | RunSimConfig) -> None:
    """Setup simulator-specific imports without side effects.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator settings.
    """
    set_simulator_type(config.simulator)
    simulator_type = get_simulator_type()

    if simulator_type == SimulatorType.MUJOCO:
        import mujoco

        assert mujoco is not None
    elif simulator_type == SimulatorType.ISAACGYM:
        import isaacgym

        assert isaacgym is not None

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


def setup_isaaclab_launcher(config: ExperimentConfig | RunSimConfig, device: str | None = None) -> Any | None:
    """Handle IsaacSim-specific launcher setup.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator and training settings.
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

    # Set values from config — divide by world_size for multi-GPU so each rank's
    # AppLauncher only allocates resources for its share of environments.
    # (The full num_envs is divided again in train_agent.train(), but AppLauncher
    # needs the per-rank count at init time to avoid over-allocating GPU memory.)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args_cli.num_envs = config.training.num_envs // world_size if world_size > 1 else config.training.num_envs
    args_cli.seed = config.training.seed
    args_cli.env_spacing = config.scene.env_spacing
    args_cli.output_dir = config.logger.base_dir
    args_cli.headless = config.training.headless
    if world_size > 1:
        # Distribute simulator across GPUs when using multi-gpu training
        args_cli.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        args_cli.distributed = True
    elif device is not None:
        # Use the resolved device
        args_cli.device = device
    else:  # AppLauncher auto-detects
        pass

    # LiDAR ray queries do not need RTX; only video and TiledCamera sensors enable the renderer.
    video_enabled = config.logger.video.enabled or config.logger.headless_recording
    cameras_enabled = any(isinstance(sensor, CameraSensorConfig) for sensor in getattr(config, "sensor", {}).values())
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


def setup_simulation_environment(
    config: ExperimentConfig | RunSimConfig, device: str | None = None
) -> tuple[Any, str, Any]:
    """Setup simulation environment with shared infrastructure.

    This function handles common setup for training, evaluation and direct simulation:
    - Simulator imports and initialization
    - Device selection and seeding
    - Environment creation
    - Keyboard listener setup (if not headless)

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing all simulation settings.
    device : str | None, optional
        Device to use for simulation. If None, auto-detects CUDA availability.

    Returns
    -------
    tuple[Any, str, Any]
        Tuple of (environment, device_string, simulation_app).
        simulation_app is None for simulators that don't need it (MuJoCo, IsaacGym).
    """
    logger.info("🚀 Setting up simulation environment...")

    # Setup simulator imports
    setup_simulator_imports(config)

    # Device selection - must happen before IsaacSim launcher setup
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Handle IsaacSim launcher if needed (for both ExperimentConfig and RunSimConfig)
    simulation_app = None
    if get_simulator_type() == SimulatorType.ISAACSIM:
        simulation_app = setup_isaaclab_launcher(config, device)

    try:
        env = _create_simulation_environment(config, device)
    except BaseException:
        try:
            close_simulation_resources(None, simulation_app)
        except Exception:
            with suppress(Exception):
                logger.exception("Simulation app cleanup failed while preserving the setup error")
        raise
    return env, device, simulation_app


def _create_simulation_environment(config: ExperimentConfig | RunSimConfig, device: str) -> Any:
    """Construct the configured environment after its provider app is available."""
    # Set random seed if specified (only for ExperimentConfig)
    if isinstance(config, ExperimentConfig) and config.training.seed is not None:
        seeding(config.training.seed, torch_deterministic=config.training.torch_deterministic)
        logger.info(f"Seed: {config.training.seed}")

    # For RunSimConfig, we need a different approach since it doesn't have env_class or training configs
    if isinstance(config, RunSimConfig):
        # For run_sim.py, we'll create the simulator directly instead of using environment wrapper
        logger.info("Direct simulation mode - creating simulator directly, without experiment config")

        # Cross-camera validation (Warp render-flag agreement) across the assembled camera dict.
        validate_camera_dict(
            {name: sensor for name, sensor in config.sensor.items() if isinstance(sensor, CameraSensorConfig)}
        )

        # Create FullSimConfig from RunSimConfig.
        full_config = FullSimConfig(
            simulator=config.simulator.config,
            robot=config.robot,
            scene=config.scene,
            # The CLI declares mounted sensors per key (key = sensor name).
            sensors=dict(config.sensor),
            training=config.training,
            logger=config.logger,
            plugin=config.plugin,
            experiment_dir=None,
        )

        # For compatibility, minimal proxy for TerrainManager since it depends on env.
        # Carries the requested num_envs through to the terrain manager.
        class EnvProxy:
            def __init__(self, device: str, num_envs: int) -> None:
                self.num_envs = num_envs
                self.device = device

        # For compatibility, wrap in a minimal object that has .sim attribute
        class DirectSimWrapper:
            def __init__(self, simulator: Any) -> None:
                self.sim = simulator

            def reset(self) -> None:
                # Basic reset - just initialize the simulator if needed
                if hasattr(self.sim, "reset"):
                    self.sim.reset()

            def close(self) -> None:
                if hasattr(self.sim, "close"):
                    self.sim.close()

        # Use terrain configuration from RunSimConfig
        terrain_manager = TerrainManager(config.terrain, env=EnvProxy(device, config.training.num_envs), device=device)

        # Create simulator using get_class() to avoid circular imports
        simulator_class = get_class(config.simulator._target_)
        simulator = simulator_class(full_config, terrain_manager, device)

        # Now we have an "env" to return which is actually the direct simulator
        env = DirectSimWrapper(simulator)
        logger.debug("Direct simulator created successfully!")

    else:
        # Original ExperimentConfig path
        env_target = config.env_class
        tyro_env_config = get_tyro_env_config(config)

        logger.info(f"Creating environment: {env_target}")
        env_class = get_class(env_target)
        env = env_class(tyro_env_config, device=device)

        logger.debug("Environment created successfully!")

        # Setup keyboard listener if not headless
        if not config.training.headless:
            setup_keyboard_listener(env)

    return env


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
        The simulation app instance returned by init_sim_imports().
        Can be None for simulators that don't have an app (e.g., IsaacGym).
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


@contextmanager
def simulation_resource_session(env: Any, simulation_app: Any) -> Iterator[Any]:
    """Close one environment/app pair while preserving an active body exception."""
    try:
        yield env
    except BaseException:
        try:
            close_simulation_resources(env, simulation_app)
        except Exception:
            with suppress(Exception):
                logger.exception("Simulation cleanup failed while preserving the active exception")
        raise
    else:
        close_simulation_resources(env, simulation_app)


class DirectSimulation:
    """Encapsulates direct simulation logic for run_sim.py.

    This class provides a clean interface for running direct simulations without
    training or evaluation environments, handling all initialization,
    loop management, and cleanup logic.

    Can be used as a context manager for resource management.

    Examples
    --------
    >>> with DirectSimulation(config, env, device, simulation_app) as sim:
    ...     sim.run()
    """

    def __init__(self, config: RunSimConfig, env: Any, device: str, simulation_app: Any):
        """Initialize DirectSimulation instance.

        Parameters
        ----------
        config : RunSimConfig
            Configuration containing all simulation settings.
        env : Any
            Environment wrapper containing the simulator.
        device : str
            Device for tensor operations.
        simulation_app : Any
            Simulation app instance (if any).
        """
        self.config = config
        self.env = env
        self.device = device
        self.simulation_app = simulation_app
        self.simulator = env.sim

    def __enter__(self) -> Self:
        """Context manager entry - initialize the simulation.

        Returns
        -------
        Self
            Self for use in the with statement.
        """
        try:
            self.initialize()
        except BaseException:
            try:
                self.cleanup()
            except Exception:
                with suppress(Exception):
                    logger.exception("Direct simulation cleanup failed after initialization error")
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - cleanup the simulation.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception instance if an exception occurred.
        exc_tb : traceback or None
            Traceback if an exception occurred.
        """
        try:
            self.cleanup()
        except Exception:
            if exc_val is None:
                raise
            with suppress(Exception):
                logger.exception("Direct simulation cleanup failed while preserving the active exception")

    def initialize(self) -> None:
        """Handle the complete simulator initialization sequence.

        Performs the initialization process required for proper simulator
        lifecycle management. Ideally this is moved into the simulator interface and
        to simplify training, evaluation and direct usage.
        """
        logger.debug("Initializing simulator...")

        # Headless lives in training config. Honoring it here is what keeps --training.headless True
        # from bringing up the viewer and pulling in isaacsim.util.debug_draw with it.
        self.simulator.set_headless(self.config.training.headless)

        # Step 1: Basic setup
        self.simulator.setup()
        logger.debug("simulator.setup() completed")

        # Step 2: Setup terrain
        self.simulator.setup_terrain()
        logger.debug("simulator.setup_terrain() completed")

        # Step 3: Load assets (this initializes the bridge!)
        self.simulator.load_assets()
        logger.debug("simulator.load_assets() completed - bridge should now be initialized")

        # Step 4: Direct sim2sim controls one environment at the origin.
        env_origins = torch.zeros(1, 3, device=self.device)

        # Create base_init_state from robot config
        base_init_state = self._create_base_init_state()

        self.simulator.create_envs(1, env_origins, base_init_state)
        logger.debug("simulator.create_envs() completed")

        # Step 5: Prepare simulation
        self.simulator.prepare_sim()
        logger.debug("simulator.prepare_sim() completed")

        # Step 6: Setup viewer if not headless
        if not self.config.training.headless:
            self.simulator.setup_viewer()
            logger.debug("simulator.setup_viewer() completed")

        self.simulator.install_plugins()

        # Step 6.5: Initialize episode (positions virtual gantry, starts lifecycle participants, etc.)
        self.simulator.hooks.emit(Phase.EPISODE_START, 0)
        logger.debug("simulator episode-start hooks completed")

        logger.info("Simulator initialized")

        # Step 7: Toggle start recording if enabled
        if self.simulator.video_recorder and self.simulator.video_recorder.enabled:
            # arbitrary episode ID given this is sim2sim, we may want to
            # actually support toggling recording and with better filenames too
            self.simulator.video_recorder.start_recording(episode_id=0)

    def run(self) -> None:
        """Run the direct simulation loop with viewer sync and performance logging.

        Manages the complete simulation loop including rate limiting,
        viewer synchronization, performance logging, and error handling.
        """
        # Setup rate limiting
        sim_frequency = self.config.simulator.config.sim.fps
        control_decimation = self.config.simulator.config.sim.control_decimation_steps
        control_frequency = sim_frequency / control_decimation
        time_scale = self.config.time_scale
        rate_limiter = RateLimiter(sim_frequency * time_scale) if time_scale > 0 else None

        # Calculate viewer sync frequency
        viewer_steps = self._calculate_viewer_steps()

        logger.info(
            f"Configured simulation rates: {sim_frequency} Hz physics "
            f"({1.0 / sim_frequency * 1000:.2f} ms/step), {control_frequency:g} Hz control "
            f"(every {control_decimation} physics steps)"
        )
        logger.info(f"Viewer rate: {1 / self.config.viewer_dt:.1f} Hz (sync every {viewer_steps} steps)")

        # Determine refresh strategy based on simulator type
        # IsaacGym/IsaacSim: need pre-step to refresh tensors to sync simulator state
        # MuJoCo: no pre-step refresh needed because we are NOT running an envs/tasks requiring
        #         those tensors e.g, _rigid_body_rot, _rigid_body_vel, etc.
        simulator_type = get_simulator_type()
        if simulator_type in [SimulatorType.ISAACGYM, SimulatorType.ISAACSIM]:
            refresh_tensors = self.simulator.refresh_sim_tensors
        else:
            refresh_tensors = lambda: None  # noqa: E731  (No-op for MuJoCo)

        logger.info("Starting direct simulation loop...")
        logger.info("Press Ctrl+C to stop simulation")

        # Direct simulation loop (like holosoma_inference's simulation_thread)
        step_count = 0
        control_step_count = 0
        window_physics_steps = 0
        window_control_steps = 0
        start_time = time.perf_counter()
        performance_window_start = start_time

        while True:
            try:
                # Direct simulator step with the same phase schedule BaseTask uses: one FRAME_BEGIN,
                # then every substep of the control step, then one FRAME_END.
                is_control_step = step_count % control_decimation == 0
                is_frame_end = step_count % control_decimation == control_decimation - 1

                # A plugin may end a finite run (e.g. motion playback reaching the end of its clip).
                # request_shutdown promises to stop after the current frame, so the flag is acted on
                # only at a frame boundary: a frame already under way runs its remaining substeps and
                # egresses, rather than losing its render and telemetry to a mid-frame exit.
                if is_control_step and self.simulator.shutdown_requested:
                    logger.info("Simulation loop exiting: simulator shutdown requested.")
                    break

                # Refresh tensors if needed (no-op for MuJoCo) so PRE_STEP writers see current state.
                refresh_tensors()

                if is_control_step:
                    self.simulator.hooks.emit(Phase.FRAME_BEGIN)
                self.simulator.hooks.emit(Phase.PRE_STEP)
                self.simulator.simulate_at_each_physics_step()
                self.simulator.hooks.emit(Phase.POST_STEP)

                # Mounted cameras render and egress consumers publish here, as FRAME_END plugins
                # (once per control step). render_sensors is registered before any consumer, so
                # buffers are fresh; both no-op when no cameras/consumers are configured.
                #
                # Closing on the control step's LAST substep is what makes the image a consumer reads
                # the one drawn for the state this step wrote: a backend draws when its substep counter
                # reaches render_interval. Every frame that opens closes, since plugins pair per-frame
                # work across the two phases; one that must not egress its last frame asks for shutdown
                # from FRAME_END (motion playback does).
                if is_frame_end:
                    # Resync first, as BaseTask._post_physics_step does, so state published beside the
                    # image describes the same substep. The refresh above predates this substep's step.
                    refresh_tensors()
                    self.simulator.hooks.emit(Phase.FRAME_END)

                # Headless backends render cameras/video on their own render schedule.
                if not self.config.training.headless and step_count % viewer_steps == 0:
                    self.simulator.render()

                step_count += 1
                window_physics_steps += 1
                if is_control_step:
                    control_step_count += 1
                    window_control_steps += 1

                if window_physics_steps == _PERFORMANCE_LOG_INTERVAL_STEPS:
                    performance_window_start = self._log_performance(
                        window_physics_steps,
                        window_control_steps,
                        performance_window_start,
                    )
                    window_physics_steps = 0
                    window_control_steps = 0

                if rate_limiter is not None:
                    rate_limiter.sleep()

            except KeyboardInterrupt:
                with suppress(Exception):
                    logger.info("Simulation interrupted by user (Ctrl+C)")
                raise
            except Exception:
                # Native cleanup reports the traceback before environment teardown.
                traceback_reported_by_cleanup = (
                    simulator_type == SimulatorType.ISAACSIM and self.simulation_app is not None
                )
                with suppress(Exception):
                    logger.opt(exception=not traceback_reported_by_cleanup).error(
                        f"Error during simulation step {step_count}"
                    )
                raise

        # Final statistics
        total_elapsed = time.perf_counter() - start_time
        logger.info(f"Simulation completed after {step_count} physics steps and {control_step_count} control steps")
        logger.info(
            _format_simulation_performance(
                step_count,
                control_step_count,
                1.0 / sim_frequency,
                total_elapsed,
                prefix="Average simulation performance",
            )
        )

    def cleanup(self) -> None:
        """Handle simulation cleanup."""
        close_simulation_resources(self.env, self.simulation_app)

    def _create_base_init_state(self) -> torch.Tensor:
        """Create base initialization state tensor from robot configuration.

        Returns
        -------
        torch.Tensor
            Base initialization state tensor.
        """
        base_init_state_list = (
            self.config.robot.init_state.pos
            + self.config.robot.init_state.rot
            + self.config.robot.init_state.lin_vel
            + self.config.robot.init_state.ang_vel
        )
        return to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _calculate_viewer_steps(self) -> int:
        """Calculate viewer synchronization frequency.

        Returns
        -------
        int
            Number of simulation steps between viewer updates.
        """
        viewer_dt = self.config.viewer_dt
        sim_dt = 1.0 / self.config.simulator.config.sim.fps
        return max(1, int(viewer_dt / sim_dt))

    def _log_performance(
        self,
        physics_steps: int,
        control_steps: int,
        window_start_time: float,
    ) -> float:
        """Log simulation performance over the latest measurement window.

        Parameters
        ----------
        physics_steps : int
            Number of physics steps completed in the measurement window.
        control_steps : int
            Number of control steps completed in the measurement window.
        window_start_time : float
            Monotonic start time for the measurement window.

        Returns
        -------
        float
            Monotonic start time for the next measurement window.
        """
        now = time.perf_counter()
        logger.info(
            _format_simulation_performance(
                physics_steps,
                control_steps,
                1.0 / self.config.simulator.config.sim.fps,
                now - window_start_time,
            )
        )
        return now
