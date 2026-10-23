"""Isaac Lab 3.0 runtime lifecycle and independent rendering/viewer selection."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any

KIT_VISUALIZER = "kit"
_active_runtime: ContextVar[Any] = ContextVar("holosoma_lab3_runtime", default=None)


def apply_launcher_args(
    args: argparse.Namespace, *, headless: bool, needs_rendering: bool, visualizers: Sequence[str] = ()
) -> None:
    """Request offscreen rendering without turning it into a viewer request."""
    args.visualizer = list(dict.fromkeys(visualizers)) if visualizers else ([] if headless else [KIT_VISUALIZER])
    args.enable_cameras = needs_rendering


def active_simulation_cfg() -> Any:
    runtime = _active_runtime.get()
    if runtime is None:
        raise RuntimeError("Launch Isaac Lab with setup_simulation_environment before constructing IsaacSim.")
    return runtime.cfg


class LaunchedSimulation:
    """Keep launch_simulation's context alive until Holosoma closes its simulator."""

    def __init__(self, context: Any, cfg: Any):
        self._context = context
        self.cfg = cfg
        self._closed = False
        context.__enter__()
        # Callers may discard the returned app handle after creating the simulator.
        # Retain the context owner until explicit shutdown; otherwise generator GC
        # closes Kit in the middle of a rollout, sometimes with exit status zero.
        self._token = _active_runtime.set(self)

    def is_running(self) -> bool:
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        return not self._closed and (sim is None or sim.is_running())

    def close_with_failure(self, failure: BaseException | None) -> None:
        if self._closed:
            return
        self._closed = True
        _active_runtime.reset(self._token)
        self._context.__exit__(type(failure) if failure else None, failure, failure.__traceback__ if failure else None)

    def close(self, **kwargs: Any) -> None:
        self.close_with_failure(sys.exc_info()[1])


def launch(config: Any, device: str | None = None) -> LaunchedSimulation:
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.sim import SimulationCfg, RigidBodyMaterialCfg
    from isaaclab.sensors import CameraCfg
    from .physics_backend import build_physics_cfg, use_newton_actuators_for
    from holosoma.config_types.sensor import CameraSensorConfig

    parser = argparse.ArgumentParser(description="Holosoma on Isaac Lab 3.0")
    add_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    # Holosoma's USD/URDF import and custom USD spawners use Kit extension APIs.
    # Kit can run without a visualizer for both PhysX and Newton.
    args.require_kit = True
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if world_size > 1 else (device or "cuda:0")
    args.distributed = world_size > 1
    needs_rendering = (
        config.logger.video.enabled
        or config.logger.headless_recording
        or any(isinstance(sensor, CameraSensorConfig) for sensor in config.sensor.values())
    )
    apply_launcher_args(
        args,
        headless=config.training.headless,
        needs_rendering=needs_rendering,
        visualizers=config.simulator.config.isaac_visualizers,
    )
    for name, value in config.simulator.config.isaacsim.app_launcher_args().items():
        if not hasattr(args, name):
            raise ValueError(f"Isaac Lab 3.0 does not accept launcher setting {name!r}.")
        setattr(args, name, value)
    init = config.simulator.config
    sim_cfg = SimulationCfg(
        dt=1.0 / init.sim.fps,
        render_interval=init.sim.render_interval_steps,
        device=args.device,
        physics=build_physics_cfg(init),
        use_newton_actuators=use_newton_actuators_for(init),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    # The scan only needs the renderer declaration here. Actual camera mounts are resolved
    # from the authored robot/object scene, after launch.
    tree = SimpleNamespace(
        sim=sim_cfg, camera=CameraCfg(prim_path="/World/holosomaCamera") if needs_rendering else None
    )
    launched = LaunchedSimulation(launch_simulation(tree, args), sim_cfg)
    sys.argv = [sys.argv[0], *unknown]
    return launched
