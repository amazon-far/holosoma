"""Shared boot/step helpers for the standalone cross-backend assertion harnesses.

``behavior_assert`` / ``static_move_assert`` / ``subset_write_assert`` / ``scene_spawn_assert`` each
build a sim, step it, and check physical outcomes. The truly common pieces — the sim-time -> step
count conversion, the per-step advance (+ optional render), and the bridge/gantry-disabled
``RunSimConfig`` build — live here so they are defined once. The scenario-specific ``main`` flows
(which scenes, which asserts) stay in each harness; only the duplicated plumbing is shared.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NoReturn

import tyro

from holosoma.config_types.run_sim import RunSimConfig


def run_and_hard_exit(main: Callable[[], int]) -> NoReturn:
    """Run a harness ``main`` and terminate the process immediately with its return code.

    Standalone IsaacSim harnesses MUST end this way instead of ``sys.exit(main())``. On IsaacSim,
    the interpreter's normal atexit teardown deadlocks in ``carbOnPluginShutdown`` while unwinding
    the ``omni.syntheticdata`` / OmniGraph render-product graph a TiledCamera creates, so the
    process never exits — the parent's ``subprocess`` timeout then SIGKILLs it, turning a PASS
    (verdict already written to ``--result-file``) into a spurious timeout failure. Flushing then
    ``os._exit`` skips the atexit teardown entirely; rendering itself is fine, only clean shutdown
    hangs. (behavior_assert / scene_spawn_assert use the same escape for the MuJoCo-viewer race.)

    Use in a harness entrypoint::

        if __name__ == "__main__":
            run_and_hard_exit(main)
    """
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def steps_for_seconds(sim: Any, seconds: float) -> int:
    """Number of physics steps that advance ``seconds`` of SIM-TIME on this backend.

    Every backend advances exactly one ``sim.sim_dt`` physics step per
    ``simulate_at_each_physics_step`` / ``backend.step()`` call, so a wall of sim-time is a simple
    per-backend step count — one ``SECONDS`` constant means the same physical duration on all four
    backends instead of a step count tuned to the fastest one.
    """
    return max(1, math.ceil(seconds / sim.sim_dt))


def step(sim: Any, n: int, *, render: bool = False) -> None:
    """Advance ``n`` physics steps, rendering each step when ``render`` is set or the backend needs it.

    ``render`` is on for the recording harness (so a video captures the motion). Backends whose
    offscreen sensors (e.g. IsaacSim RTX cameras) only update when rendered always render.
    """
    for _ in range(n):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
        if render or sim.requires_step_render():
            sim.render()


def launch_simulator(config: RunSimConfig, device: str | None = None) -> tuple[Any, str, Any]:
    """Construct the configured simulator without running its setup lifecycle.

    Harnesses drive ``setup -> setup_terrain -> load_assets -> create_envs -> prepare_sim``
    themselves so they can act between stages; tasks use ``RunSimConfig.create_task()`` instead.

    Returns
    -------
    tuple
        ``(simulator, device, simulation_app)``; ``simulation_app`` is None outside IsaacSim.
    """
    from holosoma.config_types.env import get_run_sim_env_config
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.managers.terrain.manager import TerrainManager
    from holosoma.utils.helpers import get_class
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.sim_utils import setup_isaaclab_launcher, setup_simulator_imports
    from holosoma.utils.simulator_config import SimulatorType, get_simulator_type

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_config = get_run_sim_env_config(config)
    setup_simulator_imports(env_config)
    simulation_app = None
    if get_simulator_type() == SimulatorType.ISAACSIM:
        simulation_app = setup_isaaclab_launcher(env_config, device)

    full_config = FullSimConfig(
        simulator=config.simulator.config,
        robot=config.robot,
        scene=config.scene,
        sensors=dict(config.sensor),
        training=config.training,
        logger=config.logger,
        plugin=config.plugin,
    )
    # Without a task, TerrainManager construction needs only num_envs and device from its env.
    terrain_env = SimpleNamespace(num_envs=config.training.num_envs, device=device)
    terrain_manager = TerrainManager(config.terrain, env=terrain_env, device=device)
    simulator = get_class(config.simulator._target_)(full_config, terrain_manager, device)
    return simulator, device, simulation_app


def build_run_sim_config(
    simulator: str,
    scene: str,
    robot: str,
    terrain: str,
    *,
    sensors: dict[str, Any] | None = None,
    record_dir: str | None = None,
    video_camera: Any = None,
    show_command_overlay: bool = True,
) -> RunSimConfig:
    """Parse a ``RunSimConfig`` as the CLI would, with the bridge + virtual gantry disabled.

    The bridge binds a ZMQ clock publisher that lingers between rapid subprocess runs, and the gantry
    applies a robot-only force tensor that errors once a scene adds bodies — neither is relevant to a
    spawn/behavior check, so both are turned off.

    ``sensors`` is a mounted-camera dict (``{name: CameraSensorConfig}``) injected directly onto the
    parsed config's ``sensor`` field — the harness builds rigs in Python rather than through the
    per-key ``--sensor`` CLI (a multi-camera rig can't be named by one token). When ``record_dir`` is
    set, enable video recording every episode to that dir (mp4, no wandb). ``video_camera`` overrides
    the camera config (e.g. an object-framing camera); ``show_command_overlay`` toggles the
    robot-command text overlay (off for object scenarios where it is irrelevant).
    """
    from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg

    argv = [f"simulator:{simulator}", f"robot:{robot}", f"terrain:{terrain}", f"scene:{scene}"]
    config = tyro.cli(RunSimConfig, args=argv)
    if sensors is not None:
        config = dataclasses.replace(config, sensor=dict(sensors))
    sim_cfg = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(
            config.simulator.config,
            bridge=BridgeConfig(enabled=False),
            virtual_gantry=VirtualGantryCfg(enabled=False),
        ),
    )
    config = dataclasses.replace(config, simulator=sim_cfg)

    if record_dir is not None:
        video_kwargs: dict[str, Any] = {
            "enabled": True,
            "interval": 1,
            "save_dir": record_dir,
            "output_format": "mp4",
            "upload_to_wandb": False,
            "show_command_overlay": show_command_overlay,
        }
        if video_camera is not None:
            video_kwargs["camera"] = video_camera
        video = dataclasses.replace(config.logger.video, **video_kwargs)
        config = dataclasses.replace(config, logger=dataclasses.replace(config.logger, video=video))
    return config
