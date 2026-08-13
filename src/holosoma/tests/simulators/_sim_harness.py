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
from typing import Any

import tyro

from holosoma.config_types.run_sim import RunSimConfig


def steps_for_seconds(sim: Any, seconds: float) -> int:
    """Number of physics steps that advance ``seconds`` of SIM-TIME on this backend.

    Every backend advances exactly one ``sim.sim_dt`` physics step per
    ``simulate_at_each_physics_step`` / ``backend.step()`` call, so a wall of sim-time is a simple
    per-backend step count — one ``SECONDS`` constant means the same physical duration on all four
    backends instead of a step count tuned to the fastest one.
    """
    return max(1, math.ceil(seconds / sim.sim_dt))


def step(sim: Any, n: int, *, render: bool = False) -> None:
    """Advance ``n`` physics steps. With ``render=True`` also drive the render path each step.

    Every harness sim exposes ``simulate_at_each_physics_step``. ``render`` is on for the recording
    harness (so a video captures the motion) and off for the headless physics-only checks.
    """
    for _ in range(n):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
        if render:
            sim.render()


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
