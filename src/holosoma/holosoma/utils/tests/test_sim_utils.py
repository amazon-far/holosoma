"""Unit tests for the shared simulation utilities (pure, no simulator backend)."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Any

import pytest

from holosoma.config_types.env import get_run_sim_env_config
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.config_types.simulator import IsaacLab3Config
from holosoma.config_values.run_sim import isaaclab3
from holosoma.simulator.isaacsim.backends.isaaclab3 import IsaacLab3Backend
from holosoma.utils import sim_utils
from holosoma.utils.sim_utils import format_simulation_performance


@pytest.mark.parametrize("camera_enabled", [False, True])
def test_launcher_uses_task_sensors_resolved_device_and_per_process_env_count(
    monkeypatch: pytest.MonkeyPatch, camera_enabled: bool
) -> None:
    """Replace SDK argument/launch boundaries while exercising real task-config conversion."""
    launched: list[tuple[argparse.Namespace, bool]] = []
    activated: list[bool] = []
    app = object()

    def add_args(_backend: IsaacLab3Backend, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--enable_cameras", action="store_true")
        parser.add_argument("--distributed", action="store_true")

    def launch(backend: IsaacLab3Backend, args: argparse.Namespace, *, sim_config: Any) -> object:
        launched.append((args, backend._rendering))
        return app

    monkeypatch.setattr(IsaacLab3Backend, "add_launcher_args", add_args)
    monkeypatch.setattr(IsaacLab3Backend, "launch_app", launch)
    monkeypatch.setattr(sim_utils, "_ensure_render_pipeline_active", lambda *, isaaclab3: activated.append(isaaclab3))
    monkeypatch.setattr(sys, "argv", ["test-launcher"])
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = RunSimConfig(
        simulator=dataclasses.replace(
            isaaclab3,
            config=dataclasses.replace(
                isaaclab3.config,
                sim=dataclasses.replace(
                    isaaclab3.config.sim, isaaclab3=IsaacLab3Config(physics_backend="newton", renderer="ovrtx")
                ),
            ),
        ),
        sensor={"front": CameraSensorConfig(mount=SensorMountConfig(target_kind="world"))} if camera_enabled else {},
    )
    config = dataclasses.replace(config, training=dataclasses.replace(config.training, num_envs=8, headless=True))
    assert sim_utils.setup_isaaclab_launcher(get_run_sim_env_config(config), "cuda:1") is app
    args, rendering = launched[0]
    assert args.device == "cuda:1"
    assert args.num_envs == 8
    assert args.distributed
    assert args.enable_cameras is camera_enabled
    assert rendering is camera_enabled
    assert activated == ([True] if camera_enabled else [])


def test_format_simulation_performance_names_rates_and_time_ratio() -> None:
    message = format_simulation_performance(
        physics_steps=1000,
        control_steps=250,
        sim_dt=0.005,
        wall_time=2.5,
    )

    assert message == (
        "Simulation performance: 2.00x real time (5.00 sim s / 2.50 wall s) | "
        "400.0 physics steps/s | 100.0 control steps/s"
    )


def test_format_simulation_performance_handles_empty_measurement() -> None:
    assert (
        format_simulation_performance(
            physics_steps=0,
            control_steps=0,
            sim_dt=0.005,
            wall_time=0.0,
        )
        == "Simulation performance: unavailable (no wall time elapsed)"
    )
