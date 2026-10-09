"""IsaacSim headless-render contract: the run loop must step and render without a display.

The scene_spawn_assert.py harness drives render() every step, as the real run_sim loop does.
IsaacSim's render() goes through self.sim.render() and never touches self.viewer, so this test
locks in headless rendering without requiring a display.

The launcher-configuration tests are CPU-only and replace Isaac Lab only at the external
``AppLauncher`` import boundary. The live render cases are marked ``isaacsim`` so the IsaacSim
CI job collects them; each sim runs in a subprocess because SimulationContext is a process
singleton.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import IsaacSimRenderConfig
from holosoma.config_values import run_sim as run_sim_values
from holosoma.utils.sim_utils import setup_isaaclab_launcher
from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


def _has_isaacsim() -> bool:
    if importlib.util.find_spec("isaaclab") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _run_config(render_config: IsaacSimRenderConfig) -> RunSimConfig:
    simulator = dataclasses.replace(
        run_sim_values.isaacsim,
        config=dataclasses.replace(run_sim_values.isaacsim.config, isaacsim=render_config),
    )
    return RunSimConfig(simulator=simulator)


@pytest.mark.no_sim
@pytest.mark.parametrize(
    ("render_config", "expected_rendering_mode", "expected_kit_args"),
    [
        (IsaacSimRenderConfig(), "upstream-render-default", "upstream-kit-default"),
        (
            IsaacSimRenderConfig(
                rendering_mode="quality",
                kit_args="--/holosoma/tests/launcherKitArg=cpu-boundary",
            ),
            "quality",
            "--/holosoma/tests/launcherKitArg=cpu-boundary",
        ),
    ],
)
def test_launcher_config_reaches_app_launcher_namespace(
    monkeypatch: pytest.MonkeyPatch,
    render_config: IsaacSimRenderConfig,
    expected_rendering_mode: str,
    expected_kit_args: str,
) -> None:
    """Exercise Holosoma's real launcher setup with Isaac Lab replaced only at its import boundary."""
    captured_args: dict[str, Any] = {}
    app_sentinel = object()

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--rendering_mode", default="upstream-render-default")
            parser.add_argument("--kit_args", default="upstream-kit-default")

        def __init__(self, args) -> None:
            captured_args.update(vars(args))
            self.app = app_sentinel

    isaaclab_module = ModuleType("isaaclab")
    isaaclab_app_module = ModuleType("isaaclab.app")
    isaaclab_app_module.__dict__["AppLauncher"] = FakeAppLauncher
    isaaclab_module.__dict__["app"] = isaaclab_app_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app_module)
    monkeypatch.setattr(sys, "argv", ["test-render-modes"])
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    app = setup_isaaclab_launcher(_run_config(render_config))

    assert app is app_sentinel
    assert captured_args["rendering_mode"] == expected_rendering_mode
    assert captured_args["kit_args"] == expected_kit_args


@pytest.mark.no_sim
def test_launcher_config_rejects_unknown_rendering_mode() -> None:
    with pytest.raises(ValidationError):
        IsaacSimRenderConfig(rendering_mode="cinematic")  # type: ignore[arg-type]


@pytest.mark.isaacsim
@pytest.mark.skipif(not _has_isaacsim(), reason="IsaacSim requires Isaac Lab and a CUDA device")
def test_render_headless():
    """Step and render a free-box scene without a display."""
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--scene",
        "g1-largebox",
        "--headless",
        "true",
        label="isaacsim render (headless=true)",
        timeout=900,
    )
