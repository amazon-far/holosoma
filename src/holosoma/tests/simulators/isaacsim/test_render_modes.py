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
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from holosoma.config_types.env import get_run_sim_env_config
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import IsaacLabBackend, IsaacSimRenderConfig
from holosoma.config_values import run_sim as run_sim_values
from holosoma.utils.sim_utils import setup_isaaclab_launcher

_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


def _has_isaacsim() -> bool:
    if importlib.util.find_spec("isaaclab") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _run_config(
    render_config: IsaacSimRenderConfig, backend: IsaacLabBackend = IsaacLabBackend.ISAACLAB2
) -> RunSimConfig:
    simulator = dataclasses.replace(
        run_sim_values.isaacsim,
        config=dataclasses.replace(run_sim_values.isaacsim.config, isaacsim=render_config, isaaclab_backend=backend),
    )
    return RunSimConfig(simulator=simulator)


@pytest.mark.no_sim
@pytest.mark.parametrize("backend", list(IsaacLabBackend))
@pytest.mark.parametrize("headless_requires_kit", [False, True], ids=["viewer", "explicit-kit"])
@pytest.mark.parametrize(
    ("render_config", "expected_rendering_mode", "expected_kit_args"),
    [
        (IsaacSimRenderConfig(), "upstream-render-default", "upstream-kit-default"),
        (IsaacSimRenderConfig(rendering_mode="performance"), "performance", "upstream-kit-default"),
        (IsaacSimRenderConfig(rendering_mode="balanced"), "balanced", "upstream-kit-default"),
        (
            IsaacSimRenderConfig(
                rendering_mode="quality",
                kit_args="--/holosoma/tests/launcherKitArg=cpu-boundary",
            ),
            "quality",
            "--/holosoma/tests/launcherKitArg=cpu-boundary",
        ),
        (
            IsaacSimRenderConfig(rendering_mode="quality", kit_args="--/rtx/post/dlss/execMode=0"),
            "quality",
            "--/rtx/post/dlss/execMode=0",
        ),
    ],
)
def test_launcher_config_reaches_app_launcher_namespace(
    monkeypatch: pytest.MonkeyPatch,
    render_config: IsaacSimRenderConfig,
    expected_rendering_mode: str,
    expected_kit_args: str,
    backend: IsaacLabBackend,
    headless_requires_kit: bool,
) -> None:
    """Exercise Holosoma's real launcher setup with Isaac Lab replaced only at its import boundary."""
    monkeypatch.setenv(
        "HOLOSOMA_ISAACSIM_BACKEND", "isaaclab2" if backend == IsaacLabBackend.ISAACLAB3 else "isaaclab3"
    )
    captured_args: dict[str, Any] = {}
    captured_config: list[Any] = []
    app_sentinel = object()

    @dataclasses.dataclass
    class FakeIsaacRtxRendererCfg:
        pass

    @dataclasses.dataclass
    class FakeNewtonWarpRendererCfg:
        pass

    @dataclasses.dataclass
    class FakePhysxCfg:
        bounce_threshold_velocity: float = 0.5
        solver_type: int = 1
        max_position_iteration_count: int = 255
        max_velocity_iteration_count: int = 255
        gpu_max_rigid_patch_count: int = 10 * 2**15

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            if backend == IsaacLabBackend.ISAACLAB2:
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
    physx_module = ModuleType("isaaclab_physx")
    monkeypatch.setitem(sys.modules, "isaaclab_physx", physx_module)
    closed_statuses = []
    if backend == IsaacLabBackend.ISAACLAB3:
        physx_module.__spec__ = ModuleSpec("isaaclab_physx", loader=None)
        physics_module = ModuleType("isaaclab_physx.physics")
        physics_module.__dict__["PhysxCfg"] = FakePhysxCfg
        monkeypatch.setitem(sys.modules, "isaaclab_physx.physics", physics_module)
        for module_name, class_name, cfg_type in [
            ("isaaclab_physx.renderers", "IsaacRtxRendererCfg", FakeIsaacRtxRendererCfg),
            ("isaaclab_newton.renderers", "NewtonWarpRendererCfg", FakeNewtonWarpRendererCfg),
        ]:
            module = ModuleType(module_name)
            module.__dict__[class_name] = cfg_type
            monkeypatch.setitem(sys.modules, module_name, module)
        isaaclab_app_module.__dict__["add_launcher_args"] = FakeAppLauncher.add_app_launcher_args

        @contextmanager
        def launch_simulation(cfg, args):
            captured_args.update(args)
            captured_config.append(cfg)
            try:
                yield cfg
            except SystemExit as error:
                closed_statuses.append(error.code)

        isaaclab_app_module.__dict__["launch_simulation"] = launch_simulation
        kit_app = ModuleType("omni.kit.app")
        kit_app.__dict__["get_app"] = lambda: app_sentinel
        kit_module = ModuleType("omni.kit")
        kit_module.__dict__["app"] = kit_app
        omni_module = ModuleType("omni")
        omni_module.__dict__["kit"] = kit_module
        monkeypatch.setitem(sys.modules, "omni", omni_module)
        monkeypatch.setitem(sys.modules, "omni.kit", kit_module)
        monkeypatch.setitem(sys.modules, "omni.kit.app", kit_app)
    monkeypatch.setattr(sys, "argv", ["test-render-modes"])
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    config = _run_config(render_config, backend)
    sim_cfg = config.simulator.config
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, headless=headless_requires_kit),
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                sim_cfg,
                sim=dataclasses.replace(
                    sim_cfg.sim, isaaclab3=dataclasses.replace(sim_cfg.sim.isaaclab3, require_kit=headless_requires_kit)
                ),
            ),
        ),
    )
    app = setup_isaaclab_launcher(get_run_sim_env_config(config), "cuda:0")
    assert app is not None
    from holosoma.simulator.isaacsim.backends import get_backend

    assert get_backend(backend).isaaclab_major == (3 if backend == IsaacLabBackend.ISAACLAB3 else 2)

    if backend == IsaacLabBackend.ISAACLAB3:
        assert app._app is app_sentinel
        assert len(captured_config) == 1
        # A viewer or explicit integration requirement needs Kit without
        # initializing an unused camera renderer.
        assert captured_config[0].renderer is None
        assert vars(captured_config[0].physics) == {
            "bounce_threshold_velocity": config.simulator.config.sim.physx.bounce_threshold_velocity,
            "solver_type": config.simulator.config.sim.physx.solver_type,
            "max_position_iteration_count": config.simulator.config.sim.physx.num_position_iterations,
            "max_velocity_iteration_count": config.simulator.config.sim.physx.num_velocity_iterations,
            "gpu_max_rigid_patch_count": 10 * 2**15,
        }
        assert captured_args["require_kit"] is True
        assert captured_args.get("visualizer") == (None if headless_requires_kit else "kit")
        app.close(exit_code=1)
        assert closed_statuses == [1]
        assert "rendering_mode" not in captured_args
        native_kit_args = captured_args["kit_args"].split()
        assert native_kit_args[-2:] == [
            expected_kit_args,
            "--/exts/omni.kit.telemetry/skipDeferredStartup=true",
        ]
        if render_config.rendering_mode is None:
            assert len(native_kit_args) == 2
        else:
            expected_dlss, expected_quality = {
                "performance": (0, "false"),
                "balanced": (1, "false"),
                "quality": (2, "true"),
            }[render_config.rendering_mode]
            assert f"--/rtx/post/dlss/execMode={expected_dlss}" in native_kit_args
            assert f"--/rtx/reflections/enabled={expected_quality}" in native_kit_args
            assert f"--/rtx/indirectDiffuse/enabled={expected_quality}" in native_kit_args
            if expected_kit_args == "--/rtx/post/dlss/execMode=0":
                dlss_args = [arg for arg in native_kit_args if arg.startswith("--/rtx/post/dlss/execMode=")]
                assert dlss_args == ["--/rtx/post/dlss/execMode=2", "--/rtx/post/dlss/execMode=0"]
    else:
        assert app is app_sentinel
        assert captured_args["rendering_mode"] == expected_rendering_mode
        assert captured_args["kit_args"] == expected_kit_args


@pytest.mark.no_sim
def test_launcher_config_rejects_unknown_rendering_mode() -> None:
    with pytest.raises(ValidationError):
        IsaacSimRenderConfig(rendering_mode="cinematic")  # type: ignore[arg-type]


@pytest.mark.isaacsim
@pytest.mark.skipif(not _has_isaacsim(), reason="IsaacSim requires Isaac Lab and a CUDA device")
def test_render_headless(tmp_path, run_isaacsim_harness):
    """Step and render a free-box scene without a display."""
    result_file = tmp_path / "render_headless.txt"
    run_isaacsim_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--scene",
        "g1-largebox",
        "--headless",
        "true",
        "--result-file",
        str(result_file),
        label="isaacsim render (headless=true)",
        timeout=900,
        result_file=result_file,
    )
