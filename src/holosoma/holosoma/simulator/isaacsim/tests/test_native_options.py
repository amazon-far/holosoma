"""Native choices are typed, independent and reject unsupported camera projections."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from holosoma.config_types.simulator import IsaacSim6RendererConfig
from holosoma.config_types.simulator import IsaacSim6Config, IsaacSim6NewtonConfig
from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("enabled", [False, True])
def test_ovrtx_rejects_ineffective_shadow_control_before_importing_sdk(enabled: bool) -> None:
    before = set(sys.modules)
    backend = IsaacSim6Backend(
        IsaacSim6Config(renderer="ovrtx", renderer_options=IsaacSim6RendererConfig(enable_shadows=enabled))
    )
    with pytest.raises(ValueError, match="renderer_options.enable_shadows is Warp-only"):
        backend.renderer_config()
    assert not any(name.startswith("isaaclab_ov.") for name in set(sys.modules) - before)


@pytest.mark.parametrize("field", ["nconmax", "njmax", "solver_iterations"])
def test_newton_rejects_duplicate_shared_solver_options(field: str) -> None:
    kwargs: dict[str, Any] = {field: 128}
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        IsaacSim6NewtonConfig(**kwargs)


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_newton_rejects_invalid_default_shape_margin(value: float) -> None:
    with pytest.raises(ValidationError, match="default_shape_margin must be finite and nonnegative"):
        IsaacSim6NewtonConfig(default_shape_margin=value)


def test_warp_rejects_usd_fisheye_before_constructing_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    sensors = ModuleType("isaaclab.sensors")
    sensors.CameraCfg = SimpleNamespace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    backend = IsaacSim6Backend(IsaacSim6Config(renderer="newton_warp"))
    with pytest.raises(ValueError, match="USD 'fisheyePolynomial'"):
        backend.create_camera_cfg(
            offset_position=(0.0, 0.0, 0.0),
            offset_orientation=(1.0, 0.0, 0.0, 0.0),
            spawn=SimpleNamespace(projection_type="fisheyePolynomial"),
        )


@pytest.mark.parametrize("renderer", ["isaac_rtx", "newton_warp"])
def test_depth_misses_preserve_shared_clipping_semantics(
    renderer: Literal["isaac_rtx", "newton_warp"],
) -> None:
    import torch

    backend = IsaacSim6Backend(IsaacSim6Config(renderer=renderer))
    source = torch.tensor([0.0, 0.89, 2.5, float("inf")])
    actual = backend.camera_depth(SimpleNamespace(torch=source))
    expected = torch.tensor([float("inf") if renderer == "newton_warp" else 0.0, 0.89, 2.5, float("inf")])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, torch.tensor([0.0, 0.89, 2.5, float("inf")]), rtol=0, atol=0)


@pytest.mark.parametrize("options", [{"rendering_mode": "quality"}, {"kit_args": "--/rtx/shadows/enabled=false"}])
def test_ovrtx_rejects_kit_options_before_starting_runtime(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    app = ModuleType("isaaclab.app")
    app.launch_simulation = lambda *_args: pytest.fail("invalid OVRTX config started a runtime")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    backend = IsaacSim6Backend(IsaacSim6Config(renderer="ovrtx"))
    with pytest.raises(ValueError, match="This run is Kit-less"):
        backend.launch_app({"headless": True, **options})
