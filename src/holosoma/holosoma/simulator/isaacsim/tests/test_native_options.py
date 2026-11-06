"""Native choices are typed, independent and reject unsupported camera projections."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from holosoma.config_types.simulator import IsaacLab3Config, IsaacLab3NewtonConfig
from holosoma.simulator.isaacsim.backends.isaaclab3 import IsaacLab3Backend

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("field", ["nconmax", "njmax", "solver_iterations"])
@pytest.mark.parametrize("value", [0, -1])
def test_newton_rejects_nonpositive_solver_capacity(field: str, value: int) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValidationError, match=f"{field} must be positive"):
        IsaacLab3NewtonConfig(**kwargs)


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_newton_rejects_invalid_default_shape_margin(value: float) -> None:
    with pytest.raises(ValidationError, match="default_shape_margin must be finite and nonnegative"):
        IsaacLab3NewtonConfig(default_shape_margin=value)


def test_warp_rejects_usd_fisheye_before_constructing_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    sensors = ModuleType("isaaclab.sensors")
    sensors.CameraCfg = SimpleNamespace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    backend = IsaacLab3Backend(IsaacLab3Config(renderer="newton_warp"))
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

    backend = IsaacLab3Backend(IsaacLab3Config(renderer=renderer))
    source = torch.tensor([0.0, 0.89, 2.5, float("inf")])
    actual = backend.camera_depth(SimpleNamespace(torch=source))
    expected = torch.tensor([float("inf") if renderer == "newton_warp" else 0.0, 0.89, 2.5, float("inf")])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, torch.tensor([0.0, 0.89, 2.5, float("inf")]), rtol=0, atol=0)
