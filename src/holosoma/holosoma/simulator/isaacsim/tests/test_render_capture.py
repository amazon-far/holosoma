"""Isaac renders applied state with physics held stopped, without Kit."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from holosoma.simulator.isaacsim.render_capture import RenderCapture

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim

pytestmark = pytest.mark.no_sim


@pytest.fixture
def replica(monkeypatch: pytest.MonkeyPatch) -> tuple[SimpleNamespace, list[object], list[bool]]:
    calls: list[object] = []
    flags: list[bool] = []
    settings = SimpleNamespace(set_bool=lambda _key, value: flags.append(value))
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    @contextmanager
    def mutation() -> Iterator[None]:
        calls.append("flush")
        try:
            yield
        finally:
            calls.append("stale")

    simulator = SimpleNamespace(
        sim_device="cuda:0",
        sim_dt=0.005,
        tiled_cameras={
            "camera": SimpleNamespace(
                cfg=SimpleNamespace(update_latest_camera_pose=False),
                update=lambda _dt, force_recompute: None,
                data=SimpleNamespace(output={}),
            )
        },
        sensor_manager=SimpleNamespace(lidars=[]),
        sim=SimpleNamespace(current_time=0.01, current_time_step_index=2),
        state=SimpleNamespace(mutation=mutation),
        scene=SimpleNamespace(update=lambda dt: calls.append(("refresh", dt))),
        write_state_updates=lambda: calls.append("write"),
        render=lambda: calls.append("render"),
    )
    return simulator, calls, flags


def _prepared(simulator: SimpleNamespace, calls: list[object]) -> RenderCapture:
    capture = RenderCapture(cast("IsaacSim", simulator))
    capture.initialize_cameras()
    del calls[:]
    return capture


def test_render_writes_renders_and_refreshes_inside_one_mutation(
    replica: tuple[SimpleNamespace, list[object], list[bool]],
) -> None:
    simulator, calls, flags = replica
    capture = _prepared(simulator, calls)
    capture.render()
    assert calls == ["flush", "write", "render", ("refresh", 0.005), "stale"]
    assert flags and not any(flags)


def test_render_failure_leaves_physics_stopped(replica: tuple[SimpleNamespace, list[object], list[bool]]) -> None:
    simulator, calls, flags = replica
    capture = _prepared(simulator, calls)

    def fail() -> None:
        raise RuntimeError("render failed")

    simulator.render = fail
    with pytest.raises(RuntimeError, match="render failed"):
        capture.render()
    assert flags[-1] is False and calls[-1] == "stale"


def test_physics_advancing_is_detected(replica: tuple[SimpleNamespace, list[object], list[bool]]) -> None:
    simulator, calls, _flags = replica
    capture = _prepared(simulator, calls)
    simulator.render = lambda: setattr(simulator.sim, "current_time_step_index", 3)
    with pytest.raises(RuntimeError, match="advanced Isaac physics"):
        capture.render()
    with pytest.raises(RuntimeError, match="between rendered frames"):
        capture.render()


def test_lidars_are_rejected(replica: tuple[SimpleNamespace, list[object], list[bool]]) -> None:
    simulator, _calls, _flags = replica
    simulator.sensor_manager.lidars = [object()]
    with pytest.raises(ValueError, match="cameras only"):
        RenderCapture(cast("IsaacSim", simulator))
