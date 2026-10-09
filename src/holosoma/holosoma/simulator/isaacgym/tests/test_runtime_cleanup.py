from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.isaacgym

pytest.importorskip("isaacgym")

from holosoma.config_types.video import VideoConfig  # noqa: E402
from holosoma.utils.simulator_config import SimulatorType, set_simulator_type_enum  # noqa: E402

set_simulator_type_enum(SimulatorType.ISAACGYM)

from holosoma.simulator.isaacgym import isaacgym as isaacgym_module  # noqa: E402
from holosoma.simulator.isaacgym.isaacgym import IsaacGym  # noqa: E402


def _simulator() -> IsaacGym:
    simulator = object.__new__(IsaacGym)
    simulator.viewer = object()
    simulator.sim = object()  # type: ignore[assignment]
    simulator.gym = MagicMock()
    return simulator


def _initialized_simulator(monkeypatch: pytest.MonkeyPatch) -> IsaacGym:
    simulator_config = SimpleNamespace(
        debug_viz=False,
        sim=SimpleNamespace(kinematic_playback=False, fps=200.0, control_decimation_steps=4),
    )
    tyro_config = SimpleNamespace(
        training=SimpleNamespace(),
        simulator=simulator_config,
        scene=SimpleNamespace(lights={}),
        sensors={},
        robot=SimpleNamespace(),
        logger=SimpleNamespace(video=VideoConfig(enabled=False), headless_recording=False),
        plugin={},
        experiment_dir=None,
    )
    simulator = IsaacGym(tyro_config, terrain_manager=SimpleNamespace(), device="cpu")  # type: ignore[arg-type]
    simulator.headless = False
    simulator._parse_sim_params = lambda: SimpleNamespace(  # type: ignore[method-assign]
        dt=0.005,
        up_axis=isaacgym_module.gymapi.UP_AXIS_Z,
        use_gpu_pipeline=False,
        physx=SimpleNamespace(use_gpu=False),
    )
    gym = MagicMock()
    sim_handle = object()
    viewer_handle = object()
    gym.create_sim.return_value = sim_handle
    gym.create_viewer.return_value = viewer_handle
    monkeypatch.setattr(isaacgym_module.gymapi, "acquire_gym", lambda: gym)
    monkeypatch.setattr(isaacgym_module.gymutil, "parse_device_str", lambda _device: ("cpu", 0))

    simulator.setup()
    simulator.setup_viewer()
    return simulator


def test_setup_registers_viewer_before_simulator_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = _initialized_simulator(monkeypatch)
    events: list[str] = []
    simulator.gym.destroy_viewer.side_effect = lambda _viewer: events.append("viewer")
    simulator.gym.destroy_sim.side_effect = lambda _sim: events.append("sim")

    simulator.close()
    simulator.close()

    assert events == ["viewer", "sim"]


def test_native_close_callbacks_are_idempotent() -> None:
    simulator = _simulator()
    events: list[str] = []
    simulator.gym.destroy_viewer.side_effect = lambda _viewer: events.append("viewer")
    simulator.gym.destroy_sim.side_effect = lambda _sim: events.append("sim")

    simulator._close_viewer()
    simulator._close_viewer()
    simulator._close_sim()
    simulator._close_sim()

    assert events == ["viewer", "sim"]
    assert simulator.viewer is None
    assert simulator.sim is None


def test_viewer_close_failure_does_not_touch_simulator() -> None:
    simulator = _simulator()
    viewer = simulator.viewer
    simulator.gym.destroy_viewer.side_effect = RuntimeError("viewer failed")

    with pytest.raises(RuntimeError, match="viewer failed"):
        simulator._close_viewer()

    simulator.gym.destroy_sim.assert_not_called()
    simulator._close_viewer()
    simulator.gym.destroy_viewer.assert_called_once_with(viewer)
