from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from holosoma.envs.base_task import base_task as base_task_module
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def test_task_setup_failure_closes_created_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = SimpleNamespace(close=MagicMock())

    def fail_after_simulator_creation(self: BaseTask, *_args: object, **_kwargs: object) -> None:
        self.simulator = simulator  # type: ignore[assignment]
        raise ValueError("manager setup failed")

    monkeypatch.setattr(BaseTask, "_initialize_base_task", fail_after_simulator_creation)

    with pytest.raises(ValueError, match="manager setup failed"):
        BaseTask(object())  # type: ignore[arg-type]

    simulator.close.assert_called_once_with()


def test_task_preserves_interrupt_when_constructor_cleanup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = SimpleNamespace(close=MagicMock(side_effect=RuntimeError("close failed")))

    def interrupt_after_simulator_creation(self: BaseTask, *_args: object, **_kwargs: object) -> None:
        self.simulator = simulator  # type: ignore[assignment]
        raise KeyboardInterrupt

    monkeypatch.setattr(BaseTask, "_initialize_base_task", interrupt_after_simulator_creation)

    with pytest.raises(KeyboardInterrupt):
        BaseTask(object())  # type: ignore[arg-type]

    simulator.close.assert_called_once_with()


def test_task_runs_close_hooks_when_simulator_construction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    def fail_inside_simulator_construction(self: BaseTask, *_args: object, **_kwargs: object) -> None:
        self.hooks = HookRegistry()
        self.hooks.add(Phase.CLOSE, lambda: closed.append("backend"))
        raise RuntimeError("backend failed")

    monkeypatch.setattr(BaseTask, "_initialize_base_task", fail_inside_simulator_construction)

    with pytest.raises(RuntimeError, match="backend failed"):
        BaseTask(object())  # type: ignore[arg-type]

    assert closed == ["backend"]


def test_task_installs_plugins_after_task_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _StopAfterInstallError(Exception):
        pass

    class _Simulator:
        def __init__(self, *, hooks: HookRegistry, **_kwargs: object) -> None:
            self.sim_dt = 0.005
            self.simulator_config = SimpleNamespace(reset_manager=object())
            self.hooks = hooks
            self.robot_root_states = torch.zeros(1, 13)
            self.dof_pos = torch.zeros(1, 1)
            self.dof_vel = torch.zeros(1, 1)
            self.prepared = False
            events.append("simulator")

        def set_headless(self, _headless: bool) -> None:
            events.append("headless")

        def setup(self) -> None:
            events.append("setup")

        def setup_terrain(self) -> None:
            events.append("terrain")

        def get_dof_limits_properties(self) -> tuple[object, object, object]:
            events.append("limits")
            return object(), object(), object()

        def prepare_sim(self) -> None:
            self.prepared = True
            events.append("prepare")

        def setup_viewer(self) -> None:
            assert self.prepared
            self.viewer = object()
            events.append("viewer")

        def prepare_manager_fields(self, **_managers: object) -> None:
            events.append("manager_fields")

        def close(self) -> None:
            events.append("close")

    class _ProbePlugin:
        def __init__(self, _cfg: object, task: BaseTask) -> None:
            assert task.hooks is task.simulator.hooks
            events.append("plugins")
            raise _StopAfterInstallError

    simulator_config = SimpleNamespace(
        _target_="test.Simulator",
        config=SimpleNamespace(
            name="test",
            sim=SimpleNamespace(fps=200, control_decimation_steps=4, max_episode_length_s=20.0),
        ),
    )
    config = SimpleNamespace(
        simulator=simulator_config,
        terrain=object(),
        robot=SimpleNamespace(),
        randomization=object(),
        training=SimpleNamespace(num_envs=1, seed=0, headless=False, torch_deterministic=False),
        logger=object(),
        scene=object(),
        sensors={},
        plugin={"probe": SimpleNamespace(get_cls=lambda: _ProbePlugin)},
        device="cpu",
        time_scale=0.0,
        viewer_dt=1 / 60,
    )

    monkeypatch.setattr(base_task_module, "get_class", lambda _target: _Simulator)
    monkeypatch.setattr(base_task_module, "FullSimConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        base_task_module, "TerrainManager", lambda *_args: SimpleNamespace(setup=lambda: events.append("terrain_setup"))
    )
    monkeypatch.setattr(
        base_task_module,
        "RandomizationManager",
        lambda *_args: SimpleNamespace(setup=lambda: events.append("randomization")),
    )

    def create_reset_manager(*_args: object) -> object:
        events.append("reset_manager")
        return object()

    monkeypatch.setattr(base_task_module, "ResetEventManager", create_reset_manager)
    monkeypatch.setattr(BaseTask, "_load_assets", lambda _task: events.append("assets"))
    monkeypatch.setattr(BaseTask, "_create_envs", lambda _task: events.append("envs"))
    monkeypatch.setattr(BaseTask, "_setup_robot_body_indices", lambda _task: events.append("body_indices"))
    monkeypatch.setattr(base_task_module, "get_timestamp", lambda: "timestamp")
    monkeypatch.setattr(base_task_module, "get_experiment_dir", lambda *_args, **_kwargs: ".")

    with pytest.raises(_StopAfterInstallError):
        BaseTask(config)  # type: ignore[arg-type]

    assert events.index("viewer") < events.index("reset_manager") < events.index("plugins")
    assert events[-2:] == ["plugins", "close"]
