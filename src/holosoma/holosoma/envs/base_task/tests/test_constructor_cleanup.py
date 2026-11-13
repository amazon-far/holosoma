from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from holosoma.envs.base_task import base_task as base_task_module
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.hooks import HookRegistry, Phase

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


def test_task_builds_plugins_last_with_pacing_after_them(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _Simulator:
        def __init__(self, *, hooks: HookRegistry, **_kwargs: object) -> None:
            self.sim_dt = 0.005
            self.simulator_config = SimpleNamespace(reset_manager=object())
            self.hooks = hooks
            self.prepared = False
            events.append("simulator")

        def set_headless(self, _headless: bool) -> None:
            pass

        def setup(self) -> None:
            pass

        def setup_terrain(self) -> None:
            pass

        def get_dof_limits_properties(self) -> tuple[object, object, object]:
            return object(), object(), object()

        def prepare_sim(self) -> None:
            self.prepared = True

        def setup_viewer(self) -> None:
            assert self.prepared
            self.viewer = object()
            events.append("viewer")

    class _ProbePlugin:
        def __init__(self, _cfg: object, task: BaseTask) -> None:
            assert task.hooks is task.simulator.hooks
            events.append("plugins")
            task.hooks.add(Phase.POST_STEP, lambda: None, name="probe.post_step")

    def create_reset_manager(*_args: object) -> object:
        events.append("reset_manager")
        return object()

    simulator_config = SimpleNamespace(
        _target_="test.Simulator",
        config=SimpleNamespace(name="test", sim=SimpleNamespace(fps=200, control_decimation_steps=4)),
    )
    config = SimpleNamespace(
        simulator=simulator_config,
        terrain=object(),
        robot=SimpleNamespace(),
        training=SimpleNamespace(num_envs=1, seed=0, headless=False, torch_deterministic=False),
        logger=object(),
        scene=object(),
        sensors={},
        plugin={"probe": SimpleNamespace(get_cls=lambda: _ProbePlugin)},
        device="cpu",
        viewer_update_decimation=1,
        time_scale=0.0,
    )

    monkeypatch.setattr(base_task_module, "get_class", lambda _target: _Simulator)
    monkeypatch.setattr(base_task_module, "FullSimConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(base_task_module, "TerrainManager", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(base_task_module, "ResetEventManager", create_reset_manager)
    monkeypatch.setattr(BaseTask, "_load_assets", lambda _task: None)
    monkeypatch.setattr(BaseTask, "_create_envs", lambda _task: None)
    monkeypatch.setattr(BaseTask, "_setup_robot_body_indices", lambda _task: None)
    monkeypatch.setattr(BaseTask, "_initialize_task_managers", lambda _task: events.append("managers"))
    monkeypatch.setattr(BaseTask, "_init_buffers", lambda _task: None)
    monkeypatch.setattr(BaseTask, "_prepare_manager_fields", lambda _task: None)
    monkeypatch.setattr(BaseTask, "_setup_task_managers", lambda _task: None)
    monkeypatch.setattr(base_task_module, "get_timestamp", lambda: "timestamp")
    monkeypatch.setattr(base_task_module, "get_experiment_dir", lambda *_args, **_kwargs: ".")

    task = BaseTask(config)  # type: ignore[arg-type]

    assert events == ["simulator", "viewer", "managers", "reset_manager", "plugins"]
    assert task.hooks.names(Phase.POST_STEP) == ("probe.post_step", "task.viewer")
