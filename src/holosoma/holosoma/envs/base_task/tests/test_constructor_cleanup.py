from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from holosoma.envs.base_task import base_task as base_task_module
from holosoma.envs.base_task.base_task import BaseTask

pytestmark = pytest.mark.no_sim


def test_task_setup_failure_closes_created_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = SimpleNamespace(close=MagicMock())

    def fail_after_simulator_creation(self: BaseTask, *_args: object, **_kwargs: object) -> None:
        self.simulator = simulator  # type: ignore[assignment]
        raise ValueError("manager setup failed")

    monkeypatch.setattr(BaseTask, "_initialize_base_task", fail_after_simulator_creation)

    with pytest.raises(ValueError, match="manager setup failed"):
        BaseTask(object(), device="cpu")  # type: ignore[arg-type]

    simulator.close.assert_called_once_with()


def test_task_preserves_interrupt_when_constructor_cleanup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = SimpleNamespace(close=MagicMock(side_effect=RuntimeError("close failed")))

    def interrupt_after_simulator_creation(self: BaseTask, *_args: object, **_kwargs: object) -> None:
        self.simulator = simulator  # type: ignore[assignment]
        raise KeyboardInterrupt

    monkeypatch.setattr(BaseTask, "_initialize_base_task", interrupt_after_simulator_creation)

    with pytest.raises(KeyboardInterrupt):
        BaseTask(object(), device="cpu")  # type: ignore[arg-type]

    simulator.close.assert_called_once_with()


def test_task_installs_plugins_only_after_provider_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _StopAfterInstallError(Exception):
        pass

    class _Simulator:
        def __init__(self, **_kwargs: object) -> None:
            self.sim_dt = 0.005
            self.simulator_config = SimpleNamespace(reset_manager=object())
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

        def install_plugins(self) -> None:
            assert self.prepared
            assert hasattr(self, "viewer")
            events.append("plugins")

        def close(self) -> None:
            events.append("close")

    def load_assets(_task: BaseTask) -> None:
        events.append("assets")

    def create_envs(_task: BaseTask) -> None:
        events.append("envs")

    def create_observation_manager(*_args: object, **_kwargs: object) -> None:
        events.append("manager")
        raise _StopAfterInstallError

    simulator_config = SimpleNamespace(
        _target_="test.Simulator",
        config=SimpleNamespace(
            sim=SimpleNamespace(control_decimation_steps=4, max_episode_length_s=20.0),
        ),
    )
    config = SimpleNamespace(
        observation=object(),
        simulator=simulator_config,
        terrain=object(),
        robot=SimpleNamespace(policy_obs_dim=1, critic_obs_dim=1, actions_dim=1),
        action=object(),
        reward=object(),
        termination=object(),
        randomization=object(),
        command=object(),
        curriculum=object(),
        training=SimpleNamespace(num_envs=1, seed=0, headless=False),
        logger=object(),
        scene=object(),
        sensors={},
        plugin={},
    )

    monkeypatch.setattr(base_task_module, "get_class", lambda _target: _Simulator)
    monkeypatch.setattr(base_task_module, "FullSimConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(base_task_module, "TerrainManager", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(base_task_module, "ObservationManager", create_observation_manager)
    monkeypatch.setattr(BaseTask, "_load_assets", load_assets)
    monkeypatch.setattr(BaseTask, "_create_envs", create_envs)
    monkeypatch.setattr(BaseTask, "_setup_robot_body_indices", lambda _task: events.append("body_indices"))
    monkeypatch.setattr("holosoma.utils.experiment_paths.get_timestamp", lambda: "timestamp")
    monkeypatch.setattr("holosoma.utils.experiment_paths.get_experiment_dir", lambda *_args, **_kwargs: ".")

    with pytest.raises(_StopAfterInstallError):
        BaseTask(config, device="cpu")  # type: ignore[arg-type]

    assert events.index("prepare") < events.index("viewer") < events.index("plugins") < events.index("manager")
    assert events[-1] == "close"
