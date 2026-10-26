from __future__ import annotations

import dataclasses
import signal
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import holosoma.replay as replay_module
from holosoma import eval_agent, train_agent
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.simulator import session
from holosoma.utils import experiment_paths
from holosoma.utils.simulator_config import SimulatorType

pytestmark = pytest.mark.no_sim


class _Environment:
    def __init__(self, events: list[str], *, close_fails: bool = False) -> None:
        self.events = events
        self.close_fails = close_fails
        self.device = "cpu"
        self.observation_manager = object()
        self.simulator = SimpleNamespace(sim=SimpleNamespace(step=lambda: None))

    def close(self) -> None:
        self.events.append("env.close")
        if self.close_fails:
            raise RuntimeError("env close failed")


def _patch_app_close(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> object:
    app = object()
    monkeypatch.setattr("holosoma.utils.sim_utils.get_simulator_type", lambda: SimulatorType.ISAACSIM)

    def close_app(closed_app: object, failure: BaseException | None) -> None:
        assert failure is sys.exc_info()[1]
        if closed_app is None:
            return
        assert closed_app is app
        events.append("app.close")

    monkeypatch.setattr("holosoma.utils.sim_utils._close_simulation_app", close_app)
    return app


def _patch_session(monkeypatch: pytest.MonkeyPatch, env: _Environment, app: object) -> SimpleNamespace:
    """Make ``task_session`` launch ``app`` and yield ``env``; return a config whose create_task uses it."""
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def launch(_config: object, _device: str) -> object:
        assert signal.getsignal(signal.SIGTERM) is not previous_sigterm
        return app

    class _Task(BaseTask):
        def __new__(cls, _config: object) -> Any:
            return env

    monkeypatch.setattr(session, "setup_simulator_imports", lambda _config: None)
    monkeypatch.setattr(session, "resolve_simulation_device", lambda *_args: "cpu")
    monkeypatch.setattr(session, "get_simulator_type", lambda: SimulatorType.ISAACSIM)
    monkeypatch.setattr(session, "setup_isaaclab_launcher", launch)
    monkeypatch.setattr(session, "get_class", lambda _target: _Task)
    env_config = SimpleNamespace(env_class="test.Task", training=SimpleNamespace(headless=True))
    return SimpleNamespace(create_task=lambda: session.task_session(env_config, BaseTask))  # type: ignore[arg-type]


def test_eval_interrupt_closes_environment_before_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    env = _Environment(events)
    app = _patch_app_close(monkeypatch, events)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    class _Algo:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def setup(self) -> None:
            pass

        def attach_checkpoint_metadata(self, *_args: Any) -> None:
            pass

        def load(self, _checkpoint: str) -> None:
            pass

        def evaluate_policy(self, *, max_eval_steps: int) -> None:
            assert max_eval_steps == 5
            raise KeyboardInterrupt

    config = _patch_session(monkeypatch, env, app)
    config.logger = object()
    config.training = SimpleNamespace(export_onnx=False, max_eval_steps=5)
    config.algo = SimpleNamespace(_target_="test.Algo", config=object())
    config.save_config = MagicMock()
    checkpoint_cfg = SimpleNamespace(checkpoint="model.pt")

    monkeypatch.setattr(eval_agent, "get_experiment_dir", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(eval_agent, "load_checkpoint", lambda *_args: tmp_path / "model.pt")
    monkeypatch.setattr(eval_agent, "get_class", lambda _target: _Algo)

    with pytest.raises(KeyboardInterrupt):
        eval_agent.run_eval_with_tyro(config, checkpoint_cfg, config, None)  # type: ignore[arg-type]

    assert events == ["env.close", "app.close"]
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm


def test_replay_interrupt_closes_environment_before_app(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    env = _Environment(events)
    app = _patch_app_close(monkeypatch, events)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_replay(_motion: Any) -> bool:
        raise KeyboardInterrupt

    env.step_visualize_motion = stop_replay  # type: ignore[attr-defined]
    config = _patch_session(monkeypatch, env, app)

    with pytest.raises(KeyboardInterrupt):
        replay_module.replay(config)  # type: ignore[arg-type]

    assert events == ["env.close", "app.close"]
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm


@pytest.mark.parametrize("environment_close_fails", [False, True])
def test_training_interrupt_closes_environment_before_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment_close_fails: bool,
) -> None:
    events: list[str] = []
    env = _Environment(events, close_fails=environment_close_fails)
    app = _patch_app_close(monkeypatch, events)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    class _Algo:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def setup(self) -> None:
            pass

        def attach_checkpoint_metadata(self, *_args: Any) -> None:
            pass

        def learn(self) -> None:
            raise KeyboardInterrupt

    config = _patch_session(monkeypatch, env, app)
    config.logger = SimpleNamespace(type="local")
    config.training = SimpleNamespace(
        project=None,
        name=None,
        checkpoint=None,
        teacher_checkpoint=None,
        num_envs=1,
    )
    config.robot = SimpleNamespace(asset=SimpleNamespace(robot_type="test"))
    config.algo = SimpleNamespace(_target_="test.Algo", config=object())
    config.env_class = "test.Task"
    config.save_config = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", types.ModuleType("wandb"))
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    monkeypatch.setattr(train_agent, "configure_multi_gpu", lambda: None)
    monkeypatch.setattr(train_agent, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(train_agent, "get_class", lambda _target: _Algo)
    monkeypatch.setattr(experiment_paths, "get_timestamp", lambda: "timestamp")
    monkeypatch.setattr(experiment_paths, "get_experiment_dir", lambda *_args, **_kwargs: tmp_path)

    with pytest.raises(KeyboardInterrupt):
        train_agent.train(config)  # type: ignore[arg-type]

    expected_events = ["env.close"] if environment_close_fails else ["env.close", "app.close"]
    assert events == expected_events
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm


@pytest.mark.parametrize("environment_close_fails", [False, True])
def test_task_session_closes_task_before_app(
    monkeypatch: pytest.MonkeyPatch,
    environment_close_fails: bool,
) -> None:
    events: list[str] = []
    env = _Environment(events, close_fails=environment_close_fails)
    app = _patch_app_close(monkeypatch, events)
    config = _patch_session(monkeypatch, env, app)

    def run_session() -> None:
        with config.create_task() as task:
            assert task is env

    if environment_close_fails:
        with pytest.raises(RuntimeError, match="Environment cleanup failed"):
            run_session()
    else:
        run_session()

    expected_events = ["env.close"] if environment_close_fails else ["env.close", "app.close"]
    assert events == expected_events


@pytest.mark.parametrize(("world_size", "expected"), [(None, 4096), ("1", 4096), ("4", 1024)])
def test_split_envs_across_ranks_gives_each_rank_its_share(
    monkeypatch: pytest.MonkeyPatch, world_size: str | None, expected: int
) -> None:
    from holosoma.config_values import experiment

    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    config = dataclasses.replace(
        experiment.g1_29dof, training=dataclasses.replace(experiment.g1_29dof.training, num_envs=4096)
    )

    assert train_agent.split_envs_across_ranks(config).training.num_envs == expected
