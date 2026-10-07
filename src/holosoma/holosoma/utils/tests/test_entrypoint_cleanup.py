from __future__ import annotations

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
from holosoma.utils import common, experiment_paths
from holosoma.utils.simulator_config import SimulatorType

pytestmark = pytest.mark.no_sim


class _Environment:
    def __init__(self, events: list[str], *, close_fails: bool = False) -> None:
        self.events = events
        self.close_fails = close_fails
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

    def close_current_app(closed_app: object) -> None:
        close_app(closed_app, sys.exc_info()[1])

    monkeypatch.setattr("holosoma.utils.sim_utils._close_simulation_app", close_app)
    monkeypatch.setattr(train_agent, "close_simulation_app", close_current_app)
    return app


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

    config = SimpleNamespace(
        logger=object(),
        training=SimpleNamespace(export_onnx=False, max_eval_steps=5),
        algo=SimpleNamespace(_target_="test.Algo", config=object()),
        save_config=MagicMock(),
    )
    checkpoint_cfg = SimpleNamespace(checkpoint="model.pt")

    def setup_environment(_config: object) -> tuple[_Environment, str, object]:
        assert signal.getsignal(signal.SIGTERM) is not previous_sigterm
        return env, "cpu", app

    monkeypatch.setattr(eval_agent, "setup_simulation_environment", setup_environment)
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

    def init_sim_imports(_config: object) -> object:
        assert signal.getsignal(signal.SIGTERM) is not previous_sigterm
        return app

    monkeypatch.setattr(replay_module, "init_sim_imports", init_sim_imports)
    monkeypatch.setattr(replay_module, "get_tyro_env_config", lambda _config: object())
    monkeypatch.setattr(replay_module, "get_class", lambda _target: lambda *_args, **_kwargs: env)
    monkeypatch.setattr(common, "seeding", lambda *_args, **_kwargs: None)

    with pytest.raises(KeyboardInterrupt):
        replay_module.replay(SimpleNamespace(env_class="test.Environment"))  # type: ignore[arg-type]

    assert events == ["env.close", "app.close"]
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm


@pytest.mark.parametrize(
    ("external_context", "environment_close_fails"),
    [(False, False), (True, False), (True, True)],
)
def test_training_interrupt_closes_environment_before_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    external_context: bool,
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

    config = SimpleNamespace(
        logger=SimpleNamespace(type="local"),
        training=SimpleNamespace(
            seed=1,
            torch_deterministic=False,
            project=None,
            name=None,
            checkpoint=None,
            num_envs=1,
        ),
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="test")),
        algo=SimpleNamespace(_target_="test.Algo", config=object()),
        env_class="test.Environment",
        save_config=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "wandb", types.ModuleType("wandb"))

    def init_sim_imports(_config: object) -> object:
        assert signal.getsignal(signal.SIGTERM) is not previous_sigterm
        return app

    monkeypatch.setattr(train_agent, "init_sim_imports", init_sim_imports)
    monkeypatch.setattr(train_agent, "configure_multi_gpu", lambda: None)
    monkeypatch.setattr(train_agent, "get_device", lambda *_args: "cpu")
    monkeypatch.setattr(train_agent, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(train_agent, "get_tyro_env_config", lambda _config: object())
    monkeypatch.setattr(
        train_agent,
        "get_class",
        lambda target: (lambda *_args, **_kwargs: env) if target == "test.Environment" else _Algo,
    )
    monkeypatch.setattr(common, "seeding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_paths, "get_timestamp", lambda: "timestamp")
    monkeypatch.setattr(experiment_paths, "get_experiment_dir", lambda *_args, **_kwargs: tmp_path)

    def run_training() -> None:
        if external_context:
            with train_agent.training_context(config) as context:  # type: ignore[arg-type]
                context.train()
        else:
            train_agent.train(config)  # type: ignore[arg-type]

    with pytest.raises(KeyboardInterrupt):
        run_training()

    expected_events = ["env.close"] if environment_close_fails else ["env.close", "app.close"]
    assert events == expected_events
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm


def test_training_context_owns_only_the_provider_app(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    env = _Environment(events)
    app = object()
    close_app = MagicMock()
    monkeypatch.setattr(train_agent, "init_sim_imports", lambda _config: app)
    monkeypatch.setattr(train_agent, "close_simulation_app", close_app)

    with train_agent.training_context(object()) as context:  # type: ignore[arg-type]
        assert context.simulation_app is app
        context.environment = env  # type: ignore[attr-defined]

    close_app.assert_called_once_with(app)
    assert events == []


@pytest.mark.parametrize("environment_close_fails", [False, True])
def test_training_context_session_preserves_provider_order(
    monkeypatch: pytest.MonkeyPatch,
    environment_close_fails: bool,
) -> None:
    events: list[str] = []
    env = _Environment(events, close_fails=environment_close_fails)
    app = _patch_app_close(monkeypatch, events)
    monkeypatch.setattr(train_agent, "init_sim_imports", lambda _config: app)

    def run_session() -> None:
        with train_agent.training_context(object()) as context, context.simulation_session(  # type: ignore[arg-type]
            env
        ) as owned_env:
            assert owned_env is env

    if environment_close_fails:
        with pytest.raises(RuntimeError, match="Environment cleanup failed"):
            run_session()
    else:
        run_session()

    expected_events = ["env.close"] if environment_close_fails else ["env.close", "app.close"]
    assert events == expected_events
