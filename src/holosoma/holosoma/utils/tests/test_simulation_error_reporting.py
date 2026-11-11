"""Failures must survive provider cleanup that terminates the Python process."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.utils import sim_utils
from holosoma.utils.simulator_config import SimulatorConfig, SimulatorType
from holosoma.utils.tests._simulation_error_child import run_task_session

pytestmark = pytest.mark.no_sim


def _child_command(
    entrypoint: str,
    error: str,
    *,
    sink_kind: str = "direct",
    quit_kind: str = "normal",
    diagnostic: str = "normal",
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("_simulation_error_child.py")),
        entrypoint,
        error,
        "--sink",
        sink_kind,
        "--quit",
        quit_kind,
        "--diagnostic",
        diagnostic,
    ]


def _plain_text(text: str) -> str:
    # ANSI styling can split the exception type from its message in CI.
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _run_child(entrypoint: str, error: str, *, color: str = "NO", **options: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        _child_command(entrypoint, error, **options),
        env={**os.environ, "LOGURU_COLORIZE": color},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    result.stderr = _plain_text(result.stderr)
    return result


@pytest.fixture
def failed_cleanup(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    cleanup_error = RuntimeError("Environment close exploded")
    events: list[str] = []
    logs: list[Any] = []

    def close_environment() -> None:
        events.append("env.close")
        raise cleanup_error

    env = SimpleNamespace(close=close_environment)
    app = SimpleNamespace(
        app=SimpleNamespace(post_uncancellable_quit=lambda _code: events.append("quit")),
        close=lambda **_kwargs: events.append("app.close"),
    )
    monkeypatch.setattr(sim_utils, "get_simulator_type", lambda: SimulatorType.ISAACSIM)
    capture_sink = sim_utils.logger.add(logs.append, level="ERROR")
    try:
        yield SimpleNamespace(env=env, app=app, error=cleanup_error, events=events, logs=logs)
    finally:
        sim_utils.logger.remove(capture_sink)


@pytest.mark.parametrize("entrypoint", ["construct", "app", "run_reset", "run_step", "session_body"])
def test_failure_is_reported_before_native_shutdown(entrypoint: str) -> None:
    result = _run_child(entrypoint, "runtime")

    assert result.returncode == 1, result.stderr
    expected = [] if entrypoint in {"construct", "app"} else ["ENV_CLOSE"]
    assert result.stdout.splitlines() == [*expected, "WORKAROUNDS_OK", "APP_CLOSE"], result.stderr
    assert result.stderr.count("Traceback (most recent call last)") == 1
    assert "RuntimeError: Missing LiDAR prim: /World/missing_chair" in result.stderr


@pytest.mark.parametrize(
    ("error", "return_code"),
    [("none", 0), ("interrupt", 130), ("terminate", 143), ("exit_code", 7), ("exit_message", 1), ("exit_success", 0)],
)
def test_native_shutdown_preserves_exit_intent(error: str, return_code: int) -> None:
    result = _run_child("session_body", error)

    assert result.returncode == return_code, result.stderr
    assert result.stdout.splitlines() == ["ENV_CLOSE", "WORKAROUNDS_OK", "APP_CLOSE"], result.stderr
    assert "Traceback (most recent call last)" not in result.stderr
    if error == "exit_message":
        assert "Invalid simulator configuration" in result.stderr


@pytest.mark.parametrize("entrypoint", ["run_reset", "run_step", "session_body"])
@pytest.mark.parametrize("error", ["runtime", "interrupt", "exit_code"])
def test_environment_cleanup_failure_reports_without_requesting_quit(
    entrypoint: str, error: str, failed_cleanup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = failed_cleanup
    failure = {
        "runtime": RuntimeError("Simulation failed"),
        "interrupt": KeyboardInterrupt(),
        "exit_code": SystemExit(7),
    }[error]

    def fail() -> None:
        raise failure

    with pytest.raises(type(failure)) as caught:
        run_task_session(monkeypatch, entrypoint, fail, case.env, case.app)

    assert caught.value is failure
    assert case.events == ["env.close"]
    if error == "runtime":
        assert case.logs[0].record["exception"].value is failure


def test_environment_cleanup_failure_without_body_error_is_reported(failed_cleanup: SimpleNamespace) -> None:
    case = failed_cleanup
    with pytest.raises(RuntimeError, match="Environment cleanup failed; provider application remains open") as caught:
        sim_utils.close_simulation_resources(case.env, case.app)

    assert caught.value.__cause__ is case.error
    assert case.events == ["env.close"]
    assert case.logs[0].record["exception"].value is case.error


@pytest.mark.parametrize("failure", [None, RuntimeError("Simulation failed")])
def test_unconfigured_backend_still_closes_environment(
    monkeypatch: pytest.MonkeyPatch, failure: RuntimeError | None
) -> None:
    monkeypatch.setattr(SimulatorConfig, "_simulator_type", None)
    events: list[str] = []
    env = SimpleNamespace(close=lambda: events.append("env.close"))

    def fail() -> None:
        if failure is not None:
            raise failure

    with pytest.raises(RuntimeError) as caught:
        run_task_session(monkeypatch, "session_body", fail, env, object())

    assert events == ["env.close"]
    if failure is not None:
        assert caught.value is failure
    else:
        assert str(caught.value) == "Simulation app cleanup failed"
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "Simulator type not set" in str(caught.value.__cause__)


@pytest.mark.parametrize("sink_kind", ["direct", "queued"])
def test_traceback_survives_forced_termination_during_environment_cleanup(tmp_path: Path, sink_kind: str) -> None:
    ready_file = tmp_path / "cleanup-started"
    stderr_path = tmp_path / "stderr.log"
    command = _child_command("run_step", "runtime", sink_kind=sink_kind)
    command.extend(["--ready-file", str(ready_file)])
    with (
        stderr_path.open("w") as stderr,
        subprocess.Popen(
            command,
            env={**os.environ, "LOGURU_COLORIZE": "NO"},
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        ) as process,
    ):
        try:
            deadline = time.monotonic() + 30
            while not ready_file.exists():
                assert process.poll() is None, stderr_path.read_text()
                assert time.monotonic() < deadline, "Child never entered environment cleanup"
                time.sleep(0.02)
        finally:
            process.kill()
            stdout, _ = process.communicate(timeout=10)

    assert process.returncode == -signal.SIGKILL
    assert stdout.splitlines() == ["ENV_CLOSE"]
    diagnostic = _plain_text(stderr_path.read_text())
    assert diagnostic.count("Traceback (most recent call last)") == 1
    assert "RuntimeError: Missing LiDAR prim: /World/missing_chair" in diagnostic
