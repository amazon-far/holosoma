"""Failures must survive provider cleanup that terminates the Python process."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, cast

import pytest

from holosoma.utils import sim_utils
from holosoma.utils.simulator_config import SimulatorConfig, SimulatorType
from holosoma.utils.tests._simulation_error_child import configure_logging

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

    env = SimpleNamespace(sim=object(), close=close_environment)
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


@pytest.mark.parametrize(
    "entrypoint", ["setup", "app", "direct_initialize", "direct_body", "direct_run", "resource_body"]
)
def test_failure_is_reported_before_native_shutdown(entrypoint: str) -> None:
    result = _run_child(entrypoint, "runtime")

    assert result.returncode == 1, result.stderr
    expected = [] if entrypoint in {"setup", "app"} else ["ENV_CLOSE"]
    assert result.stdout.splitlines() == [*expected, "WORKAROUNDS_OK", "APP_CLOSE"], result.stderr
    assert result.stderr.count("Traceback (most recent call last)") == 1
    assert "RuntimeError: Missing LiDAR prim: /World/missing_chair" in result.stderr
    if entrypoint == "direct_run":
        assert "Error during simulation step 0" in result.stderr


@pytest.mark.parametrize("error", ["runtime", "exit_message"])
@pytest.mark.parametrize(
    ("color", "sink_kind"),
    [("YES", "direct"), ("YES", "queued"), ("NO", "queued")],
    ids=["color", "color-queue", "plain-queue"],
)
def test_diagnostics_are_delivered_before_native_shutdown(error: str, sink_kind: str, color: str) -> None:
    # Plain direct diagnostics are covered by the entrypoint and exit-intent tests.
    result = _run_child("resource_body", error, sink_kind=sink_kind, color=color)

    assert result.returncode == 1, result.stderr
    assert result.stdout.splitlines() == ["ENV_CLOSE", "WORKAROUNDS_OK", "APP_CLOSE"], result.stderr
    if error == "runtime":
        assert "Traceback (most recent call last)" in result.stderr
        assert "RuntimeError: Missing LiDAR prim: /World/missing_chair" in result.stderr
    else:
        assert "Invalid simulator configuration" in result.stderr
        assert "Traceback (most recent call last)" not in result.stderr


@pytest.mark.parametrize(
    ("error", "return_code"),
    [("none", 0), ("interrupt", 130), ("terminate", 143), ("exit_code", 7), ("exit_message", 1), ("exit_success", 0)],
)
def test_native_shutdown_preserves_exit_intent(error: str, return_code: int) -> None:
    result = _run_child("resource_body", error)

    assert result.returncode == return_code, result.stderr
    assert result.stdout.splitlines() == ["ENV_CLOSE", "WORKAROUNDS_OK", "APP_CLOSE"], result.stderr
    assert "Traceback (most recent call last)" not in result.stderr
    if error == "exit_message":
        assert "Invalid simulator configuration" in result.stderr


@pytest.mark.parametrize("entrypoint", ["direct_initialize", "direct_body", "resource_body"])
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

    def initialize(_self: sim_utils.DirectSimulation) -> None:
        if entrypoint == "direct_initialize":
            raise failure

    monkeypatch.setattr(sim_utils.DirectSimulation, "initialize", initialize)

    def run_failure() -> None:
        if entrypoint == "resource_body":
            with sim_utils.simulation_resource_session(case.env, case.app):
                raise failure
        else:
            config = cast("sim_utils.RunSimConfig", object())
            with sim_utils.DirectSimulation(config, case.env, "cpu", case.app):
                if entrypoint == "direct_initialize":
                    raise AssertionError("Initialization unexpectedly succeeded")
                raise failure

    with pytest.raises(type(failure)) as caught:
        run_failure()

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

    def run_session() -> None:
        with sim_utils.simulation_resource_session(env, object()):
            if failure is not None:
                raise failure

    with pytest.raises(RuntimeError) as caught:
        run_session()

    assert events == ["env.close"]
    if failure is not None:
        assert caught.value is failure
    else:
        assert str(caught.value) == "Simulation app cleanup failed"
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "Simulator type not set" in str(caught.value.__cause__)


@pytest.mark.parametrize("quit_kind", ["missing_app", "binding_error"])
@pytest.mark.parametrize("sink_kind", ["direct", "queued"])
def test_unavailable_quit_api_can_exit_zero_but_still_cleans_up(quit_kind: str, sink_kind: str) -> None:
    """Simulate failures looking up or calling Kit's exit-status API."""
    result = _run_child("resource_body", "runtime", quit_kind=quit_kind, sink_kind=sink_kind)

    # The unavailable Kit API cannot override its default status, but the
    # diagnostic and both shutdown workarounds must survive through app.close().
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["ENV_CLOSE", "WORKAROUNDS_OK", "APP_CLOSE"]
    assert "RuntimeError: Missing LiDAR prim: /World/missing_chair" in result.stderr
    assert "Could not set Isaac Sim exit status to 1" in result.stderr


@pytest.mark.parametrize(
    ("entrypoint", "diagnostic"),
    [
        ("resource_body", "sink_error"),
        ("resource_body", "sink_all"),
        ("resource_body", "drain"),
        ("resource_body", "flush"),
        ("app", "sink_error"),
        ("app", "drain"),
        ("app", "flush"),
        ("direct_run", "sink_error"),
    ],
)
def test_diagnostic_failure_does_not_skip_exit_status_or_native_cleanup(entrypoint: str, diagnostic: str) -> None:
    result = _run_child(entrypoint, "runtime", diagnostic=diagnostic)

    assert result.returncode == 1, result.stderr
    expected = [] if entrypoint == "app" else ["ENV_CLOSE"]
    assert result.stdout.splitlines() == [*expected, "WORKAROUNDS_OK", "APP_CLOSE"], result.stderr
    assert f"DIAGNOSTIC_FAILURE {diagnostic}" in result.stderr
    assert result.stderr.count("Traceback (most recent call last)") == 1
    assert "RuntimeError: Missing LiDAR prim: /World/missing_chair" in result.stderr


@pytest.mark.parametrize("error", ["none", "runtime"])
@pytest.mark.parametrize("diagnostic", ["sink_error", "drain", "flush"])
def test_diagnostic_failure_preserves_cleanup_error_and_original_exception(
    error: str,
    diagnostic: str,
    failed_cleanup: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    case = failed_cleanup
    failure = RuntimeError("Simulation failed")

    def run_failure() -> None:
        if error == "runtime":
            with sim_utils.simulation_resource_session(case.env, case.app):
                raise failure
        else:
            sim_utils.close_simulation_resources(case.env, case.app)

    with monkeypatch.context() as fault_patch:
        fault_sink = configure_logging(fault_patch, "direct", diagnostic)
        try:
            with pytest.raises(RuntimeError) as caught:
                run_failure()
        finally:
            if fault_sink is not None:
                sim_utils.logger.remove(fault_sink)

    if error == "runtime":
        assert caught.value is failure
    else:
        assert str(caught.value) == "Environment cleanup failed; provider application remains open"
        assert caught.value.__cause__ is case.error
    assert case.events == ["env.close"]
    assert f"DIAGNOSTIC_FAILURE {diagnostic}" in capfd.readouterr().err


@pytest.mark.parametrize("sink_kind", ["direct", "queued"])
def test_traceback_survives_forced_termination_during_environment_cleanup(tmp_path: Path, sink_kind: str) -> None:
    ready_file = tmp_path / "cleanup-started"
    stderr_path = tmp_path / "stderr.log"
    command = _child_command("direct_run", "runtime", sink_kind=sink_kind)
    command.extend(["--ready-file", str(ready_file)])
    with stderr_path.open("w") as stderr, subprocess.Popen(
        command,
        env={**os.environ, "LOGURU_COLORIZE": "NO"},
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
    ) as process:
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
