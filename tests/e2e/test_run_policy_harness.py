"""Exercise the E2E launcher with real processes, without training or SDKs."""

from __future__ import annotations

import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("policy_e2e", Path(__file__).with_name("test_run_policy.py"))
assert SPEC is not None and SPEC.loader is not None
policy_e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy_e2e)


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    # Keep bash as group leader and a distinct Python child holding the pipes.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    policy_dir = tmp_path / "src/holosoma_inference/holosoma_inference"
    policy_dir.mkdir(parents=True)
    install = tmp_path / "install.py"
    policy = policy_dir / "run_policy.py"
    executable = shlex.quote(sys.executable)
    (scripts / "source_inference_setup.sh").write_text(
        f'pip() {{ command {executable} {shlex.quote(str(install))}; }}\npython() {{ command {executable} "$@"; }}\n'
    )
    monkeypatch.setattr(policy_e2e, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(policy_e2e, "_STOP_TIMEOUT", 1, raising=False)
    processes = []
    launches = []
    real_popen = subprocess.Popen

    def popen(command, **kwargs):
        proc = real_popen(command, **kwargs)
        processes.append(proc)
        deadlines = []
        launches.append((command, deadlines))
        communicate = proc.communicate

        def measured_communicate(*args, **kwargs):
            deadlines.append(kwargs.get("timeout"))
            return communicate(*args, **kwargs)

        proc.communicate = measured_communicate
        marker = tmp_path / ("policy.ready" if "run_policy.py" in command[-1] else "install.ready")
        deadline = time.monotonic() + 10
        # Scheduling delay is not part of the timeout being tested. The child
        # must have established its signal handler/output before the probe.
        while not marker.exists() and proc.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("fixture child never became ready")
            time.sleep(0.01)
        if marker.exists():
            child_pid, child_group = map(int, marker.read_text().split())
            assert child_pid != proc.pid
            assert child_group == proc.pid
        return proc

    monkeypatch.setattr(policy_e2e.subprocess, "Popen", popen)

    def configure(*, output="RL FPS: 50", ignore_term=False, close_output=False, exit_code=None, install_code=""):
        def ready_script(name):
            target = str(tmp_path / name)
            return (
                f"Path({target + '.tmp'!r}).write_text(f'{{os.getpid()}} {{os.getpgrp()}}')\n"
                f"os.replace({target + '.tmp'!r}, {target!r})\n"
            )

        install.write_text(
            "import os, time, signal\nfrom pathlib import Path\n"
            + ready_script("install.ready")
            + install_code
            + f"\nPath({str(tmp_path / 'installed')!r}).touch()\n"
        )
        policy.write_text(
            "import os, sys, time, signal\nfrom pathlib import Path\n"
            f"assert Path({str(tmp_path / 'installed')!r}).exists()\n"
            + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
            + f"print({output!r}, flush=True)\n"
            + ("os.close(1)\nos.close(2)\n" if close_output else "")
            + ready_script("policy.ready")
            + ("time.sleep(60)\n" if exit_code is None else f"sys.exit({exit_code})\n")
        )

    yield configure, launches, tmp_path
    for proc in processes:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate(timeout=5)


def test_setup_and_policy_have_independent_deadlines(launcher):
    configure, launches, _ = launcher
    configure()
    policy_e2e.assert_run_policy_with_hsinference("synthetic", "model.onnx", timeout=0.1)
    assert len(launches) == 2
    assert "pip install" in launches[0][0][-1]
    assert "run_policy.py" not in launches[0][0][-1]
    assert "pip install" not in launches[1][0][-1]
    assert launches[0][1][0] == 300
    assert launches[1][1][0] == 0.1


def test_missing_inference_output_fails_with_diagnostics(launcher):
    configure, _, _ = launcher
    configure(
        output="startup diagnostic: no observations",
        install_code="print('RL FPS: setup output must not count', flush=True)\n",
    )
    with pytest.raises(AssertionError, match="startup diagnostic: no observations"):
        policy_e2e.assert_run_policy_with_hsinference("synthetic", "model.onnx", timeout=0.1)


def test_policy_crash_after_inference_output_is_not_success(launcher):
    configure, _, _ = launcher
    configure(exit_code=7)
    with pytest.raises(AssertionError, match="Policy exited.*7"):
        policy_e2e.assert_run_policy_with_hsinference("synthetic", "model.onnx", timeout=0.1)


def test_failed_setup_does_not_launch_policy(launcher):
    configure, launches, _ = launcher
    configure(install_code="print('installation failed', flush=True)\nraise SystemExit(19)\n")
    with pytest.raises(AssertionError, match="installation failed"):
        policy_e2e.assert_run_policy_with_hsinference("synthetic", "model.onnx", timeout=0.1)
    assert len(launches) == 1
    assert "run_policy.py" not in launches[0][0][-1]


def test_setup_timeout_is_bounded_and_preserves_diagnostics(launcher, monkeypatch):
    configure, launches, _ = launcher
    configure(install_code="print('setup still running', flush=True)\ntime.sleep(60)\n")
    monkeypatch.setattr(policy_e2e, "_SETUP_TIMEOUT", 0.1, raising=False)
    with pytest.raises(AssertionError, match="(?s)Setup timed out.*setup still running"):
        policy_e2e.assert_run_policy_with_hsinference("synthetic", "model.onnx", timeout=0.1)
    assert len(launches) == 1
