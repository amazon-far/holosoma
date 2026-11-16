"""Bound the training E2E harness with real processes, without a simulator or GPUs."""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_sim

SPEC = importlib.util.spec_from_file_location("training_e2e", Path(__file__).with_name("test_training.py"))
assert SPEC is not None and SPEC.loader is not None
training_e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_e2e)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_teed_call_returns_the_exit_code_and_tees_output():
    out, err = io.BytesIO(), io.BytesIO()
    cmd = [sys.executable, "-c", "import sys; print('to-out'); print('to-err', file=sys.stderr); sys.exit(3)"]

    assert training_e2e.teed_call(cmd, stdout=out, stderr=err, timeout=30) == 3
    assert out.getvalue().strip() == b"to-out"
    assert err.getvalue().strip() == b"to-err"


def test_teed_call_times_out_and_stops_the_whole_process_group(tmp_path, monkeypatch):
    # A launcher with a child rank that inherits the pipes, like torch.distributed.run.
    monkeypatch.setattr(training_e2e, "_STOP_TIMEOUT", 2)
    pid_file = tmp_path / "rank.pid"
    launcher = textwrap.dedent(
        f"""
        import subprocess, sys, time
        rank = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pid_file)!r}, "w").write(str(rank.pid))
        print("launched", flush=True)
        time.sleep(120)
        """
    )
    cmd = [sys.executable, "-c", launcher]
    out = io.BytesIO()

    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        training_e2e.teed_call(cmd, stdout=out, stderr=io.BytesIO(), timeout=3)
    elapsed = time.monotonic() - start

    assert excinfo.value.cmd == cmd
    assert excinfo.value.timeout == 3
    assert elapsed < 20, f"teed_call took {elapsed:.1f}s to give up"
    assert b"launched" in out.getvalue()
    rank_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(rank_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(rank_pid), "the child rank outlived the timeout"


@pytest.mark.parametrize("multi_gpu", [False, True])
def test_workflow_bounds_train_and_eval(monkeypatch, multi_gpu):
    calls = {}

    def fake_teed_call(cmd, stdout=None, stderr=None, timeout=None, env=None):
        calls["train"] = {"timeout": timeout, "env": env}
        stdout.write(b"Saving checkpoint to logs/x/y/model_2.pt\n")
        return 0

    def fake_check_call(cmd, timeout=None):
        calls["eval"] = {"timeout": timeout}
        return 0

    monkeypatch.setattr(training_e2e, "teed_call", fake_teed_call)
    monkeypatch.setattr(training_e2e.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(training_e2e.torch.cuda, "device_count", lambda: 4)

    training_e2e.assert_training_and_eval_workflow("g1-29dof", multi_gpu=multi_gpu)

    assert calls["train"]["timeout"] == 900
    assert calls["eval"]["timeout"] == 600
    if multi_gpu:
        dist_timeout = int(calls["train"]["env"]["TORCH_DIST_INIT_TIMEOUT_S"])
        assert dist_timeout == 300
        assert dist_timeout < calls["train"]["timeout"], "ranks must report before the harness kills them"
    else:
        assert calls["train"]["env"] is None
