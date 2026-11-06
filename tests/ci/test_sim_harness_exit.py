"""Verify real process verdicts survive a teardown callback that would replace them."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    ("action", "status", "error_text"),
    [
        ("return 0", 0, None),
        ("return 19", 19, None),
        ("raise RuntimeError('controlled harness failure')", 1, "RuntimeError: controlled harness failure"),
        ("raise KeyboardInterrupt", 130, None),
        ("raise SystemExit(17)", 17, None),
        ("raise SystemExit()", 0, None),
        ("raise SystemExit('controlled explicit failure')", 1, "controlled explicit failure"),
    ],
)
def test_hard_exit_preserves_real_child_verdict(action: str, status: int, error_text: str | None) -> None:
    script = f"""
import atexit
import os
import sys

from tests.simulators._sim_harness import run_and_hard_exit

# Model the native launcher teardown replacing a normal exception exit with zero.
atexit.register(lambda: os._exit(0))

def main():
    print("buffered harness stdout", end="")
    print("buffered harness stderr", end="", file=sys.stderr)
    {action}

run_and_hard_exit(main)
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert child.returncode == status, child.stderr
    assert "buffered harness stdout" in child.stdout
    assert "buffered harness stderr" in child.stderr
    if error_text is not None:
        assert error_text in child.stderr
    if action.startswith("raise RuntimeError"):
        assert "Traceback (most recent call last)" in child.stderr
