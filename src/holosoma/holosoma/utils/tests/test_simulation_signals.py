"""Process-level tests for simulation signal semantics."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.no_sim,
    pytest.mark.skipif(os.name != "posix", reason="POSIX signal semantics required"),
]

_CHILD = textwrap.dedent(
    """
    import time
    from types import SimpleNamespace

    import holosoma.run_sim as runner

    class Env:
        sim = object()

        def close(self):
            print("CLEANUP", flush=True)

    def setup(_config, *, device):
        return Env(), device, None

    def initialize(self):
        pass

    def run(self):
        try:
            print("READY", flush=True)
            time.sleep(30)
        except KeyboardInterrupt:
            print("KEYBOARD_INTERRUPT", flush=True)
            raise

    runner.setup_simulation_environment = setup
    runner.DirectSimulation.initialize = initialize
    runner.DirectSimulation.run = run
    config = SimpleNamespace(
        device="cpu",
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="test")),
        simulator=SimpleNamespace(_target_="test.Simulator"),
        terrain=SimpleNamespace(terrain_term=SimpleNamespace(mesh_type="plane", func="test")),
    )
    runner.run_simulation(config)
    """
)


def _signal_child(sig: signal.Signals) -> tuple[int, str, str]:
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "READY"
    child.send_signal(sig)
    stdout, stderr = child.communicate(timeout=10)
    return child.returncode, stdout, stderr


def test_sigterm_unwinds_cleanup_and_exits_143() -> None:
    returncode, stdout, _stderr = _signal_child(signal.SIGTERM)

    assert returncode == 128 + signal.SIGTERM
    assert stdout.splitlines() == ["CLEANUP"]


def test_sigint_remains_keyboard_interrupt_and_unwinds_cleanup() -> None:
    returncode, stdout, stderr = _signal_child(signal.SIGINT)

    assert returncode != 0
    assert stdout.splitlines() == ["KEYBOARD_INTERRUPT", "CLEANUP"]
    assert "KeyboardInterrupt" in stderr
