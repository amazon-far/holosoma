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
    from holosoma.config_types import env as env_types
    from holosoma.config_values.simulator import SIMULATOR_REGISTRY
    from holosoma.envs.simulation_task import SimulationTask
    from holosoma.simulator import session
    from holosoma.utils.simulator_config import SimulatorType
    from holosoma.utils.tests._simulation_error_child import EnvConfigStub

    class Task(SimulationTask):
        def __init__(self, _config):
            pass

        def run(self):
            try:
                print("READY", flush=True)
                time.sleep(30)
            except KeyboardInterrupt:
                print("KEYBOARD_INTERRUPT", flush=True)
                raise

        def close(self):
            print("CLEANUP", flush=True)

    session.setup_simulator_imports = lambda _config: None
    session.resolve_simulation_device = lambda *_args: "cpu"
    session.get_simulator_type = lambda: SimulatorType.MUJOCO
    session.get_class = lambda _target: Task
    env_config = EnvConfigStub(env_class="test.Task", training=SimpleNamespace(headless=True))
    env_types.get_run_sim_env_config = lambda _config: env_config
    config = SimpleNamespace(
        device="cpu",
        # A real preset, so the fields run_simulation reads (``config.name``, ``config.mujoco_backend``)
        # have the shapes they have in production.
        simulator=SIMULATOR_REGISTRY.get("mujoco"),
        create_task=lambda: session.task_session(env_config, SimulationTask),
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
