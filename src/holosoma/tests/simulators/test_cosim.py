"""Co-simulation across processes: a leader's camera observations rendered by a follower.

The follower runs ``run_sim`` with the cosim-follower plugin in its own process, activated with
its simulator's env setup script; the leader runs ``cosim_assert.py`` under this interpreter, so
run each pair from the leader simulator's env. Pairs that need both the MuJoCo and Isaac Sim envs
are marked ``cross_sim``; run them on a machine with both, e.g. ``-m cross_sim`` from either env.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests.simulators._run_harness import run_harness

_REPO = Path(__file__).resolve().parents[4]
_HARNESS = Path(__file__).resolve().parent / "cosim_assert.py"
_READY = "cosim follower bound"
_ENV = {"mujoco": "mujoco", "mjwarp": "mujoco", "isaacsim": "isaacsim"}
_TYPE = {"mujoco": "mujoco", "mjwarp": "mujoco", "isaacsim": "isaacsim"}


def _pair(leader: str, follower: str) -> pytest.ParameterSet:
    marks: list[pytest.MarkDecorator]
    if _ENV[leader] != _ENV[follower]:
        marks = [pytest.mark.cross_sim]
    elif leader == "isaacsim":
        marks = [pytest.mark.isaacsim]
    elif leader == follower == "mujoco":
        marks = [pytest.mark.mujoco, pytest.mark.mujoco_classic]
    else:
        marks = [pytest.mark.mujoco, pytest.mark.mujoco_warp]
    return pytest.param(leader, follower, marks=marks, id=f"{leader}-{follower}")


PAIRS = [_pair(leader, follower) for leader in _ENV for follower in _ENV]


def _start_follower(simulator: str, endpoint: str, log: Path) -> subprocess.Popen:
    command = (
        f"source scripts/source_{_ENV[simulator]}_setup.sh >/dev/null && "
        f"exec python src/holosoma/holosoma/run_sim.py simulator:{simulator} robot:g1-29dof sensor.head:g1-head "
        f"plugin.cosim:cosim-follower --plugin.cosim.endpoint {endpoint} "
        "--simulator.config.sim.kinematic_playback True --simulator.config.sim.fps 50 "
        "--simulator.config.sim.control_decimation 1 --simulator.config.sim.render_interval 1 "
        "--simulator.config.bridge.enabled False --time-scale 0 --training.headless True"
    )
    with log.open("w") as out:
        return subprocess.Popen(
            ["/bin/bash", "-c", command],
            cwd=_REPO,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )


def _wait_ready(follower: subprocess.Popen, log: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while _READY not in log.read_text():
        if follower.poll() is not None:
            pytest.fail(f"follower exited ({follower.returncode}) before binding:\n{log.read_text()[-3000:]}")
        if time.monotonic() > deadline:
            pytest.fail(f"follower did not bind within {timeout} s:\n{log.read_text()[-3000:]}")
        time.sleep(0.5)


@pytest.mark.parametrize(("leader", "follower"), PAIRS)
def test_leader_observes_follower_renders(leader: str, follower: str, tmp_path: Path) -> None:
    endpoint = f"ipc://{tmp_path}/cosim"
    log = tmp_path / "follower.log"
    process = _start_follower(follower, endpoint, log)
    try:
        _wait_ready(process, log, timeout=600)
        result_file = tmp_path / "result.txt"
        run_harness(
            _HARNESS,
            "--simulator",
            leader,
            "--follower-type",
            _TYPE[follower],
            "--endpoint",
            endpoint,
            "--result-file",
            str(result_file),
            label=f"cosim {leader} -> {follower}",
            timeout=900,
            result_file=result_file,
        )
        process.wait(timeout=120)
        assert "cosim leader closed" in log.read_text(), log.read_text()[-3000:]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
