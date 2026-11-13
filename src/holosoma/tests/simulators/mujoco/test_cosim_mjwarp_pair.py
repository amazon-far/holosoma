"""An MJWarp render task renders exactly what an MJWarp source publishes over the co-sim bridge."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("zmq")

pytestmark = pytest.mark.mujoco_warp
if not torch.cuda.is_available():
    pytest.skip("Co-simulation shares CUDA memory between processes", allow_module_level=True)

_ROLES = Path(__file__).with_name("cosim_pair_roles.py")
_TIMEOUT_S = 600


def _role(role: str, endpoint: str, output: Path, num_envs: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            str(_ROLES),
            role,
            "--endpoint",
            endpoint,
            "--output",
            str(output),
            "--num-envs",
            str(num_envs),
        ],
        stdout=(output / f"{role}.log").open("wb"),
        stderr=subprocess.STDOUT,
    )


@pytest.mark.parametrize("num_envs", [1, 2])
def test_mjwarp_receiver_renders_the_published_state(num_envs: int) -> None:
    # Unix socket paths are limited to 107 characters; keep the endpoint short.
    with tempfile.TemporaryDirectory(prefix="cosim-") as directory:
        output = Path(directory)
        endpoint = f"ipc://{output}/control.sock"
        receiver = _role("receiver", endpoint, output, num_envs)
        try:
            deadline = time.monotonic() + _TIMEOUT_S
            while not (output / "ready").exists():
                assert receiver.poll() is None, (output / "receiver.log").read_text()
                assert time.monotonic() < deadline, "receiver did not start"
                time.sleep(0.2)
            source = _role("source", endpoint, output, num_envs)
            assert source.wait(_TIMEOUT_S) == 0, (output / "source.log").read_text()
            assert receiver.wait(_TIMEOUT_S) == 0, (output / "receiver.log").read_text()
        finally:
            receiver.kill()
        sent = torch.load(output / "source.pt")
        rendered = torch.load(output / "receiver.pt")

    assert sent["frames"] == rendered["frames"] > 1
    torch.testing.assert_close(rendered["robot_root_states"], sent["robot_root_states"])
    torch.testing.assert_close(rendered["dof_pos"], sent["dof_pos"])
    assert rendered["rgb"].shape == sent["rgb"].shape
    assert torch.equal(rendered["rgb"], sent["rgb"])
