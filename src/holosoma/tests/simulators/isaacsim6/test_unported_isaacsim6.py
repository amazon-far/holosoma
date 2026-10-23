"""The three capabilities the Isaac Sim 6.1 lane does not implement must fail loudly.

Asserting a *rejection* is worth a GPU slot here because the alternative to raising is degrading,
and degrading is what these would do: a LiDAR that ignores its configured scan pattern, a camera
parented to a pose that never updates, a recorder that stops following the robot. None of those
fail -- they produce plausible output that disagrees with the configuration, which for a
perception sensor is the worst available outcome.

``capabilities.validate`` is unit-tested on the CPU lane, so what these add is that the gate is
actually *reached* on a live run: wired into the real construction path, with the real sensor
config, before any simulator state exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim6

pytest.importorskip("isaaclab")
pytest.importorskip("isaaclab_physx")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires a CUDA device", allow_module_level=True)

_HARNESS_DIR = Path(__file__).resolve().parents[1]


def _run(harness: str, *args: str) -> subprocess.CompletedProcess:
    """Run a harness and return the result without asserting success.

    Deliberately not ``_run_harness.run_harness``: that asserts exit 0, and here a non-zero exit
    carrying the right message *is* the expected outcome.
    """
    import sys

    return subprocess.run(
        [sys.executable, "-u", str(_HARNESS_DIR / harness), *args],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


@pytest.mark.parametrize(
    ("harness", "extra", "expected"),
    [
        ("lidar_assert.py", (), "LiDAR"),
        ("camera_actor_mount_assert.py", (), "actor"),
    ],
    ids=["lidar", "actor-mounted-camera"],
)
def test_unported_capability_is_refused(harness: str, extra: tuple[str, ...], expected: str) -> None:
    """The run must stop with a message naming the capability and the alternative.

    Matching on the capability name rather than on ``NotImplementedError`` alone: the point of
    these messages is that a user learns *what* is missing and that ``simulator:isaacsim`` still
    does it, so a bare exception type would pass while being useless.
    """
    result = _run(harness, "--simulator", "isaacsim6", *extra)
    combined = result.stdout + result.stderr

    # Judged on the raised message, NOT on the return code. Isaac Sim's teardown corrupts the
    # exit status -- measured: these harnesses exit 0 even though `capabilities.validate` raised
    # NotImplementedError and nothing ran. That is the same reason the rest of this suite reads a
    # `--result-file` sentinel instead of `returncode`, and asserting `returncode != 0` here made
    # both of these fail against a lane that was behaving correctly.
    assert "NotImplementedError" in combined, combined[-3000:]
    assert expected.lower() in combined.lower(), combined[-3000:]
    assert "simulator:isaacsim" in combined, combined[-3000:]

    # And it must not also claim success: a gate that raises while the harness still prints its
    # pass verdict would be a worse outcome than either alone.
    assert "PASS" not in combined, combined[-3000:]
