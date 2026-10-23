"""Headless Isaac Lab 3.0 randomization, RGB and depth regressions.

Property randomization is checked by live read-back on both engines. Camera
checks use offscreen rendering and never request desktop visualizers."""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim6

pytest.importorskip("isaaclab")
pytest.importorskip("isaaclab_physx")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402
from tests.simulators.isaacsim6.test_state_isaacsim6 import BACKENDS  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parents[1]

INSTALLED_VISUALIZERS = [""]
"""GPU regression tests capture offscreen frames without opening desktop viewers."""


@pytest.mark.parametrize("simulator", BACKENDS)
def test_dr_matrix(simulator: str, tmp_path: pathlib.Path) -> None:
    """Mass, friction, inertia and CoM randomization, verified by reading the values back."""
    # Isaac Sim's teardown can hard-terminate the process, masking the exit code and swallowing
    # late stdout, so the verdict comes from the sentinel the harness writes after its checks.
    sentinel = tmp_path / "dr_matrix_result.txt"
    run_harness(
        _HARNESS_DIR / "dr_matrix_assert.py",
        "--simulator",
        simulator,
        "--result-file",
        str(sentinel),
        label=f"{simulator}/dr-matrix",
        timeout=1200,
        result_file=sentinel,
    )


@pytest.mark.parametrize("simulator", BACKENDS)
@pytest.mark.parametrize("visualizers", INSTALLED_VISUALIZERS, ids=lambda v: v or "headless")
def test_camera_across_visualizers(simulator: str, visualizers: str, tmp_path: pathlib.Path) -> None:
    """RGB capture must produce non-blank frames under every physics x renderer combination.

    The headless case is not redundant: a ``TiledCamera`` is an offscreen RTX sensor, so Isaac
    Lab's renderer enables what it needs on demand and cameras work with no visualizer at all.
    That is worth pinning, because the obvious reading of "no visualizer" is "no pixels".

    This parametrization is also the regression test for a defect that made the whole axis
    untestable: ``isaac_visualizers`` was validated at startup and then never read, so every run
    resolved to ``kit`` regardless of configuration and five "verified" selections were one
    configuration measured five times.
    """
    args = ["--simulator", simulator]
    if visualizers:
        args += ["--visualizers", visualizers]
    sentinel = tmp_path / "camera_result.txt"
    run_harness(
        _HARNESS_DIR / "camera_assert.py",
        *args,
        "--result-file",
        str(sentinel),
        label=f"{simulator}/camera (visualizers={visualizers or 'headless'})",
        timeout=1200,
        result_file=sentinel,
    )


@pytest.mark.parametrize("simulator", BACKENDS)
def test_depth(simulator: str, tmp_path: pathlib.Path) -> None:
    """Depth capture, which shares the RTX path with RGB but its own annotator."""
    sentinel = tmp_path / "depth_result.txt"
    run_harness(
        _HARNESS_DIR / "depth_assert.py",
        "--simulator",
        simulator,
        "--result-file",
        str(sentinel),
        label=f"{simulator}/depth",
        timeout=1200,
        result_file=sentinel,
    )


@pytest.mark.parametrize("simulator", BACKENDS)
def test_world_mounted_camera(simulator: str, tmp_path: pathlib.Path) -> None:
    """A camera mounted to the environment rather than to a robot link.

    The world mount must remain fixed while the robot moves, independently in each environment.
    """
    sentinel = tmp_path / "world_camera_result.txt"
    run_harness(
        _HARNESS_DIR / "world_camera_assert.py",
        "--simulator",
        simulator,
        "--result-file",
        str(sentinel),
        label=f"{simulator}/world-camera",
        timeout=1200,
        result_file=sentinel,
    )
