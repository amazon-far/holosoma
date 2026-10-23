"""Live Isaac Sim 6.1 domain randomization and camera capture, across both axes.

Isaac Lab 3.0 made the physics engine *and* the renderer independent configuration choices, so
these cover both: domain randomization per physics backend, and cameras per visualizer selection.

Domain randomization is the highest-value sim test in this lane. Its terms read and write mass,
inertia, centre of mass and shape friction through what used to be the PhysX tensor view, and
Newton's ``ArticulationView`` provides *none* of those seven members -- the adapter in
``physics_properties`` reconciles three separate layout differences (a per-articulation axis,
``mat33`` inertia, host-vs-device placement), each of which produced a distinct failure during the
port and none of which any CPU test can reach. ``dr_matrix_assert`` verifies by read-back, so it
catches a write that silently lands nowhere.
"""

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

INSTALLED_VISUALIZERS = ["", "kit", "newton"]
"""Visualizer selections this environment can actually run.

``rerun`` and ``viser`` are valid *names* -- ``capabilities.KNOWN_VISUALIZERS`` keeps them, and
``--visualizer`` accepts them -- but Isaac Lab resolves a visualizer by import path
(``isaaclab_visualizers.rerun.rerun_visualizer:RerunVisualizer``) and raises
``ModuleNotFoundError`` when the backing distribution is absent, which it is here. Listing them
would test the environment rather than the lane; ``setup_isaacsim6.sh`` installing the extra is
what would change that.
"""


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

    Kept distinct from the robot-link mount because the actor mount -- the third kind -- renders
    blank on this lane for reasons still not established, so which mounts are proven matters.
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
