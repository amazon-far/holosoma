"""Native Lab 3 supports independently declared physics and camera rendering."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim


@pytest.mark.parametrize("physics", ["physx", "newton"])
@pytest.mark.parametrize("renderer", ["isaac_rtx", "newton_warp"])
def test_physics_and_camera_renderer_matrix(physics, renderer, isaacsim_backend, run_isaacsim_harness, tmp_path):
    if isaacsim_backend != "isaaclab3":
        pytest.skip("physics/renderer selection requires the explicit Isaac Lab 3 profile")
    result_file = tmp_path / "native-options.txt"
    result = run_isaacsim_harness(
        Path(__file__).resolve().parents[1] / "isaaclab3_options_assert.py",
        "--physics",
        physics,
        "--renderer",
        renderer,
        "--result-file",
        str(result_file),
        label=f"isaaclab3/{physics}/{renderer}",
        timeout=900,
        result_file=result_file,
    )
    if physics == "newton":
        assert "narrowphase overflow" not in (result.stdout + result.stderr).lower(), result.stdout[-3000:]
