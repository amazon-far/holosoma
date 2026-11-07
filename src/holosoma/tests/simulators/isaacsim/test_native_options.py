"""Native Lab 3 supports independently declared physics and camera rendering."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim


@pytest.mark.parametrize("physics", ["physx", "newton"])
@pytest.mark.parametrize("renderer", ["isaac_rtx", "newton_warp", "ovrtx"])
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


@pytest.mark.parametrize("physics", ["physx", "newton"])
@pytest.mark.parametrize("renderer", ["newton_warp", "ovrtx"])
def test_native_logger_movie(physics, renderer, isaacsim_backend, run_isaacsim_harness, tmp_path):
    if isaacsim_backend != "isaaclab3":
        pytest.skip("OVRTX requires the explicit Isaac Lab 3 profile")
    result_file = tmp_path / "native-movie.txt"
    run_isaacsim_harness(
        Path(__file__).resolve().parents[1] / "scene_spawn_assert.py",
        "--physics-backend",
        physics,
        "--renderer",
        renderer,
        "--num-envs",
        "2",
        "--record",
        str(tmp_path / "movies"),
        "--probe-video-freshness",
        "--result-file",
        str(result_file),
        label=f"isaaclab3/{physics}/{renderer}-movie",
        timeout=900,
        result_file=result_file,
    )


@pytest.mark.parametrize("physics", ["physx", "newton"])
@pytest.mark.parametrize("renderer", ["isaac_rtx", "newton_warp", "ovrtx"])
def test_physics_only_startup(physics, renderer, isaacsim_backend, run_isaacsim_harness, tmp_path):
    if isaacsim_backend != "isaaclab3":
        pytest.skip("standalone startup requires the explicit Isaac Lab 3 profile")
    result_file = tmp_path / "physics-only.txt"
    run_isaacsim_harness(
        Path(__file__).resolve().parents[1] / "isaaclab3_startup_assert.py",
        "--physics",
        physics,
        "--renderer",
        renderer,
        "--result-file",
        str(result_file),
        label=f"isaaclab3/{physics}/{renderer}/physics-only",
        timeout=900,
        result_file=result_file,
    )


@pytest.mark.parametrize("physics", ["physx", "newton"])
def test_ovrtx_fisheye_projection(physics, isaacsim_backend, run_isaacsim_harness, tmp_path):
    if isaacsim_backend != "isaaclab3":
        pytest.skip("OVRTX requires the explicit Isaac Lab 3 profile")
    result_file = tmp_path / "ovrtx-fisheye.txt"
    run_isaacsim_harness(
        Path(__file__).resolve().parents[1] / "camera_projection_assert.py",
        "--physics-backend",
        physics,
        "--renderer",
        "ovrtx",
        "--num-envs",
        "2",
        "--result-file",
        str(result_file),
        label=f"isaaclab3/{physics}/ovrtx-fisheye",
        timeout=900,
        result_file=result_file,
    )
