"""Native Lab 3 supports independently declared physics and camera rendering."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim


@pytest.mark.parametrize(
    ("physics", "renderer"),
    [
        ("physx", "isaac_rtx"),
        ("newton", "isaac_rtx"),
        ("physx", "ovrtx"),
        ("newton", "ovrtx"),
        ("physx", "newton_warp"),
        ("newton", "newton_warp"),
    ],
)
@pytest.mark.parametrize("decimations", ["1,5", "2,3", "1,5,2", "3,3", "3"])
def test_native_camera_observation_cadence(
    physics, renderer, decimations, isaaclab_backend, run_isaacsim_harness, tmp_path
):
    if isaaclab_backend != "isaaclab3":
        pytest.skip("Native camera batching requires Isaac Lab 3.")
    result_file = tmp_path / "camera-cadence.txt"
    result = run_isaacsim_harness(
        Path(__file__).resolve().parents[1] / "camera_cadence_assert.py",
        "--physics",
        physics,
        "--renderer",
        renderer,
        "--decimations",
        decimations,
        "--result-file",
        str(result_file),
        label=f"isaaclab3/{physics}/{renderer}/camera-cadence-{decimations}",
        timeout=900,
        result_file=result_file,
    )
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize("physics", ["physx", "newton"])
@pytest.mark.parametrize("renderer", ["isaac_rtx", "ovrtx", "newton_warp"])
def test_camera_cadence_with_movie(physics, renderer, isaaclab_backend, run_isaacsim_harness, tmp_path):
    if isaaclab_backend != "isaaclab3":
        pytest.skip("Native camera batching requires Isaac Lab 3.")
    result_file = tmp_path / "camera-cadence-movie.txt"
    result = run_isaacsim_harness(
        Path(__file__).resolve().parents[1] / "camera_cadence_assert.py",
        "--physics",
        physics,
        "--renderer",
        renderer,
        "--decimations",
        "1,5,2",
        "--record",
        str(tmp_path / "movies"),
        "--result-file",
        str(result_file),
        label=f"isaaclab3/{physics}/{renderer}/camera-cadence-movie",
        timeout=900,
        result_file=result_file,
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_newton_without_cuda_graph_uses_shared_solver_settings(isaaclab_backend, run_isaacsim_harness, tmp_path):
    if isaaclab_backend != "isaaclab3":
        pytest.skip("Newton requires Isaac Lab 3")
    result_file = tmp_path / "newton-without-cuda-graph.txt"
    run_isaacsim_harness(
        Path(__file__).resolve().parents[1] / "isaaclab3_schema_assert.py",
        "--physics",
        "newton",
        "--renderer",
        "newton_warp",
        "--disable-cuda-graph",
        "--result-file",
        str(result_file),
        label="isaaclab3/newton/newton_warp/no-cuda-graph",
        timeout=900,
        result_file=result_file,
    )


@pytest.mark.parametrize(
    ("physics", "renderer", "collision_source"),
    [("newton", "newton_warp", "mujoco"), ("newton", "ovrtx", "newton"), ("physx", "ovrtx", "mujoco")],
)
def test_native_schema_solver_and_renderer_overrides(
    physics, renderer, collision_source, isaaclab_backend, run_isaacsim_harness, tmp_path
):
    if isaaclab_backend != "isaaclab3":
        pytest.skip("native schema fragments require Isaac Lab 3")
    result_file = tmp_path / "native-schema.txt"
    run_isaacsim_harness(
        Path(__file__).resolve().parents[1] / "isaaclab3_schema_assert.py",
        "--physics",
        physics,
        "--renderer",
        renderer,
        "--collision-source",
        collision_source,
        "--result-file",
        str(result_file),
        label=f"isaaclab3/{physics}/{renderer}/{collision_source}/schema",
        timeout=900,
        result_file=result_file,
    )


@pytest.mark.parametrize("physics", ["physx", "newton"])
@pytest.mark.parametrize("renderer", ["isaac_rtx", "newton_warp", "ovrtx"])
def test_physics_and_camera_renderer_matrix(physics, renderer, isaaclab_backend, run_isaacsim_harness, tmp_path):
    if isaaclab_backend != "isaaclab3":
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
    if renderer != "isaac_rtx":
        # These harnesses intentionally exercise ordinary native teardown.
        assert result.returncode == 0, result.stderr[-3000:]
    if physics == "newton":
        assert "narrowphase overflow" not in (result.stdout + result.stderr).lower(), result.stdout[-3000:]


@pytest.mark.parametrize("physics", ["physx", "newton"])
@pytest.mark.parametrize("renderer", ["isaac_rtx", "newton_warp", "ovrtx"])
def test_native_logger_movie(physics, renderer, isaaclab_backend, run_isaacsim_harness, tmp_path):
    if isaaclab_backend != "isaaclab3":
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
def test_physics_only_startup(physics, renderer, isaaclab_backend, run_isaacsim_harness, tmp_path):
    if isaaclab_backend != "isaaclab3":
        pytest.skip("standalone startup requires the explicit Isaac Lab 3 profile")
    result_file = tmp_path / "physics-only.txt"
    result = run_isaacsim_harness(
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
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize("physics", ["physx", "newton"])
def test_ovrtx_fisheye_projection(physics, isaaclab_backend, run_isaacsim_harness, tmp_path):
    if isaaclab_backend != "isaaclab3":
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
