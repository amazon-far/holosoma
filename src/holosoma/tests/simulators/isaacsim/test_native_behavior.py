"""Rendered outcomes through the shared geometry and mount harnesses, with real Newton dynamics."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim
_HARNESS_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("renderer", ["isaac_rtx", "ovrtx"])
@pytest.mark.parametrize(
    ("harness", "args"),
    [
        ("camera_geometry_assert.py", []),
        ("camera_follow_assert.py", []),
        ("camera_follow_assert.py", ["--articulated-link"]),
        ("camera_actor_mount_assert.py", []),
    ],
    ids=["geometry", "robot-mount", "articulated-link-mount", "actor-mount"],
)
def test_newton_rendered_outcomes(harness, args, renderer, isaacsim_backend, run_isaacsim_harness, tmp_path):
    if isaacsim_backend != "isaacsim6":
        pytest.skip("Newton renderer selection requires Isaac Lab 3")
    result_file = tmp_path / "rendered-outcomes.txt"
    result = run_isaacsim_harness(
        _HARNESS_DIR / harness,
        "--physics-backend",
        "newton",
        "--renderer",
        renderer,
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        *args,
        label=f"isaacsim6/newton/{renderer}/{harness}/{args}",
        timeout=900,
        result_file=result_file,
    )
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize("scene", ["usd-box", "static-box", "velocity-box", "friction-slide", "multibody-override"])
def test_newton_scene_spawn(scene, isaacsim_backend, run_isaacsim_harness, tmp_path):
    if isaacsim_backend != "isaacsim6":
        pytest.skip("Newton requires Isaac Lab 3")
    result_file = tmp_path / "scene-spawn.txt"
    run_isaacsim_harness(
        _HARNESS_DIR / "scene_spawn_assert.py",
        "--physics-backend",
        "newton",
        "--scene",
        scene,
        "--num-envs",
        "4",
        "--steps",
        "150",
        "--result-file",
        str(result_file),
        label=f"isaacsim6/newton/{scene}",
        timeout=900,
        result_file=result_file,
    )
