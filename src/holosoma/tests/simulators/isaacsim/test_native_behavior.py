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


@pytest.mark.parametrize(
    "scene",
    ["usd-box", "g1-largebox", "static-box", "static-box-usd", "velocity-box", "friction-slide", "multibody-override"],
)
def test_newton_scene_spawn(scene, isaacsim_backend, run_isaacsim_harness, tmp_path):
    if isaacsim_backend != "isaacsim6":
        pytest.skip("Newton requires Isaac Lab 3")
    result_file = tmp_path / "scene-spawn.txt"
    try:
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
    except AssertionError as error:
        # Keep exercising URDF loading. Only the pinned SDK's known view-binding
        # error is expected; assertion failures and other import errors still fail.
        actor = {"g1-largebox": "box"}.get(scene)
        expected = f"No articulations matching pattern '/World/envs/env_*/{actor}/Geometry/baseLink'"
        if actor and expected in str(error) and "isaaclab_newton/assets/rigid_object/rigid_object.py" in str(error):
            pytest.xfail("Pinned Newton URDF rigid-object view binds the body path against an asset-root articulation")
        raise


@pytest.mark.parametrize("renderer", ["isaac_rtx", "ovrtx"])
@pytest.mark.parametrize("scenario", ["collide-into-fixed", "momentum-transfer"])
def test_newton_physics_with_active_movie(scenario, renderer, isaacsim_backend, run_isaacsim_harness, tmp_path):
    if isaacsim_backend != "isaacsim6":
        pytest.skip("Newton requires Isaac Lab 3")
    result_file = tmp_path / "rendered-physics.txt"
    run_isaacsim_harness(
        _HARNESS_DIR / "behavior_assert.py",
        "--physics-backend",
        "newton",
        "--renderer",
        renderer,
        "--scenario",
        scenario,
        "--robot",
        "onelink-box" if scenario == "collide-into-fixed" else "g1-29dof",
        "--num-envs",
        "4",
        "--record",
        str(tmp_path / "movies"),
        "--result-file",
        str(result_file),
        label=f"isaacsim6/newton/{renderer}/{scenario}/active-movie",
        timeout=900,
        result_file=result_file,
    )


@pytest.mark.parametrize(
    "renderer", [None, "isaac_rtx", "ovrtx", "newton_warp"], ids=["physics-only", "rtx", "ovrtx", "warp"]
)
def test_newton_fixed_fixtures(renderer, isaacsim_backend, run_isaacsim_harness, tmp_path):
    """Exact fixed topology plus contacts, relocation, isolation and reset outcomes."""
    if isaacsim_backend != "isaacsim6":
        pytest.skip("Newton requires Isaac Lab 3")
    result_file = tmp_path / "fixed-fixtures.txt"
    run_isaacsim_harness(
        _HARNESS_DIR / "fixed_object_assert.py",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        *(["--renderer", renderer] if renderer else []),
        label=f"isaacsim6/newton/{renderer or 'physics-only'}/fixed-fixtures",
        timeout=900,
        result_file=result_file,
    )


@pytest.mark.parametrize("renderer", [None, "isaac_rtx", "ovrtx"], ids=["physics-only", "rtx", "ovrtx"])
def test_newton_uca1_fixed_fixtures(renderer, isaacsim_backend, run_isaacsim_harness, tmp_path):
    """Full external apartment dataset; skip explicitly when its assets are absent."""
    if isaacsim_backend != "isaacsim6":
        pytest.skip("Newton requires Isaac Lab 3")
    scene_config = Path("/session/resolved-config.json")
    if not scene_config.is_file():
        pytest.skip("UCA1 dataset requires the existing /session/resolved-config.json and referenced assets")
    result_file = tmp_path / "uca1-fixtures.txt"
    run_isaacsim_harness(
        _HARNESS_DIR / "apartment_fixed_assert.py",
        "--scene-config",
        str(scene_config),
        "--num-envs",
        "2",
        "--result-file",
        str(result_file),
        *(["--renderer", renderer] if renderer else []),
        label=f"isaacsim6/newton/{renderer or 'physics-only'}/uca1-fixed-fixtures",
        timeout=900,
        result_file=result_file,
    )
