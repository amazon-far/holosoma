"""Live Isaac Sim mutable-pattern MultiMeshRayCaster LiDAR test."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

_HARNESS = Path(__file__).resolve().parents[1] / "lidar_assert.py"


def test_lidar(tmp_path, run_isaacsim_harness):
    result_file = tmp_path / "lidar_isaacsim.txt"
    run_isaacsim_harness(
        _HARNESS,
        "--num-envs",
        "3",
        "--result-file",
        str(result_file),
        label="isaacsim/lidar",
        timeout=700,
        result_file=result_file,
    )


def test_lidar_newton(tmp_path, run_isaacsim_harness, isaacsim_backend):
    if isaacsim_backend != "isaaclab3":
        pytest.skip("Newton LiDAR requires the explicit Isaac Lab 3 profile")
    result_file = tmp_path / "lidar_newton.txt"
    result = run_isaacsim_harness(
        _HARNESS,
        "--physics-backend",
        "newton",
        "--num-envs",
        "3",
        "--result-file",
        str(result_file),
        label="isaacsim3/newton-lidar",
        timeout=900,
        result_file=result_file,
    )
    assert "narrowphase overflow" not in (result.stdout + result.stderr).lower(), result.stdout[-3000:]


@pytest.mark.parametrize("physics", ["physx", "newton"])
def test_lidar_ovrtx_runtime(physics, tmp_path, run_isaacsim_harness, isaacsim_backend):
    if isaacsim_backend != "isaaclab3":
        pytest.skip("OVRTX LiDAR requires the explicit Isaac Lab 3 profile")
    result_file = tmp_path / "lidar-ovrtx.txt"
    run_isaacsim_harness(
        _HARNESS,
        "--physics-backend",
        physics,
        "--renderer",
        "ovrtx",
        "--num-envs",
        "3",
        "--result-file",
        str(result_file),
        label=f"isaacsim3/{physics}/ovrtx-lidar",
        timeout=900,
        result_file=result_file,
    )
