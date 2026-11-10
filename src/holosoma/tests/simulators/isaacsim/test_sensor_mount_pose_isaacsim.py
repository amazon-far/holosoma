"""Run the mounting-pose behavioral harness with Isaac Sim sensors and Fabric."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim
pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires CUDA", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "sensor_mount_pose_assert.py"


@pytest.mark.parametrize("num_envs", [1, 3])
def test_runtime_sensor_mounts(tmp_path, num_envs):
    result = tmp_path / "mounts_isaacsim.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        str(num_envs),
        "--result-file",
        str(result),
        label="isaacsim/runtime-sensor-mounts",
        timeout=900,
        result_file=result,
    )
