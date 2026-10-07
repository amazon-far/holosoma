# mypy: disable-error-code="arg-type"
"""Live Isaac Gym terrain LiDAR test."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("isaacgym")

from holosoma.config_types.sensor import (
    CameraSensorConfig,
    LidarBodyFilterConfig,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.simulator.isaacgym.sensor_setup import (
    _mount_pose,
    create_sensors,
    register_lidars,
    render_lidars,
)
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch
from tests.simulators._run_harness import run_harness

pytestmark = pytest.mark.isaacgym

_HARNESS = Path(__file__).resolve().parents[1] / "lidar_assert.py"


def test_camera_only_sensor_manager_never_looks_up_lidar_terrain() -> None:
    manager = SensorManager(device="cpu", control_hz=50.0)
    manager.register_camera("camera", CameraSensorConfig(mount=SensorMountConfig(target_kind="world")))

    def unexpected_terrain_lookup(name: str) -> None:
        raise AssertionError(f"camera-only render attempted terrain lookup for {name!r}")

    sim = SimpleNamespace(
        sensor_manager=manager,
        terrain_manager=SimpleNamespace(get_state=unexpected_terrain_lookup),
    )
    render_lidars(sim)
    assert manager.last_lidar_due == set()


def test_robot_link_lidar_mount_uses_its_body_index_cache() -> None:
    half_sqrt_two = math.sqrt(0.5)
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link",
            target="pelvis",
            position=[0.1, -0.2, 0.3],
            orientation=[half_sqrt_two, half_sqrt_two, 0.0, 0.0],
        )
    )
    sim = SimpleNamespace(
        sensor_config={"scan": lidar},
        sensor_manager=None,
        device="cpu",
        simulator_config=SimpleNamespace(sim=SimpleNamespace(fps=100.0, control_decimation_steps=2)),
        find_rigid_body_indice=lambda name: 1 if name == "pelvis" else -1,
        rigid_body_pos_w=torch.tensor([[[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]]]),
        rigid_body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt_two, half_sqrt_two]]]),
        num_envs=1,
    )

    manager = SensorManager(device="cpu", control_hz=50.0)
    sim.sensor_manager = manager
    register_lidars(sim, manager)
    record = sim.sensor_manager.get_lidar("scan")
    origin, orientation = _mount_pose(sim, record)

    assert record.backend_cache["isaacgym_body_index"] == 1
    torch.testing.assert_close(origin, torch.tensor([[2.2, 3.1, 4.3]]))
    torch.testing.assert_close(orientation, torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    del record.backend_cache["isaacgym_body_index"]
    with pytest.raises(RuntimeError, match="mount-body index"):
        _mount_pose(sim, record)


def test_robot_filter_is_configuration_compatible_terrain_only_noop() -> None:
    lidar = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="world", position=[0.1, -0.2, 0.3]),
        body_filter=LidarBodyFilterConfig(target_kind="robot"),
    )

    def unexpected_body_lookup(name: str) -> int:
        raise AssertionError(f"terrain-only robot filter looked up body {name!r}")

    sim = SimpleNamespace(
        sensor_config={"scan": lidar},
        device="cpu",
        num_envs=2,
        env_origins=torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]]),
        find_rigid_body_indice=unexpected_body_lookup,
    )
    manager = SensorManager(device="cpu", control_hz=50.0)

    register_lidars(sim, manager)
    record = manager.get_lidar("scan")
    origin, orientation = _mount_pose(sim, record)

    assert record.config.body_filter == LidarBodyFilterConfig(target_kind="robot")
    assert "isaacgym_body_index" not in record.backend_cache
    torch.testing.assert_close(
        origin,
        torch.tensor([[0.1, -0.2, 0.3], [2.1, 2.8, 4.3]]),
    )
    torch.testing.assert_close(
        orientation,
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
    )


def test_create_sensors_registers_mixed_camera_and_lidar_rig() -> None:
    sim = SimpleNamespace(
        sensor_config={
            "camera": CameraSensorConfig(mount=SensorMountConfig(target_kind="world")),
            "scan": LidarSensorConfig(mount=SensorMountConfig(target_kind="world")),
        },
        sensor_manager=None,
        device="cpu",
        simulator_config=SimpleNamespace(sim=SimpleNamespace(fps=100.0, control_decimation_steps=2)),
        find_rigid_body_indice=lambda _name: -1,
    )

    create_sensors(sim)

    assert sim.sensor_manager.names == ["camera", "scan"]
    assert sim.sensor_manager.has_camera("camera")
    assert sim.sensor_manager.has_lidar("scan")


def test_lidar(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("IsaacGym requires a CUDA device")
    result_file = tmp_path / "lidar_isaacgym.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        "3",
        "--result-file",
        str(result_file),
        label="isaacgym/lidar",
        timeout=600,
        result_file=result_file,
    )
