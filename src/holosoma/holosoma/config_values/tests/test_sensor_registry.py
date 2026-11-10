"""CLI resolution tests for the heterogeneous mounted-sensor registry."""

from __future__ import annotations

import pytest

from holosoma.config_types.plugin import ROS2PointCloudPluginConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.sensor import (
    CameraSensorConfig,
    CustomLidarRayPatternConfig,
    GridLidarRayPatternConfig,
    LidarSensorConfig,
)
from holosoma.config_values.sensor import SENSOR_REGISTRY
from holosoma.utils.config_registry import parse_config

pytestmark = pytest.mark.no_sim


def test_sensor_registry_contains_camera_and_lidar_presets() -> None:
    assert isinstance(SENSOR_REGISTRY["g1-head"], CameraSensorConfig)
    assert isinstance(SENSOR_REGISTRY["g1-pelvis-lidar"], LidarSensorConfig)
    assert isinstance(SENSOR_REGISTRY["custom-lidar"].pattern, CustomLidarRayPatternConfig)


def test_cli_composes_camera_lidar_and_pointcloud_route() -> None:
    config = parse_config(
        RunSimConfig,
        args=[
            "simulator:mujoco",
            "sensor.head:g1-head",
            "sensor.front_scan:g1-pelvis-lidar",
            "--sensor.front-scan.pattern.vertical-angles=[-15,0,15]",
            "--sensor.front-scan.update-decimation=2",
            "plugin.points:ros2-pointcloud",
            "--plugin.points.routes.lidar.lidar=front_scan",
            "--plugin.points.routes.lidar.topic=/front_scan/points",
            "--plugin.points.routes.lidar.frame-id=front_scan_link",
        ],
    )

    assert isinstance(config.sensor["head"], CameraSensorConfig)
    lidar = config.sensor["front_scan"]
    assert isinstance(lidar, LidarSensorConfig)
    assert isinstance(lidar.pattern, GridLidarRayPatternConfig)
    assert lidar.pattern.vertical_angles == [-15.0, 0.0, 15.0]
    assert lidar.update_decimation == 2

    plugin = config.plugin["points"]
    assert isinstance(plugin, ROS2PointCloudPluginConfig)
    route = plugin.routes["lidar"]
    assert (route.lidar, route.topic, route.frame_id) == (
        "front_scan",
        "/front_scan/points",
        "front_scan_link",
    )


def test_cli_rejects_removed_isaacsim_camera_depth_clipping_field(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse_config(
            RunSimConfig,
            args=[
                "simulator:isaacsim",
                "sensor.depth:g1-head",
                "sensor.depth.isaacsim:isaac-sim-camera-config",
                "--sensor.depth.isaacsim.depth-clipping-behavior=max",
            ],
        )
    assert "depth-clipping-behavior" in capsys.readouterr().err
