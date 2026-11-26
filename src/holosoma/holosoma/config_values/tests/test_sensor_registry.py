"""CLI resolution tests for the heterogeneous mounted-sensor registry."""

from __future__ import annotations

import pytest

from holosoma.config_types.plugin import ROS2PointCloudPluginConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.sensor import (
    CameraSensorConfig,
    ContactSensorConfig,
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
    assert isinstance(SENSOR_REGISTRY["robot-contacts"], ContactSensorConfig)


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


def test_cli_switches_the_raycast_preset_to_rtx_with_one_override() -> None:
    config = parse_config(
        RunSimConfig,
        args=["simulator:isaacsim", "sensor.depth:g1-head-d435i-raycast-depth", "--sensor.depth.isaacsim.renderer=rtx"],
    )
    camera = config.sensor["depth"]
    assert isinstance(camera, CameraSensorConfig)
    assert camera.isaacsim.renderer == "rtx"


def test_cli_composes_contact_sensors_with_source_and_counterpart_overrides() -> None:
    config = parse_config(
        RunSimConfig,
        args=[
            "simulator:mujoco",
            "sensor.ground:robot-terrain-contacts",
            "sensor.hand:robot-terrain-contacts",
            "--sensor.hand.sources.0.target-kind=robot_link",
            "--sensor.hand.sources.0.target=left_wrist_yaw_link",
            "--sensor.hand.counterparts.0.target-kind=actor",
            "--sensor.hand.counterparts.0.target=box",
            "--sensor.hand.force-mode=total",
            "--sensor.hand.update-decimation=25Hz",
        ],
    )
    ground = config.sensor["ground"]
    hand = config.sensor["hand"]
    assert isinstance(ground, ContactSensorConfig)
    assert isinstance(hand, ContactSensorConfig)
    assert ground.counterparts[0].label == "terrain"
    assert hand.sources[0].label == "robot_link:left_wrist_yaw_link"
    assert hand.counterparts[0].label == "actor:box"
    assert hand.force_mode == "total"
    assert hand.update_decimation == "25Hz"
