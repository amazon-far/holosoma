"""CLI resolution tests for the heterogeneous mounted-sensor registry."""

from __future__ import annotations

import pytest

from holosoma.config_types.plugin import LidarVizPluginConfig, ROS2PointCloudPluginConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.sensor import (
    CameraSensorConfig,
    CustomLidarRayPatternConfig,
    GridLidarRayPatternConfig,
    IsaacSimCameraConfig,
    IsaacSimFisheyeConfig,
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


def test_cli_round_trips_lidar_range_and_backend_query_options() -> None:
    config = parse_config(
        RunSimConfig,
        args=[
            "simulator:mujoco",
            "sensor.depth:g1-head",
            "sensor.lidar:g1-pelvis-lidar",
            "sensor.lidar.mujoco:mujoco-lidar-config",
            "sensor.lidar.isaacsim:isaac-sim-lidar-config",
            "--sensor.depth.depth-clipping-behavior=zero",
            "--sensor.lidar.range-clipping-behavior=zero",
            "--sensor.lidar.mujoco.geom-groups=[True,False,True,False,True,False]",
            "--sensor.lidar.body-filter.target-kind=robot_link",
            "--sensor.lidar.body-filter.target=head_link",
            "--sensor.lidar.isaacsim.mesh-prim-paths=['/World/ground','/World/props/box']",
        ],
    )

    lidar = config.sensor["lidar"]
    assert isinstance(lidar, LidarSensorConfig)
    assert lidar.range_clipping_behavior == "zero"
    assert lidar.mujoco is not None
    assert lidar.mujoco.geom_groups == [True, False, True, False, True, False]
    assert lidar.body_filter.target_kind == "robot_link"
    assert lidar.body_filter.target == "head_link"
    assert lidar.isaacsim is not None
    assert lidar.isaacsim.mesh_prim_paths == ["/World/ground", "/World/props/box"]
    depth = config.sensor["depth"]
    assert isinstance(depth, CameraSensorConfig)
    assert depth.depth_clipping_behavior == "zero"


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


def test_cli_configures_isaacsim_fisheye_projection() -> None:
    config = parse_config(
        RunSimConfig,
        args=[
            "simulator:isaacsim",
            "sensor.head:g1-head",
            "--sensor.head.isaacsim.projection-type=fisheyeKannalaBrandtK3",
            "--sensor.head.isaacsim.fisheye.nominal-width=1920",
            "--sensor.head.isaacsim.fisheye.nominal-height=1080",
            "--sensor.head.isaacsim.fisheye.optical-centre-x=960",
            "--sensor.head.isaacsim.fisheye.optical-centre-y=540",
            "--sensor.head.isaacsim.fisheye.max-fov=210",
            "--sensor.head.isaacsim.fisheye.polynomial-a=0.01",
        ],
    )

    camera = config.sensor["head"]
    assert isinstance(camera, CameraSensorConfig)
    assert isinstance(camera.isaacsim, IsaacSimCameraConfig)
    assert isinstance(camera.isaacsim.fisheye, IsaacSimFisheyeConfig)
    assert camera.isaacsim.projection_type == "fisheyeKannalaBrandtK3"
    assert camera.isaacsim.fisheye.nominal_width == 1920.0
    assert camera.isaacsim.fisheye.optical_centre_x == 960.0
    assert camera.isaacsim.fisheye.max_fov == 210.0
    assert camera.isaacsim.fisheye.polynomial_a == 0.01


def test_cli_customizes_explicit_lidar_directions() -> None:
    config = parse_config(
        RunSimConfig,
        args=[
            "simulator:mujoco",
            "sensor.scan:custom-lidar",
            "--sensor.scan.pattern.ray-directions=[[-1,0,-3],[4,5,-6]]",
            "--sensor.scan.pattern.organized-shape=[1,2]",
        ],
    )

    lidar = config.sensor["scan"]
    assert isinstance(lidar, LidarSensorConfig)
    assert isinstance(lidar.pattern, CustomLidarRayPatternConfig)
    assert lidar.pattern.ray_directions == [[-1.0, 0.0, -3.0], [4.0, 5.0, -6.0]]
    assert lidar.pattern.organized_shape == [1, 2]


def test_cli_configures_lidar_viz_saved_scan_selection() -> None:
    config = parse_config(
        RunSimConfig,
        args=[
            "simulator:mujoco",
            "sensor.front_scan:custom-lidar",
            "plugin.debug:lidar-viz-save",
            "--plugin.debug.lidars=['front_scan']",
            "--plugin.debug.env-ids=[0]",
            "--plugin.debug.max-points=500",
            "--plugin.debug.save-dir=logs/debug-scans",
        ],
    )

    plugin = config.plugin["debug"]
    assert isinstance(plugin, LidarVizPluginConfig)
    assert plugin.save_scans and not plugin.live_window
    assert plugin.lidars == ["front_scan"]
    assert plugin.env_ids == [0]
    assert plugin.max_points == 500
    assert plugin.save_dir == "logs/debug-scans"
