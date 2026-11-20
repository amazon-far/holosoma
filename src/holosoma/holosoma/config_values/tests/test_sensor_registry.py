"""CLI resolution tests for the heterogeneous mounted-sensor registry."""

from __future__ import annotations

import pytest

from holosoma.config_types.experiment import SimulationExperimentConfig
from holosoma.config_types.plugin import LidarVizPluginConfig, ROS2PointCloudPluginConfig
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
        SimulationExperimentConfig,
        args=[
            "env.simulator:sim-mujoco",
            "env.sensor.head:g1-head",
            "env.sensor.front_scan:g1-pelvis-lidar",
            "--env.sensor.front-scan.pattern.vertical-angles=[-15,0,15]",
            "--env.sensor.front-scan.update-decimation=2",
            "env.plugin.points:ros2-pointcloud",
            "--env.plugin.points.routes.lidar.lidar=front_scan",
            "--env.plugin.points.routes.lidar.topic=/front_scan/points",
            "--env.plugin.points.routes.lidar.frame-id=front_scan_link",
        ],
    )

    assert isinstance(config.env.sensor["head"], CameraSensorConfig)
    lidar = config.env.sensor["front_scan"]
    assert isinstance(lidar, LidarSensorConfig)
    assert isinstance(lidar.pattern, GridLidarRayPatternConfig)
    assert lidar.pattern.vertical_angles == [-15.0, 0.0, 15.0]
    assert lidar.update_decimation == 2

    plugin = config.env.plugin["points"]
    assert isinstance(plugin, ROS2PointCloudPluginConfig)
    route = plugin.routes["lidar"]
    assert (route.lidar, route.topic, route.frame_id) == (
        "front_scan",
        "/front_scan/points",
        "front_scan_link",
    )


def test_cli_round_trips_lidar_range_and_backend_query_options() -> None:
    config = parse_config(
        SimulationExperimentConfig,
        args=[
            "env.simulator:sim-mujoco",
            "env.sensor.depth:g1-head",
            "env.sensor.lidar:g1-pelvis-lidar",
            "env.sensor.lidar.mujoco:mujoco-lidar-config",
            "env.sensor.lidar.isaacsim:isaac-sim-lidar-config",
            "--env.sensor.depth.depth-clipping-behavior=zero",
            "--env.sensor.lidar.range-clipping-behavior=zero",
            "--env.sensor.lidar.mujoco.geom-groups=[True,False,True,False,True,False]",
            "--env.sensor.lidar.body-filter.target-kind=robot_link",
            "--env.sensor.lidar.body-filter.target=head_link",
            "--env.sensor.lidar.isaacsim.mesh-prim-paths=['/World/ground','/World/props/box']",
        ],
    )

    lidar = config.env.sensor["lidar"]
    assert isinstance(lidar, LidarSensorConfig)
    assert lidar.range_clipping_behavior == "zero"
    assert lidar.mujoco is not None
    assert lidar.mujoco.geom_groups == [True, False, True, False, True, False]
    assert lidar.body_filter.target_kind == "robot_link"
    assert lidar.body_filter.target == "head_link"
    assert lidar.isaacsim is not None
    assert lidar.isaacsim.mesh_prim_paths == ["/World/ground", "/World/props/box"]
    depth = config.env.sensor["depth"]
    assert isinstance(depth, CameraSensorConfig)
    assert depth.depth_clipping_behavior == "zero"


def test_cli_rejects_removed_isaacsim_camera_depth_clipping_field(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse_config(
            SimulationExperimentConfig,
            args=[
                "env.simulator:sim-isaacsim",
                "env.sensor.depth:g1-head",
                "env.sensor.depth.isaacsim:isaac-sim-camera-config",
                "--env.sensor.depth.isaacsim.depth-clipping-behavior=max",
            ],
        )
    assert "depth-clipping-behavior" in capsys.readouterr().err


def test_cli_configures_isaacsim_fisheye_projection() -> None:
    config = parse_config(
        SimulationExperimentConfig,
        args=[
            "env.simulator:sim-isaacsim",
            "env.sensor.head:g1-head",
            "--env.sensor.head.isaacsim.projection-type=fisheyeKannalaBrandtK3",
            "--env.sensor.head.isaacsim.fisheye.nominal-width=1920",
            "--env.sensor.head.isaacsim.fisheye.nominal-height=1080",
            "--env.sensor.head.isaacsim.fisheye.optical-centre-x=960",
            "--env.sensor.head.isaacsim.fisheye.optical-centre-y=540",
            "--env.sensor.head.isaacsim.fisheye.max-fov=210",
            "--env.sensor.head.isaacsim.fisheye.polynomial-a=0.01",
        ],
    )

    camera = config.env.sensor["head"]
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
        SimulationExperimentConfig,
        args=[
            "env.simulator:sim-mujoco",
            "env.sensor.scan:custom-lidar",
            "--env.sensor.scan.pattern.ray-directions=[[-1,0,-3],[4,5,-6]]",
            "--env.sensor.scan.pattern.organized-shape=[1,2]",
        ],
    )

    lidar = config.env.sensor["scan"]
    assert isinstance(lidar, LidarSensorConfig)
    assert isinstance(lidar.pattern, CustomLidarRayPatternConfig)
    assert lidar.pattern.ray_directions == [[-1.0, 0.0, -3.0], [4.0, 5.0, -6.0]]
    assert lidar.pattern.organized_shape == [1, 2]


def test_cli_configures_lidar_viz_saved_scan_selection() -> None:
    config = parse_config(
        SimulationExperimentConfig,
        args=[
            "env.simulator:sim-mujoco",
            "env.sensor.front_scan:custom-lidar",
            "env.plugin.debug:lidar-viz-save",
            "--env.plugin.debug.lidars=['front_scan']",
            "--env.plugin.debug.env-ids=[0]",
            "--env.plugin.debug.max-points=500",
            "--env.plugin.debug.save-dir=logs/debug-scans",
        ],
    )

    plugin = config.env.plugin["debug"]
    assert isinstance(plugin, LidarVizPluginConfig)
    assert plugin.save_scans and not plugin.live_window
    assert plugin.lidars == ["front_scan"]
    assert plugin.env_ids == [0]
    assert plugin.max_points == 500
    assert plugin.save_dir == "logs/debug-scans"


def test_cli_switches_the_raycast_preset_to_rtx_with_one_override() -> None:
    config = parse_config(
        SimulationExperimentConfig,
        args=[
            "env.simulator:sim-isaacsim",
            "env.sensor.depth:g1-head-d435i-raycast-depth",
            "--env.sensor.depth.isaacsim.renderer=rtx",
        ],
    )
    camera = config.env.sensor["depth"]
    assert isinstance(camera, CameraSensorConfig)
    assert camera.isaacsim.renderer == "rtx"
