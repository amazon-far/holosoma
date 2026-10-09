"""Unit tests for camera-egress plugin config validators (pure, no simulator, no ROS).

Construction-time validation checks on the frozen pydantic dataclasses in
``config_types/plugin.py``. No rclpy is imported; ``get_cls`` is never touched.
Runtime behavior is covered by the consumer-hook/ROS tests in ``simulator/plugins/tests/``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from holosoma.config_types.plugin import (
    LidarVizPluginConfig,
    ROS2ImagePluginConfig,
    ROS2ImageRoute,
    ROS2OdometryPluginConfig,
    ROS2PointCloudPluginConfig,
    ROS2PointCloudRoute,
)

pytestmark = pytest.mark.no_sim


def test_ros_publishers_default_to_inline_and_allow_async_mode() -> None:
    assert ROS2ImagePluginConfig(routes={}).async_publish is False
    assert ROS2PointCloudPluginConfig(routes={}).async_publish is False
    assert ROS2ImagePluginConfig(async_publish=True, routes={}).async_publish is True
    assert ROS2PointCloudPluginConfig(async_publish=True, routes={}).async_publish is True


def test_jpeg_quality_validated() -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        ROS2ImagePluginConfig(jpeg_quality=0, routes={})
    with pytest.raises(ValueError, match="jpeg_quality"):
        ROS2ImagePluginConfig(jpeg_quality=101, routes={})


def test_queue_maxlen_validated() -> None:
    with pytest.raises(ValueError, match="queue_maxlen"):
        ROS2ImagePluginConfig(queue_maxlen=0, routes={})


def test_qos_validated() -> None:
    with pytest.raises(ValueError, match="qos"):
        ROS2ImagePluginConfig(qos="bogus", routes={})


def test_duplicate_topics_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate topics"):
        ROS2ImagePluginConfig(
            routes={
                "a": ROS2ImageRoute(camera="a", topic="/same", modality="rgb", format="jpeg"),
                "b": ROS2ImageRoute(camera="b", topic="/same", modality="rgb", format="jpeg"),
            }
        )


def test_route_format_modality_mismatch_rejected() -> None:
    # rgb modality still requires an rgb format (a depth format has nothing to colorize).
    with pytest.raises(ValueError, match="needs an rgb format"):
        ROS2ImageRoute(camera="a", topic="/t", modality="rgb", format="32FC1")


def test_depth_route_accepts_rgb_format_for_colorization() -> None:
    # A depth route MAY pick an rgb format: it is colorized to RGB before encoding. Both a raw depth
    # format and a colorizing rgb format must construct cleanly.
    raw = ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="32FC1")
    color = ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg", depth_colormap="turbo")
    assert raw.format == "32FC1"
    assert color.format == "jpeg" and color.depth_colormap == "turbo"


def test_route_depth_colormap_validated() -> None:
    with pytest.raises(ValueError, match="depth_colormap"):
        ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg", depth_colormap="bogus")


def test_route_depth_range_validated() -> None:
    with pytest.raises(ValueError, match="depth_range"):
        ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg", depth_range=[5.0, 1.0])
    with pytest.raises(ValueError, match="depth_range"):
        ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg", depth_range=[1.0])


def test_route_requires_camera_and_topic() -> None:
    with pytest.raises(ValueError, match="non-empty camera"):
        ROS2ImageRoute(camera="", topic="/t", modality="rgb", format="jpeg")
    with pytest.raises(ValueError, match="non-empty topic"):
        ROS2ImageRoute(camera="a", topic="", modality="rgb", format="jpeg")


def test_odometry_frame_transform_validated() -> None:
    cfg = ROS2OdometryPluginConfig(
        body_name="pelvis",
        position=[0.1, 0.2, 0.3],
        orientation=[1.000001, 0.0, 0.0, 0.0],
    )
    assert cfg.body_name == "pelvis"
    assert cfg.orientation == [1.0, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="position must contain 3 finite values"):
        ROS2OdometryPluginConfig(position=[0.0, 0.0])
    with pytest.raises(ValueError, match=r"(?s)orientation.*at least 4"):
        ROS2OdometryPluginConfig(orientation=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-zero quaternion"):
        ROS2OdometryPluginConfig(orientation=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Quaternion norm"):
        ROS2OdometryPluginConfig(orientation=[2.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="body_name must be non-empty or None"):
        ROS2OdometryPluginConfig(body_name="")


def test_odometry_orientation_error_identifies_field() -> None:
    with pytest.raises(ValidationError) as error:
        ROS2OdometryPluginConfig(orientation=[2.0, 0.0, 0.0, 0.0])
    assert error.value.errors()[0]["loc"] == ("orientation",)


def test_odometry_rejects_unknown_fields() -> None:
    kwargs: dict[str, Any] = {"orientaton": [1.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValidationError, match="orientaton"):
        ROS2OdometryPluginConfig(**kwargs)


@pytest.mark.parametrize("value", [True, False, 2.0, b"50Hz", 0, "50"])
def test_odometry_rejects_invalid_rates_before_coercion(value: Any) -> None:
    with pytest.raises(ValueError, match="publish_every"):
        ROS2OdometryPluginConfig(publish_every=value)


@pytest.mark.parametrize("value", [1, 2, "50Hz", ">20Hz", "<20Hz"])
def test_odometry_preserves_valid_rates(value: int | str) -> None:
    assert ROS2OdometryPluginConfig(publish_every=value).publish_every == value


@pytest.mark.parametrize(
    "value",
    [
        [0.0, float("nan"), 0.0],
        [0.0, float("inf"), 0.0],
    ],
)
def test_odometry_frame_transform_rejects_non_finite_position(value: list[float]) -> None:
    with pytest.raises(ValueError, match=r"position must contain .* finite values"):
        ROS2OdometryPluginConfig(position=value)


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 0.0, float("nan"), 0.0],
        [1.0, 0.0, float("inf"), 0.0],
    ],
)
def test_odometry_frame_transform_rejects_non_finite_orientation(value: list[float]) -> None:
    with pytest.raises(ValueError, match=r"(?s)orientation.*only finite"):
        ROS2OdometryPluginConfig(orientation=value)


def test_odometry_frame_transform_defaults_are_independent() -> None:
    first = ROS2OdometryPluginConfig()
    second = ROS2OdometryPluginConfig()

    first.position[0] = 1.0
    first.orientation[0] = 0.0

    assert second.position == [0.0, 0.0, 0.0]
    assert second.orientation == [1.0, 0.0, 0.0, 0.0]


def test_pointcloud_route_and_plugin_validation() -> None:
    route = ROS2PointCloudRoute(lidar="scan", topic="/scan/points", frame_id="scan_link")
    assert route.frame_id == "scan_link"
    assert route.point_translation == [0.0, 0.0, 0.0]
    assert route.point_rotation == [1.0, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="non-empty sensor"):
        ROS2PointCloudRoute(lidar="", topic="/scan")
    with pytest.raises(ValueError, match="non-empty topic"):
        ROS2PointCloudRoute(lidar="scan", topic="")
    with pytest.raises(ValueError, match="point_translation"):
        ROS2PointCloudRoute(lidar="scan", topic="/scan", point_translation=[0.0, 0.0])
    with pytest.raises(ValueError, match="point_rotation"):
        ROS2PointCloudRoute(lidar="scan", topic="/scan", point_rotation=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="point_rotation"):
        ROS2PointCloudRoute(lidar="scan", topic="/scan", point_rotation=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="queue_maxlen"):
        ROS2PointCloudPluginConfig(queue_maxlen=0)
    with pytest.raises(ValueError, match="node_name"):
        ROS2PointCloudPluginConfig(node_name=" ")
    with pytest.raises(ValueError, match="duplicate topics"):
        ROS2PointCloudPluginConfig(
            routes={
                "a": ROS2PointCloudRoute(lidar="a", topic="/same"),
                "b": ROS2PointCloudRoute(lidar="b", topic="/same"),
            }
        )


@pytest.mark.parametrize("value", [[0.0, float("nan"), 0.0], [0.0, float("inf"), 0.0]])
def test_pointcloud_route_rejects_non_finite_translation(value: list[float]) -> None:
    with pytest.raises(ValueError, match="point_translation"):
        ROS2PointCloudRoute(lidar="scan", topic="/scan", point_translation=value)


@pytest.mark.parametrize("value", [[1.0, 0.0, float("nan"), 0.0], [1.0, 0.0, float("inf"), 0.0]])
def test_pointcloud_route_rejects_non_finite_rotation(value: list[float]) -> None:
    with pytest.raises(ValueError, match="point_rotation"):
        ROS2PointCloudRoute(lidar="scan", topic="/scan", point_rotation=value)


def test_pointcloud_route_transform_defaults_are_independent() -> None:
    first = ROS2PointCloudRoute(lidar="first", topic="/first")
    second = ROS2PointCloudRoute(lidar="second", topic="/second")

    first.point_translation[0] = 1.0
    first.point_rotation[0] = 0.0

    assert second.point_translation == [0.0, 0.0, 0.0]
    assert second.point_rotation == [1.0, 0.0, 0.0, 0.0]


def test_lidar_viz_validation() -> None:
    config = LidarVizPluginConfig(
        live_window=True,
        save_scans=True,
        env_ids=[0, 2],
        lidars=["front_scan"],
        point_size=3.0,
        max_points=None,
        view_elevation=15.0,
        view_azimuth=-45.0,
        save_dir="logs/scans",
    )
    assert config.lidars == ["front_scan"]

    with pytest.raises(ValueError, match="env_ids must be non-empty"):
        LidarVizPluginConfig(env_ids=[])
    with pytest.raises(ValueError, match="env_ids"):
        LidarVizPluginConfig(env_ids=[-1])
    with pytest.raises(ValueError, match="point_size"):
        LidarVizPluginConfig(point_size=0.0)
    with pytest.raises(ValueError, match="max_points"):
        LidarVizPluginConfig(max_points=0)
    with pytest.raises(ValueError, match="save_dir"):
        LidarVizPluginConfig(save_dir="")


@pytest.mark.parametrize("field", ["point_size", "view_elevation", "view_azimuth"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_lidar_viz_rejects_nonfinite_draw_options(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        LidarVizPluginConfig(**{field: value})  # type: ignore[arg-type]
