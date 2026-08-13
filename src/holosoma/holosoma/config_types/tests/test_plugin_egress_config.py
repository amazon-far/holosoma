"""Unit tests for camera-egress plugin config validators (pure, no simulator, no ROS).

Construction-time validation checks on the frozen pydantic dataclasses in
``config_types/plugin.py``. No rclpy is imported; ``get_cls`` is never touched.
Runtime behavior is covered by the consumer-hook/ROS tests in ``simulator/plugins/tests/``.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.plugin import ROS2ImagePluginConfig, ROS2ImageRoute

pytestmark = pytest.mark.no_sim


def test_inline_mode_is_configurable():
    assert ROS2ImagePluginConfig(async_publish=False, routes={}).async_publish is False


def test_jpeg_quality_validated():
    with pytest.raises(ValueError, match="jpeg_quality"):
        ROS2ImagePluginConfig(jpeg_quality=0, routes={})
    with pytest.raises(ValueError, match="jpeg_quality"):
        ROS2ImagePluginConfig(jpeg_quality=101, routes={})


def test_queue_maxlen_validated():
    with pytest.raises(ValueError, match="queue_maxlen"):
        ROS2ImagePluginConfig(queue_maxlen=0, routes={})


def test_qos_validated():
    with pytest.raises(ValueError, match="qos"):
        ROS2ImagePluginConfig(qos="bogus", routes={})


def test_duplicate_topics_rejected():
    with pytest.raises(ValueError, match="duplicate topics"):
        ROS2ImagePluginConfig(
            routes={
                "a": ROS2ImageRoute(camera="a", topic="/same", modality="rgb", format="jpeg"),
                "b": ROS2ImageRoute(camera="b", topic="/same", modality="rgb", format="jpeg"),
            }
        )


def test_route_format_modality_mismatch_rejected():
    # rgb modality still requires an rgb format (a depth format has nothing to colorize).
    with pytest.raises(ValueError, match="needs an rgb format"):
        ROS2ImageRoute(camera="a", topic="/t", modality="rgb", format="32FC1")


def test_depth_route_accepts_rgb_format_for_colorization():
    # A depth route MAY pick an rgb format: it is colorized to RGB before encoding. Both a raw depth
    # format and a colorizing rgb format must construct cleanly.
    raw = ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="32FC1")
    color = ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg", depth_colormap="turbo")
    assert raw.format == "32FC1"
    assert color.format == "jpeg" and color.depth_colormap == "turbo"


def test_route_depth_colormap_validated():
    with pytest.raises(ValueError, match="depth_colormap"):
        ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg", depth_colormap="bogus")


def test_route_depth_range_validated():
    with pytest.raises(ValueError, match="depth_range"):
        ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg", depth_range=[5.0, 1.0])
    with pytest.raises(ValueError, match="depth_range"):
        ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg", depth_range=[1.0])


def test_route_requires_camera_and_topic():
    with pytest.raises(ValueError, match="non-empty camera"):
        ROS2ImageRoute(camera="", topic="/t", modality="rgb", format="jpeg")
    with pytest.raises(ValueError, match="non-empty topic"):
        ROS2ImageRoute(camera="a", topic="", modality="rgb", format="jpeg")
