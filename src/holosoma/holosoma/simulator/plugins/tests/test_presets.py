"""Tests for the core camera-egress plugin presets and the real ROS2ImagePlugin wiring (no ROS env).

Egress sinks are plugins: each preset is a single ``ROS2ImagePluginConfig`` / ``CameraVizPluginConfig``
(a ``PluginConfig``) registered in ``PLUGIN_REGISTRY``, selected per-key as ``plugin.<key>:<variant>``.
The ROS2ImagePlugin impl defers rclpy to start(), so we can construct it (which validates streams +
registers callbacks, no rclpy) and inspect wiring — only start()/publish() would need a ROS env. This
pins the preset / topic guarantees (incl. the rfmpi teleop topics the removed holosoma_sim_stereo
sidecar published).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from holosoma.config_types.plugin import ROS2ImagePluginConfig, ROS2ImageRoute
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.config_values.plugin import PLUGIN_REGISTRY
from holosoma.config_values.wbt.g1 import sensor as g1_cameras
from holosoma.simulator.base_simulator.hooks import HookRegistry
from holosoma.simulator.plugins.camera_consumer import CameraIntrinsics, FramePacket

pytestmark = pytest.mark.no_sim

# Camera rigs are composed per-key on the CLI (``--sensor.<name>:<variant>``), so a rig is just a
# ``{sensor_name: CameraSensorConfig}`` dict. The egress presets reference cameras by conventional
# names (``head_cam_left``, ``waist_front_cam``, ...); these rigs pair those names with the G1 camera
# building blocks, exactly as a user would key them to line up with the egress routes.
_G1_STEREO_RIG = {
    "head_cam_left": g1_cameras.stereo_head_camera_left,
    "head_cam_right": g1_cameras.stereo_head_camera_right,
}
_G1_STEREO_WRISTS_RIG = {
    **_G1_STEREO_RIG,
    "left_wrist_cam": g1_cameras.left_wrist_camera,
    "right_wrist_cam": g1_cameras.right_wrist_camera,
}
_G1_WAIST_RIG = {
    "waist_front_cam": g1_cameras.waist_front_camera,
    "waist_back_cam": g1_cameras.waist_back_camera,
}


class _FakeSimEngineCfg:
    fps = 200.0
    control_decimation_steps = 4


class _FakeSimulatorConfig:
    sim = _FakeSimEngineCfg()


class _FakeVideoConfig:
    save_dir = None


class _FakeTrainingConfig:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs


class _FakeSimulator:
    """Minimal stand-in exposing what a camera-consumer hook reads/registers on (no backend)."""

    def __init__(self, sensors_config, num_envs=1):
        self.hooks = HookRegistry()
        self.sensor_config = sensors_config
        self.training_config = _FakeTrainingConfig(num_envs)
        self.headless = True
        self.simulator_config = _FakeSimulatorConfig()
        self.video_config = _FakeVideoConfig()
        self.sensor_manager = None

    def sensor_config_by_name(self, name):
        return self.sensor_config[name]


def test_core_presets_present():
    assert set(PLUGIN_REGISTRY) >= {"none", "ros2-image", "ros2-stereo", "viz", "viz-record"}


def test_presets_resolve_get_cls_without_rclpy():
    # Resolving get_cls imports the ros2_image_egress MODULE but must NOT import rclpy (deferred to
    # start()), so presets are inspectable in a non-ROS env.
    for name in ("ros2-image", "ros2-stereo"):
        cls = PLUGIN_REGISTRY[name].get_cls()
        assert cls.__name__ == "ROS2ImagePlugin"
    assert "rclpy" not in sys.modules


def test_stereo_preset_publishes_rfmpi_teleop_topics():
    # The rfmpi teleop stack subscribes to /ros_camera/rgb/{left,right}/compressed. Topics are
    # published VERBATIM (no auto-suffix), so the preset spells the full topics out.
    inst = PLUGIN_REGISTRY["ros2-stereo"]
    assert isinstance(inst, ROS2ImagePluginConfig)
    topics = {r.topic for r in inst.routes.values()}
    assert topics == {"/ros_camera/rgb/left/compressed", "/ros_camera/rgb/right/compressed"}
    cams = {r.camera for r in inst.routes.values()}
    assert cams == {"head_cam_left", "head_cam_right"}


def test_stereo_egress_cameras_exist_in_g1_stereo_rig():
    # The egress route cameras must exist in a rig you'd pair it with, else the hook fails loud at
    # construction. Pin that ros2-stereo lines up with a stereo head rig keyed head_cam_left/right.
    stereo_cams = set(_G1_STEREO_RIG)
    route_cams = {r.camera for r in PLUGIN_REGISTRY["ros2-stereo"].routes.values()}
    assert route_cams <= stereo_cams


def test_g1_stereo_wrists_rig_composes_from_building_blocks():
    # The stereo-head-plus-wrists rig (previously a whole-config preset) now composes from the G1
    # camera building blocks, keyed to the names the egress routes reference.
    cams = set(_G1_STEREO_WRISTS_RIG)
    assert {"head_cam_left", "head_cam_right", "left_wrist_cam", "right_wrist_cam"} <= cams


def test_waist_depth_color_preset_colorizes_depth_over_rgb_format():
    # The colorized-depth preset publishes DEPTH cameras on an RGB (jpeg) format => colorized to RGB.
    # Cameras must exist in the paired waist rig (else the hook fails loud at construction).
    inst = PLUGIN_REGISTRY["ros2-waist-depth-color"]
    assert isinstance(inst, ROS2ImagePluginConfig)
    for route in inst.routes.values():
        assert route.modality == "depth" and route.format == "jpeg"  # depth + rgb format = colorize
        assert route.depth_colormap == "turbo"
    assert {r.camera for r in inst.routes.values()} <= set(_G1_WAIST_RIG)


def test_waist_depth_raw_and_color_preset_shares_one_snapshot_per_camera():
    # The combined preset publishes each waist camera BOTH ways (raw 32FC1 + colorized jpeg) from a
    # single node. The two routes per camera share the same (camera, "depth", env) stream key, so
    # wanted_streams collapses to one triple per camera => ONE cached device->host copy each, not two.
    inst = PLUGIN_REGISTRY["ros2-waist-depth-raw+color"]
    assert isinstance(inst, ROS2ImagePluginConfig)
    # Four distinct topics (2 cameras x 2 formats), all unique (the config validator also enforces).
    topics = {r.topic for r in inst.routes.values()}
    assert len(topics) == len(inst.routes) == 4
    # Both a raw depth-format route and a colorized rgb-format route are present.
    formats = {r.format for r in inst.routes.values()}
    assert "32FC1" in formats and "jpeg" in formats
    assert any(r.format == "jpeg" and r.depth_colormap == "turbo" for r in inst.routes.values())
    assert {r.camera for r in inst.routes.values()} <= set(_G1_WAIST_RIG)
    # The single-copy guarantee: 4 routes but only 2 wanted streams (one per camera).
    egress = inst.get_cls()(inst, _FakeSimulator(_G1_WAIST_RIG))
    assert egress.wanted_streams() == {("waist_front_cam", "depth", 0), ("waist_back_cam", "depth", 0)}
    assert "rclpy" not in sys.modules


def test_real_egress_wanted_streams_and_async_default():
    # Construct the REAL ROS2ImagePlugin (no start -> no ROS) and check its wiring: wanted_streams
    # from routes, and async_publish on by default.
    inst = PLUGIN_REGISTRY["ros2-stereo"]
    egress = inst.get_cls()(inst, _FakeSimulator(_G1_STEREO_RIG))
    assert egress.wanted_streams() == {("head_cam_left", "rgb", 0), ("head_cam_right", "rgb", 0)}
    assert inst.async_publish is True
    assert "rclpy" not in sys.modules


def _depth_frame(camera, value=2.0, h=6, w=6):
    arr = np.full((h, w, 1), value, np.float32)  # [H,W,1] float meters, as get_camera_data gives
    intr = CameraIntrinsics(width=w, height=h, vertical_fov=45.0, near=0.01, far=100.0)
    return FramePacket(camera=camera, modality="depth", env_id=0, array=arr, sim_time=0.0, intrinsics=intr)


def test_encode_threads_route_colormap_and_range_without_ros():
    # ROS2ImagePlugin._encode is ROS-free (never touches the node/simulator), so we can verify the
    # route's depth_colormap/depth_range actually reach encode_frame WITHOUT an rclpy env. Each knob
    # is isolated (routes differing in ONLY that field) so the test guards BOTH independently.
    mount = SensorMountConfig(target_kind="robot_link", target="torso_link")
    sensors = {"waist": CameraSensorConfig(mount=mount, data_types=["depth"])}

    def _route(topic, colormap, drange):
        return ROS2ImageRoute(
            camera="waist", topic=topic, modality="depth", format="rgb8", depth_colormap=colormap, depth_range=drange
        )

    # Same range, different colormap -> isolates route.depth_colormap threading.
    cmap_a = _route("/t/turbo", "turbo", [0.1, 5.0])
    cmap_b = _route("/t/gray", "gray", [0.1, 5.0])
    # Same colormap, different range -> isolates route.depth_range threading.
    range_a = _route("/t/tight", "gray", [0.1, 3.0])
    range_b = _route("/t/wide", "gray", [0.1, 20.0])
    cfg = ROS2ImagePluginConfig(
        publish_camera_info=False,
        routes={"ca": cmap_a, "cb": cmap_b, "ra": range_a, "rb": range_b},
    )
    egress = cfg.get_cls()(cfg, _FakeSimulator(sensors))  # no start() -> no rclpy
    frame = _depth_frame("waist")
    enc = {k: egress._encode(r, frame).data for k, r in cfg.routes.items()}
    assert enc["ca"] != enc["cb"]  # colormap reaches encode_frame (would collide if defaulted)
    assert enc["ra"] != enc["rb"]  # depth_range reaches encode_frame (would collide if defaulted)
    assert "rclpy" not in sys.modules
