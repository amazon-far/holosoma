"""Registered plugin presets, selectable as ``plugin.<key>:<variant>`` on the CLI.

Holds the in-tree plugin presets: the ``none`` no-op, the ROS2 example plugins
(``clock_publish``/``gantry_control``/``odometry``), and the camera-frame egress sinks
(``ros2-image`` and friends, ``viz``/``viz-record``). All live in one ``PLUGIN_REGISTRY`` and
compose per-key on the CLI, e.g. ``plugin.clk:clock_publish plugin.head:ros2-image plugin.rec:viz``.

Imported at CLI-build time; stays ROS-free (a preset's impl loads only when its ``get_cls`` fires).
"""

from holosoma.config_types.plugin import (
    CameraVizPluginConfig,
    ClockPublishPluginConfig,
    FrameWriterPluginConfig,
    GantryControlPluginConfig,
    LidarVizPluginConfig,
    MotionPlaybackPluginConfig,
    NoOpPluginConfig,
    PluginConfig,
    ROS2ImagePluginConfig,
    ROS2ImageRoute,
    ROS2OdometryPluginConfig,
    ROS2PointCloudPluginConfig,
    ROS2PointCloudRoute,
)
from holosoma.utils.config_registry import ConfigRegistry, deprecated_defaults_alias

PLUGIN_REGISTRY = ConfigRegistry(PluginConfig, group="holosoma.config.plugin")

# `none` disables a slot (plugin.<key>:none), mirroring every other config family's `none`
# preset. Unlike the scalar families (which register a literal None), this dict field
# registers a real no-op config so the field type stays uniform.
none = PLUGIN_REGISTRY.add("none", NoOpPluginConfig())

# ROS2 example presets. Their impls import rclpy (optional dep: holosoma[ros2]); the
# configs stay rclpy-free, so registering them here does not require ROS.
clock_publish = PLUGIN_REGISTRY.add("clock_publish", ClockPublishPluginConfig())
gantry_control = PLUGIN_REGISTRY.add("gantry_control", GantryControlPluginConfig())

# Robot-attached frame pose/velocity as nav_msgs/Odometry. Defaults to the robot root for backward
# compatibility; body_name + position/orientation can select an offset frame on any robot body.
odometry = PLUGIN_REGISTRY.add("odometry", ROS2OdometryPluginConfig())

# Kinematic motion playback: drive the robot (and optionally one scene object) from a recorded
# motion NPZ. Rendering-only replay of external trajectories; combine with
# --simulator.config.sim.kinematic_playback True to replace dynamics with forward kinematics,
# plus a mounted camera + a frame consumer, e.g.
#   plugin.play:motion-playback '--plugin.play.motion_files=["clip.npz"]' \
#   --sensor.head_cam:g1-head plugin.out:frame-writer
motion_playback = PLUGIN_REGISTRY.add("motion-playback", MotionPlaybackPluginConfig())

# ------------------------------------------------------------------------------------------------ #
# Camera-frame egress presets. Compose several on the CLI by giving each its own key, e.g.
#   plugin.head:ros2-image plugin.rec:viz-record --plugin.head.node_name cams
# Field overrides key off the plugin name, e.g. --plugin.waist.routes.front.depth_range '[0.1, 6.0]'.
# ------------------------------------------------------------------------------------------------ #

# One ROS2 node publishing the G1 head camera as JPEG plus latched CameraInfo. Pair with a sensor
# rig that defines `head_cam` (e.g. `--sensor.head_cam:g1-head`).
ros2_image = PLUGIN_REGISTRY.add(
    "ros2-image",
    ROS2ImagePluginConfig(
        node_name="sim_cameras_head",
        routes={
            "head": ROS2ImageRoute(
                camera="head_cam", topic="/sim_cameras/head/image/compressed", modality="rgb", format="jpeg"
            )
        },
    ),
)

# Stereo head pair as CompressedImage on the exact topics the rfmpi teleop stack subscribes to:
# /ros_camera/rgb/{left,right}/compressed. Pair with a stereo head rig keyed head_cam_left/right.
# CameraInfo off for this teleop stack.
ros2_stereo = PLUGIN_REGISTRY.add(
    "ros2-stereo",
    ROS2ImagePluginConfig(
        node_name="sim_cameras_head_stereo",
        publish_camera_info=False,
        routes={
            "left": ROS2ImageRoute(
                camera="head_cam_left", topic="/ros_camera/rgb/left/compressed", modality="rgb", format="jpeg"
            ),
            "right": ROS2ImageRoute(
                camera="head_cam_right", topic="/ros_camera/rgb/right/compressed", modality="rgb", format="jpeg"
            ),
        },
    ),
)

# One ROS2 node publishing the G1 waist forward/back depth cameras as raw float32-meter Image
# (sensor_msgs/Image, 32FC1) plus latched CameraInfo. Pair with a rig keyed waist_front_cam /
# waist_back_cam (`--sensor.waist_front_cam:g1-waist-front --sensor.waist_back_cam:g1-waist-back`).
ros2_waist_depth = PLUGIN_REGISTRY.add(
    "ros2-waist-depth",
    ROS2ImagePluginConfig(
        node_name="sim_cameras_waist",
        routes={
            "front": ROS2ImageRoute(
                camera="waist_front_cam", topic="/sim_cameras/waist_front/depth", modality="depth", format="32FC1"
            ),
            "back": ROS2ImageRoute(
                camera="waist_back_cam", topic="/sim_cameras/waist_back/depth", modality="depth", format="32FC1"
            ),
        },
    ),
)

# Like `ros2-waist-depth`, but publishes the waist depth cameras COLORIZED to RGB (turbo colormap)
# as CompressedImage (jpeg) — a human-viewable depth stream for rviz/teleop rather than raw metric
# depth. Depth modality + an rgb format triggers colorization; depth_range fixes the color scale.
ros2_waist_depth_color = PLUGIN_REGISTRY.add(
    "ros2-waist-depth-color",
    ROS2ImagePluginConfig(
        node_name="sim_cameras_waist_color",
        routes={
            "front": ROS2ImageRoute(
                camera="waist_front_cam",
                topic="/sim_cameras/waist_front/depth_color/compressed",
                modality="depth",
                format="jpeg",
                depth_colormap="turbo",
                depth_range=[0.1, 4.0],
            ),
            "back": ROS2ImageRoute(
                camera="waist_back_cam",
                topic="/sim_cameras/waist_back/depth_color/compressed",
                modality="depth",
                format="jpeg",
                depth_colormap="turbo",
                depth_range=[0.1, 4.0],
            ),
        },
    ),
)

# Both at once from ONE node: each waist camera published as raw metric depth (32FC1) AND colorized
# RGB (jpeg) — the machine-readable stream for perception plus the human-viewable stream for
# rviz/teleop. The raw and colorized routes for a camera share the same (camera, depth) stream key,
# so the consumer does ONE cached device->host copy per camera and each route encodes it its own way
# (no extra render/copy cost). Pair with a waist rig.
ros2_waist_depth_raw_and_color = PLUGIN_REGISTRY.add(
    "ros2-waist-depth-raw+color",
    ROS2ImagePluginConfig(
        node_name="sim_cameras_waist",
        routes={
            "front_raw": ros2_waist_depth.routes["front"],
            "back_raw": ros2_waist_depth.routes["back"],
            "front_color": ros2_waist_depth_color.routes["front"],
            "back_color": ros2_waist_depth_color.routes["back"],
        },
    ),
)

# Visualize all configured cameras (every modality) in a live cv2 window. cameras=None watches them
# all; add ``--plugin.<key>.record_video True`` to also write an mp4 at teardown.
viz = PLUGIN_REGISTRY.add("viz", CameraVizPluginConfig(live_window=True))

# Record all configured cameras to an mp4 at teardown (no live window; runs headless).
viz_record = PLUGIN_REGISTRY.add("viz-record", CameraVizPluginConfig(record_video=True))

# Inspect fresh LiDAR scans in local Matplotlib windows. In a headless run select
# `lidar-viz-save` for a numbered PNG sequence or `lidar-viz-record` for MP4 output.
lidar_viz = PLUGIN_REGISTRY.add("lidar-viz", LidarVizPluginConfig(live_window=True))
lidar_viz_save = PLUGIN_REGISTRY.add("lidar-viz-save", LidarVizPluginConfig(save_scans=True))
lidar_viz_record = PLUGIN_REGISTRY.add("lidar-viz-record", LidarVizPluginConfig(record_video=True))

# Write every configured camera's frames to disk as numbered images + index.jsonl (dataset capture;
# lossless, every frame, inline on the sim thread). --plugin.<key>.output_dir picks the destination.
frame_writer = PLUGIN_REGISTRY.add("frame-writer", FrameWriterPluginConfig())

# Sensor-local XYZ points as an organized PointCloud2. Pair with
# `sensor.lidar:g1-pelvis-lidar`; the sensor key matches this route's `lidar`.
ros2_pointcloud = PLUGIN_REGISTRY.add(
    "ros2-pointcloud",
    ROS2PointCloudPluginConfig(
        routes={
            "lidar": ROS2PointCloudRoute(
                lidar="lidar",
                topic="/lidar/points",
            )
        }
    ),
)

__getattr__ = deprecated_defaults_alias(__name__, PLUGIN_REGISTRY)
