"""Named mounted-sensor building blocks, selectable per key via the dynamic ``--sensor`` dict.

Each preset is one camera or LiDAR config. Compose a rig by giving each sensor its own key,
e.g. ``sensor.my_head:g1-head sensor.lidar:g1-pelvis-lidar``; the key becomes the name used by
``get_camera_data`` or ``get_lidar_data``. Fields can be overridden per key, e.g.
``--sensor.my_head.width 224``.

Combination presets (a head + two wrists, stereo + wrists, ...) are intentionally gone: the dynamic
dict composes them from these building blocks at the CLI.

Extensions add sensors through ``SENSOR_REGISTRY`` (e.g. via a ``holosoma.config.sensor`` entry
point). ``CAMERA_REGISTRY`` remains as a compatibility alias.
"""

from holosoma.config_types.sensor import (
    CameraSensorConfig,
    CustomLidarRayPatternConfig,
    GridLidarRayPatternConfig,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.config_values.wbt.g1.sensor import (
    head_camera,
    left_wrist_camera,
    right_wrist_camera,
    stereo_head_camera_left,
    stereo_head_camera_right,
    waist_back_camera,
    waist_front_camera,
)
from holosoma.utils.config_registry import ConfigRegistry

SENSOR_REGISTRY = ConfigRegistry((CameraSensorConfig, LidarSensorConfig), group="holosoma.config.sensor")
# Compatibility alias for extensions written before the registry became heterogeneous.
CAMERA_REGISTRY = SENSOR_REGISTRY

# Free-floating overview camera: fixed at an elevated corner of each env, looking down at the scene
# center in an angled ISOMETRIC view. The ``world`` mount anchors to the env frame (not any body),
# so it never moves with the robot — useful for logging/overview. Placed at (2.5, -2.5, 2.5) m
# (front-right, up); the orientation is the look-at quaternion aiming the optical axis (-Z) at the
# origin with +Y up, giving a ~35.26deg downward elevation (the true isometric angle). Verified: the
# camera's -Z maps to the normalized eye->origin direction and +Y stays up. Robot/scene-agnostic,
# 640x480.
_ISO_LOOK_AT_ORIGIN_WXYZ = [0.820473, 0.424708, 0.17592, 0.339851]
overview_camera = CameraSensorConfig(
    mount=SensorMountConfig(target_kind="world", position=[2.5, -2.5, 2.5], orientation=_ISO_LOOK_AT_ORIGIN_WXYZ),
    width=640,
    height=480,
    vertical_fov=60.0,
    data_types=["rgb"],
)

# Generic planar scanner mounted on the G1 pelvis. Override the grid's angles per key for
# multi-channel devices, for example ``--sensor.lidar.pattern.vertical-angles '[-15, 0, 15]'``.
g1_pelvis_lidar = LidarSensorConfig(
    mount=SensorMountConfig(target_kind="robot_link", target="pelvis", position=[0.1, 0.0, 0.2]),
    pattern=GridLidarRayPatternConfig(horizontal_resolution=1.0),
    near=0.05,
    far=30.0,
)

# Minimal explicit-direction scanner. Override ``pattern.ray_directions`` and the mount per
# sensor key to describe a device-specific custom pattern from the CLI.
custom_lidar = LidarSensorConfig(
    mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
    pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]),
)

# Egocentric G1 head camera, forward-facing.
CAMERA_REGISTRY.add("g1-head", head_camera)
# G1 stereo head eyes (compose both for a stereo pair).
CAMERA_REGISTRY.add("g1-stereo-left", stereo_head_camera_left)
CAMERA_REGISTRY.add("g1-stereo-right", stereo_head_camera_right)
# G1 wrist grasp cameras.
CAMERA_REGISTRY.add("g1-left-wrist", left_wrist_camera)
CAMERA_REGISTRY.add("g1-right-wrist", right_wrist_camera)
# G1 waist-height forward/back depth cameras.
CAMERA_REGISTRY.add("g1-waist-front", waist_front_camera)
CAMERA_REGISTRY.add("g1-waist-back", waist_back_camera)
# Free-floating angled isometric overview camera looking at the scene center (640x480).
CAMERA_REGISTRY.add("overview", overview_camera)
# Generic 360-degree planar LiDAR on the G1 pelvis.
SENSOR_REGISTRY.add("g1-pelvis-lidar", g1_pelvis_lidar)
# Explicit optical-frame directions, intended for per-key overrides.
SENSOR_REGISTRY.add("custom-lidar", custom_lidar)
