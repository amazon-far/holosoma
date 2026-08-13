"""Named single-camera building blocks, selectable per key via the dynamic ``--sensor`` dict.

Each preset is ONE :class:`CameraSensorConfig`. Compose a rig by giving each camera its own key,
e.g. ``--sensor.my_head:g1-head --sensor.my_left_wrist:g1-left-wrist``; the key becomes the sensor
name that ``get_camera_data`` and the observation terms address. Any camera field can be overridden
per key, e.g. ``--sensor.my_head.width 224``.

Combination presets (a head + two wrists, stereo + wrists, ...) are intentionally gone: the dynamic
dict composes them from these building blocks at the CLI.

Extensions add cameras by registering into ``CAMERA_REGISTRY`` (e.g. via a ``holosoma.config.sensor``
entry point).
"""

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
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

CAMERA_REGISTRY = ConfigRegistry(CameraSensorConfig, group="holosoma.config.sensor")

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
