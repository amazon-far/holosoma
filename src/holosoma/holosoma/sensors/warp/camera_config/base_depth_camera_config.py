from __future__ import annotations

from dataclasses import MISSING as _DATACLASS_MISSING
from typing import Any

# The dataclasses sentinel for required fields, typed as ``Any`` so each field
# can declare its real type while still defaulting to the sentinel.
MISSING: Any = _DATACLASS_MISSING


class BaseSensorConfigCamera:
    num_sensors: int = MISSING
    sensor_type: str = MISSING

    randomize_placement: bool = False
    min_translation: dict[str, list[float]] = MISSING
    max_translation: dict[str, list[float]] = MISSING
    min_euler_rotation_deg: dict[str, list[float]] = MISSING
    max_euler_rotation_deg: dict[str, list[float]] = MISSING


class BaseDepthCameraConfig(BaseSensorConfigCamera):
    num_sensors: int = MISSING  # number of sensors of this type

    sensor_type: str = "camera"  # sensor type

    # camera params VFOV is calcuated from the aspect ratio and HFOV
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)

    height: int = MISSING
    width: int = MISSING
    horizontal_fov_deg: float = MISSING
    max_range: float = MISSING
    min_range: float = MISSING

    # Declared (no default) because they are supplied by the robot-specific
    # subclasses; the Warp sensors require all of them.
    dynamic_meshes: bool
    asset_meshes_root: str
    base_link_frame: dict[str, str]
    offset: dict[str, dict[str, tuple[float, float, float]]]
    offset_rot_base: list[float]
    ray_cast_bodies: dict[str, str]
    add_offpath_obstacle: bool
    offpath_obstacle_meshes_root: str | None
    offpath_obstacle_bodies: dict[str, str]

    # Border crop applied before resizing the depth image. Keeping this with
    # the camera geometry prevents a generic observation term from silently
    # hard-coding one sensor's field of view.
    crop_top = 0
    crop_bottom = 0
    crop_left = 0
    crop_right = 0

    # Type of camera (depth, range, pointcloud, segmentation)
    # You can combine: (depth+segmentation), (range+segmentation), (pointcloud+segmentation)
    # Other combinations are trivial and you can add support for them in the code if you want.

    calculate_depth = True  # Get a depth image and not a range image. False will result in a range image
    return_pointcloud = (
        False  # Return a pointcloud instead of an image. Above depth option will be ignored if this is set to True
    )
    pointcloud_in_world_frame = False
    segmentation_camera = False

    # transform from sensor element coordinate frame to sensor_base_link frame
    base_offset_pos = MISSING
    base_offset_rot = MISSING

    # randomize placement of the sensor
    randomize_placement = False
    min_translation = MISSING
    max_translation = MISSING
    min_euler_rotation_deg = MISSING
    max_euler_rotation_deg = MISSING
