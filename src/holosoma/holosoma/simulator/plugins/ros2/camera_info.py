"""ROS-free pinhole-intrinsics math for the ROS2 image egress CameraInfo.

Derives the 3x3 K, 3x4 P, distortion D, and rectification R from a camera's agnostic intrinsics
(``CameraIntrinsics``: width/height/vertical_fov). Kept separate from ``ros2_image_egress.py`` so
the projection math is unit-tested without a ROS environment; the egress just copies these arrays
into a ``sensor_msgs/CameraInfo``.

Holosoma cameras are ideal pinholes (no distortion), with a symmetric vertical FOV. fy = fx
(square pixels), principal point at the image center.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from holosoma.simulator.plugins.camera_consumer import CameraIntrinsics


@dataclass(frozen=True)
class CameraInfoData:
    """Plain CameraInfo payload (lists, ROS-free) the egress copies into the message."""

    width: int
    height: int
    k: list[float]  # 3x3 row-major
    p: list[float]  # 3x4 row-major
    d: list[float]  # distortion (empty model -> zeros)
    r: list[float]  # 3x3 rectification (identity for a monocular pinhole)
    distortion_model: str = "plumb_bob"


def focal_length_px(height: int, vertical_fov_deg: float) -> float:
    """Pixel focal length from image height and vertical FOV: f = (H/2) / tan(vfov/2)."""
    half = math.radians(vertical_fov_deg) / 2.0
    return (height / 2.0) / math.tan(half)


def camera_info_from_intrinsics(intr: CameraIntrinsics) -> CameraInfoData:
    """Build the CameraInfo payload for an ideal pinhole with square pixels, centered principal pt."""
    w, h = intr.width, intr.height
    f = focal_length_px(h, intr.vertical_fov)  # fx == fy (square pixels)
    cx, cy = w / 2.0, h / 2.0

    k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
    # Monocular P = [K | 0] (no stereo baseline term on Tx).
    p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    d = [0.0, 0.0, 0.0, 0.0, 0.0]  # plumb_bob, no distortion
    return CameraInfoData(width=w, height=h, k=k, p=p, d=d, r=r)
