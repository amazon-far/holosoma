"""ROS-free image encoding for the ROS2 image egress.

Pure numpy/cv2 (no rclpy): given a :class:`~holosoma.simulator.plugins.camera_consumer.FramePacket` array and a
target wire format, produce the raw bytes + the ROS image fields (encoding string, dimensions,
step, and whether it is a CompressedImage). Kept here so every encoding/format rule is unit-tested
without a ROS environment; ``ros2_image_egress.py`` only wraps these bytes in messages.

Format → wire mapping (see ``ROS2ImageFormat`` in config_types/plugin.py):
  rgb8  -> raw sensor_msgs/Image, encoding "rgb8"      (R,G,B, NO swap — swap is a JPEG/cv2 artifact)
  jpeg  -> sensor_msgs/CompressedImage, format "jpeg"  (cv2 wants BGR)
  png   -> sensor_msgs/CompressedImage, format "png"   (cv2 wants BGR; lossless)
  32FC1 -> raw sensor_msgs/Image, encoding "32FC1"     (float32 meters, as get_camera_data gives)
  16UC1 -> raw sensor_msgs/Image, encoding "16UC1"     (uint16 millimeters; +inf/no-hit -> 0)

Modality decides how the array is read before it hits a format: an ``rgb`` frame goes straight to the
rgb encoders; a ``depth`` frame goes to the raw depth encoders for a depth format, or is COLORIZED to
an RGB image (via :func:`~holosoma.simulator.plugins.depth_color.colorize_depth`) and then rgb-encoded for
an rgb format. The colorized path is what gives a human-viewable depth stream over jpeg/png/rgb8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from holosoma.simulator.plugins.depth_color import DEFAULT_DEPTH_RANGE, colorize_depth

_RGB_FORMATS = ("rgb8", "jpeg", "png")
_DEPTH_FORMATS = ("32FC1", "16UC1")


@dataclass(frozen=True)
class EncodedImage:
    """Encoder output: bytes plus the ROS message fields needed to publish them.

    ``compressed`` selects the message type at the publish site: True -> CompressedImage (use
    ``compressed_format`` + ``data``); False -> Image (use ``encoding``/``height``/``width``/
    ``step`` + ``data``).
    """

    data: bytes
    compressed: bool
    compressed_format: str = ""  # "jpeg" | "png" for CompressedImage; "" for raw Image
    encoding: str = ""  # "rgb8" | "32FC1" | "16UC1" for raw Image; "" for compressed
    height: int = 0
    width: int = 0
    step: int = 0  # bytes per row for raw Image; 0 for compressed


def encode_frame(
    array: np.ndarray,
    fmt: str,
    *,
    modality: str = "rgb",
    jpeg_quality: int = 50,
    depth_colormap: str = "inferno",
    depth_range: tuple[float, float] | None = None,
) -> EncodedImage:
    """Encode one frame array to ``fmt``. Raises ValueError on a format/array mismatch.

    ``modality`` (``"rgb"`` | ``"depth"``) says how to read ``array`` before encoding: an rgb frame is
    an ``[H,W,3]`` uint8 image; a depth frame is ``[H,W]``/``[H,W,1]`` float32 meters. A depth frame
    bound for an rgb format is COLORIZED to RGB first (``depth_colormap``/``depth_range``); a depth
    frame bound for a depth format is encoded raw. ``depth_range`` ``None`` uses the shared default.
    """
    if fmt in _RGB_FORMATS:
        if modality == "depth":
            # Colorize float depth to an [H,W,3] uint8 RGB image, then encode like any rgb frame.
            array = colorize_depth(array, depth_range or DEFAULT_DEPTH_RANGE, depth_colormap)
        return _encode_rgb(array, fmt, jpeg_quality=jpeg_quality)
    if fmt in _DEPTH_FORMATS:
        return _encode_depth(array, fmt)
    raise ValueError(f"Unknown image egress format '{fmt}'. Known: {(*_RGB_FORMATS, *_DEPTH_FORMATS)}.")


def _encode_rgb(array: np.ndarray, fmt: str, *, jpeg_quality: int) -> EncodedImage:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"RGB format '{fmt}' needs an [H,W,3] array, got shape {array.shape}.")
    rgb = np.ascontiguousarray(array, dtype=np.uint8)
    h, w = rgb.shape[:2]

    if fmt == "rgb8":
        # Raw Image, R,G,B order preserved (NO BGR swap — that is only for cv2's JPEG/PNG encoders).
        return EncodedImage(data=rgb.tobytes(), compressed=False, encoding="rgb8", height=h, width=w, step=3 * w)

    # jpeg / png: cv2 encodes BGR, so swap channels first.
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:  # png
        ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError(f"cv2 failed to encode a {w}x{h} image as {fmt}.")
    return EncodedImage(data=buf.tobytes(), compressed=True, compressed_format=fmt)


def _encode_depth(array: np.ndarray, fmt: str) -> EncodedImage:
    # Accept [H,W] or [H,W,1] (get_camera_data gives [H,W,1] float32 meters); squeeze the channel.
    arr = array
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Depth format '{fmt}' needs an [H,W] or [H,W,1] array, got shape {array.shape}.")
    h, w = arr.shape[:2]

    if fmt == "32FC1":
        # float32 meters, verbatim (+inf no-hit preserved — the standard depth-image no-hit value).
        depth = np.ascontiguousarray(arr, dtype=np.float32)
        return EncodedImage(data=depth.tobytes(), compressed=False, encoding="32FC1", height=h, width=w, step=4 * w)

    # 16UC1: millimeters, uint16. +inf/no-hit -> 0 (the OpenNI/REP-118 "no measurement" value),
    # and clamp to the uint16 range so a far-but-finite hit does not wrap.
    mm = arr.astype(np.float32) * 1000.0
    mm[~np.isfinite(mm)] = 0.0
    mm = np.clip(mm, 0.0, 65535.0)
    depth_mm = np.ascontiguousarray(mm, dtype=np.uint16)
    return EncodedImage(data=depth_mm.tobytes(), compressed=False, encoding="16UC1", height=h, width=w, step=2 * w)
