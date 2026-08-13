"""Unit tests for the ROS-free ROS2-egress helpers (encode / camera_info / worker).

These cover the parts of the ROS2 image egress that carry real logic but need no ROS environment:
wire encoding + format rules, pinhole-K math, and the drop-oldest backpressure worker. The thin
``ros2_image_egress.py`` rclpy shell is exercised on the cluster (Phase 4), not here.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from holosoma.simulator.plugins.camera_consumer import CameraIntrinsics
from holosoma.simulator.plugins.ros2.camera_info import camera_info_from_intrinsics, focal_length_px
from holosoma.simulator.plugins.ros2.encode import encode_frame
from holosoma.simulator.plugins.ros2.worker import PublishWorker

pytestmark = pytest.mark.no_sim


# ----- encode -----


def _rgb(h=4, w=6):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[..., 0] = 10  # R
    a[..., 1] = 20  # G
    a[..., 2] = 30  # B
    return a


def test_rgb8_is_raw_rgb_no_swap():
    enc = encode_frame(_rgb(2, 2), "rgb8")
    assert not enc.compressed
    assert enc.encoding == "rgb8"
    assert (enc.height, enc.width, enc.step) == (2, 2, 6)  # step = 3*w
    # R,G,B order preserved (first pixel = 10,20,30), proving NO BGR swap on the raw path.
    assert list(enc.data[:3]) == [10, 20, 30]


def test_jpeg_is_compressed_and_decodes_back_to_rgb():
    import cv2

    enc = encode_frame(_rgb(8, 8), "jpeg", jpeg_quality=95)
    assert enc.compressed and enc.compressed_format == "jpeg"
    # Decode: cv2 gives BGR; the original solid color round-trips (B≈30,G≈20,R≈10) within JPEG noise.
    bgr = cv2.imdecode(np.frombuffer(enc.data, np.uint8), cv2.IMREAD_COLOR)
    assert bgr.shape == (8, 8, 3)
    b, g, r = bgr[0, 0]
    assert abs(int(b) - 30) <= 3 and abs(int(g) - 20) <= 3 and abs(int(r) - 10) <= 3


def test_png_is_lossless_compressed():
    import cv2

    enc = encode_frame(_rgb(8, 8), "png")
    assert enc.compressed and enc.compressed_format == "png"
    bgr = cv2.imdecode(np.frombuffer(enc.data, np.uint8), cv2.IMREAD_COLOR)
    assert list(map(int, bgr[0, 0])) == [30, 20, 10]  # exact: lossless, BGR


def test_depth_32fc1_preserves_meters_and_inf():
    depth = np.array([[1.5, np.inf], [0.0, 3.25]], dtype=np.float32)[..., None]  # [H,W,1]
    enc = encode_frame(depth, "32FC1")
    assert not enc.compressed and enc.encoding == "32FC1"
    assert (enc.height, enc.width, enc.step) == (2, 2, 8)  # step = 4*w
    out = np.frombuffer(enc.data, dtype=np.float32).reshape(2, 2)
    assert out[0, 0] == 1.5 and out[1, 1] == 3.25
    assert math.isinf(out[0, 1])  # +inf no-hit preserved verbatim


def test_depth_16uc1_millimeters_and_nohit_zero():
    depth = np.array([[1.5, np.inf], [0.001, 100.0]], dtype=np.float32)
    enc = encode_frame(depth, "16UC1")
    assert enc.encoding == "16UC1" and enc.step == 4  # 2 bytes * 2 cols
    out = np.frombuffer(enc.data, dtype=np.uint16).reshape(2, 2)
    assert out[0, 0] == 1500  # 1.5 m -> 1500 mm
    assert out[0, 1] == 0  # +inf no-hit -> 0
    assert out[1, 0] == 1  # 0.001 m -> 1 mm
    assert out[1, 1] == 65535  # 100 m = 100000 mm clamped to uint16 max


def test_format_array_mismatch_raises():
    with pytest.raises(ValueError, match="needs an"):
        encode_frame(np.zeros((4, 4), np.uint8), "rgb8")  # 2-D into an rgb format
    with pytest.raises(ValueError, match="Unknown image egress format"):
        encode_frame(_rgb(), "bogus")


def test_depth_colorized_to_rgb8_is_raw_rgb_image():
    # A depth frame in an rgb format is colorized to an [H,W,3] uint8 RGB image, then rgb-encoded.
    depth = np.full((5, 7, 1), 1.0, np.float32)  # [H,W,1] float meters, as get_camera_data gives
    enc = encode_frame(depth, "rgb8", modality="depth", depth_range=(0.1, 5.0))
    assert not enc.compressed and enc.encoding == "rgb8"
    assert (enc.height, enc.width, enc.step) == (5, 7, 21)  # 3*w; colorized to 3 channels
    assert len(enc.data) == 5 * 7 * 3


def test_depth_colorized_to_jpeg_is_compressed():
    depth = np.full((8, 8), 1.0, np.float32)
    enc = encode_frame(depth, "jpeg", modality="depth", jpeg_quality=90, depth_colormap="turbo")
    assert enc.compressed and enc.compressed_format == "jpeg"
    assert len(enc.data) > 0


def test_depth_colormap_changes_encoded_bytes():
    # Different colormaps colorize the same depth differently, so the raw rgb bytes differ.
    depth = np.linspace(0.2, 4.0, 64, dtype=np.float32).reshape(8, 8)
    turbo = encode_frame(depth, "rgb8", modality="depth", depth_colormap="turbo").data
    gray = encode_frame(depth, "rgb8", modality="depth", depth_colormap="gray").data
    assert turbo != gray


def test_depth_colorized_near_brighter_than_far_grayscale():
    # Sanity that the colorization scale reaches the encoder: near reads brighter than far (gray).
    kw = {"modality": "depth", "depth_colormap": "gray", "depth_range": (0.1, 5.0)}
    near = encode_frame(np.full((4, 4), 0.1, np.float32), "rgb8", **kw)
    far = encode_frame(np.full((4, 4), 5.0, np.float32), "rgb8", **kw)
    near_mean = np.frombuffer(near.data, np.uint8).mean()
    far_mean = np.frombuffer(far.data, np.uint8).mean()
    assert near_mean > far_mean


def test_depth_range_value_changes_encoded_bytes():
    # depth_range must actually drive normalization: the SAME depth normalizes differently under two
    # ranges, so the encoded bytes differ. Guards against depth_range being threaded but ignored.
    depth = np.full((4, 4), 2.0, np.float32)  # a mid-scene depth, well inside both ranges
    kw = {"modality": "depth", "depth_colormap": "gray"}
    tight = np.frombuffer(encode_frame(depth, "rgb8", depth_range=(0.1, 3.0), **kw).data, np.uint8)
    wide = np.frombuffer(encode_frame(depth, "rgb8", depth_range=(0.1, 20.0), **kw).data, np.uint8)
    # Nearer end of a tight range => 2m sits darker; in a wide range 2m is close to bright. Different.
    assert not np.array_equal(tight, wide)
    assert tight.mean() != wide.mean()


# ----- camera_info -----


def test_focal_length_matches_fov():
    # 90deg vertical FOV over 200px -> f = 100 (since tan(45deg)=1).
    assert focal_length_px(200, 90.0) == pytest.approx(100.0)


def test_camera_info_k_p_layout():
    intr = CameraIntrinsics(width=320, height=240, vertical_fov=60.0, near=0.01, far=100.0)
    info = camera_info_from_intrinsics(intr)
    f = focal_length_px(240, 60.0)
    assert info.width == 320 and info.height == 240
    # K = [f 0 cx; 0 f cy; 0 0 1]; principal point centered.
    assert info.k[0] == pytest.approx(f) and info.k[4] == pytest.approx(f)
    assert info.k[2] == pytest.approx(160.0) and info.k[5] == pytest.approx(120.0)
    assert info.k[8] == 1.0
    # P = [K | 0]: first 3 cols match K rows, 4th col is zero.
    assert info.p[3] == 0.0 and info.p[7] == 0.0 and info.p[11] == 0.0
    assert info.p[0] == pytest.approx(f)
    assert info.r == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert info.d == [0.0, 0.0, 0.0, 0.0, 0.0]


# ----- worker (drop-oldest backpressure) -----
#
# submit() only appends to the deque (and signals), so it works before start(). Queuing all items
# BEFORE starting the consumer thread makes these tests deterministic (no producer/consumer race),
# then we poll w.published to know the worker has drained before stop().


def _wait_published(w: PublishWorker, n: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while w.published < n and time.monotonic() < deadline:
        time.sleep(0.005)


def test_worker_publishes_all_when_not_overflowing():
    seen: list[int] = []
    # maxlen >= count, so nothing is dropped even though everything is queued before draining.
    w = PublishWorker(seen.append, maxlen=64, name="t")
    for i in range(20):
        w.submit(i)
    w.start()
    _wait_published(w, 20)
    w.stop()
    assert seen == list(range(20))
    assert w.dropped == 0
    assert w.published == 20


def test_worker_drops_oldest_keeps_latest():
    # Queue 10 items into a maxlen=2 queue BEFORE starting the consumer: deque(maxlen) evicts the
    # oldest on each append, so only [8, 9] remain and 0..7 are counted as drops. Deterministic.
    received: list[int] = []
    w = PublishWorker(received.append, maxlen=2, name="t")
    for i in range(10):
        w.submit(i)
    assert w.dropped == 8  # 0..7 evicted, counted (not silent)
    w.start()
    _wait_published(w, 2)
    w.stop()
    assert received == [8, 9]  # the two NEWEST survived; oldest dropped, order preserved


def test_worker_stop_is_idempotent():
    w = PublishWorker(lambda _: None, maxlen=2, name="t")
    w.start()
    w.stop()
    w.stop()  # must not raise
