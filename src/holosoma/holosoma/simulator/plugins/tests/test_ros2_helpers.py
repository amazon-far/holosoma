"""Unit tests for the ROS-free ROS2-egress helpers (encode / camera_info / worker).

These cover the parts of the ROS2 image egress that carry real logic but need no ROS environment:
wire encoding + format rules, pinhole-K math, and the drop-oldest backpressure worker. The thin
``ros2_image_egress.py`` rclpy shell is exercised on the cluster (Phase 4), not here.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from holosoma.config_types.sensor import IsaacSimCameraConfig, IsaacSimFisheyeConfig
from holosoma.simulator.plugins.camera_consumer import CameraIntrinsics
from holosoma.simulator.plugins.ros2.camera_info import camera_info_from_intrinsics, focal_length_px
from holosoma.simulator.plugins.ros2.encode import encode_frame
from holosoma.simulator.plugins.ros2.pointcloud2 import encode_xyz_points
from holosoma.simulator.plugins.ros2.worker import PublishWorker

pytestmark = pytest.mark.no_sim


# ----- encode -----


def _rgb(h: int = 4, w: int = 6) -> npt.NDArray[np.uint8]:
    a: npt.NDArray[np.uint8] = np.zeros((h, w, 3), dtype=np.uint8)
    a[..., 0] = 10  # R
    a[..., 1] = 20  # G
    a[..., 2] = 30  # B
    return a


def test_rgb8_is_raw_rgb_no_swap() -> None:
    enc = encode_frame(_rgb(2, 2), "rgb8")
    assert not enc.compressed
    assert enc.encoding == "rgb8"
    assert (enc.height, enc.width, enc.step) == (2, 2, 6)  # step = 3*w
    # R,G,B order preserved (first pixel = 10,20,30), proving NO BGR swap on the raw path.
    assert list(enc.data[:3]) == [10, 20, 30]


def test_jpeg_is_compressed_and_decodes_back_to_rgb() -> None:
    import cv2

    enc = encode_frame(_rgb(8, 8), "jpeg", jpeg_quality=95)
    assert enc.compressed and enc.compressed_format == "jpeg"
    # Decode: cv2 gives BGR; the original solid color round-trips (B≈30,G≈20,R≈10) within JPEG noise.
    bgr = cv2.imdecode(np.frombuffer(enc.data, np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None  # imdecode returns None on invalid input (typed Optional)
    assert bgr.shape == (8, 8, 3)
    b, g, r = bgr[0, 0]
    assert abs(int(b) - 30) <= 3 and abs(int(g) - 20) <= 3 and abs(int(r) - 10) <= 3


def test_png_is_lossless_compressed() -> None:
    import cv2

    enc = encode_frame(_rgb(8, 8), "png")
    assert enc.compressed and enc.compressed_format == "png"
    bgr = cv2.imdecode(np.frombuffer(enc.data, np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None  # imdecode returns None on invalid input (typed Optional)
    assert list(map(int, bgr[0, 0])) == [30, 20, 10]  # exact: lossless, BGR


def test_depth_32fc1_preserves_meters_and_inf() -> None:
    depth: npt.NDArray[np.float32] = np.array([[1.5, np.inf], [0.0, 3.25]], dtype=np.float32)[..., None]  # [H,W,1]
    enc = encode_frame(depth, "32FC1")
    assert not enc.compressed and enc.encoding == "32FC1"
    assert (enc.height, enc.width, enc.step) == (2, 2, 8)  # step = 4*w
    out: npt.NDArray[np.float32] = np.frombuffer(enc.data, dtype=np.float32).reshape(2, 2)
    assert out[0, 0] == 1.5 and out[1, 1] == 3.25
    assert math.isinf(out[0, 1])  # +inf no-hit preserved verbatim


def test_depth_16uc1_millimeters_and_nohit_zero() -> None:
    depth: npt.NDArray[np.float32] = np.array([[1.5, np.inf], [0.001, 100.0]], dtype=np.float32)
    enc = encode_frame(depth, "16UC1")
    assert enc.encoding == "16UC1" and enc.step == 4  # 2 bytes * 2 cols
    out: npt.NDArray[np.uint16] = np.frombuffer(enc.data, dtype=np.uint16).reshape(2, 2)
    assert out[0, 0] == 1500  # 1.5 m -> 1500 mm
    assert out[0, 1] == 0  # +inf no-hit -> 0
    assert out[1, 0] == 1  # 0.001 m -> 1 mm
    assert out[1, 1] == 65535  # 100 m = 100000 mm clamped to uint16 max


def test_format_array_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="needs an"):
        encode_frame(np.zeros((4, 4), np.uint8), "rgb8")  # 2-D into an rgb format
    with pytest.raises(ValueError, match="Unknown image egress format"):
        encode_frame(_rgb(), "bogus")


def test_depth_colorized_to_rgb8_is_raw_rgb_image() -> None:
    # A depth frame in an rgb format is colorized to an [H,W,3] uint8 RGB image, then rgb-encoded.
    depth: npt.NDArray[np.float32] = np.full(
        (5, 7, 1), 1.0, np.float32
    )  # [H,W,1] float meters, as get_camera_data gives
    enc = encode_frame(depth, "rgb8", modality="depth", depth_range=(0.1, 5.0))
    assert not enc.compressed and enc.encoding == "rgb8"
    assert (enc.height, enc.width, enc.step) == (5, 7, 21)  # 3*w; colorized to 3 channels
    assert len(enc.data) == 5 * 7 * 3


def test_depth_colorized_to_jpeg_is_compressed() -> None:
    depth: npt.NDArray[np.float32] = np.full((8, 8), 1.0, np.float32)
    enc = encode_frame(depth, "jpeg", modality="depth", jpeg_quality=90, depth_colormap="turbo")
    assert enc.compressed and enc.compressed_format == "jpeg"
    assert len(enc.data) > 0


def test_depth_colormap_changes_encoded_bytes() -> None:
    # Different colormaps colorize the same depth differently, so the raw rgb bytes differ.
    depth: npt.NDArray[np.float32] = np.linspace(0.2, 4.0, 64, dtype=np.float32).reshape(8, 8)
    turbo = encode_frame(depth, "rgb8", modality="depth", depth_colormap="turbo").data
    gray = encode_frame(depth, "rgb8", modality="depth", depth_colormap="gray").data
    assert turbo != gray


def test_depth_colorized_near_brighter_than_far_grayscale() -> None:
    # Sanity that the colorization scale reaches the encoder: near reads brighter than far (gray).
    kw: dict[str, Any] = {"modality": "depth", "depth_colormap": "gray", "depth_range": (0.1, 5.0)}
    near = encode_frame(np.full((4, 4), 0.1, np.float32), "rgb8", **kw)
    far = encode_frame(np.full((4, 4), 5.0, np.float32), "rgb8", **kw)
    near_mean = np.frombuffer(near.data, np.uint8).mean()
    far_mean = np.frombuffer(far.data, np.uint8).mean()
    assert near_mean > far_mean


def test_depth_range_value_changes_encoded_bytes() -> None:
    # depth_range must actually drive normalization: the SAME depth normalizes differently under two
    # ranges, so the encoded bytes differ. Guards against depth_range being threaded but ignored.
    depth: npt.NDArray[np.float32] = np.full((4, 4), 2.0, np.float32)  # a mid-scene depth, well inside both ranges
    kw: dict[str, Any] = {"modality": "depth", "depth_colormap": "gray"}
    tight: npt.NDArray[np.uint8] = np.frombuffer(
        encode_frame(depth, "rgb8", depth_range=(0.1, 3.0), **kw).data, np.uint8
    )
    wide: npt.NDArray[np.uint8] = np.frombuffer(
        encode_frame(depth, "rgb8", depth_range=(0.1, 20.0), **kw).data, np.uint8
    )
    # Nearer end of a tight range => 2m sits darker; in a wide range 2m is close to bright. Different.
    assert not np.array_equal(tight, wide)
    assert tight.mean() != wide.mean()


# ----- camera_info -----


def test_focal_length_matches_fov() -> None:
    # 90deg vertical FOV over 200px -> f = 100 (since tan(45deg)=1).
    assert focal_length_px(200, 90.0) == pytest.approx(100.0)


def test_camera_info_k_p_layout() -> None:
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


def test_camera_info_maps_linear_f_theta_to_equidistant_and_scales_intrinsics() -> None:
    intr = CameraIntrinsics(
        width=640,
        height=480,
        vertical_fov=60.0,
        near=0.01,
        far=100.0,
        backend_config=IsaacSimCameraConfig(
            projection_type="fisheyePolynomial",
            fisheye=IsaacSimFisheyeConfig(
                nominal_width=320.0,
                nominal_height=240.0,
                optical_centre_x=150.0,
                optical_centre_y=110.0,
                max_fov=180.0,
                polynomial_b=0.01,
            ),
        ),
    )
    info = camera_info_from_intrinsics(intr)

    assert info.distortion_model == "equidistant"
    assert info.k[0] == pytest.approx(200.0)
    assert info.k[4] == pytest.approx(200.0)
    assert info.k[2] == pytest.approx(300.0)
    assert info.k[5] == pytest.approx(220.0)
    assert info.d == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-10)


def test_camera_info_equidistant_fit_reconstructs_nonlinear_f_theta_radius() -> None:
    fisheye = IsaacSimFisheyeConfig(
        nominal_width=800.0,
        nominal_height=600.0,
        optical_centre_x=400.0,
        optical_centre_y=300.0,
        max_fov=180.0,
        polynomial_b=0.004,
        polynomial_c=1e-6,
    )
    info = camera_info_from_intrinsics(
        CameraIntrinsics(
            width=800,
            height=600,
            vertical_fov=60.0,
            near=0.01,
            far=100.0,
            backend_config=IsaacSimCameraConfig(
                projection_type="fisheyeKannalaBrandtK3",
                fisheye=fisheye,
            ),
        )
    )

    radii = np.linspace(0.0, 350.0, 64)
    theta = fisheye.polynomial_b * radii + fisheye.polynomial_c * radii**2
    k1, k2, k3, k4 = info.d
    reconstructed = info.k[0] * theta * (1.0 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    assert np.max(np.abs(reconstructed - radii)) < 0.25
    assert max(abs(value) for value in info.d) > 0.01


def test_camera_info_rejects_non_monotonic_f_theta_calibration() -> None:
    intr = CameraIntrinsics(
        width=320,
        height=240,
        vertical_fov=60.0,
        near=0.01,
        far=100.0,
        backend_config=IsaacSimCameraConfig(
            projection_type="fisheyePolynomial",
            fisheye=IsaacSimFisheyeConfig(
                nominal_width=200.0,
                nominal_height=200.0,
                optical_centre_x=100.0,
                optical_centre_y=100.0,
                max_fov=180.0,
                polynomial_b=0.01,
                polynomial_c=-0.0001,
            ),
        ),
    )

    with pytest.raises(ValueError, match="non-monotonic"):
        camera_info_from_intrinsics(intr)


# ----- worker (drop-oldest backpressure) -----
#
# submit() only appends to the deque (and signals), so it works before start(). Queuing all items
# BEFORE starting the consumer thread makes these tests deterministic (no producer/consumer race),
# then we poll w.published to know the worker has drained before stop().


def _wait_published(w: PublishWorker[int], n: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while w.published < n and time.monotonic() < deadline:
        time.sleep(0.005)


def test_worker_publishes_all_when_not_overflowing() -> None:
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


def test_worker_drops_oldest_keeps_latest() -> None:
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


def test_worker_stop_is_idempotent() -> None:
    w: PublishWorker[int] = PublishWorker(lambda _: None, maxlen=2, name="t")
    w.start()
    w.stop()
    w.stop()  # must not raise


def test_worker_stop_reports_a_blocked_publisher() -> None:
    entered = threading.Event()
    release = threading.Event()

    def _publish(_: int) -> None:
        entered.set()
        release.wait()

    w = PublishWorker(_publish, maxlen=1, name="blocked")
    w.start()
    w.submit(1)
    assert entered.wait(timeout=2.0)
    with pytest.raises(RuntimeError, match="did not stop"):
        w.stop(timeout=0.01)
    release.set()
    w.stop(timeout=2.0)


def test_worker_publish_failure_does_not_block_later_item() -> None:
    seen = []

    def _publish(value: int) -> None:
        if value == 1:
            raise RuntimeError("intentional")
        seen.append(value)

    w = PublishWorker(_publish, maxlen=2, name="t")
    w.submit(1)
    w.submit(2)
    w.start()
    _wait_published(w, 1)
    w.stop()
    assert seen == [2]
    assert w.published == 1


def test_pointcloud2_xyz_layout_and_nan_density() -> None:
    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [float("nan"), float("nan"), float("nan")],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    encoded = encode_xyz_points(points, height=2, width=2)
    assert (encoded.height, encoded.width) == (2, 2)
    assert encoded.point_step == 12
    assert encoded.row_step == 24
    assert not encoded.is_dense
    decoded = np.frombuffer(encoded.data, dtype="<f4").reshape(4, 3)
    assert decoded.dtype == np.dtype("<f4")
    assert np.array_equal(decoded[:2], points[:2].astype(np.float32))
    assert np.isnan(decoded[2]).all()
    assert np.array_equal(decoded[3], points[3].astype(np.float32))


def test_pointcloud2_infinity_is_not_dense() -> None:
    encoded = encode_xyz_points(
        np.array([[1.0, float("inf"), 3.0]], dtype=np.float32),
        height=1,
        width=1,
    )
    assert not encoded.is_dense
    assert np.isinf(np.frombuffer(encoded.data, dtype="<f4")[1])


def test_pointcloud2_rejects_inconsistent_shape() -> None:
    with pytest.raises(ValueError, match="does not contain"):
        encode_xyz_points(np.zeros((3, 3), dtype=np.float32), height=2, width=2)


def test_worker_rejects_new_work_after_stop() -> None:
    received: list[int] = []
    w = PublishWorker(received.append, maxlen=2, name="t")
    w.start()
    w.stop()
    w.submit(1)

    assert received == []
    assert list(w._queue) == []


def test_worker_timeout_retains_live_thread_until_retry() -> None:
    entered = threading.Event()
    release = threading.Event()

    def _blocked_publish(_item: int) -> None:
        entered.set()
        release.wait(timeout=2.0)

    worker = PublishWorker(_blocked_publish, maxlen=1, name="blocked")
    worker.start()
    worker.submit(1)
    try:
        assert entered.wait(timeout=1.0)

        with pytest.raises(RuntimeError, match="did not stop"):
            worker.stop(timeout=0.01)

        assert worker._thread.is_alive()
    finally:
        release.set()
        worker.stop(timeout=1.0)
    assert not worker._thread.is_alive()
