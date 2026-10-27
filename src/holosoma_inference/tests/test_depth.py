"""Unit tests for the contextual policy's depth input."""

from types import SimpleNamespace

import numpy as np
import pytest

from holosoma_inference.depth import (
    FAR_M,
    HEIGHT,
    NEAR_M,
    WIDTH,
    frame_from_image,
    preprocess,
    stamp_seconds,
)


def test_resizes_a_camera_frame_to_the_policy_input():
    """The ZED X Mini's trained resolution is 112x63; the policy takes 58x87."""
    out = preprocess(np.full((63, 112), 1.0, np.float32))
    assert out.shape == (HEIGHT, WIDTH)
    assert out.dtype == np.float32


def test_maps_the_clip_range_onto_the_trained_band():
    near = preprocess(np.full((HEIGHT, WIDTH), NEAR_M, np.float32))
    far = preprocess(np.full((HEIGHT, WIDTH), FAR_M, np.float32))
    mid = preprocess(np.full((HEIGHT, WIDTH), (NEAR_M + FAR_M) / 2, np.float32))
    assert near.min() == pytest.approx(-0.5)
    assert far.max() == pytest.approx(+0.5)
    # Zero is the middle of the band, not an absence of reading.
    assert mid.mean() == pytest.approx(0.0, abs=1e-6)


def test_clips_beyond_the_trained_range():
    out = preprocess(np.array([[0.05, 40.0], [1.0, -3.0]], np.float32))
    assert out.min() >= -0.5 - 1e-6
    assert out.max() <= +0.5 + 1e-6


def test_non_finite_pixels_read_as_far_not_as_holes():
    """The SDK emits NaN/inf for no-return; the policy never saw holes."""
    frame = np.full((HEIGHT, WIDTH), 1.0, np.float32)
    frame[0, 0], frame[0, 1] = np.nan, np.inf
    out = preprocess(frame)
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(+0.5)
    assert out[0, 1] == pytest.approx(+0.5)


def test_preserves_the_depth_gradient_direction():
    """A ramp must stay monotonic through resize and normalisation."""
    ramp = np.linspace(NEAR_M, FAR_M, 63, dtype=np.float32)[:, None] * np.ones((1, 112), np.float32)
    rows = preprocess(ramp).mean(axis=1)
    assert np.all(np.diff(rows) > 0)


def test_rejects_a_non_2d_frame():
    with pytest.raises(ValueError, match="2-D depth frame"):
        preprocess(np.zeros((HEIGHT, WIDTH, 3), np.float32))


class _Image:
    """Shape-compatible stand-in for far_zed_driver's published sensor_msgs/Image."""

    def __init__(self, frame, encoding="32FC1", step=None):
        self.header = SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nanosec=500_000_000), frame_id="zed_depth_optical_frame"
        )
        self.height, self.width = frame.shape
        self.encoding = encoding
        self.is_bigendian = 0
        self.step = msg_step = self.width * 4 if step is None else step
        self.data = frame.astype(np.float32).tobytes()
        assert msg_step


def test_decodes_a_driver_depth_image():
    frame = np.linspace(NEAR_M, FAR_M, 63 * 112, dtype=np.float32).reshape(63, 112)
    out = frame_from_image(_Image(frame))
    assert out.shape == (63, 112)
    np.testing.assert_allclose(out, frame)


def test_stamp_is_read_in_seconds():
    assert stamp_seconds(_Image(np.ones((4, 4), np.float32))) == pytest.approx(12.5)


def test_rejects_a_non_float_encoding():
    """The driver sets 32FC1 explicitly; anything else means millimetres or colour."""
    with pytest.raises(ValueError, match="expected 32FC1"):
        frame_from_image(_Image(np.ones((4, 4), np.float32), encoding="16UC1"))


def test_rejects_padded_rows():
    with pytest.raises(ValueError, match="unpadded rows"):
        frame_from_image(_Image(np.ones((4, 4), np.float32), step=64))


def test_zero_and_negative_read_as_far_not_near():
    """The ZED reports invalid pixels as 0 as well as NaN; clipping would read 0 as an obstacle."""
    frame = np.full((HEIGHT, WIDTH), 1.0, np.float32)
    frame[0, 0], frame[0, 1] = 0.0, -1.0
    out = preprocess(frame)
    assert out[0, 0] == pytest.approx(+0.5)
    assert out[0, 1] == pytest.approx(+0.5)


def test_accepts_a_trailing_singleton_channel():
    out = preprocess(np.full((63, 112, 1), 1.0, np.float32))
    assert out.shape == (HEIGHT, WIDTH)
