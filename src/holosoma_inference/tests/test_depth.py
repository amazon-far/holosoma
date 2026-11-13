"""Unit tests for the contextual policy's depth input."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from holosoma_inference.depth import (
    BICUBIC_ANTIALIAS,
    FAR_M,
    FRAME_SHAPE_MISMATCH,
    HEIGHT,
    NEAR_M,
    UNFRAMED_CONTRACT,
    WIDTH,
    DepthConsumer,
    DepthSettings,
    frame_from_image,
    preprocess,
    resize_bicubic_antialias,
    settings_for,
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


@pytest.mark.parametrize(("fill", "expected"), [("far", +0.5), ("near", -0.5)])
def test_pixels_with_no_reading_take_the_configured_fill(fill, expected):
    """NaN and non-positive depth take the fill plane; -inf/+inf stay near/far under either fill."""
    frame = np.full((HEIGHT, WIDTH), 1.0, np.float32)
    frame[0, 0], frame[0, 1], frame[0, 2] = np.nan, 0.0, -3.0
    frame[1, 0], frame[1, 1] = -np.inf, np.inf
    out = preprocess(frame, invalid_fill=fill)
    assert out[0, :3] == pytest.approx([expected] * 3)
    assert out[1, 0] == pytest.approx(-0.5)
    assert out[1, 1] == pytest.approx(+0.5)


def test_the_default_fill_is_the_far_plane_the_primitive_models_trained_on():
    frame = np.full((HEIGHT, WIDTH), 1.0, np.float32)
    frame[0, 0] = np.nan
    np.testing.assert_array_equal(preprocess(frame), preprocess(frame, invalid_fill="far"))
    assert preprocess(frame)[0, 0] == pytest.approx(+0.5)


def test_rejects_an_unknown_fill():
    with pytest.raises(ValueError, match="invalid_fill"):
        preprocess(np.ones((HEIGHT, WIDTH), np.float32), invalid_fill="zero")


GOLDEN = np.load(Path(__file__).parent / "data" / "depth_resize_golden.npz")
CASES = ("noise", "ramp", "step", "up")
EXPORT_CONTRACT = {
    "near_m": 0.3,
    "far_m": 2.0,
    "invalid_fill": "near",
    "raw_shape_hw": (63, 112),
    "resize": BICUBIC_ANTIALIAS,
    "name": "gma_zed_x_mini_v1",
}


@pytest.mark.parametrize("case", CASES)
def test_the_antialiased_resize_matches_torch(case):
    """Golden from torch.nn.functional.interpolate(bicubic, antialias=True), which training used."""
    ours = resize_bicubic_antialias(GOLDEN[f"in_{case}"], (HEIGHT, WIDTH))
    np.testing.assert_allclose(ours, GOLDEN[f"out_{case}"], rtol=0, atol=5e-5)


def test_the_spline_zoom_is_not_the_trained_resize():
    """Control: the primitive path's resize differs from training's, so the golden discriminates."""
    frame = GOLDEN["in_step"]
    zoomed = ndimage.zoom(frame, (HEIGHT / frame.shape[0], WIDTH / frame.shape[1]), order=3, mode="nearest")
    assert np.abs(zoomed - GOLDEN["out_step"]).max() > 1e-2


def test_a_declared_contract_preprocesses_as_training_did():
    """Clamp, antialiased resize, clamp, normalise: the order the export declares."""
    near, far = EXPORT_CONTRACT["near_m"], EXPORT_CONTRACT["far_m"]
    s = settings_for(EXPORT_CONTRACT)
    out = preprocess(
        GOLDEN["in_noise"], s.near_m, s.far_m, s.invalid_fill, raw_shape_hw=s.raw_shape_hw, resize=s.resize
    )
    expected = (np.clip(GOLDEN["out_noise"], near, far) - near) / (far - near) - 0.5
    np.testing.assert_allclose(out, expected, rtol=0, atol=5e-5)


def test_a_model_without_a_depth_contract_keeps_the_default_preprocessing():
    assert settings_for(None) == DepthSettings(NEAR_M, FAR_M, "far", None, None)
    custom = DepthSettings(near_m=0.1, far_m=5.0)
    assert settings_for(None, custom) is custom


def test_a_declared_contract_sets_every_setting_including_a_clip_range_of_its_own():
    contract = {**EXPORT_CONTRACT, "near_m": 0.5, "far_m": 3.0, "invalid_fill": "far"}
    assert settings_for(contract) == DepthSettings(0.5, 3.0, "far", (63, 112), BICUBIC_ANTIALIAS)
    assert settings_for(EXPORT_CONTRACT).invalid_fill == "near"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"raw_shape_hw": None}, "depth_raw_shape_hw and depth_resize"),
        ({"resize": None}, "depth_raw_shape_hw and depth_resize"),
        ({"resize": "bilinear"}, "depth_resize 'bilinear' is not one of"),
    ],
)
def test_a_contract_that_does_not_say_how_to_frame_a_topic_is_refused(change, message):
    with pytest.raises(ValueError, match=UNFRAMED_CONTRACT) as refused:
        settings_for({**EXPORT_CONTRACT, **change})
    assert message in str(refused.value)


def test_a_frame_of_another_size_is_refused_for_a_declared_frame_size():
    frame = np.full((376, 672), 1.0, np.float32)  # the ZED X Mini at its native resolution
    with pytest.raises(ValueError, match=FRAME_SHAPE_MISMATCH) as refused:
        preprocess(frame, raw_shape_hw=(63, 112), resize=BICUBIC_ANTIALIAS)
    assert "got 376x672, the model expects 63x112" in str(refused.value)
    assert preprocess(np.full((63, 112), 1.0, np.float32), raw_shape_hw=(63, 112)).shape == (HEIGHT, WIDTH)


class _Policy:
    def __init__(self, contract):
        self.depth_contract = contract
        self.frames = []

    def set_depth(self, frame, stamp_s=None):
        self.frames.append((frame, stamp_s))
        return True


class _OfflineConsumer(DepthConsumer):
    """The consumer without its ROS subscription."""

    def _attach(self, topic):
        pass


def test_the_topic_consumer_follows_the_active_models_contract_across_a_switch():
    policy = _Policy(EXPORT_CONTRACT)
    consumer = _OfflineConsumer(policy)
    frame = np.full((63, 112), 1.0, np.float32)
    frame[0, 0] = np.nan
    assert consumer.push(frame, 1.0)
    first = policy.frames[-1][0]
    assert first[1:, 1:].mean() == pytest.approx((1.0 - 0.3) / 1.7 - 0.5, abs=1e-4)
    # A switch to a model with another clip range and fill: the next frame uses them.
    policy.depth_contract = {**EXPORT_CONTRACT, "near_m": 0.5, "far_m": 3.0, "invalid_fill": "far"}
    assert consumer.push(frame, 2.0)
    second = policy.frames[-1][0]
    assert second[1:, 1:].mean() == pytest.approx((1.0 - 0.5) / 2.5 - 0.5, abs=1e-4)
    assert first[0, 0] < second[0, 0]  # the hole took near, then far
    # And to a primitive model, which declares none: today's defaults, any frame size.
    policy.depth_contract = None
    assert consumer.push(np.full((HEIGHT, WIDTH), 1.0, np.float32), 3.0)
    assert (consumer.frames, consumer.rejected) == (3, 0)


def test_the_topic_consumer_drops_a_frame_the_active_contract_cannot_take():
    policy = _Policy(EXPORT_CONTRACT)
    consumer = _OfflineConsumer(policy)
    assert not consumer.push(np.full((376, 672), 1.0, np.float32), 1.0)
    # A switch to a model whose contract cannot frame a topic drops frames rather than guessing.
    policy.depth_contract = {**EXPORT_CONTRACT, "raw_shape_hw": None}
    assert not consumer.push(np.full((63, 112), 1.0, np.float32), 2.0)
    assert (consumer.frames, consumer.rejected, policy.frames) == (0, 2, [])


def test_the_topic_consumer_is_refused_up_front_for_a_contract_that_cannot_frame_a_topic():
    with pytest.raises(ValueError, match=UNFRAMED_CONTRACT):
        _OfflineConsumer(_Policy({**EXPORT_CONTRACT, "resize": None}))
