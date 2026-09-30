"""Unit tests for ``Ros2DepthConsumer`` frame-delay buffering.

These exercise the ``frame_delay_ms`` look-back path without ROS traffic by
driving ``_store`` directly and stubbing the monotonic clock, so they cover the
selection logic (``get_latest`` returns the freshest set at least the configured
delay old) independent of any publisher.

``sensors`` imports cv2/message_filters/rclpy/numpy at module load, so skip
cleanly when those aren't present (host env); the bazel pytest_test target has
them.
"""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("rclpy")
pytest.importorskip("message_filters")

import numpy as np

from holosoma_service.policy_control import sensors
from holosoma_service.policy_control.sensors import Ros2DepthConsumer


class _FakeNode:
    """Records subscriptions; ApproximateTimeSynchronizer is never invoked."""

    def create_subscription(self, *args, **kwargs):
        return object()


def _make(monkeypatch, frame_delay_ms, clock):
    """Build a single-camera consumer with a controllable monotonic clock.

    ``clock`` is a one-element list holding the current fake time; tests mutate
    ``clock[0]`` to advance it. Patches ``time.monotonic`` in the sensors module.
    """
    monkeypatch.setattr(sensors.time, "monotonic", lambda: clock[0])
    return Ros2DepthConsumer(
        _FakeNode(),
        topics=["/depth"],
        resized_height=4,
        resized_width=4,
        timeout=0.0,  # disable staleness gating; isolate the delay logic
        frame_delay_ms=frame_delay_ms,
    )


def _push(consumer, value):
    """Store a frame whose pixels are all ``value`` so we can identify it."""
    consumer._store([np.full((4, 4), float(value), dtype=np.float32)])


def test_zero_delay_returns_freshest(monkeypatch):
    clock = [0.0]
    c = _make(monkeypatch, frame_delay_ms=0.0, clock=clock)
    _push(c, 1)
    clock[0] = 0.1
    _push(c, 2)
    out = c.get_latest()
    assert out is not None
    # value 2 was the most recent; normalize() shifts values, so compare the
    # newest stored set rather than raw pixels.
    assert np.array_equal(out, c._buffer[-1][1])


def test_delay_returns_older_frame(monkeypatch):
    clock = [0.0]
    c = _make(monkeypatch, frame_delay_ms=200.0, clock=clock)
    _push(c, 1)  # t=0.0
    clock[0] = 0.1
    _push(c, 2)  # t=0.1
    clock[0] = 0.25
    _push(c, 3)  # t=0.25
    # now=0.25, target=0.25-0.2=0.05 -> freshest set at/under 0.05 is the t=0.0 one.
    out = c.get_latest()
    assert out is not None
    assert np.array_equal(out, c._buffer[0][1])


def test_delay_none_until_enough_history(monkeypatch):
    clock = [0.0]
    c = _make(monkeypatch, frame_delay_ms=200.0, clock=clock)
    _push(c, 1)  # t=0.0
    clock[0] = 0.05  # only 50ms elapsed; nothing is 200ms old yet
    assert c.get_latest() is None


def test_empty_buffer_returns_none(monkeypatch):
    clock = [0.0]
    c = _make(monkeypatch, frame_delay_ms=0.0, clock=clock)
    assert c.get_latest() is None


def test_timeout_zeros_on_dead_stream(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(sensors.time, "monotonic", lambda: clock[0])
    c = Ros2DepthConsumer(_FakeNode(), topics=["/depth"], resized_height=4, resized_width=4, timeout=0.5)
    _push(c, 1)
    clock[0] = 1.0  # newest set is now 1s old, past the 0.5s timeout
    assert c.get_latest() is None


def test_negative_delay_rejected():
    with pytest.raises(ValueError, match="frame_delay_ms must be >= 0"):
        Ros2DepthConsumer(_FakeNode(), topics=["/depth"], frame_delay_ms=-1.0)


# --- no-return pixel sanitization (REP-117 non-finite depth) --------------------


def _consumer_for_preprocess(size=8, near=0.1, far=2.0):
    """A same-size consumer (resized_h/w == frame size). The bicubic resize is not
    an identity even at equal size, so per-pixel value assertions use UNIFORM
    frames (bicubic of a constant region is that constant)."""
    return Ros2DepthConsumer(
        _FakeNode(), topics=["/depth"], resized_height=size, resized_width=size, near_clip=near, far_clip=far
    )


def _normalize(value, near=0.1, far=2.0):
    """Mirror _preprocess's clip+normalize for an already-sanitized metric depth."""
    clipped = min(max(value, near), far)
    return (clipped - near) / (far - near) - 0.5


@pytest.mark.parametrize(
    "fill,expected_metric",
    [
        (np.inf, 2.0),  # +Inf -> far
        (-np.inf, 0.1),  # -Inf -> near
        (np.nan, 2.0),  # NaN  -> far
        (0.0, 2.0),  # <=0  -> far
        (-5.0, 2.0),  # negative -> far
    ],
)
def test_preprocess_maps_no_return_fill(fill, expected_metric):
    """Each no-return sentinel maps to the right clip; a uniform frame stays uniform
    through the bicubic resize, so we can assert the exact normalized value."""
    c = _consumer_for_preprocess()
    out = c._preprocess(np.full((8, 8), fill, dtype=np.float32))
    assert out.shape == (1, 8, 8)
    assert np.all(np.isfinite(out))
    assert out == pytest.approx(_normalize(expected_metric), abs=1e-5)


def test_preprocess_finite_uniform_depth_value():
    """A uniform in-range depth normalizes to its exact value (resize-invariant)."""
    c = _consumer_for_preprocess()
    out = c._preprocess(np.full((8, 8), 0.5, dtype=np.float32))
    assert out == pytest.approx(_normalize(0.5), abs=1e-5)


def test_preprocess_output_always_in_range():
    """Any input (mixed finite + no-return) yields a finite stack within [-0.5, 0.5]."""
    c = _consumer_for_preprocess()
    frame = np.array([[np.inf, -np.inf, 0.5], [np.nan, 0.0, 5.0], [-1.0, 1.0, 0.05]], dtype=np.float32)
    c2 = _consumer_for_preprocess(size=3)
    out = c2._preprocess(frame)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -0.5 - 1e-4) and np.all(out <= 0.5 + 1e-4)


def test_preprocess_does_not_mutate_readonly_source():
    """Decoded frames are read-only np.frombuffer views; _preprocess must copy."""
    c = _consumer_for_preprocess(size=2)
    frame = np.array([[np.inf, 1.0], [np.nan, 0.0]], dtype=np.float32)
    frame.setflags(write=False)  # emulate the np.frombuffer view from _decode_depth
    out = c._preprocess(frame)  # must not raise
    assert np.all(np.isfinite(out))


def test_preprocess_all_no_return_frame_is_finite():
    """A frame with no valid returns still yields a finite, in-range stack."""
    c = _consumer_for_preprocess()
    out = c._preprocess(np.full((8, 8), np.nan, dtype=np.float32))
    assert np.all(np.isfinite(out))
    assert np.all(out >= -0.5) and np.all(out <= 0.5)
