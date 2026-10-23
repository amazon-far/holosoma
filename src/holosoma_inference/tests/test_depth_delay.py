"""Unit tests for ``depth.DepthDelay``."""

import numpy as np
import pytest

from holosoma_inference.depth import DepthDelay

SHAPE = (58, 87)


def _frame(value: float) -> np.ndarray:
    return np.full(SHAPE, value, dtype=np.float32)


def test_withholds_until_a_frame_is_old_enough():
    """Engage must not start on a frame fresher than the trained latency."""
    delay = DepthDelay(0.3, shape=SHAPE)
    for tick in range(9):  # 9 frames at 30 Hz spans 0.267 s, short of 0.3
        delay.push(_frame(tick), tick / 30)
    assert not delay.ready()
    delay.push(_frame(9), 9 / 30)
    assert delay.ready()


def test_serves_the_delayed_frame_not_the_freshest():
    delay = DepthDelay(0.3, shape=SHAPE)
    for tick in range(20):
        delay.push(_frame(tick), tick / 30)
    # Newest stamp is 19/30; the newest frame at or before 19/30 - 0.3 is tick 10.
    assert delay.latest()[0, 0] == pytest.approx(10.0)


def test_zero_latency_serves_the_freshest_frame():
    delay = DepthDelay(0.0, shape=SHAPE)
    delay.push(_frame(1), 0.0)
    delay.push(_frame(2), 0.1)
    assert delay.latest()[0, 0] == pytest.approx(2.0)


def test_repeat_and_stale_stamps_are_rejected():
    """sensor_msgs/Image has no repeat flag, so an unchanged stamp is the only signal."""
    delay = DepthDelay(0.1, shape=SHAPE)
    assert delay.push(_frame(1), 1.0)
    assert delay.push(_frame(2), 1.1)
    assert not delay.push(_frame(3), 1.1)
    assert not delay.push(_frame(4), 1.05)


def test_a_stalled_publisher_holds_rather_than_going_blind():
    """Frozen stamps freeze the cutoff, so latest() keeps returning the last delayed frame."""
    delay = DepthDelay(0.3, shape=SHAPE)
    for tick in range(20):
        delay.push(_frame(tick), tick / 30)
    held = delay.latest()[0, 0]
    for _ in range(50):
        delay.push(_frame(99), 19 / 30)
    assert delay.latest()[0, 0] == pytest.approx(held)


def test_a_rewound_clock_starts_a_new_epoch():
    """A sim reset rewinds MuJoCo time to 0; the buffer must refill, not reject forever."""
    delay = DepthDelay(0.3, shape=SHAPE)
    for tick in range(20):
        delay.push(_frame(tick), tick / 30)
    assert delay.ready()
    assert delay.push(_frame(100), 0.0)
    assert not delay.ready()
    for tick in range(1, 20):
        delay.push(_frame(100 + tick), tick / 30)
    assert delay.latest()[0, 0] == pytest.approx(110.0)


def test_a_reordered_frame_inside_the_window_is_still_stale():
    """Only a stamp preceding everything retained counts as a rewind."""
    delay = DepthDelay(0.3, shape=SHAPE)
    for tick in range(20):
        delay.push(_frame(tick), tick / 30)
    assert not delay.push(_frame(99), 15 / 30)
    assert delay.latest()[0, 0] == pytest.approx(10.0)


def test_rejects_a_wrong_shape_or_non_finite_frame():
    delay = DepthDelay(0.1, shape=SHAPE)
    with pytest.raises(ValueError, match="expected depth"):
        delay.push(np.zeros((58, 86), dtype=np.float32), 0.0)
    bad = _frame(1.0)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="must clip and fill"):
        delay.push(bad, 0.0)


def test_rejects_a_negative_latency():
    with pytest.raises(ValueError, match="nonnegative"):
        DepthDelay(-0.1)
