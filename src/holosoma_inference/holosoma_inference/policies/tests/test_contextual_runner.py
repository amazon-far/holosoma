"""Unit tests for ``NativeContextualRunner``'s per-tick depth input and engage gate.

Constructed with ``object.__new__`` because a real instance builds an SDK interface; these cover
the two methods the contextual policy adds, not the transport around them.
"""

import time
from types import SimpleNamespace

import numpy as np

from holosoma_inference.depth import DepthDelay
from holosoma_inference.policies.native_runner import NativeContextualRunner

SHAPE = (58, 87)


class _StubNative:
    def __init__(self):
        self.inputs = []
        self.commands = []

    def step_contextual(self, state, inputs, tick_ms=0):
        self.inputs.append(inputs)
        return "command"

    def command(self, name):
        self.commands.append(name)


class _StubLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message)

    warning = info


def _runner(depth):
    runner = object.__new__(NativeContextualRunner)
    runner.depth = depth
    runner._monotonic_base = time.monotonic()
    runner._synthetic_stamp = 0.0
    runner._start_pending = False
    runner.native = _StubNative()
    runner.logger = _StubLogger()
    runner._clock = None
    runner.config = SimpleNamespace(task=SimpleNamespace(rl_rate=50.0))
    return runner


def _filled(latency_s=0.3):
    delay = DepthDelay(latency_s, shape=SHAPE)
    for tick in range(20):
        delay.push(np.full(SHAPE, tick, dtype=np.float32), tick / 30)
    return delay


def test_step_passes_the_delayed_frame_flattened():
    runner = _runner(_filled())
    assert runner._step_native(np.zeros(71)) == "command"
    (inputs,) = runner.native.inputs
    assert len(inputs.depth) == SHAPE[0] * SHAPE[1]
    # Frame 10 is the delayed one at 30 Hz with 0.3 s of latency.
    assert inputs.depth[0] == 10.0


def test_step_supplies_an_input_even_with_no_frame():
    """The kernel throws without a ContextualInput, so one must go in regardless of depth."""
    runner = _runner(DepthDelay(0.3, shape=SHAPE))
    runner._step_native(np.zeros(71))
    (inputs,) = runner.native.inputs
    assert inputs.depth is None


def test_monotonic_time_is_finite_and_nonnegative():
    """set_input rejects anything else."""
    runner = _runner(_filled())
    runner._step_native(np.zeros(71))
    elapsed = runner.native.inputs[0].monotonic_time
    assert np.isfinite(elapsed) and elapsed >= 0


def test_set_depth_without_a_stamp_advances_at_the_policy_rate():
    """An in-process caller repeating one frame every tick still fills the buffer."""
    runner = _runner(DepthDelay(0.2, shape=SHAPE))
    frame = np.ones(SHAPE, dtype=np.float32)
    # Stamps start one tick in, so the span from the oldest frame reaches 0.2 s on the 11th push.
    for _ in range(10):
        assert runner.set_depth(frame)
    assert not runner.depth.ready()
    runner.set_depth(frame)
    assert runner.depth.ready()


def test_set_depth_honours_a_publisher_stamp():
    runner = _runner(DepthDelay(0.0, shape=SHAPE))
    assert runner.set_depth(np.ones(SHAPE, dtype=np.float32), stamp_s=12.5)
    assert not runner.set_depth(np.ones(SHAPE, dtype=np.float32), stamp_s=12.5)


def test_engage_is_deferred_until_the_buffer_fills_then_completes():
    """Start arrives before the first frame, so it must be held and retried, not dropped."""
    runner = _runner(DepthDelay(0.2, shape=SHAPE))
    runner._state_ready = True
    runner.interface = SimpleNamespace(get_low_state=lambda: np.zeros((1, 71)))
    runner.native.set_state = lambda *_, **__: None

    runner._handle_start_policy()
    assert runner._start_pending
    assert runner.native.commands == []

    frame = np.ones(SHAPE, dtype=np.float32)
    for _ in range(10):
        runner.set_depth(frame)
    assert runner.native.commands == []
    runner.set_depth(frame)
    assert runner.native.commands == ["START"]
    assert not runner._start_pending
