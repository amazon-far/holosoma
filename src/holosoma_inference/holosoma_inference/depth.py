"""Stamp-keyed depth delay, for simulation only.

Deployment never delays depth -- see NativeContextualRunner. This exists so a sim harness can
emulate the pipeline latency that real hardware supplies on its own.
"""

from __future__ import annotations

import collections

import numpy as np


class DepthDelay:
    """Holds depth frames and optionally hands back an older one, keyed on publisher stamps.

    Training models the latency of a real depth pipeline, and on hardware that latency is already
    there: frames off the camera arrive about that stale by the time the policy sees them, so the
    freshest frame available is the right one and the delay here must be zero. Adding a delay on
    hardware doubles the staleness.

    The delay exists for simulation, where rendered or analytic depth is instantaneous and the
    pipeline latency has to be emulated to match what the policy was trained against.

    Delay is measured against the newest stamp received rather than a local clock, so no clock
    sync with the publisher is needed.
    """

    def __init__(self, latency_s: float, shape: tuple[int, int] = (58, 87)):
        if not np.isfinite(latency_s) or latency_s < 0:
            raise ValueError("latency_s must be finite and nonnegative")
        self.latency_s = float(latency_s)
        self.shape = shape
        self._frames: collections.deque[tuple[float, np.ndarray]] = collections.deque()
        self._newest_stamp: float | None = None

    def push(self, frame, stamp_s: float) -> bool:
        """Record a frame; returns False for a repeat (unchanged stamp) or a stale one."""
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != self.shape:
            raise ValueError(f"expected depth {self.shape}, got {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError("depth frame is not finite; the server must clip and fill")
        # A stamp preceding everything retained is a new epoch, not a reordered frame -- a sim
        # reset rewinds MuJoCo's clock to 0, and rejecting those would wedge the buffer for good.
        if self._frames and stamp_s < self._frames[0][0]:
            self.reset()
        if self._newest_stamp is not None and stamp_s <= self._newest_stamp:
            return False
        self._newest_stamp = stamp_s
        self._frames.append((stamp_s, frame))
        # Two latencies of history is enough to serve the delayed frame and nothing more.
        horizon = stamp_s - 2 * self.latency_s
        while len(self._frames) > 1 and self._frames[0][0] < horizon:
            self._frames.popleft()
        return True

    def latest(self) -> np.ndarray | None:
        """The newest frame at least ``latency_s`` old, or None if the buffer has not filled."""
        if self._newest_stamp is None:
            return None
        # Slop well under any real timing resolution, so a stamp landing on the boundary does not
        # flip on accumulated float error.
        cutoff = self._newest_stamp - self.latency_s + 1e-9
        delayed = [frame for stamp, frame in self._frames if stamp <= cutoff]
        return delayed[-1] if delayed else None

    def ready(self) -> bool:
        return self.latest() is not None

    def reset(self) -> None:
        self._frames.clear()
        self._newest_stamp = None
