"""DepthShmConsumer: shared-memory depth frames reach a contextual policy once each, never as zeros."""

from __future__ import annotations

from multiprocessing import shared_memory
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from holosoma_inference.depth import DepthShmConsumer
from holosoma_inference.run_policy import _start_depth_source
from holosoma_inference.sensors.depth_heartbeat import STATE_FAILED, STATE_OK, DepthLiveness

pytestmark = pytest.mark.no_sim

SHAPE = (1, 1, 2, 3)


class _FakeSharedMemory:
    blocks: dict[str, bytearray] = {}

    def __init__(self, *, name: str, create: bool = False, size: int = 0) -> None:
        if create:
            self.blocks[name] = bytearray(size)
        elif name not in self.blocks:
            raise FileNotFoundError(name)
        self.name = name
        self.buf = self.blocks[name]
        self.size = len(self.buf)

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def fake_shared_memory(monkeypatch: pytest.MonkeyPatch):
    _FakeSharedMemory.blocks.clear()
    monkeypatch.setattr(shared_memory, "SharedMemory", _FakeSharedMemory)
    yield
    _FakeSharedMemory.blocks.clear()


class _Policy:
    def __init__(self) -> None:
        self.frames: list[tuple[np.ndarray, float | None]] = []

    def set_depth(self, frame, stamp_s=None) -> bool:
        if stamp_s is not None and self.frames and self.frames[-1][1] is not None and stamp_s <= self.frames[-1][1]:
            return False
        self.frames.append((np.array(frame), stamp_s))
        return True


class _Heartbeat:
    """Serves scripted DepthLiveness values in place of the zmq monitor."""

    def __init__(self) -> None:
        self.next = DepthLiveness(checked=False, state=None, stale_s=0.0, seq=None, restarted=False)

    def beat(self, seq: int, t_capture: float, state: str = STATE_OK, restarted: bool = False) -> None:
        self.next = DepthLiveness(
            checked=True, state=state, stale_s=0.0, seq=seq, restarted=restarted, t_capture=t_capture
        )

    def poll(self) -> DepthLiveness:
        return self.next

    def close(self) -> None:
        pass


def _write(name: str, value: float) -> None:
    block = _FakeSharedMemory.blocks.setdefault(name, bytearray(int(np.prod(SHAPE)) * 4))
    np.ndarray(SHAPE, dtype=np.float32, buffer=block)[:] = value


def _consumer(*, heartbeat: bool = True) -> tuple[DepthShmConsumer, _Policy, _Heartbeat, str]:
    name = f"depth_test_{uuid4().hex}"
    policy = _Policy()
    consumer = DepthShmConsumer(policy, shape=SHAPE, name=name, heartbeat=heartbeat)
    fake = _Heartbeat()
    if heartbeat:
        consumer._sensor._heartbeat = fake  # stand in for the zmq monitor start() would create
    return consumer, policy, fake, name


def test_frames_are_pushed_once_per_heartbeat_seq_with_their_capture_time() -> None:
    consumer, policy, heartbeat, name = _consumer()
    assert not consumer.poll_once()  # no block yet: an unmapped read would be zeros

    _write(name, 0.25)
    assert not consumer.poll_once()  # block mapped, but no heartbeat yet
    heartbeat.beat(seq=1, t_capture=10.0)
    assert consumer.poll_once()
    assert not consumer.poll_once()  # same seq: nothing new
    _write(name, -0.25)
    heartbeat.beat(seq=2, t_capture=10.1)
    assert consumer.poll_once()

    assert [stamp for _, stamp in policy.frames] == [10.0, 10.1]
    assert policy.frames[0][0].shape == SHAPE[2:]
    np.testing.assert_array_equal(policy.frames[1][0], np.full(SHAPE[2:], -0.25, dtype=np.float32))


def test_a_failed_producer_pushes_nothing() -> None:
    consumer, policy, heartbeat, name = _consumer()
    _write(name, 0.25)
    heartbeat.beat(seq=1, t_capture=10.0, state=STATE_FAILED)

    assert not consumer.poll_once()
    assert policy.frames == []


def test_a_restarted_producer_resumes_from_its_new_sequence() -> None:
    consumer, policy, heartbeat, name = _consumer()
    _write(name, 0.25)
    heartbeat.beat(seq=7, t_capture=10.0)
    assert consumer.poll_once()

    # The producer restarts: seq goes back to 1, with a later capture time.
    heartbeat.beat(seq=1, t_capture=20.0, restarted=True)
    assert consumer.poll_once()
    assert [stamp for _, stamp in policy.frames] == [10.0, 20.0]


def test_without_a_heartbeat_every_poll_is_pushed_unstamped() -> None:
    consumer, policy, _, name = _consumer(heartbeat=False)
    _write(name, 0.25)

    assert consumer.poll_once() and consumer.poll_once()
    assert [stamp for _, stamp in policy.frames] == [None, None]


def test_a_depth_source_needs_a_contextual_policy() -> None:
    blind = SimpleNamespace(contextual=None)
    with pytest.raises(ValueError, match="contextual"):
        _start_depth_source(object(), blind, "depth_img_shm", None)
    contextual = SimpleNamespace(contextual=SimpleNamespace(depth_height=2, depth_width=3))
    with pytest.raises(ValueError, match="not both"):
        _start_depth_source(_Policy(), contextual, "depth_img_shm", "depth/topic")
    assert _start_depth_source(_Policy(), contextual, None, None) is None


@pytest.fixture
def logs():
    from loguru import logger

    records: list[str] = []
    sink = logger.add(lambda message: records.append(message.record["message"]), level="WARNING")
    yield records
    logger.remove(sink)


def test_a_missing_heartbeat_is_reported_once_as_no_frames_delivered(logs: list[str]) -> None:
    now = [0.0]
    name = f"depth_test_{uuid4().hex}"
    consumer = DepthShmConsumer(_Policy(), shape=SHAPE, name=name, clock=lambda: now[0])
    consumer._sensor._heartbeat = _Heartbeat()
    _write(name, 0.25)

    for now[0] in (0.0, 1.0, 2.5, 5.0):
        assert not consumer.poll_once()

    warnings = [m for m in logs if "no depth frames will be delivered" in m]
    assert len(warnings) == 1 and "--no-depth-heartbeat" in warnings[0]


def test_an_error_on_the_polling_thread_is_logged_and_raised_at_stop(logs: list[str]) -> None:
    import time

    consumer, _, _, name = _consumer(heartbeat=False)
    _write(name, 0.25)

    def broken():
        raise ValueError("size mismatch")

    consumer._sensor.get_latest = broken  # type: ignore[method-assign]
    consumer.start()
    for _ in range(100):
        if consumer.error is not None:
            break
        time.sleep(0.01)

    assert isinstance(consumer.error, ValueError)
    assert any(name in m and "size mismatch" in m for m in logs)
    with pytest.raises(RuntimeError, match=name) as raised:
        consumer.stop()
    assert raised.value.__cause__ is consumer.error


def test_waiting_for_a_restarted_producer_logs_once(logs: list[str]) -> None:
    consumer, policy, heartbeat, name = _consumer()
    _write(name, 0.25)
    heartbeat.beat(seq=7, t_capture=10.0)
    assert consumer.poll_once()

    del _FakeSharedMemory.blocks[name]  # the producer restarts and has not recreated its block yet
    heartbeat.beat(seq=1, t_capture=20.0, restarted=True)
    for _ in range(50):
        assert not consumer.poll_once()
    _write(name, -0.25)
    heartbeat.beat(seq=2, t_capture=20.1)
    assert consumer.poll_once()

    assert sum("waiting for the restarted producer" in m for m in logs) == 1
    assert not any("zero depth" in m for m in logs)
    assert [stamp for _, stamp in policy.frames] == [10.0, 20.1]


def test_no_depth_heartbeat_reads_the_block_without_one() -> None:
    contextual = SimpleNamespace(contextual=SimpleNamespace(depth_height=2, depth_width=3))
    source = _start_depth_source(_Policy(), contextual, "depth_img_shm", None, depth_heartbeat=False)
    try:
        assert source._sensor._heartbeat_endpoint is None
    finally:
        source.stop()
