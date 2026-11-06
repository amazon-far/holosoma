"""The depth producer's liveness heartbeat: wire format, staleness, and restart handling."""

from __future__ import annotations

import math
import time
from multiprocessing import shared_memory
from uuid import uuid4

import numpy as np
import pytest

from holosoma_inference.sensors.depth_heartbeat import (
    STATE_FAILED,
    STATE_OK,
    STATE_RECONNECTING,
    DepthHeartbeatMonitor,
    DepthHeartbeatPublisher,
    DepthLiveness,
    decode_heartbeat,
    default_heartbeat_endpoint,
    encode_heartbeat,
)
from holosoma_inference.sensors.depth_shm import DepthShmSensor

pytestmark = pytest.mark.no_sim

SHAPE = (1, 1, 2, 3)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


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


@pytest.fixture
def endpoint() -> str:
    return f"ipc:///tmp/hb-test-{uuid4().hex[:12]}"


def _write_block(name: str, value: float) -> None:
    """Create (or recreate) a producer's block holding ``value`` everywhere."""
    nbytes = int(np.prod(SHAPE)) * np.dtype(np.float32).itemsize
    block = _FakeSharedMemory(name=name, create=True, size=nbytes)
    np.ndarray(SHAPE, dtype=np.float32, buffer=block.buf)[:] = value


def _deliver(publisher: DepthHeartbeatPublisher, poll, seq: int, state: str = STATE_OK) -> DepthLiveness:
    """Publish until the consumer sees ``(seq, state)``: zmq PUB/SUB drops messages until it joins."""
    for _ in range(400):
        publisher.publish(seq, state)
        time.sleep(0.005)
        status = poll()
        if status.checked and status.seq == seq and status.state == state:
            return status
    raise AssertionError(f"heartbeat seq={seq} state={state} was never delivered")


def test_heartbeat_round_trips() -> None:
    beat = decode_heartbeat(encode_heartbeat(3, STATE_RECONNECTING, 12.5))
    assert (beat["seq"], beat["state"], beat["t_capture"]) == (3, STATE_RECONNECTING, 12.5)
    assert default_heartbeat_endpoint("depth_img_shm") == "ipc:///tmp/depth_img_shm.heartbeat"


@pytest.mark.parametrize(
    ("message", "match"),
    [
        (b"not json", "not UTF-8 JSON"),
        (b'{"v": 2, "seq": 1, "state": "OK"}', "Unsupported heartbeat"),
        (b'{"v": 1, "seq": 1, "state": "MAYBE"}', "needs an int seq"),
    ],
)
def test_malformed_heartbeats_are_rejected(message: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        decode_heartbeat(message)


def test_unknown_state_cannot_be_encoded() -> None:
    with pytest.raises(ValueError, match="state must be one of"):
        encode_heartbeat(1, "MAYBE", 0.0)


def test_monitor_is_unchecked_until_the_first_heartbeat(endpoint: str) -> None:
    """A producer without a heartbeat keeps today's behavior: nothing is flagged as stale."""
    clock = _Clock()
    monitor = DepthHeartbeatMonitor(endpoint, clock=clock)
    try:
        clock.now += 60.0
        status = monitor.poll()
        assert not status.checked and status.stale_s == 0.0
    finally:
        monitor.close()


def test_staleness_grows_while_seq_stops_advancing(endpoint: str) -> None:
    clock = _Clock()
    publisher = DepthHeartbeatPublisher(endpoint)
    monitor = DepthHeartbeatMonitor(endpoint, clock=clock)
    try:
        status = _deliver(publisher, monitor.poll, seq=1)
        assert status.stale_s == 0.0 and status.state == STATE_OK

        # The camera drops out: the producer keeps publishing its last seq, as RECONNECTING.
        clock.now += 0.3
        status = _deliver(publisher, monitor.poll, seq=1, state=STATE_RECONNECTING)
        assert status.stale_s == pytest.approx(0.3)

        # A new frame resets the clock.
        clock.now += 0.2
        status = _deliver(publisher, monitor.poll, seq=2)
        assert status.stale_s == 0.0 and status.state == STATE_OK
    finally:
        monitor.close()
        publisher.close()


def test_seq_going_backwards_reports_a_restart_once(endpoint: str) -> None:
    clock = _Clock()
    publisher = DepthHeartbeatPublisher(endpoint)
    monitor = DepthHeartbeatMonitor(endpoint, clock=clock)
    try:
        _deliver(publisher, monitor.poll, seq=50)
        assert _deliver(publisher, monitor.poll, seq=1).restarted
        assert not _deliver(publisher, monitor.poll, seq=2).restarted
    finally:
        monitor.close()
        publisher.close()


def test_sensor_reattaches_when_the_producer_restarts(endpoint: str) -> None:
    """A restarted producer recreates the block; the sensor must read the new one, not the old."""
    name = f"hb-shm-{uuid4().hex[:8]}"
    _write_block(name, 1.0)
    publisher = DepthHeartbeatPublisher(endpoint)
    sensor = DepthShmSensor(SHAPE, name=name, heartbeat_endpoint=endpoint)
    sensor.start()
    try:
        _deliver(publisher, sensor.liveness, seq=40)
        assert sensor.get_latest().flat[0] == 1.0

        _write_block(name, 2.0)  # the restarted producer's new block under the same name
        _deliver(publisher, sensor.liveness, seq=1)
        assert sensor.get_latest().flat[0] == 2.0
    finally:
        sensor.stop()
        publisher.close()


def test_failed_reattach_reports_failed_until_the_block_returns(endpoint: str) -> None:
    """Never keep serving a dead producer's block as if it were live."""
    name = f"hb-shm-{uuid4().hex[:8]}"
    _write_block(name, 1.0)
    publisher = DepthHeartbeatPublisher(endpoint)
    sensor = DepthShmSensor(SHAPE, name=name, heartbeat_endpoint=endpoint)
    sensor.start()
    try:
        _deliver(publisher, sensor.liveness, seq=40)
        del _FakeSharedMemory.blocks[name]  # restarted producer has not created its block yet

        for _ in range(400):
            publisher.publish(1, STATE_OK)
            time.sleep(0.005)
            status = sensor.liveness()
            if status is not None and status.seq == 1:
                break
        assert status is not None
        assert status.state == STATE_FAILED and math.isinf(status.stale_s)
        status = sensor.liveness()  # still pending: no fresh-looking status while on the old block
        assert status is not None and status.state == STATE_FAILED

        _write_block(name, 3.0)
        status = sensor.liveness()
        assert status is not None and status.state == STATE_OK
        assert sensor.get_latest().flat[0] == 3.0
    finally:
        sensor.stop()
        publisher.close()


def test_liveness_is_none_without_a_heartbeat_endpoint() -> None:
    name = f"hb-shm-{uuid4().hex[:8]}"
    _write_block(name, 1.0)
    sensor = DepthShmSensor(SHAPE, name=name)
    sensor.start()
    try:
        assert sensor.liveness() is None
    finally:
        sensor.stop()
