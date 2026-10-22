from __future__ import annotations

from multiprocessing import shared_memory
from uuid import uuid4

import numpy as np
import pytest

from holosoma_inference.sensors.depth_shm import DepthShmSensor

pytestmark = pytest.mark.no_sim

SHAPE = (1, 1, 2, 3)
NBYTES = int(np.prod(SHAPE)) * np.dtype(np.float32).itemsize


class _FakeSharedMemory:
    blocks: dict[str, bytearray] = {}

    def __init__(self, *, name: str, create: bool = False, size: int = 0) -> None:
        if create:
            if name in self.blocks:
                raise FileExistsError(name)
            self.blocks[name] = bytearray(size)
        elif name not in self.blocks:
            raise FileNotFoundError(name)
        self.name = name
        self.buf = self.blocks[name]
        self.size = len(self.buf)

    def close(self) -> None:
        pass

    def unlink(self) -> None:
        if self.name not in self.blocks:
            raise FileNotFoundError(self.name)
        del self.blocks[self.name]


@pytest.fixture(autouse=True)
def fake_shared_memory(monkeypatch: pytest.MonkeyPatch):
    _FakeSharedMemory.blocks.clear()
    monkeypatch.setattr(shared_memory, "SharedMemory", _FakeSharedMemory)
    yield
    _FakeSharedMemory.blocks.clear()


def _name() -> str:
    return f"holosoma_depth_test_{uuid4().hex}"


def test_sensor_attaches_only_to_exact_shape() -> None:
    name = _name()
    block = shared_memory.SharedMemory(name=name, create=True, size=NBYTES + 4)
    sensor = DepthShmSensor(shape=SHAPE, name=name)

    try:
        with pytest.raises(ValueError, match=f"is {NBYTES + 4} bytes"):
            sensor.start()
    finally:
        sensor.stop()
        block.close()
        block.unlink()


def test_sensor_returns_a_stable_copy() -> None:
    name = _name()
    block = shared_memory.SharedMemory(name=name, create=True, size=NBYTES)
    producer_view = np.ndarray(SHAPE, dtype=np.float32, buffer=block.buf)
    producer_view[:] = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    sensor = DepthShmSensor(shape=SHAPE, name=name)

    try:
        sensor.start()
        first = sensor.get_latest()
        producer_view.fill(-1.0)

        np.testing.assert_array_equal(
            first,
            np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE),
        )
        assert not np.shares_memory(first, producer_view)
    finally:
        sensor.stop()
        del producer_view
        block.close()
        block.unlink()


def test_optional_missing_sensor_serves_zeros() -> None:
    sensor = DepthShmSensor(shape=SHAPE, name=_name(), required=False)

    sensor.start()

    np.testing.assert_array_equal(sensor.get_latest(), np.zeros(SHAPE, dtype=np.float32))
