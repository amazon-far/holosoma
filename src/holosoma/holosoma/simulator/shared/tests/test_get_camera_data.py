"""Unit tests for the shared ``BaseSimulator.get_camera_data`` accessor (pure, no simulator).

The accessor lives once on the base class and reads buffers filled by a backend's ``render_sensors``.
These pin its two sim-free failure modes and the env-id slicing, on a minimal stub, so the behavior
holds identically regardless of backend.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.camera_sensor import CameraRuntime
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _Manager:
    """Minimal SensorManager-shaped stub: name -> CameraRuntime with prefilled buffers."""

    def __init__(self, runtimes: dict[str, CameraRuntime]):
        self._runtimes = runtimes

    def has_camera(self, name: str) -> bool:
        return name in self._runtimes

    def get(self, name: str) -> CameraRuntime:
        return self._runtimes[name]

    @property
    def names(self) -> list[str]:
        return list(self._runtimes)


def _sim(manager) -> BaseSimulator:
    sim = BaseSimulator.__new__(BaseSimulator)  # skip __init__ (needs a full config)
    sim.sensor_manager = manager
    return sim


def test_no_sensor_manager_raises_not_implemented():
    sim = _sim(None)
    with pytest.raises(NotImplementedError, match="no camera 'head'"):
        sim.get_camera_data("head", "rgb")


def test_missing_camera_raises_not_implemented():
    sim = _sim(_Manager({}))
    with pytest.raises(NotImplementedError, match="no camera 'head'"):
        sim.get_camera_data("head", "rgb")


def test_missing_buffer_raises_runtime_error():
    # test stub: buffer-only runtime, no real config; no buffers filled (render_sensors never ran).
    rt = CameraRuntime(name="head", config=None)  # type: ignore[arg-type]
    sim = _sim(_Manager({"head": rt}))
    with pytest.raises(RuntimeError, match="no 'rgb' frame"):
        sim.get_camera_data("head", "rgb")


def test_returns_full_buffer_and_env_subset():
    buf = torch.arange(3 * 2 * 2 * 3, dtype=torch.uint8).reshape(3, 2, 2, 3)
    rt = CameraRuntime(name="head", config=None, buffers={"rgb": buf})  # type: ignore[arg-type]  # test stub: buffer-only runtime, no real config
    sim = _sim(_Manager({"head": rt}))
    assert torch.equal(sim.get_camera_data("head", "rgb"), buf)  # env_ids=None -> all
    subset = sim.get_camera_data("head", "rgb", env_ids=[0, 2])
    assert torch.equal(subset, buf[[0, 2]])


def test_device_none_returns_sim_buffer_directly_no_cache():
    # device=None (default) is the identity path: the sim-device buffer itself, no cross-device copy.
    buf = torch.arange(2 * 2 * 2 * 3, dtype=torch.uint8).reshape(2, 2, 2, 3)  # already cpu in this env
    rt = CameraRuntime(name="head", config=None, buffers={"rgb": buf})  # type: ignore[arg-type]  # test stub: buffer-only runtime, no real config
    sim = _sim(_Manager({"head": rt}))
    assert sim.get_camera_data("head", "rgb", device=None) is buf  # no copy
    assert rt._device_cache == {}  # nothing cached


def test_same_device_as_buffer_is_passthrough():
    # Asking for the device the buffer already lives on returns it directly (no cache entry).
    buf = torch.zeros(1, 2, 2, 3, dtype=torch.uint8)  # cpu
    rt = CameraRuntime(name="head", config=None, buffers={"rgb": buf})  # type: ignore[arg-type]  # test stub: buffer-only runtime, no real config
    sim = _sim(_Manager({"head": rt}))
    assert sim.get_camera_data("head", "rgb", device="cpu") is buf
    assert rt._device_cache == {}


def test_env_subset_slices_the_device_buffer():
    buf = torch.arange(3 * 1 * 1 * 3, dtype=torch.uint8).reshape(3, 1, 1, 3)
    rt = CameraRuntime(name="head", config=None, buffers={"rgb": buf})  # type: ignore[arg-type]  # test stub: buffer-only runtime, no real config
    sim = _sim(_Manager({"head": rt}))
    # device + env_ids compose: slice the (cached) full buffer for that device.
    subset = sim.get_camera_data("head", "rgb", env_ids=[1, 2], device="cpu")
    assert torch.equal(subset, buf[[1, 2]])
