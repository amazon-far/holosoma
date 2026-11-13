"""Custom cache storage retains SimState's freshness and write contracts."""

from __future__ import annotations

import pytest
import torch

from holosoma.simulator.base_simulator.state import SimState

pytestmark = pytest.mark.no_sim


def test_allocator_receives_each_name_once_and_keeps_storage() -> None:
    state = SimState(device="cpu")
    source = (torch.ones(2, 3, dtype=torch.float64), torch.full((2, 4), 2.0))
    owners: dict[str, torch.Tensor] = {}

    def allocate(name: str, value: torch.Tensor, device: torch.device | str | None) -> torch.Tensor:
        assert name not in owners and device == "cpu"
        owners[name] = torch.empty(value.shape, dtype=torch.float32, device=device)
        return owners[name]

    state.set_buffer_allocator(allocate)
    state.register(("pose", "velocity"), lambda: source)
    held = state.get("pose")
    assert held is owners["pose"] and state.get("velocity") is owners["velocity"]
    assert held.dtype == torch.float32 and torch.equal(held, source[0].float())
    source[0].add_(5)
    state.mark_stale()
    assert state.get("pose") is held and torch.equal(held, source[0].float())
    assert tuple(owners) == ("pose", "velocity")


def test_allocator_must_precede_first_cached_read() -> None:
    state = SimState()
    state.register("pose", lambda: torch.zeros(2, 3))
    state.get("pose")
    with pytest.raises(RuntimeError, match="before the first cached read"):
        state.set_buffer_allocator(lambda _name, value, _device: torch.empty_like(value))


@pytest.mark.parametrize(
    "storage",
    [
        torch.empty(2, 4),
        torch.empty(2, 3, dtype=torch.float64),
        torch.empty(2, 3, device="meta"),
    ],
    ids=["shape", "dtype", "device"],
)
def test_invalid_allocator_storage_is_rejected(storage: torch.Tensor) -> None:
    state = SimState(device="cpu")
    state.set_buffer_allocator(lambda _name, _value, _device: storage)
    state.register("pose", lambda: torch.zeros(2, 3))
    with pytest.raises(ValueError, match="shape, float32 dtype and device"):
        state.get("pose")
    state.set_buffer_allocator(lambda _name, value, _device: torch.empty_like(value))
    assert state.get("pose").shape == (2, 3)


@pytest.mark.parametrize("custom", [False, True])
def test_device_alias_keeps_default_allocation_behavior(custom: bool) -> None:
    state = SimState(device="cpu:0")
    if custom:
        state.set_buffer_allocator(
            lambda _name, value, device: torch.empty(value.shape, dtype=torch.float32, device=device)
        )
    state.register("pose", lambda: torch.zeros(2, 3))
    assert state.get("pose").device == torch.device("cpu")


def test_custom_allocation_inside_inference_mode_remains_writable() -> None:
    state = SimState()
    state.set_buffer_allocator(lambda _name, value, _device: torch.empty_like(value))
    backing = torch.zeros(2, 3)

    def write(rows: torch.Tensor, value: torch.Tensor) -> None:
        backing[rows] = value[rows]

    state.register("pose", lambda: backing, write=write)
    with torch.inference_mode():
        held = state.get("pose")
    assert not held.is_inference()
    held[1] = 7
    state.flush_writes()
    assert torch.equal(backing[1], torch.full((3,), 7.0))


def test_mutation_flushes_custom_storage_then_refreshes_same_buffer() -> None:
    state = SimState()
    state.set_buffer_allocator(lambda _name, value, _device: torch.empty_like(value))
    backing = torch.zeros(2, 3)
    written: list[list[int]] = []

    def write(rows: torch.Tensor, value: torch.Tensor) -> None:
        written.append(rows.tolist())
        backing[rows] = value[rows]

    state.register("pose", lambda: backing, write=write)
    held = state.get("pose")
    held[1] = 3
    with state.mutation():
        assert written == [[1]] and torch.equal(backing[1], torch.full((3,), 3.0))
        backing[0] = 9
    assert state.get("pose") is held and torch.equal(held, backing)
