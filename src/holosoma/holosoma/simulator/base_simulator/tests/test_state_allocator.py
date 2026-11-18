"""Custom cache storage retains SimState's freshness and write contracts."""

from __future__ import annotations

import pytest
import torch

from holosoma.simulator.base_simulator.state import DERIVED, STATE, SimState

pytestmark = pytest.mark.no_sim


def test_allocator_receives_each_name_once_and_keeps_storage() -> None:
    state = SimState(device="cpu")
    source = (torch.ones(2, 3, dtype=torch.float64), torch.full((2, 4), 2.0))
    owners: dict[str, torch.Tensor] = {}

    def allocate(name: str, value: torch.Tensor, device: torch.device | str | None, dtype: torch.dtype) -> torch.Tensor:
        assert name not in owners and device == "cpu"
        owners[name] = torch.empty(value.shape, dtype=dtype, device=device)
        return owners[name]

    state.set_buffer_allocator(allocate)
    state.register(("pose", "velocity"), lambda: source, validation_groups=(DERIVED,), invalidates=())
    held = state.get("pose")
    assert held is owners["pose"] and state.get("velocity") is owners["velocity"]
    assert held.dtype == torch.float32 and torch.equal(held, source[0].float())
    source[0].add_(5)
    state.mark_stale()
    assert state.get("pose") is held and torch.equal(held, source[0].float())
    assert tuple(owners) == ("pose", "velocity")


def test_allocator_must_precede_first_cached_read() -> None:
    state = SimState()
    state.register("pose", lambda: torch.zeros(2, 3), validation_groups=(DERIVED,), invalidates=())
    state.get("pose")
    with pytest.raises(RuntimeError, match="before the first cached read"):
        state.set_buffer_allocator(lambda _name, value, _device, _dtype: torch.empty_like(value))


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
    state.set_buffer_allocator(lambda _name, _value, _device, _dtype: storage)
    state.register("pose", lambda: torch.zeros(2, 3), validation_groups=(DERIVED,), invalidates=())
    with pytest.raises(ValueError, match="shape, the field's dtype and device"):
        state.get("pose")
    state.set_buffer_allocator(lambda _name, value, _device, _dtype: torch.empty_like(value))
    assert state.get("pose").shape == (2, 3)


def test_device_alias_is_accepted_from_a_custom_allocator() -> None:
    state = SimState(device="cpu:0")
    state.set_buffer_allocator(lambda _name, value, device, dtype: torch.empty(value.shape, dtype=dtype, device=device))
    state.register("pose", lambda: torch.zeros(2, 3), validation_groups=(DERIVED,), invalidates=())
    assert state.get("pose").device == torch.device("cpu")


def test_custom_allocation_inside_inference_mode_remains_writable() -> None:
    state = SimState()
    state.set_buffer_allocator(lambda _name, value, _device, _dtype: torch.empty_like(value))
    state.register(
        "pose",
        lambda: torch.zeros(2, 3),
        write=lambda _rows, _value: None,
        validation_groups=(STATE,),
        invalidates=(DERIVED,),
    )
    with torch.inference_mode():
        held = state.get("pose")
    assert not held.is_inference()


def test_a_field_keeps_its_dtype() -> None:
    state = SimState()
    state.register(
        "ids", lambda: torch.tensor([[1, -1]]), dtype=torch.int32, validation_groups=(DERIVED,), invalidates=()
    )
    assert state.get("ids").dtype == torch.int32 and state._read_fresh("ids").dtype == torch.int32
