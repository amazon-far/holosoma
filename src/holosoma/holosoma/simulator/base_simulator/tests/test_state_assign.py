"""Whole-tensor assignment reaches the simulation without reading it back."""

from __future__ import annotations

import pytest
import torch

from holosoma.simulator.base_simulator.state import SimState

pytestmark = pytest.mark.no_sim


class _Backend:
    def __init__(self) -> None:
        self.pose = torch.zeros(3, 2)
        self.reads = 0
        self.written: list[list[int]] = []

    def read(self) -> torch.Tensor:
        self.reads += 1
        return self.pose

    def write(self, rows: torch.Tensor, value: torch.Tensor) -> None:
        self.written.append(rows.tolist())
        self.pose[rows] = value[rows]


def _state(backend: _Backend) -> SimState:
    state = SimState(device="cpu")
    state.register("pose", backend.read, write=backend.write)
    state.register("contact", lambda: torch.zeros(3, 1))
    return state


def test_assignment_writes_every_row_without_reading_the_simulation() -> None:
    backend = _Backend()
    state = _state(backend)
    state.get("pose")
    state.mark_stale()
    backend.reads = 0
    value = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
    state.assign({"pose": value})
    assert torch.equal(state.get("pose"), value)
    state.flush_writes()
    assert backend.reads == 0 and backend.written == [[0, 1, 2]] and torch.equal(backend.pose, value)
    assert torch.equal(state.get("pose"), value) and backend.reads == 0
    with state.mutation():
        backend.pose += 1
    assert torch.equal(state.get("pose"), value + 1) and backend.reads == 1


def test_first_assignment_reads_once_to_allocate_through_the_installed_allocator() -> None:
    backend = _Backend()
    state = _state(backend)
    allocated: list[str] = []

    def allocate(name: str, value: torch.Tensor, device: torch.device | str | None) -> torch.Tensor:
        allocated.append(name)
        return torch.empty(value.shape, dtype=torch.float32, device=device)

    state.set_buffer_allocator(allocate)
    state.assign({"pose": torch.ones(3, 2)})
    held = state.get("pose")
    state.flush_writes()
    state.assign({"pose": torch.full((3, 2), 2.0)})
    state.flush_writes()
    assert allocated == ["pose"] and state.get("pose") is held and backend.reads == 1


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"pose": torch.ones(3, 2), "contact": torch.zeros(3, 1)}, "not writable"),
        ({"pose": torch.ones(3, 2), "missing": torch.zeros(3, 1)}, "not writable"),
        ({"pose": torch.zeros(2, 2)}, "must be"),
        ({"pose": torch.zeros(3, 2, dtype=torch.float64)}, "must be"),
    ],
    ids=["read-only", "unknown", "shape", "dtype"],
)
def test_invalid_assignment_writes_nothing(values: dict[str, torch.Tensor], match: str) -> None:
    backend = _Backend()
    state = _state(backend)
    with pytest.raises(ValueError, match=match):
        state.assign(values)
    state.flush_writes()
    assert backend.written == [] and not backend.pose.any()
