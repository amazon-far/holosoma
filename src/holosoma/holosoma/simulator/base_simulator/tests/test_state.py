from __future__ import annotations

import pytest
import torch

from holosoma.simulator.base_simulator.state import Columns, SimState, StateTensor

pytestmark = pytest.mark.no_sim


def _counting_state(writable: bool = True, rows: int = 2) -> tuple[SimState, torch.Tensor, list[int], list[list[int]]]:
    """A state serving ``x`` from ``backing`` (written back by flushes), counting reads and flushed rows."""
    state = SimState()
    backing = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    reads: list[int] = []
    flushed: list[list[int]] = []

    def read() -> torch.Tensor:
        reads.append(1)
        return backing

    def write(env_rows: torch.Tensor, out: torch.Tensor) -> None:
        flushed.append(env_rows.tolist())
        backing[env_rows] = out[env_rows]

    state.register("x", read, write=write if writable else None)
    return state, backing, reads, flushed


def test_reads_once_per_state_change() -> None:
    state, backing, reads, _ = _counting_state()
    served = state.get("x")
    assert torch.equal(served, backing) and state.get("x") is served and len(reads) == 1

    backing += 10
    assert not torch.equal(state.get("x"), backing) and len(reads) == 1
    state.mark_stale()
    assert torch.equal(state.get("x"), backing) and state.get("x") is served and len(reads) == 2


def test_in_place_writes_are_flushed_before_a_mutation() -> None:
    state, backing, _, _ = _counting_state()
    state.get("x")[0] = -1.0
    assert not torch.any(backing[0] == -1.0)

    state.flush_writes()
    state.mark_stale()
    assert torch.all(backing[0] == -1.0) and torch.all(state.get("x")[0] == -1.0)


def test_a_flush_writes_only_changed_rows() -> None:
    state, backing, _, flushed = _counting_state(rows=4)
    served = state.get("x")
    served[1] = -1.0
    served[3, 2] = -2.0
    served[2] = served[2].clone()  # rewritten with the same values: not a change
    state.flush_writes()
    assert flushed == [[1, 3]]
    assert torch.equal(backing, served)

    state.flush_writes()
    assert flushed == [[1, 3]]


def test_unflushable_writes_raise() -> None:
    state, _, _, _ = _counting_state(writable=False)
    state.get("x")[0] = -1.0
    with pytest.raises(RuntimeError, match="read-only"):
        state.flush_writes()

    state, _, _, _ = _counting_state()
    served = state.get("x")
    state.mark_stale()
    served[0] = -1.0
    with pytest.raises(RuntimeError, match="written in place after"):
        state.get("x")


def test_one_read_serves_several_names() -> None:
    state = SimState()
    reads: list[int] = []
    values = (torch.arange(6.0).reshape(2, 3), torch.ones(2, 4, dtype=torch.float64))

    def read() -> tuple[torch.Tensor, ...]:
        reads.append(1)
        return values

    state.register(("a", "b"), read)
    assert torch.equal(state.get("b"), values[1].float()) and state.get("b").dtype == torch.float32
    state.get("a")
    assert len(reads) == 1
    assert torch.equal(state.read_fresh("a"), values[0])

    state.register("c", lambda: values)
    with pytest.raises(ValueError, match="returned 2 tensors"):
        state.get("c")


def test_columns_return_the_selection() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    state = SimState()
    state.register("run", Columns(source, [2, 3, 4]))
    state.register("gathered", Columns(source, [4, 2]))
    state.register("callable", Columns(lambda: source, [2, 3, 4]))
    state.register("float64", Columns(source.double(), [2, 3, 4]))

    assert torch.equal(state.get("run"), source[:, 2:5])
    assert state.get("run").untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert torch.equal(state.get("gathered"), source[:, [4, 2]])
    assert state.get("float64").dtype == torch.float32

    source += 100
    assert not torch.equal(state.get("run"), source[:, 2:5])
    state.mark_stale()
    assert torch.equal(state.get("run"), source[:, 2:5])
    assert torch.equal(state.get("callable"), source[:, 2:5])


def test_state_tensor_assignment_serves_the_value() -> None:
    class Sim:
        x = StateTensor("x")

    sim = Sim()
    with pytest.raises(AttributeError):
        _ = sim.x
    value = torch.ones(3)
    sim.x = value
    assert sim.x is value


def test_mutation_flushes_first_and_marks_stale_after_even_on_error() -> None:
    state, backing, reads, _ = _counting_state()
    state.get("x")[0] = -1.0
    flushed_on_entry: list[bool] = []

    class MutationError(Exception):
        pass

    def failing_mutation() -> None:
        with state.mutation():
            flushed_on_entry.append(bool(torch.all(backing[0] == -1.0)))
            raise MutationError

    with pytest.raises(MutationError):
        failing_mutation()
    assert flushed_on_entry == [True]
    state.get("x")
    assert len(reads) == 2
