from __future__ import annotations

import pytest
import torch

from holosoma.simulator.base_simulator.state import Columns, SimState, StateTensor, sim_state

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

    backing[0, 0] = float("nan")  # a diverged env: NaN rows are unchanged unless written
    state.mark_stale()
    state.get("x")[2] = -3.0
    state.flush_writes()
    assert flushed == [[1, 3], [2]]


def test_a_read_only_write_raises_once_and_flushes_nothing() -> None:
    state, backing, _, flushed = _counting_state()
    read_only = torch.zeros(2, 3)
    state.register("y", lambda: read_only)
    state.get("x")[0] = -1.0
    state.get("y")[0] = -1.0
    with pytest.raises(RuntimeError, match="read-only"):
        state.flush_writes()
    assert flushed == [] and not torch.any(backing == -1.0)

    state.flush_writes()  # the read-only write is discarded; the writable one is still pending
    assert flushed == [[0]] and torch.equal(state.get("y"), read_only)


def test_writes_after_the_simulation_changed_raise() -> None:
    state, _, _, _ = _counting_state()
    served = state.get("x")
    state.mark_stale()
    served[0] = -1.0
    with pytest.raises(RuntimeError, match="written in place after"):
        state.get("x")
    assert not torch.any(state.get("x") == -1.0)  # the write was discarded; the tensor works again


def test_a_first_read_inside_inference_mode_serves_a_normal_tensor() -> None:
    state, backing, _, flushed = _counting_state()
    with torch.inference_mode():
        state.get("x")
    state.get("x")[0] = -1.0
    state.flush_writes()
    assert flushed == [[0]] and torch.all(backing[0] == -1.0)


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
    assert torch.equal(state._read_fresh("a"), values[0])

    state.register("c", lambda: values)
    with pytest.raises(ValueError, match="returned 2 tensors"):
        state.get("c")


def test_columns_return_the_selection() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    state = SimState()
    state.register("run", Columns(source, [2, 3, 4]))
    state.register("gathered", Columns(source, [4, 2]))
    state.register("callable", Columns(lambda: source, [2, 3, 4]))

    assert torch.equal(state.get("run"), source[:, 2:5])
    assert torch.equal(state.get("gathered"), source[:, [4, 2]])

    source += 100
    assert not torch.equal(state.get("run"), source[:, 2:5])
    state.mark_stale()
    assert torch.equal(state.get("run"), source[:, 2:5])
    assert torch.equal(state.get("callable"), source[:, 2:5])


def test_a_flush_keeps_written_tensors_valid() -> None:
    state, backing, _, flushed = _counting_state()
    other_reads: list[int] = []

    def read_other() -> torch.Tensor:
        other_reads.append(1)
        return torch.zeros(2, 3)

    state.register("y", read_other)
    held = state.get("x")
    state.get("y")

    held[0] = -1.0
    state.flush_writes()
    held[1] = -2.0  # through the same reference, after a flush
    state.flush_writes()
    assert flushed == [[0], [1]] and torch.equal(backing, held)
    state.get("y")
    assert len(other_reads) == 2  # tensors that were not written are re-read


def test_a_flush_keeps_other_writable_tensors_valid() -> None:
    state, backing, _, flushed = _counting_state()
    other = torch.zeros(2, 3)
    state.register("z", lambda: other, write=lambda rows, out: other.__setitem__(rows, out[rows]))
    held = state.get("z")
    state.get("x")[0] = -1.0
    state.flush_writes()
    held[1] = 5.0  # a writable input is not changed by another's flush
    state.flush_writes()
    assert torch.all(other[1] == 5.0)


def test_a_stale_write_raises_at_the_next_flush() -> None:
    state, backing, _, flushed = _counting_state()
    held = state.get("x")
    state.mark_stale()  # a physics step
    held[0] = -1.0
    with pytest.raises(RuntimeError, match="written in place after"):
        state.flush_writes()
    assert flushed == [] and not torch.any(backing == -1.0)
    state.flush_writes()  # discarded: raises once


def test_assigning_a_registered_tensor_raises() -> None:
    class Sim:
        x = StateTensor("x")

    sim = Sim()
    sim.x = torch.zeros(2)  # a placeholder before registration
    sim_state(sim).register("x", lambda: torch.ones(2))
    with pytest.raises(AttributeError, match="in place"):
        sim.x = torch.zeros(2)


def test_pending_kinematics_run_once_before_a_kinematic_read() -> None:
    state = SimState()
    calls: list[int] = []
    state.register("body", lambda: torch.zeros(2, 3), needs_kinematics=True)
    state.register("q", lambda: torch.zeros(2, 3))
    state.pending_kinematics = lambda: calls.append(1)
    state.get("q")
    assert calls == []
    state.get("body")
    state.get("body")
    assert calls == [1]


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
    reads_before = len(reads)
    state.get("x")
    assert len(reads) == reads_before + 1  # marked stale despite the error
