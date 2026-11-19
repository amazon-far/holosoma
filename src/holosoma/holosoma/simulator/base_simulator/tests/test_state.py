from __future__ import annotations

import pytest
import torch

from holosoma.simulator.base_simulator.state import DERIVED, STATE, Columns, SimState, StateTensor

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

    state.register("x", read, write=write if writable else None, stale_with=(STATE,), stales=(DERIVED,))
    return state, backing, reads, flushed


def test_reads_once_per_state_change() -> None:
    state, backing, reads, _ = _counting_state()
    served = state.get("x")
    assert torch.equal(served, backing) and state.get("x") is served and len(reads) == 1

    backing += 10
    assert not torch.equal(state.get("x"), backing) and len(reads) == 1
    state.mark_stale((STATE, DERIVED))
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
    state.mark_stale((STATE, DERIVED))
    state.get("x")[2] = -3.0
    state.flush_writes()
    assert flushed == [[1, 3], [2]]


def test_a_read_only_write_raises_once_and_flushes_nothing() -> None:
    state, backing, _, flushed = _counting_state()
    read_only = torch.zeros(2, 3)
    state.register("y", lambda: read_only, stale_with=(DERIVED,), stales=())
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
    state.mark_stale((STATE, DERIVED))
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

    state.register(("a", "b"), read, stale_with=(DERIVED,), stales=())
    assert torch.equal(state.get("b"), values[1].float()) and state.get("b").dtype == torch.float32
    state.get("a")
    assert len(reads) == 1
    assert torch.equal(state._read_fresh("a"), values[0])

    state.register("c", lambda: values, stale_with=(DERIVED,), stales=())
    with pytest.raises(ValueError, match="returned 2 tensors"):
        state.get("c")


def test_columns_return_the_selection() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    state = SimState()
    state.register("run", Columns(source, [2, 3, 4]), stale_with=(DERIVED,), stales=())
    state.register("gathered", Columns(source, [4, 2]), stale_with=(DERIVED,), stales=())
    state.register("callable", Columns(lambda: source, [2, 3, 4]), stale_with=(DERIVED,), stales=())

    assert torch.equal(state.get("run"), source[:, 2:5])
    assert torch.equal(state.get("gathered"), source[:, [4, 2]])

    source += 100
    assert not torch.equal(state.get("run"), source[:, 2:5])
    state.mark_stale((STATE, DERIVED))
    assert torch.equal(state.get("run"), source[:, 2:5])
    assert torch.equal(state.get("callable"), source[:, 2:5])


def test_a_flush_keeps_written_tensors_valid() -> None:
    state, backing, _, flushed = _counting_state()
    other_reads: list[int] = []

    def read_other() -> torch.Tensor:
        other_reads.append(1)
        return torch.zeros(2, 3)

    state.register("y", read_other, stale_with=(DERIVED,), stales=())
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
    state.register(
        "z",
        lambda: other,
        write=lambda rows, out: other.__setitem__(rows, out[rows]),
        stale_with=(STATE,),
        stales=(DERIVED,),
    )
    held = state.get("z")
    state.get("x")[0] = -1.0
    state.flush_writes()
    held[1] = 5.0  # a writable input is not changed by another's flush
    state.flush_writes()
    assert torch.all(other[1] == 5.0)


def test_a_stale_write_raises_at_the_next_flush() -> None:
    state, backing, _, flushed = _counting_state()
    held = state.get("x")
    state.mark_stale(STATE)  # as a physics step does
    held[0] = -1.0
    with pytest.raises(RuntimeError, match="written in place after"):
        state.flush_writes()
    assert flushed == [] and not torch.any(backing == -1.0)
    state.flush_writes()  # discarded: raises once


def test_assigning_a_registered_tensor_raises() -> None:
    class Sim:
        x = StateTensor("x")

        def __init__(self) -> None:
            self.state = SimState()

    sim = Sim()
    sim.x = torch.zeros(2)  # a placeholder before registration
    sim.state.register("x", lambda: torch.ones(2), stale_with=(DERIVED,), stales=())
    with pytest.raises(AttributeError, match="in place"):
        sim.x = torch.zeros(2)


def test_mutation_flushes_first_and_marks_stale_after_even_on_error() -> None:
    state, backing, reads, _ = _counting_state()
    state.get("x")[0] = -1.0
    flushed_on_entry: list[bool] = []

    class MutationError(Exception):
        pass

    def failing_mutation() -> None:
        with state.mutation(STATE):
            flushed_on_entry.append(bool(torch.all(backing[0] == -1.0)))
            raise MutationError

    with pytest.raises(MutationError):
        failing_mutation()
    assert flushed_on_entry == [True]
    reads_before = len(reads)
    state.get("x")
    assert len(reads) == reads_before + 1  # marked stale despite the error


def test_linked_tensor_is_never_re_read_and_flushes_outside_writes() -> None:
    state, backing, reads, flushed = _counting_state(rows=3)
    outside = torch.zeros(3, 3)  # memory another writer owns
    linked = state.link({"x": outside})["x"]
    torch.autograd.graph.increment_version(linked)  # the writer reports the memory it now serves
    with state.mutation(STATE):
        pass
    assert flushed == [[0, 1, 2]] and state.get("x") is linked and len(reads) == 1  # only the first read

    outside[1] = 7.0  # the writer's change, then the version bump that reports it
    torch.autograd.graph.increment_version(linked)
    with state.mutation(STATE):
        pass
    assert flushed[-1] == [0, 1, 2] and torch.equal(backing[1], torch.full((3,), 7.0))  # every row, uncompared
    with state.mutation(STATE):  # nothing reported, nothing flushed
        pass
    assert len(flushed) == 2 and len(reads) == 1
    assert state.get("x") is linked and torch.equal(linked, outside)

    state.unlink(["x"])
    assert linked.data_ptr() != outside.data_ptr()
    backing += 1
    assert torch.equal(state.get("x"), backing)


def _output(state: SimState, name: str, value: torch.Tensor) -> torch.Tensor:
    """Register ``name`` without a read and link it onto ``value``, as a sensor output."""
    state.register(name, None, stale_with=(), stales=(), dtype=value.dtype)
    return state.link({name: value})[name]


def test_a_read_less_field_accepts_its_producer_and_rejects_other_writers() -> None:
    state = SimState()
    served = _output(state, "camera/cam/rgb", torch.zeros(2, 3, dtype=torch.uint8))
    with state.producing("camera/cam/rgb"):
        served.fill_(7)
    with state.mutation(STATE):  # the producer's write is not a violation, and the field never goes stale
        pass
    assert state.get("camera/cam/rgb") is served and int(served[0, 0]) == 7

    served[0, 0] = 1  # a consumer's write
    with pytest.raises(RuntimeError, match="read-only"):
        state.flush_writes()
    served[0, 0] = 2
    with pytest.raises(RuntimeError, match="read-only"), state.producing("camera/cam/rgb"):
        pass


def test_a_linked_field_serves_outside_memory_until_unlinked() -> None:
    state = SimState()
    outside = torch.full((2, 3), 4.0)
    linked = _output(state, "camera/cam/depth", outside)
    assert linked.data_ptr() == outside.data_ptr()
    with state.producing("camera/cam/depth"):  # the owner writes the outside memory and reports it
        outside.fill_(5.0)
    state.flush_writes()
    state.unlink(["camera/cam/depth"])
    assert linked.data_ptr() != outside.data_ptr() and torch.equal(linked, outside)
    assert state.get("camera/cam/depth") is linked  # a read-less field keeps serving its own memory


def test_moving_a_linked_field_onto_other_memory_is_not_a_write() -> None:
    state = SimState()
    image = _output(state, "camera/cam/depth", torch.zeros(2, 3))
    image.set_(torch.ones(2, 3))  # e.g. onto a co-simulation export
    state.flush_writes()  # the move alone is not a consumer write
    with state.producing("camera/cam/depth"):  # nor does it fail the producer's check
        image.fill_(2.0)
    assert torch.equal(state.get("camera/cam/depth"), torch.full((2, 3), 2.0))


@pytest.mark.parametrize(
    ("event", "joint_reread", "pose_reread"),
    [("state", True, True), ("derived", False, True), ("pose write", False, False), ("joint write", False, True)],
)
def test_staleness_cascades_from_inputs_to_derived_fields(event: str, joint_reread: bool, pose_reread: bool) -> None:
    state = SimState()
    joints, poses, mount = torch.zeros(1, 2), torch.zeros(1, 3), torch.zeros(1, 7)
    reads = {"joints": 0, "poses": 0}

    def read(name: str, value: torch.Tensor):  # type: ignore[no-untyped-def]
        def counted() -> torch.Tensor:
            reads[name] += 1
            return value

        return counted

    state.register(
        "joints",
        read("joints", joints),
        write=lambda _rows, out: joints.copy_(out),
        stale_with=(STATE,),
        stales=(DERIVED,),
    )
    state.register("poses", read("poses", poses), stale_with=(DERIVED,), stales=())  # derived from the joints
    state.register("mount", lambda: mount, write=lambda _rows, out: mount.copy_(out), stale_with=(STATE,), stales=())
    for name in ("joints", "poses", "mount"):
        state.get(name)
    if event in ("state", "derived"):
        with state.mutation(STATE if event == "state" else DERIVED):
            pass
    else:
        state.get("mount" if event == "pose write" else "joints")[0, 0] = 1.0
        state.flush_writes()
    before = dict(reads)
    state.get("joints"), state.get("poses")
    assert (reads["joints"] > before["joints"], reads["poses"] > before["poses"]) == (joint_reread, pose_reread)
