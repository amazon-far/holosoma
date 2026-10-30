from __future__ import annotations

import pytest
import torch

from holosoma.simulator.base_simulator.state import SimState, StateTensor

pytestmark = pytest.mark.no_sim


def _counting_state(
    sources: tuple[str, ...] = ("joints",), writable: bool = True
) -> tuple[SimState, torch.Tensor, list[int]]:
    """A state serving ``x`` from ``backing`` (written back by flushes), counting reads."""
    state = SimState()
    backing = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    reads: list[int] = []

    def read(out: torch.Tensor) -> None:
        reads.append(1)
        out.copy_(backing)

    write = backing.copy_ if writable else None
    state.lazy("x", read, torch.zeros(2, 3), sources=sources, write=write)  # type: ignore[arg-type]
    return state, backing, reads


def test_lazy_reads_once_per_change_of_its_sources() -> None:
    state, backing, reads = _counting_state()
    served = state.get("x")
    assert torch.equal(served, backing) and state.get("x") is served and len(reads) == 1

    backing += 10
    state.changed("bodies")
    assert len(reads) == 1 and not torch.equal(state.get("x"), backing)

    state.changed("joints")
    assert torch.equal(state.get("x"), backing) and state.get("x") is served and len(reads) == 2


def test_in_place_writes_are_flushed_before_a_mutation_of_their_sources() -> None:
    state, backing, _ = _counting_state()
    state.get("x")[0] = -1.0
    state.prepare("bodies")
    assert not torch.any(backing[0] == -1.0)

    state.prepare("joints")
    state.changed("joints")
    assert torch.all(backing[0] == -1.0) and torch.all(state.get("x")[0] == -1.0)


def test_owned_writes_are_left_to_the_mutation() -> None:
    state, backing, _ = _counting_state()
    state.get("x")[0] = -1.0
    state.prepare("joints", owned=("x",))
    state.changed("joints", written=("x",))
    assert not torch.any(backing[0] == -1.0)
    state.get("x")


def test_unflushable_writes_raise() -> None:
    state, _, _ = _counting_state(writable=False)
    state.get("x")[0] = -1.0
    with pytest.raises(RuntimeError, match="read-only"):
        state.prepare()

    state, _, _ = _counting_state()
    served = state.get("x")
    state.changed()
    served[0] = -1.0
    with pytest.raises(RuntimeError, match="written in place after"):
        state.get("x")


def test_changed_without_sources_stales_every_lazy_tensor() -> None:
    state, backing, reads = _counting_state(sources=())
    state.get("x")
    state.changed()
    state.get("x")
    assert len(reads) == 2


def test_grouped_names_share_one_read() -> None:
    state = SimState()
    reads: list[int] = []
    root = torch.zeros(2, 13)

    def read(root: torch.Tensor, _quat: torch.Tensor) -> None:
        reads.append(1)
        root[:] = torch.arange(13.0)

    state.lazy(("root", "quat"), read, (root, root[:, 3:7]), sources=("root",))
    assert torch.equal(state.get("quat")[0], torch.arange(3.0, 7.0))
    state.get("root")
    assert len(reads) == 1
    assert torch.equal(state.read_fresh("quat"), root[:, 3:7])


def test_columns_are_live_only_for_a_contiguous_float32_run() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    state = SimState()
    state.columns("live", source, [2, 3, 4], sources=("joints",))
    state.columns("gathered", source, [4, 2], sources=("joints",))
    state.columns("callable", lambda: source, [2, 3, 4], sources=("joints",))
    state.columns("float64", source.double(), [2, 3, 4], sources=("joints",))

    assert state.is_live("live") and state.get("live").data_ptr() == source[:, 2].data_ptr()
    assert not any(state.is_live(name) for name in ("gathered", "callable", "float64"))
    assert torch.equal(state.get("gathered"), source[:, [4, 2]])
    assert state.get("float64").dtype == torch.float32

    source += 100
    assert torch.equal(state.get("live"), source[:, 2:5])
    state.changed("joints")
    assert torch.equal(state.get("callable"), source[:, 2:5])


def test_state_tensor_assignment_binds_a_live_tensor() -> None:
    class Sim:
        x = StateTensor("x")

    sim = Sim()
    with pytest.raises(AttributeError):
        _ = sim.x
    value = torch.ones(3)
    sim.x = value
    assert sim.x is value
