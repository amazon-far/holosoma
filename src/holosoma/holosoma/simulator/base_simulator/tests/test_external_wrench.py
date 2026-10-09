"""Unit tests for the external-force accumulator: accumulation, broadcast, and the flush+zero /
write-only-touched-columns contract (pure, no simulator, CPU-only).

Pinned against a minimal ``BaseSimulator`` subclass that records what ``_write_external_wrench_native``
received. Per-backend native writes are covered by the behavioral harness (behavior_assert.py).
"""

from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.external_wrench import ExternalWrenchAccumulator, WrenchTarget
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _FakeSim(BaseSimulator):
    """Minimal BaseSimulator: 2 robot bodies + one known object actor, on CPU.

    Bypasses ``__init__`` (which needs a full tyro config) and sets only what the external-force
    API reads, including the ``_external_wrench`` accumulator that ``__init__`` would create.
    """

    class _FakeRegistry:
        def list_all_objects(self) -> list[str]:
            return ["robot", "obj0"]

    def __init__(self, num_envs: int = 3) -> None:
        self.num_envs = num_envs
        self.sim_device = "cpu"
        self.body_names = ["pelvis", "torso_link"]
        self.object_registry = self._FakeRegistry()  # type: ignore[assignment]  # test double
        self._external_wrench = ExternalWrenchAccumulator(self)
        self._flushed: list[tuple[str, bool, list[str], torch.Tensor]] = []

    def _write_external_wrench_native(self, targets: list[WrenchTarget]) -> None:
        # Record a deep copy of each accumulator + which columns we were asked to write, at flush
        # time (before the accumulator is zeroed).
        self._flushed = [(t.actor_name, t.is_robot, list(t.body_names), t.wrench.clone()) for t in targets]
        self._flushed_write_cols = {t.actor_name: list(t.write_cols) for t in targets}

    # -- helpers for assertions --
    def flushed_for(self, actor_name: str) -> torch.Tensor:
        for name, _is_robot, _bodies, wrench in self._flushed:
            if name == actor_name:
                return wrench
        raise AssertionError(f"actor '{actor_name}' was not flushed")

    def flushed_actors(self) -> set[str]:
        return {name for name, _r, _b, _w in self._flushed}

    def write_cols_for(self, actor_name: str) -> list[int]:
        return self._flushed_write_cols[actor_name]


def test_single_force_lands_on_named_body_and_env() -> None:
    sim = _FakeSim(num_envs=3)
    sim.apply_external_force(
        "robot",
        forces=torch.tensor([10.0, 0.0, 0.0]),
        body_names=["torso_link"],
        env_ids=torch.tensor([1]),
    )
    sim.flush_external_wrench()
    w = sim.flushed_for("robot")  # [3, 2, 6]
    # Force on env 1, body col 1 (torso_link), force channel only.
    assert torch.allclose(w[1, 1, 0:3], torch.tensor([10.0, 0.0, 0.0]))
    # Everything else zero (env 0/2, body 0, all torque).
    assert w.sum().item() == pytest.approx(10.0)


def test_forces_accumulate_additively_across_calls() -> None:
    sim = _FakeSim(num_envs=1)
    sim.apply_external_force("robot", forces=torch.tensor([1.0, 0.0, 0.0]), body_names=["pelvis"])
    sim.apply_external_force("robot", forces=torch.tensor([0.0, 2.0, 0.0]), body_names=["pelvis"])
    sim.flush_external_wrench()
    w = sim.flushed_for("robot")
    assert torch.allclose(w[0, 0, 0:3], torch.tensor([1.0, 2.0, 0.0]))


def test_force_and_torque_are_peers() -> None:
    """forces= and torques= are independent kwargs: either alone, or both, with no cross-leak."""
    # Torque only (no forces=): force channel stays zero.
    sim = _FakeSim(num_envs=1)
    sim.apply_external_force("robot", torques=torch.tensor([1.0, 2.0, 3.0]), body_names=["pelvis"])
    sim.flush_external_wrench()
    w = sim.flushed_for("robot")
    assert torch.allclose(w[0, 0, 0:3], torch.zeros(3))
    assert torch.allclose(w[0, 0, 3:6], torch.tensor([1.0, 2.0, 3.0]))

    # Force only (no torques=): torque channel stays zero.
    sim2 = _FakeSim(num_envs=1)
    sim2.apply_external_force("robot", forces=torch.tensor([4.0, 5.0, 6.0]), body_names=["pelvis"])
    sim2.flush_external_wrench()
    w2 = sim2.flushed_for("robot")
    assert torch.allclose(w2[0, 0, 0:3], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.allclose(w2[0, 0, 3:6], torch.zeros(3))

    # Both: each lands in its own channel.
    sim3 = _FakeSim(num_envs=1)
    sim3.apply_external_force(
        "robot", forces=torch.tensor([1.0, 0.0, 0.0]), torques=torch.tensor([0.0, 0.0, 7.0]), body_names=["pelvis"]
    )
    sim3.flush_external_wrench()
    w3 = sim3.flushed_for("robot")
    assert torch.allclose(w3[0, 0, 0:3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(w3[0, 0, 3:6], torch.tensor([0.0, 0.0, 7.0]))


def test_no_force_or_torque_raises() -> None:
    """At least one of forces/torques is required."""
    sim = _FakeSim(num_envs=1)
    with pytest.raises(ValueError, match="provide forces, torques, or both"):
        sim.apply_external_force("robot", body_names=["pelvis"])


def test_accumulator_auto_zeros_after_flush() -> None:
    sim = _FakeSim(num_envs=1)
    sim.apply_external_force("robot", forces=torch.tensor([9.0, 0.0, 0.0]), body_names=["pelvis"])
    sim.flush_external_wrench()
    assert sim.flushed_for("robot")[0, 0, 0].item() == pytest.approx(9.0)
    # Second flush with no new force must present an all-zero wrench (one-substep contract).
    sim.flush_external_wrench()
    assert sim.flushed_for("robot").abs().sum().item() == pytest.approx(0.0)


def test_all_bodies_when_body_names_none() -> None:
    sim = _FakeSim(num_envs=1)
    # Per-body force [B, 3] broadcast over the (single) env.
    sim.apply_external_force("robot", forces=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 3.0]]))
    sim.flush_external_wrench()
    w = sim.flushed_for("robot")
    assert torch.allclose(w[0, 0, 0:3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(w[0, 1, 0:3], torch.tensor([0.0, 0.0, 3.0]))


def test_per_env_force_broadcasts_over_bodies() -> None:
    sim = _FakeSim(num_envs=2)
    # [N, 3] with a single selected body -> per-env force on that body.
    sim.apply_external_force(
        "robot",
        forces=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        body_names=["pelvis"],
    )
    sim.flush_external_wrench()
    w = sim.flushed_for("robot")
    assert w[0, 0, 0].item() == pytest.approx(1.0)
    assert w[1, 0, 0].item() == pytest.approx(2.0)


def test_object_actor_is_single_body() -> None:
    sim = _FakeSim(num_envs=2)
    sim.apply_external_force("obj0", forces=torch.tensor([0.0, 0.0, -4.0]))
    sim.flush_external_wrench()
    w = sim.flushed_for("obj0")  # [2, 1, 6]
    assert w.shape == (2, 1, 6)
    assert torch.allclose(w[:, 0, 0:3], torch.tensor([[0.0, 0.0, -4.0], [0.0, 0.0, -4.0]]))


def test_unknown_actor_raises() -> None:
    sim = _FakeSim()
    with pytest.raises(ValueError, match="unknown actor"):
        sim.apply_external_force("ghost", forces=torch.tensor([1.0, 0.0, 0.0]))


def test_unknown_body_name_raises() -> None:
    sim = _FakeSim()
    with pytest.raises(ValueError, match="not a body of actor"):
        sim.apply_external_force("robot", forces=torch.tensor([1.0, 0.0, 0.0]), body_names=["no_such_body"])


def test_bad_force_shape_raises() -> None:
    sim = _FakeSim(num_envs=2)
    with pytest.raises(ValueError, match=r"1-D forces must have shape \[3\]"):
        sim.apply_external_force("robot", forces=torch.zeros(4), body_names=["pelvis"])


def test_robot_and_object_flush_together() -> None:
    sim = _FakeSim(num_envs=1)
    sim.apply_external_force("robot", forces=torch.tensor([1.0, 0.0, 0.0]), body_names=["pelvis"])
    sim.apply_external_force("obj0", forces=torch.tensor([0.0, 0.0, -2.0]))
    sim.flush_external_wrench()
    assert sim.flushed_for("robot")[0, 0, 0].item() == pytest.approx(1.0)
    assert sim.flushed_for("obj0")[0, 0, 2].item() == pytest.approx(-2.0)


def test_no_targets_flush_is_noop() -> None:
    sim = _FakeSim()
    sim.flush_external_wrench()  # never called apply_external_force
    assert sim._flushed == []


def test_flush_writes_only_touched_columns() -> None:
    """The backend is asked to write ONLY the bodies the mixin touched — never other rows,
    so a force another writer applied to a different body is not clobbered (HIGH-1 fix)."""
    sim = _FakeSim(num_envs=1)  # robot has 2 bodies: pelvis(0), torso_link(1)
    sim.apply_external_force("robot", forces=torch.tensor([5.0, 0.0, 0.0]), body_names=["torso_link"])
    sim.flush_external_wrench()
    # Only column 1 (torso_link) should be in write_cols; column 0 (pelvis) untouched.
    assert sim.write_cols_for("robot") == [1]


def test_released_column_gets_one_zero_write_then_stops() -> None:
    """A body pushed then released is written once more (to zero it) then dropped from writes."""
    sim = _FakeSim(num_envs=1)
    sim.apply_external_force("robot", forces=torch.tensor([5.0, 0.0, 0.0]), body_names=["torso_link"])
    sim.flush_external_wrench()
    assert sim.write_cols_for("robot") == [1]
    # Next substep: no force applied. The released column 1 must be written once (zeroed).
    sim.flush_external_wrench()
    assert sim.write_cols_for("robot") == [1]
    assert sim.flushed_for("robot")[0, 1, :].abs().sum().item() == pytest.approx(0.0)
    # Third substep: nothing touched and nothing to release -> the backend is not called at all.
    sim._flushed = []
    sim.flush_external_wrench()
    assert sim._flushed == []  # _write_external_wrench_native not invoked this substep


def test_untouched_actor_not_flushed() -> None:
    """An actor never targeted is never handed to the backend (so it can't clobber it)."""
    sim = _FakeSim(num_envs=1)
    sim.apply_external_force("obj0", forces=torch.tensor([0.0, 0.0, 1.0]))
    sim.flush_external_wrench()
    assert sim.flushed_actors() == {"obj0"}  # robot never touched -> not flushed


def test_square_2d_input_is_rejected_as_ambiguous() -> None:
    """[K,3] with n_env == n_body == K is ambiguous (per-env vs per-body) -> must raise."""
    sim = _FakeSim(num_envs=2)  # robot has 2 bodies
    with pytest.raises(ValueError, match="ambiguous"):
        sim.apply_external_force(
            "robot",
            forces=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),  # [2,3], N==B==2
            env_ids=torch.tensor([0, 1]),
        )
    # The explicit [N,B,3] form is accepted.
    sim.apply_external_force(
        "robot",
        forces=torch.zeros(2, 2, 3),
        env_ids=torch.tensor([0, 1]),
    )


def test_out_of_range_env_ids_raises() -> None:
    """A bad env id fails with a clear ValueError, not an out-of-bounds index_put_."""
    sim = _FakeSim(num_envs=3)
    with pytest.raises(ValueError, match=r"env_ids must be in \[0, 3\)"):
        sim.apply_external_force("robot", forces=torch.tensor([1.0, 0.0, 0.0]), env_ids=torch.tensor([99]))
    with pytest.raises(ValueError, match=r"env_ids must be in \[0, 3\)"):
        sim.apply_external_force("robot", forces=torch.tensor([1.0, 0.0, 0.0]), env_ids=torch.tensor([-1]))
