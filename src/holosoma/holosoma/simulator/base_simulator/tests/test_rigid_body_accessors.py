"""Unit tests for the public robot rigid-body kinematics accessors (pure, no simulator, CPU-only).

Pins that the four ``rigid_body_*_w`` properties are zero-cost views onto the live ``_rigid_body_*``
storage (identity, no copy) and track reassignment (IsaacSim rebinds them each refresh).
"""

from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _FakeSim(BaseSimulator):
    """Minimal BaseSimulator standing in for a backend: 2 envs, 3 named bodies, CPU tensors.

    Bypasses ``__init__`` (which needs a full tyro config) and sets only what the accessors read.
    """

    def __init__(self) -> None:
        self.num_envs = 2
        self.num_bodies = 3
        self.body_names = ["pelvis", "torso_link", "left_ankle_roll_link"]
        # Distinct, checkable values per (env, body, field).
        e, b = self.num_envs, self.num_bodies
        base = torch.arange(e * b, dtype=torch.float32).reshape(e, b, 1)
        self._rigid_body_pos = base + torch.tensor([0.0, 0.0, 0.0])  # [E,B,3]
        self._rigid_body_rot = torch.zeros(e, b, 4) + torch.tensor([0.0, 0.0, 0.0, 1.0])  # xyzw identity
        self._rigid_body_vel = base + torch.tensor([100.0, 0.0, 0.0])
        self._rigid_body_ang_vel = base + torch.tensor([200.0, 0.0, 0.0])

    def find_rigid_body_indice(self, body_name: str) -> int:
        return self.body_names.index(body_name) if body_name in self.body_names else -1


def test_properties_alias_storage_no_copy() -> None:
    sim = _FakeSim()
    # Identity — the property must return the very same tensor object (zero-cost, no copy).
    assert sim.rigid_body_pos_w is sim._rigid_body_pos
    assert sim.rigid_body_quat_w is sim._rigid_body_rot
    assert sim.rigid_body_lin_vel_w is sim._rigid_body_vel
    assert sim.rigid_body_ang_vel_w is sim._rigid_body_ang_vel


def test_properties_track_reassignment() -> None:
    """IsaacSim reassigns _rigid_body_* each refresh; the property must reflect the latest."""
    sim = _FakeSim()
    new_pos = torch.full((2, 3, 3), 7.0)
    sim._rigid_body_pos = new_pos
    assert sim.rigid_body_pos_w is new_pos


def test_property_shapes() -> None:
    sim = _FakeSim()
    e, b = sim.num_envs, sim.num_bodies
    assert tuple(sim.rigid_body_pos_w.shape) == (e, b, 3)
    assert tuple(sim.rigid_body_quat_w.shape) == (e, b, 4)
    assert tuple(sim.rigid_body_lin_vel_w.shape) == (e, b, 3)
    assert tuple(sim.rigid_body_ang_vel_w.shape) == (e, b, 3)


def test_name_based_access_via_find_rigid_body_indice() -> None:
    """The name-based read idiom is find_rigid_body_indice + index (no separate getter needed)."""
    sim = _FakeSim()
    idx = sim.find_rigid_body_indice("left_ankle_roll_link")
    assert idx == sim.body_names.index("left_ankle_roll_link")
    # A field for a named body, per-env, is a plain indexing expression.
    foot_pos = sim.rigid_body_pos_w[:, idx, :]
    assert torch.allclose(foot_pos, sim._rigid_body_pos[:, idx, :])
