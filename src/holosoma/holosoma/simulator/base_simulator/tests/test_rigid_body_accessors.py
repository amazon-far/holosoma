"""Unit tests for the public robot rigid-body kinematics accessors (pure, no simulator, CPU-only).

Pins that the four ``rigid_body_*_w`` properties return the served ``_rigid_body_*`` tensors (identity,
no copy).
"""

from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.state import SimState
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _FakeSim(BaseSimulator):
    """Minimal BaseSimulator standing in for a backend: 2 envs, 3 named bodies, CPU tensors.

    Bypasses ``__init__`` (which needs a full tyro config) and sets only what the accessors read.
    """

    def __init__(self) -> None:
        self.state = SimState()
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
