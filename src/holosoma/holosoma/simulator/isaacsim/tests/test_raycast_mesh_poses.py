"""Robot mesh-local composition and refreshed-state reads without an Isaac SDK."""

from __future__ import annotations

from math import sqrt
from types import SimpleNamespace

import pytest
import torch

from holosoma.simulator.isaacsim.raycast_mesh_poses import RobotMeshPoses

pytestmark = pytest.mark.no_sim


def test_offsets_duplicate_body_bindings_and_unbound_slots() -> None:
    # Body 1 rotates +90 degrees around Z; two different meshes belong to it.
    sim = SimpleNamespace(
        rigid_body_pos_w=torch.tensor([[[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]]]),
        rigid_body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, sqrt(0.5), sqrt(0.5)]]]),
    )
    binding = RobotMeshPoses(
        target_paths=frozenset({"mesh_a", "mesh_b", "mesh_c"}),
        body_ids=torch.tensor([1, 0, 1]),
        mesh_slots=torch.tensor([1, 3, 4]),
        position_offsets=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]]),
        orientation_offsets=torch.tensor(
            [[[sqrt(0.5), 0.0, 0.0, sqrt(0.5)], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
        ),
    )
    positions = torch.full((1, 5, 3), -7.0)
    orientations = torch.full((1, 5, 4), -9.0)
    position_storage, orientation_storage = positions.data_ptr(), orientations.data_ptr()

    binding.update(sim, positions, orientations)  # type: ignore[arg-type]

    torch.testing.assert_close(
        positions[:, [1, 3, 4]], torch.tensor([[[0.0, 21.0, 0.0], [10.0, 2.0, 0.0], [-1.0, 20.0, 0.0]]])
    )
    # Rz(90) * Rx(90), returned in native wxyz order.
    torch.testing.assert_close(orientations[:, 1], torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    torch.testing.assert_close(orientations[:, 3], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(orientations[:, 4], torch.tensor([[sqrt(0.5), 0.0, 0.0, sqrt(0.5)]]))
    assert torch.all(positions[:, [0, 2]] == -7)
    assert torch.all(orientations[:, [0, 2]] == -9)
    assert positions.data_ptr() == position_storage
    assert orientations.data_ptr() == orientation_storage

    # refresh_sim_tensors replaces body tensors; cached bindings must reread them.
    sim.rigid_body_pos_w = sim.rigid_body_pos_w + 100.0
    binding.update(sim, positions, orientations)  # type: ignore[arg-type]
    torch.testing.assert_close(positions[:, 1], torch.tensor([[100.0, 121.0, 100.0]]))


def test_each_environment_uses_its_own_offset_and_body_state() -> None:
    sim = SimpleNamespace(
        rigid_body_pos_w=torch.tensor([[[1.0, 2.0, 3.0]], [[11.0, 12.0, 13.0]]]),
        rigid_body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0, 0.0]]]),
    )
    binding = RobotMeshPoses(
        target_paths=frozenset({"mesh"}),
        body_ids=torch.tensor([0]),
        mesh_slots=torch.tensor([0]),
        position_offsets=torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]),
        orientation_offsets=torch.tensor([[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]]),
    )
    positions, orientations = torch.zeros(2, 1, 3), torch.zeros(2, 1, 4)

    binding.update(sim, positions, orientations)  # type: ignore[arg-type]

    torch.testing.assert_close(positions, torch.tensor([[[2.0, 2.0, 3.0]], [[9.0, 12.0, 13.0]]]))
    torch.testing.assert_close(orientations, torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]]))
