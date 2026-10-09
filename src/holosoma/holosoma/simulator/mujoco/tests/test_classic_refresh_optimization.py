"""Behavioral tests for reusable-buffer classic MuJoCo tensor refresh."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

mujoco = pytest.importorskip("mujoco")

from holosoma.simulator.mujoco.backends.classic_backend import ClassicBackend  # noqa: E402
from holosoma.simulator.mujoco.mujoco import MuJoCo  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig

pytestmark = pytest.mark.no_sim

_MJCF = """
<mujoco>
  <worldbody>
    <body name="b0" pos="1 2 3" quat="0.5 0.5 0.5 0.5">
      <geom name="g0" type="sphere" size="0.1"/>
      <body name="b1" pos="0.4 0.5 0.6" quat="0.7071067811865476 0 0.7071067811865476 0">
        <geom name="g1" type="sphere" size="0.1"/>
        <body name="b2" pos="-0.2 0.3 0.7" quat="0.7071067811865476 0.7071067811865476 0 0">
          <geom name="g2" type="sphere" size="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_CONTACT_MJCF = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="b0" pos="0 0 0.05">
      <freejoint/><geom name="g0" type="sphere" size="0.1"/>
    </body>
    <body name="b1" pos="0.15 0 0.05">
      <freejoint/><geom name="g1" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _backend(xml: str = _MJCF) -> ClassicBackend:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = cast("FullSimConfig", SimpleNamespace(training=SimpleNamespace(num_envs=1)))
    return ClassicBackend(model, data, config, "cpu")


def _legacy_body_state(
    backend: ClassicBackend, body_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.stack([torch.from_numpy(backend.data.xpos[body_id]).float() for body_id in body_ids])
    orientations = torch.stack(
        [
            torch.tensor(
                backend.data.xquat[body_id][[1, 2, 3, 0]],
                dtype=torch.float32,
            )
            for body_id in body_ids
        ]
    )
    velocities = []
    for body_id in body_ids:
        body_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            backend.model,
            backend.data,
            mujoco.mjtObj.mjOBJ_XBODY,
            body_id,
            body_velocity,
            0,
        )
        velocities.append(torch.from_numpy(body_velocity).float())
    velocity_tensor = torch.stack(velocities)
    return positions, orientations, velocity_tensor[:, 3:], velocity_tensor[:, :3]


def _legacy_contact_forces(backend: ClassicBackend) -> torch.Tensor:
    forces = torch.zeros(1, backend.model.nbody, 3, dtype=torch.float32)
    force_torque = np.zeros(6, dtype=np.float64)
    for contact_index in range(backend.data.ncon):
        contact = backend.data.contact[contact_index]
        mujoco.mj_contactForce(
            backend.model,
            backend.data,
            contact_index,
            force_torque,
        )
        contact_frame = backend.data.contact[contact_index].frame.reshape(3, 3)
        force = torch.from_numpy(contact_frame.T @ force_torque[:3]).float()
        body_1 = backend.model.geom_bodyid[contact.geom1]
        body_2 = backend.model.geom_bodyid[contact.geom2]
        forces[0, body_1] -= force
        forces[0, body_2] += force
    return forces


def test_batched_body_refresh_preserves_order_quaternions_and_velocity_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    body_ids = [3, 1, 2]
    expected = _legacy_body_state(backend, body_ids)
    backend.configure_rigid_body_refresh(body_ids)

    original = mujoco.mj_objectVelocity
    calls: list[tuple[Any, int]] = []

    def record_velocity(
        model: Any,
        data: Any,
        object_type: Any,
        object_id: int,
        velocity: npt.NDArray[np.float64],
        local: int,
    ) -> None:
        calls.append((object_type, object_id))
        original(model, data, object_type, object_id, velocity, local)

    monkeypatch.setattr(mujoco, "mj_objectVelocity", record_velocity)
    actual = backend.refresh_rigid_body_states()

    assert [object_id for _, object_id in calls] == body_ids
    assert all(object_type == mujoco.mjtObj.mjOBJ_XBODY for object_type, _ in calls)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor[0].float(), expected_tensor)
    assert backend.refresh_rigid_body_states() is actual

    # Readdressing rebuilds storage and ordering.
    backend.configure_rigid_body_refresh([1])
    readdressed = backend.refresh_rigid_body_states()
    assert readdressed[0].shape == (1, 1, 3)
    assert torch.equal(readdressed[0][0].float(), expected[0][1:2])


def _set_contacts(
    backend: ClassicBackend,
    monkeypatch: pytest.MonkeyPatch,
    geom_pairs: npt.NDArray[np.int32],
    forces: npt.NDArray[np.float64],
    calls: list[int] | None = None,
) -> None:
    contact = SimpleNamespace(
        geom=geom_pairs,
        frame=np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (len(geom_pairs), 1)),
    )
    backend.data = cast("Any", SimpleNamespace(ncon=len(geom_pairs), contact=contact))

    def contact_force(
        _model: Any,
        _data: Any,
        contact_index: int,
        force_torque: npt.NDArray[np.float64],
    ) -> None:
        if calls is not None:
            calls.append(contact_index)
        force_torque.fill(0.0)
        force_torque[:3] = forces[contact_index]

    monkeypatch.setattr(mujoco, "mj_contactForce", contact_force)


def test_contact_accumulation_preserves_float32_order_and_signs(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    calls: list[int] = []
    geom_pairs = np.asarray([[0, 1], [0, 1], [0, 1]], dtype=np.int32)
    forces = np.asarray(
        [
            [16_777_217.0, 2.25, -3.5],
            [-16_777_216.0, -1.0, 1.25],
            [1.0, 0.5, 0.25],
        ],
        dtype=np.float64,
    )
    _set_contacts(backend, monkeypatch, geom_pairs, forces, calls)

    actual = backend.compute_contact_forces()[0]
    expected = torch.zeros(backend.model.nbody, 3, dtype=torch.float32)
    for geom_pair, force_64 in zip(geom_pairs, forces):
        force_32 = torch.from_numpy(force_64).float()
        body_1, body_2 = backend.model.geom_bodyid[geom_pair]
        expected[body_1] -= force_32
        expected[body_2] += force_32

    assert calls == [0, 1, 2]
    assert torch.equal(actual, expected)
    assert actual[1, 0] == -1.0
    assert actual[2, 0] == 1.0


def test_contact_refresh_zeroes_previous_values_when_there_are_no_contacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    _set_contacts(
        backend,
        monkeypatch,
        np.asarray([[0, 1]], dtype=np.int32),
        np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64),
    )
    force_tensor = backend.compute_contact_forces()
    assert torch.count_nonzero(force_tensor) > 0

    _set_contacts(
        backend,
        monkeypatch,
        np.empty((0, 2), dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
    )
    zero_tensor = backend.compute_contact_forces()
    assert zero_tensor is force_tensor
    assert torch.count_nonzero(zero_tensor) == 0


def test_contact_buffers_grow_beyond_initial_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    contact_count = backend._contact_capacity + 1
    geom_pairs = np.tile(np.asarray([[0, 1]], dtype=np.int32), (contact_count, 1))
    forces = np.ones((contact_count, 3), dtype=np.float64)
    calls: list[int] = []
    _set_contacts(backend, monkeypatch, geom_pairs, forces, calls)

    actual = backend.compute_contact_forces()[0].numpy()

    assert backend._contact_capacity >= contact_count
    assert calls == list(range(contact_count))
    np.testing.assert_array_equal(actual[1], np.full(3, -contact_count, dtype=np.float32))
    np.testing.assert_array_equal(actual[2], np.full(3, contact_count, dtype=np.float32))


class _RefreshStub:
    def __init__(self, backend: ClassicBackend, body_ids: list[int]) -> None:
        self.backend = backend
        self._body_ids_t = torch.tensor(body_ids, dtype=torch.long)
        self.num_bodies = len(body_ids)
        self._rigid_body_pos = torch.zeros(1, len(body_ids), 3)
        self._rigid_body_rot = torch.zeros(1, len(body_ids), 4)
        self._rigid_body_vel = torch.zeros(1, len(body_ids), 3)
        self._rigid_body_ang_vel = torch.zeros(1, len(body_ids), 3)
        self.contact_forces = torch.zeros(1, len(body_ids), 3)
        self.contact_forces_history = torch.arange(
            9 * len(body_ids),
            dtype=torch.float32,
        ).reshape(1, 3, len(body_ids), 3)


def test_live_refresh_matches_legacy_state_contacts_and_history() -> None:
    backend = _backend(_CONTACT_MJCF)
    mujoco.mj_step(backend.model, backend.data)
    assert backend.data.ncon == 3
    body_ids = [2, 1]
    backend.configure_rigid_body_refresh(body_ids)
    expected_state = _legacy_body_state(backend, body_ids)
    expected_full_forces = _legacy_contact_forces(backend)
    sim = _RefreshStub(backend, body_ids)
    old_history = sim.contact_forces_history.clone()
    expected_contacts = expected_full_forces[:, sim._body_ids_t]
    expected_history = torch.cat([expected_contacts.unsqueeze(1), old_history[:, :-1]], dim=1)

    MuJoCo.refresh_sim_tensors(cast("MuJoCo", sim))

    actual_state = (
        sim._rigid_body_pos,
        sim._rigid_body_rot,
        sim._rigid_body_vel,
        sim._rigid_body_ang_vel,
    )
    for actual, expected in zip(actual_state, expected_state):
        assert torch.equal(actual[0], expected)
    assert torch.equal(sim.contact_forces, expected_contacts)
    assert torch.equal(sim.contact_forces_history, expected_history)
    assert all(tensor.dtype == torch.float32 for tensor in (*actual_state, sim.contact_forces))
    assert sim.contact_forces.device.type == "cpu"
