from __future__ import annotations

import pytest
import torch

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.mujoco_classic

from holosoma.simulator.mujoco.backends.classic_backend import ClassicBackend  # noqa: E402
from holosoma.simulator.mujoco.mujoco import MuJoCo, _address_span, _configured_dof_names  # noqa: E402
from holosoma.simulator.mujoco.tensor_views import FilteredDofStateView  # noqa: E402

_INTERLEAVED_MJCF = """
<mujoco>
  <worldbody>
    <body>
      <joint name="left_motor" type="hinge"/>
      <geom type="sphere" size="0.1"/>
      <body pos="0 0 0.2">
        <joint name="passive_joint" type="hinge"/>
        <geom type="sphere" size="0.1"/>
        <body pos="0 0 0.2">
          <joint name="right_motor" type="hinge"/>
          <geom type="sphere" size="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="left_motor"/>
    <motor joint="right_motor"/>
  </actuator>
</mujoco>
"""


class Proxy:
    def __init__(self, value: torch.Tensor):
        self.value = value

    def __getitem__(self, key):
        return self.value[key]


def test_interleaved_passive_joint_is_excluded_and_preserved_on_write() -> None:
    model = mujoco.MjModel.from_xml_string(_INTERLEAVED_MJCF)
    data = mujoco.MjData(model)
    model_dofs = [model.joint(i).name for i in range(model.njnt)]
    dof_names = _configured_dof_names(model_dofs, ["left_motor", "right_motor"])
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in dof_names]
    qpos_addrs = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    qvel_addrs = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]

    assert dof_names == ["left_motor", "right_motor"]
    assert qpos_addrs == [0, 2]
    assert qvel_addrs == [0, 2]

    passive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_joint")
    passive_qpos = int(model.jnt_qposadr[passive_id])
    passive_qvel = int(model.jnt_dofadr[passive_id])
    data.qpos[passive_qpos] = 0.75
    data.qvel[passive_qvel] = -0.25

    backend = object.__new__(ClassicBackend)
    backend.model = model
    backend.data = data
    state = torch.tensor([[1.0, 0.1], [2.0, 0.2]])
    backend.set_dof_state(torch.tensor([0]), state, {"dof_qpos_addrs": qpos_addrs, "dof_qvel_addrs": qvel_addrs})

    assert data.qpos[qpos_addrs].tolist() == pytest.approx([1.0, 2.0])
    assert data.qvel[qvel_addrs].tolist() == pytest.approx([0.1, 0.2])
    assert data.qpos[passive_qpos] == pytest.approx(0.75)
    assert data.qvel[passive_qvel] == pytest.approx(-0.25)


@pytest.mark.parametrize("config_dofs", [["right_motor", "left_motor"], ["left_motor", "missing_motor"]])
def test_configured_dofs_reject_order_mismatches_and_missing_joints(config_dofs: list[str]) -> None:
    with pytest.raises(ValueError, match="DOF names/order"):
        _configured_dof_names(["left_motor", "passive_joint", "right_motor"], config_dofs)


def test_address_span_detects_gaps() -> None:
    assert _address_span([5, 6]) == (slice(5, 7), 2, [0, 1], True)
    assert _address_span([5, 7]) == (slice(5, 8), 3, [0, 2], False)


@pytest.mark.parametrize("use_proxy", [False, True], ids=["tensor", "proxy"])
def test_gather_excludes_interleaved_passive_joints(use_proxy: bool) -> None:
    simulator = object.__new__(MuJoCo)

    def wrap(value: torch.Tensor) -> Proxy | torch.Tensor:
        return Proxy(value) if use_proxy else value

    simulator._full_dof_pos = wrap(torch.tensor([[10.0, 99.0, 20.0]]))
    simulator._full_dof_vel = wrap(torch.tensor([[1.0, 9.9, 2.0]]))
    simulator._full_dof_acc = wrap(torch.tensor([[0.1, 0.99, 0.2]]))
    simulator._actuated_qpos_rel = [0, 2]
    simulator._actuated_qvel_rel = [0, 2]
    simulator.dof_pos = torch.zeros(1, 2)
    simulator.dof_vel = torch.zeros(1, 2)
    simulator.dof_acc = torch.zeros(1, 2)

    simulator._gather_actuated_dofs()

    torch.testing.assert_close(simulator.dof_pos, torch.tensor([[10.0, 20.0]]))
    torch.testing.assert_close(simulator.dof_vel, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(simulator.dof_acc, torch.tensor([[0.1, 0.2]]))


def test_filtered_state_view_writes_positions_and_velocities() -> None:
    positions = torch.zeros(1, 3)
    velocities = torch.zeros(1, 3)
    view = FilteredDofStateView(positions, velocities)
    state = torch.tensor([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])

    view[:] = state

    torch.testing.assert_close(positions, state[:, 0].unsqueeze(0))
    torch.testing.assert_close(velocities, state[:, 1].unsqueeze(0))
    torch.testing.assert_close(view.clone(), state)
