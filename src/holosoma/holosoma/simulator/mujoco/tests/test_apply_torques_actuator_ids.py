"""``MuJoCo.apply_torques_at_dof`` / ``get_dof_forces`` address ctrl and actuator_force by actuator id, not by
DOF position — pure, no simulator build.

The model authors its actuators in a different order than the DOFs and adds a spare actuator that
drives no DOF, so ``dof_actuator_ids`` (from ``validate_dof_actuators``, the production resolver) is
not ``arange(nu)``. Runs the REAL method bodies against a stub carrying exactly the attributes they read,
on both backend paths (Classic: a real ``mjData.ctrl``; Warp: a fake zero-copy ``[num_envs, nu]`` ctrl
tensor), and asserts each torque lands in its DOF's actuator column while the spare column is never
written, and that the forces read back in ``dof_names`` order. ``test_apply_torques_subset.py`` covers
the in-order (``arange(nu)``) case.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

# no_sim lane runs in an env without the mujoco package (isaacsim conda); the test
# needs the real mjModel/mjData for the classic path, so skip when unavailable.
mujoco = pytest.importorskip("mujoco")

from holosoma.simulator.mujoco.mujoco import MuJoCo  # noqa: E402  (needs mujoco importable)
from holosoma.simulator.mujoco.robot_addressing import validate_dof_actuators  # noqa: E402
from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402

pytestmark = pytest.mark.no_sim

_DOFS = ["j0", "j1", "j2"]
# ctrl columns: 0 = spare (drives j0 too, never written by holosoma), 1 = j2, 2 = j0, 3 = j1.
_EXPECTED_IDS = [2, 3, 1]
_SPARE = 0

_MJCF = """
<mujoco>
  <worldbody>
    <body name="b0"><joint name="j0" type="hinge" axis="0 0 1"/><geom size="0.1"/>
      <body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/><geom size="0.1"/>
        <body name="b2"><joint name="j2" type="hinge" axis="0 0 1"/><geom size="0.1"/></body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="spare" joint="j0"/>
    <motor name="j2" joint="j2"/><motor name="j0" joint="j0"/><motor name="j1" joint="j1"/>
  </actuator>
</mujoco>
"""


class _Stub:
    """Attributes ``apply_torques_at_dof`` / ``get_dof_forces`` read; ``ctrl`` selects the Warp fast path (a tensor)
    or Classic (None)."""

    def __init__(self, ctrl: torch.Tensor | None) -> None:
        self.root_model = mujoco.MjModel.from_xml_string(_MJCF)
        self.root_data = mujoco.MjData(self.root_model)
        self.sim_device = "cpu"
        self.dof_names = list(_DOFS)
        self.num_dof = len(_DOFS)
        # Resolve the ids exactly as _set_robot_joint_addressing does (no prefix in this minimal model).
        meta = ActorSpecMeta(prefix="", root_body="b0", dof_joint_names=list(_DOFS))
        self.dof_actuator_ids = validate_dof_actuators(self.root_model, meta, self.dof_names, {})
        self._dof_actuator_ids_t = (
            None
            if self.dof_actuator_ids == list(range(self.root_model.nu))
            else torch.tensor(self.dof_actuator_ids, dtype=torch.long)
        )
        self._ctrl = ctrl
        self.backend = type("B", (), {"get_ctrl_tensor": staticmethod(lambda: self._ctrl)})()

    def apply_torques_at_dof(self, torques: torch.Tensor, dof_indices: list[int] | None = None) -> None:
        # Run the REAL method body against this stub (it reads only the attributes above).
        MuJoCo.apply_torques_at_dof(cast("MuJoCo", self), torques, dof_indices)


def _classic() -> _Stub:
    sim = _Stub(ctrl=None)
    sim.root_data.ctrl[:] = 7.0  # co-controller's prior values
    return sim


def _warp(num_envs: int = 2) -> _Stub:
    sim = _Stub(ctrl=torch.full((num_envs, 4), 7.0))
    assert sim._ctrl is not None
    return sim


def test_ids_resolved_by_name_not_position() -> None:
    sim = _classic()
    assert sim.root_model.nu == 4
    assert sim.dof_actuator_ids == _EXPECTED_IDS
    assert sim._dof_actuator_ids_t is not None


def test_classic_full_path_writes_dof_columns_and_skips_spare() -> None:
    sim = _classic()
    sim.apply_torques_at_dof(torch.tensor([1.0, 2.0, 3.0]))
    # spare untouched; j2 -> col 1, j0 -> col 2, j1 -> col 3
    np.testing.assert_allclose(sim.root_data.ctrl, [7.0, 3.0, 1.0, 2.0])


def test_classic_get_dof_forces_gathers_by_actuator_id() -> None:
    sim = _classic()
    sim.apply_torques_at_dof(torch.tensor([1.0, 2.0, 3.0]))
    mujoco.mj_forward(sim.root_model, sim.root_data)  # motor force = gear (1) * ctrl
    np.testing.assert_allclose(sim.root_data.actuator_force, [7.0, 3.0, 1.0, 2.0])

    forces = MuJoCo.get_dof_forces(cast("MuJoCo", sim))

    # dof_names order, the spare column (7.0) never read; a [: num_dof] slice would give [7, 3, 1].
    assert forces.dtype == torch.float32
    torch.testing.assert_close(forces, torch.tensor([1.0, 2.0, 3.0]))
    with pytest.raises(RuntimeError, match="only supports single environment"):
        MuJoCo.get_dof_forces(cast("MuJoCo", sim), env_id=1)


def test_classic_subset_writes_only_owned_columns() -> None:
    sim = _classic()
    sim.apply_torques_at_dof(torch.tensor([1.5, 3.5]), dof_indices=[0, 2])
    np.testing.assert_allclose(sim.root_data.ctrl, [7.0, 3.5, 1.5, 7.0])


def test_warp_full_path_scatters_columns_and_casts_dtype() -> None:
    sim = _warp(num_envs=2)
    assert sim._ctrl is not None
    # float64 torques into a float32 ctrl: the indexed write must cast (index_put requires equal dtypes).
    sim.apply_torques_at_dof(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64))
    expected = torch.tensor([[7.0, 3.0, 1.0, 2.0], [7.0, 6.0, 4.0, 5.0]])
    torch.testing.assert_close(sim._ctrl, expected)
    assert sim._ctrl.dtype == torch.float32


def test_warp_full_path_broadcasts_single_robot_vector() -> None:
    sim = _warp(num_envs=2)
    assert sim._ctrl is not None
    sim.apply_torques_at_dof(torch.tensor([1.0, 2.0, 3.0]))
    expected = torch.tensor([[7.0, 3.0, 1.0, 2.0], [7.0, 3.0, 1.0, 2.0]])
    torch.testing.assert_close(sim._ctrl, expected)


def test_warp_subset_scatters_owned_columns_across_envs() -> None:
    sim = _warp(num_envs=2)
    assert sim._ctrl is not None
    sim.apply_torques_at_dof(torch.tensor([[1.5, 3.5], [2.5, 4.5]]), dof_indices=[0, 2])
    expected = torch.tensor([[7.0, 3.5, 1.5, 7.0], [7.0, 4.5, 2.5, 7.0]])
    torch.testing.assert_close(sim._ctrl, expected)
    assert sim._ctrl[:, _SPARE].eq(7.0).all()
