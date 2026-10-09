"""``MuJoCo.apply_torques_at_dof`` DOF-subset scatter — pure, no full simulator build.

Exercises the REAL method body (invoked against a light stub that supplies exactly the attributes it
reads, via ``cast``) over both backend paths:
  * Classic (CPU): a real tiny mjModel/mjData with named actuators; assert only the owned actuators'
    ``ctrl`` slots change and the full path still writes all of them.
  * Warp fast path: a fake zero-copy ``[num_envs, nu]`` ctrl tensor; assert the subset scatter and
    the full-width write.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

# no_sim lane runs in an env without the mujoco package (isaacsim conda); the test
# needs the real mjModel/mjData for the classic path, so skip when unavailable.
mujoco = pytest.importorskip("mujoco")

from holosoma.simulator.mujoco.mujoco import MuJoCo  # noqa: E402  (needs mujoco importable)
from holosoma.utils.safe_torch_import import torch  # noqa: E402

pytestmark = pytest.mark.no_sim

_DOFS = ["j0", "j1", "j2", "j3"]

_MJCF = """
<mujoco>
  <worldbody>
    <body name="b0"><joint name="j0" type="hinge" axis="0 0 1"/><geom size="0.1"/>
      <body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/><geom size="0.1"/>
        <body name="b2"><joint name="j2" type="hinge" axis="0 0 1"/><geom size="0.1"/>
          <body name="b3"><joint name="j3" type="hinge" axis="0 0 1"/><geom size="0.1"/></body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="j0" joint="j0"/><motor name="j1" joint="j1"/>
    <motor name="j2" joint="j2"/><motor name="j3" joint="j3"/>
  </actuator>
</mujoco>
"""


class _ClassicStub:
    """Attributes ``apply_torques_at_dof`` reads on the ClassicBackend path (ctrl tensor is None)."""

    def __init__(self) -> None:
        self.root_model = mujoco.MjModel.from_xml_string(_MJCF)
        self.root_data = mujoco.MjData(self.root_model)
        self.dof_names = list(_DOFS)
        self.num_dof = len(_DOFS)
        self.backend = type("B", (), {"get_ctrl_tensor": staticmethod(lambda: None)})()

    def _get_prefixed_name(self, clean_name: str) -> str:
        return clean_name  # no prefix in this minimal model

    def apply_torques_at_dof(self, torques: torch.Tensor, dof_indices: list[int] | None = None) -> None:
        # Run the REAL method body against this stub (it reads only the attributes above).
        MuJoCo.apply_torques_at_dof(cast("MuJoCo", self), torques, dof_indices)


class _WarpStub:
    """Attributes for the WarpBackend fast path — a fake zero-copy [num_envs, nu] ctrl tensor."""

    def __init__(self, num_envs: int = 2) -> None:
        self.root_model = mujoco.MjModel.from_xml_string(_MJCF)
        self.root_data = mujoco.MjData(self.root_model)
        self.dof_names = list(_DOFS)
        self.num_dof = len(_DOFS)
        self._ctrl = torch.zeros(num_envs, len(_DOFS))
        self.backend = type("B", (), {"get_ctrl_tensor": staticmethod(lambda: self._ctrl)})()

    def apply_torques_at_dof(self, torques: torch.Tensor, dof_indices: list[int] | None = None) -> None:
        # Run the REAL method body against this stub (it reads only the attributes above).
        MuJoCo.apply_torques_at_dof(cast("MuJoCo", self), torques, dof_indices)


def test_classic_subset_writes_only_owned_slots() -> None:
    sim = _ClassicStub()
    sim.root_data.ctrl[:] = [7.0, 7.0, 7.0, 7.0]  # co-controller's prior values
    # Own DOFs j1, j3 (indices 1, 3); torques aligned with that list.
    sim.apply_torques_at_dof(torch.tensor([1.5, 3.5]), dof_indices=[1, 3])
    np.testing.assert_allclose(sim.root_data.ctrl, [7.0, 1.5, 7.0, 3.5])


def test_classic_full_path_writes_all() -> None:
    sim = _ClassicStub()
    sim.root_data.ctrl[:] = [7.0, 7.0, 7.0, 7.0]
    sim.apply_torques_at_dof(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(sim.root_data.ctrl, [1.0, 2.0, 3.0, 4.0])


def test_classic_subset_count_mismatch_raises() -> None:
    sim = _ClassicStub()
    with pytest.raises(ValueError, match="Torque count mismatch"):
        sim.apply_torques_at_dof(torch.tensor([1.0]), dof_indices=[1, 3])


def test_warp_subset_scatters_across_envs_and_leaves_rest() -> None:
    sim = _WarpStub(num_envs=2)
    sim._ctrl[:] = 7.0
    sim.apply_torques_at_dof(torch.tensor([1.5, 3.5]), dof_indices=[1, 3])
    expected = torch.tensor([[7.0, 1.5, 7.0, 3.5], [7.0, 1.5, 7.0, 3.5]])
    torch.testing.assert_close(sim._ctrl, expected)


def test_warp_full_path_writes_all() -> None:
    sim = _WarpStub(num_envs=2)
    full = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    sim.apply_torques_at_dof(full)
    torch.testing.assert_close(sim._ctrl, full)
