"""Run the production static-geom selection and Warp repair kernel on CPU.

MuJoCo supplies independently computed body/geom transforms. Only the optional
mujoco_warp import is stubbed when absent; neither selection nor the kernel is
replaced. The CUDA companion exercises the complete backend placement/step path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")

pytestmark = pytest.mark.no_sim

WELDED_ROBOT_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="unrelated" pos="-3 2 1">
      <geom name="unrelated_geom" type="box" size=".1 .2 .3"/>
      <body name="unrelated_child" pos="0 0 1">
        <geom name="unrelated_child_geom" size=".1"/>
      </body>
    </body>
    <body name="base" pos="0 0 .5">
      <geom name="base_geom" type="box" size=".1 .2 .15" pos=".1 0 .1"/>
      <body name="mount" pos=".4 -.2 .3" euler=".2 .1 -.3">
        <geom name="mount_geom" type="box" size=".1 .05 .08" pos=".1 .2 .3" euler=".3 -.2 .1"/>
        <body name="spacer" pos=".5 0 .4">
          <body name="bracket" pos=".2 .3 .1">
            <geom name="bracket_geom" type="box" size=".1 .2 .05" euler="-.1 .4 .2"/>
          </body>
        </body>
        <body name="arm" pos="1 0 0">
          <joint name="hinge" axis="0 0 1" armature=".1"/>
          <geom name="arm_geom" type="box" size=".2 .1 .1" pos=".2 0 0"/>
          <body name="hand" pos=".5 0 0">
            <geom name="hand_geom" size=".1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="hinge" joint="hinge"/></actuator>
</mujoco>
"""
STATIC_ROBOT_GEOMS = ("base_geom", "mount_geom", "bracket_geom")


def reference_poses(
    positions: npt.NDArray[np.float64],
    quaternions: npt.NDArray[np.float64],
    joint_positions: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Independent Classic FK for each requested root pose and joint position (wxyz)."""
    model = mujoco.MjModel.from_xml_string(WELDED_ROBOT_XML)
    root = model.body("base").id
    results = []
    for position, quat, joints in zip(positions, quaternions, joint_positions):
        model.body_pos[root] = position
        model.body_quat[root] = quat
        data = mujoco.MjData(model)
        data.qpos[:] = joints
        mujoco.mj_forward(model, data)
        results.append((data.xpos.copy(), data.xquat.copy(), data.geom_xpos.copy(), data.geom_xmat.copy()))
    return tuple(np.stack(parts) for parts in zip(*results))


@pytest.fixture
def warp_backend_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    """Load the actual module under a test-local name without changing optional-backend discovery."""
    from holosoma.simulator.mujoco import backends

    if importlib.util.find_spec("mujoco_warp") is None:
        monkeypatch.setitem(sys.modules, "mujoco_warp", ModuleType("mujoco_warp"))
    name = f"{backends.__name__}._welded_geom_cpu_test"
    path = Path(backends.__file__).with_name("warp_backend.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp-cache"))
    wp.init()
    return module


def test_selection_excludes_articulated_and_unrelated_bodies_and_reuses_cache(
    warp_backend_module: ModuleType,
) -> None:
    model = mujoco.MjModel.from_xml_string(WELDED_ROBOT_XML)
    backend = object.__new__(warp_backend_module.WarpBackend)
    backend.model, backend.mjw_device = model, "cpu"
    root, mount = model.body("base").id, model.body("mount").id
    expected = [model.geom(name).id for name in STATIC_ROBOT_GEOMS]

    selection = backend._geom_ids_for_bodies([root])
    np.testing.assert_array_equal(selection.numpy(), expected)
    assert backend._geom_ids_for_bodies([root]) is selection
    # Overlapping requested roots must not launch duplicate writes to a descendant's geom.
    np.testing.assert_array_equal(backend._geom_ids_for_bodies([mount, root]).numpy(), expected)
    np.testing.assert_array_equal(
        backend._geom_ids_for_bodies([mount]).numpy(), [model.geom(name).id for name in STATIC_ROBOT_GEOMS[1:]]
    )
    assert backend._geom_ids_for_bodies([]).shape == (0,)


@pytest.mark.parametrize("selected", [[1], None], ids=["one-environment", "all-environments"])
def test_repair_translates_and_rotates_only_selected_welded_geometry(
    warp_backend_module: ModuleType, selected: list[int] | None
) -> None:
    model = mujoco.MjModel.from_xml_string(WELDED_ROBOT_XML)
    root = model.body("base").id
    num_envs = 3
    positions = np.array([[0.0, 0.0, 0.5], [5.0, 0.0, 0.5], [10.0, 0.0, 0.5]])
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (num_envs, 1))
    joints = np.array([[0.2], [0.4], [0.6]])
    _, _, old_positions, old_matrices = reference_poses(positions, quats, joints)

    # FK inputs for all environments move. The repair must write only the selected
    # environment's owned geoms, preserving other worlds' pending geometry updates.
    positions += [2.0, -1.0, 1.5]
    quats[:] = [2**-0.5, 0.0, 0.0, 2**-0.5]
    xpos, xquat, new_positions, new_matrices = reference_poses(positions, quats, joints)
    backend = object.__new__(warp_backend_module.WarpBackend)
    backend.model, backend.mjw_device = model, "cpu"
    backend.device, backend.num_envs = "cpu", num_envs
    with wp.ScopedDevice("cpu"):
        backend.mjw_model = SimpleNamespace(
            geom_bodyid=wp.array(model.geom_bodyid, dtype=wp.int32),
            geom_pos=wp.array(model.geom_pos[None], dtype=wp.vec3),
            geom_quat=wp.array(model.geom_quat[None], dtype=wp.quat),
        )
        backend.mjw_data = SimpleNamespace(
            xpos=wp.array(xpos, dtype=wp.vec3),
            xquat=wp.array(xquat, dtype=wp.quat),
            geom_xpos=wp.array(old_positions, dtype=wp.vec3),
            geom_xmat=wp.array(old_matrices.reshape(num_envs, -1, 3, 3), dtype=wp.mat33),
        )
        backend._sync_static_geom_xpos([root], None if selected is None else torch.tensor(selected))

    worlds = list(range(num_envs)) if selected is None else selected
    geoms = [model.geom(name).id for name in STATIC_ROBOT_GEOMS]
    expected_pos, expected_mat = old_positions.copy(), old_matrices.copy()
    expected_pos[np.ix_(worlds, geoms)] = new_positions[np.ix_(worlds, geoms)]
    expected_mat[np.ix_(worlds, geoms)] = new_matrices[np.ix_(worlds, geoms)]
    np.testing.assert_allclose(backend.mjw_data.geom_xpos.numpy(), expected_pos, atol=2e-6)
    np.testing.assert_allclose(backend.mjw_data.geom_xmat.numpy().reshape(num_envs, -1, 9), expected_mat, atol=2e-6)
