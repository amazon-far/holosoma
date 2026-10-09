"""Tests for the DOF-subset control surface of ``BasicSdk2Bridge`` — pure, no simulator, no DDS.

The base bridge historically assumed it owned every simulator DOF (``num_motor = simulator.num_dof``,
full-width torque writes). These tests cover the subset support that lets an SDK drive only a NAMED
subset of DOFs (``RobotBridgeConfig.controlled_dof_names`` / ``excluded_dof_names``), presenting a
possibly-different ``robot_type`` to the SDK (``sdk_robot_type``), so a co-controller can own the rest.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import numpy.typing as npt
import pytest

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge
from holosoma.config_types.robot import RobotBridgeConfig
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig
    from holosoma.config_types.simulator import BridgeConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim


class _StubBridge(BasicSdk2Bridge):
    """Minimal concrete bridge: no SDK, so the abstract hooks are trivial no-ops."""

    def _init_sdk_components(self) -> None:
        pass

    def low_cmd_handler(self, msg: Any = None) -> None:
        pass

    def publish_low_state(self) -> None:
        pass

    def compute_torques(self) -> npt.NDArray[np.floating[Any]]:
        return self.torques


class _FakeSim:
    """Exposes exactly the DOF surface the base bridge reads at construction / PD time."""

    def __init__(self, dof_names: Sequence[str], device: str = "cpu") -> None:
        self.dof_names = list(dof_names)
        self.num_dof = len(self.dof_names)
        self.device = device
        n = self.num_dof
        self.dof_pos = torch.arange(n, dtype=torch.float32).reshape(1, n) * 0.1
        self.dof_vel = torch.zeros(1, n)
        self.dof_acc = torch.zeros(1, n)


def _robot_config(
    dof_names: Sequence[str],
    *,
    controlled: tuple[str, ...] = (),
    excluded: tuple[str, ...] = (),
    sdk_robot_type: str | None = None,
    robot_type: str = "my_robot",
) -> SimpleNamespace:
    return SimpleNamespace(
        dof_effort_limit_list=[float(10 * (i + 1)) for i in range(len(dof_names))],
        asset=SimpleNamespace(robot_type=robot_type),
        bridge=RobotBridgeConfig(
            controlled_dof_names=controlled,
            excluded_dof_names=excluded,
            sdk_robot_type=sdk_robot_type,
        ),
    )


_DOFS = ["a", "b", "c", "d", "e", "f"]


def _build(**cfg_kwargs: Any) -> tuple[_StubBridge, _FakeSim]:
    sim = _FakeSim(_DOFS)
    robot = _robot_config(_DOFS, **cfg_kwargs)
    bridge = _StubBridge(
        cast("BaseSimulator", sim),
        cast("RobotConfig", robot),
        cast("BridgeConfig", SimpleNamespace(interface="lo")),
    )
    return bridge, sim


def test_default_controls_all_dofs_full_width_path() -> None:
    bridge, _sim = _build()
    assert bridge.num_motor == 6
    assert bridge.dof_indices == list(range(6))
    # Every DOF in natural order -> None so the simulator takes the fast full-width write.
    assert bridge._apply_indices is None
    np.testing.assert_array_equal(bridge.torque_limit, [10, 20, 30, 40, 50, 60])


def test_excluded_dofs_narrows_to_complement() -> None:
    bridge, _sim = _build(excluded=("c", "e"))
    assert bridge.num_motor == 4
    assert bridge.dof_indices == [0, 1, 3, 5]  # a, b, d, f
    assert bridge._apply_indices == [0, 1, 3, 5]
    # torque_limit is subset + reordered to the controlled DOFs.
    np.testing.assert_array_equal(bridge.torque_limit, [10, 20, 40, 60])
    assert bridge.torques.shape == (4,)


def test_controlled_dofs_wins_over_excluded_and_preserves_order() -> None:
    # controlled_dof_names present -> excluded is ignored, order follows controlled.
    bridge, _ = _build(controlled=("f", "a", "c"), excluded=("a",))
    assert bridge.dof_indices == [5, 0, 2]
    assert bridge.num_motor == 3
    np.testing.assert_array_equal(bridge.torque_limit, [60, 10, 30])


def test_unknown_controlled_name_raises() -> None:
    with pytest.raises(ValueError, match="controlled_dof_names"):
        _build(controlled=("a", "nope"))


def test_unknown_excluded_name_raises() -> None:
    with pytest.raises(ValueError, match="excluded_dof_names"):
        _build(excluded=("ghost",))


def test_excluding_every_dof_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _build(excluded=tuple(_DOFS))


def test_sdk_robot_type_overrides_asset_type() -> None:
    bridge, _ = _build(sdk_robot_type="g1_29dof", robot_type="my_robot")
    assert bridge.sdk_robot_type == "g1_29dof"


def test_sdk_robot_type_defaults_to_asset_type() -> None:
    bridge, _ = _build(robot_type="my_robot")
    assert bridge.sdk_robot_type == "my_robot"


def test_compute_pd_torques_broadcasts_against_the_subset() -> None:
    bridge, sim = _build(excluded=("c", "e"))  # controls a,b,d,f (idx 0,1,3,5)
    n = bridge.num_motor
    # SDK command vectors are num_motor-length; PD must line up with the subset state, not all 6 DOFs.
    tau_ff = np.ones(n)
    kp = np.full(n, 2.0)
    kd = np.zeros(n)
    q_target = np.full(n, 1.0)
    dq_target = np.zeros(n)

    torques = bridge._compute_pd_torques(tau_ff, kp, kd, q_target, dq_target)

    assert cast("tuple[int, ...]", torques.shape) == (n,)
    q_actual = sim.dof_pos[0][bridge.dof_indices].numpy()
    expected = np.clip(1.0 + 2.0 * (1.0 - q_actual), -bridge.torque_limit, bridge.torque_limit)
    np.testing.assert_allclose(torques, expected, rtol=1e-6)
