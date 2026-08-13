"""Tests for the multiprocess Unitree bridge (``UnitreeMpSdk2Bridge``) — pure, no simulator, no DDS.

Two concerns:
  1. Import isolation — importing the bridge modules must NOT pull in the ``unitree_interface`` C++
     binding (that is the whole point: keep CycloneDDS out of the rclpy process).
  2. End-to-end plumbing — with an on-disk fake ``unitree_interface`` injected, spawn the real child,
     round-trip publish_low_state / read_incoming_command / publish_wireless_controller, and confirm
     the inherited torque computation runs against the proxied command.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_FAKE = Path(__file__).parent / "_fake_unitree_interface.py"


# ───────────────────────── 1. import isolation ──────────────────────────


def test_importing_bridge_modules_does_not_load_unitree_interface():
    """Neither the direct nor the MP bridge module may import the C++ binding at module-import time."""
    guard = _ImportGuard("unitree_interface")
    sys.meta_path.insert(0, guard)
    try:
        # Force a fresh import of both modules with the guard active.
        for name in [
            "holosoma.bridge.unitree.unitree_sdk2py_bridge",
            "holosoma.bridge.unitree.unitree_sdk2py_bridge_mp",
        ]:
            sys.modules.pop(name, None)
        import holosoma.bridge.unitree.unitree_sdk2py_bridge  # noqa: F401
        import holosoma.bridge.unitree.unitree_sdk2py_bridge_mp as mp

        assert not guard.tripped, "unitree_interface was imported at module-import time"
        # The MP bridge is a UnitreeSdk2Bridge, so the inherited torque/PD logic is shared.
        assert issubclass(mp.UnitreeMpSdk2Bridge, mp.UnitreeSdk2Bridge)
    finally:
        sys.meta_path.remove(guard)


class _ImportGuard:
    """Records any attempt to import ``forbidden`` (returns None so the real import machinery runs)."""

    def __init__(self, forbidden: str):
        self.forbidden = forbidden
        self.tripped = False

    def find_spec(self, name, path=None, target=None):
        # Return None (implicitly) so the real import machinery still runs; just note the attempt.
        if name == self.forbidden:
            self.tripped = True


# ───────────────────────── 3. end-to-end via spawned child ──────────────


class _FakeSim:
    """Minimal simulator exposing exactly what the bridge reads off ``self.simulator``."""

    def __init__(self, num_motor: int, device="cpu"):
        self.num_dof = num_motor
        self.device = device
        self._n = num_motor
        self.dof_pos = torch.arange(num_motor, dtype=torch.float32).reshape(1, num_motor) * 0.1
        self.dof_vel = torch.zeros(1, num_motor)
        self.dof_acc = torch.zeros(1, num_motor)
        # root state: pos(3) quat(3:7 = x,y,z,w = identity) lin(7:10) ang(10:13)
        # Distinct, non-trivial pos/vel so publish_odom's world->body rotation and xyzw->wxyz
        # conversion are actually exercised (identity quat => body == world, so values pass through).
        root = torch.zeros(1, 13)
        root[0, 0:3] = torch.tensor([1.0, 2.0, 3.0])  # position
        root[0, 6] = 1.0  # w = 1 (identity quat)
        root[0, 7:10] = torch.tensor([0.5, 0.0, 0.0])  # world linear velocity
        root[0, 10:13] = torch.tensor([0.0, 0.0, 0.3])  # world angular velocity (yaw rate)
        self.robot_root_states = root
        self.base_linear_acc = torch.zeros(1, 3)

    def time(self):
        return 1.5

    def get_dof_forces(self, env_id):
        return torch.full((self._n,), 7.0)


def _robot_full_config(num_motor: int, robot_type="g1_29dof", sdk_type="unitree_mp"):
    from holosoma.config_types.robot import RobotBridgeConfig

    return SimpleNamespace(
        dof_effort_limit_list=[100.0] * num_motor,
        asset=SimpleNamespace(robot_type=robot_type),
        bridge=RobotBridgeConfig(sdk_type=sdk_type),
    )


@pytest.fixture
def fake_binding_on_path(tmp_path, monkeypatch):
    """Install the fake as an importable ``unitree_interface`` for both the parent and the spawned child.

    ``multiprocessing`` spawn forwards ``sys.path``, but PYTHONPATH is set too so the fresh child
    interpreter resolves the fake regardless of platform. Returns the record-file path.
    """
    pkg_dir = tmp_path / "fake_pkg"
    pkg_dir.mkdir()
    shutil.copy(_FAKE, pkg_dir / "unitree_interface.py")
    record = tmp_path / "record.jsonl"

    import os

    monkeypatch.syspath_prepend(str(pkg_dir))
    # Prepend for the spawned child interpreter (spawn forwards sys.path too, but be explicit).
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(pkg_dir) + (os.pathsep + existing if existing else ""))
    monkeypatch.setenv("FAKE_UNITREE_RECORD", str(record))
    return record


def _read_records(path: Path):
    import json

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_mp_bridge_end_to_end(fake_binding_on_path, monkeypatch):
    num_motor = 2
    monkeypatch.setenv("FAKE_UNITREE_NUM_MOTOR", str(num_motor))
    record = fake_binding_on_path

    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import LowCommand, UnitreeMpSdk2Bridge

    sim = _FakeSim(num_motor)
    robot_cfg = _robot_full_config(num_motor)
    bridge_cfg = SimpleNamespace(interface="lo")

    bridge = UnitreeMpSdk2Bridge(sim, robot_cfg, bridge_cfg)
    try:
        # Parent stand-ins are binding-free and truthy for compute_torques' guard.
        assert isinstance(bridge.low_cmd, LowCommand)

        # publish_low_state: parent computes fields from sim state, child records them.
        bridge.publish_low_state()

        # publish_odom: parent reads robot_root_states, rotates world->body, ships to child.
        bridge.publish_odom()

        # read incoming command from the child (canned deterministic values).
        bridge.low_cmd_handler()
        assert list(bridge.low_cmd.tau_ff) == [1.0] * num_motor
        assert list(bridge.low_cmd.kp) == [2.0] * num_motor
        assert list(bridge.low_cmd.q_target) == [4.0] * num_motor

        # Inherited PD torque computation runs against the proxied command; result is clipped
        # to the effort limits and returned as a numpy array of the right length.
        torques = bridge.compute_torques()
        assert isinstance(torques, np.ndarray)
        assert torques.shape == (num_motor,)
        # tau_ff(1) + kp(2)*(q_target(4) - q_actual) + kd(3)*(dq_target(5) - 0), all within +-100.
        q_actual = sim.dof_pos[0].numpy()
        expected = 1.0 + 2.0 * (4.0 - q_actual) + 3.0 * (5.0 - 0.0)
        np.testing.assert_allclose(torques, np.clip(expected, -100.0, 100.0), rtol=1e-5)
    finally:
        bridge.close()

    # The child recorded a construction with the mapped enums and one publish_low_state.
    records = _read_records(record)
    kinds = [r["kind"] for r in records]
    assert "init" in kinds
    init = next(r for r in records if r["kind"] == "init")
    assert init["payload"]["robot_type"] == "G1"  # g1_29dof -> G1
    assert init["payload"]["message_type"] == "HG"  # g1_29dof -> HG
    assert init["payload"]["interface"] == "lo"

    pub = next(r for r in records if r["kind"] == "publish_low_state")
    # q shipped as sim dof_pos (0.0, 0.1); quat converted x,y,z,w -> w,x,y,z = identity (1,0,0,0).
    np.testing.assert_allclose(pub["payload"]["q"], [0.0, 0.1], atol=1e-6)
    assert pub["payload"]["quat"] == [1.0, 0.0, 0.0, 0.0]
    assert pub["payload"]["tick"] == int(1.5 * 1e3)

    # publish_odom: position passes through; identity quat -> body velocity == world velocity;
    # quat converted x,y,z,w -> w,x,y,z = identity; yaw_speed is the body-frame z angular rate.
    odom = next(r for r in records if r["kind"] == "publish_odom_state")
    np.testing.assert_allclose(odom["payload"]["position"], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(odom["payload"]["velocity"], [0.5, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(odom["payload"]["quat"], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(odom["payload"]["yaw_speed"], 0.3, atol=1e-6)


def test_mp_bridge_close_is_idempotent(fake_binding_on_path, monkeypatch):
    monkeypatch.setenv("FAKE_UNITREE_NUM_MOTOR", "1")
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import UnitreeMpSdk2Bridge

    bridge = UnitreeMpSdk2Bridge(_FakeSim(1), _robot_full_config(1), SimpleNamespace(interface="lo"))
    bridge.close()
    bridge.close()  # must not raise
    assert not bridge._proc.is_alive()
