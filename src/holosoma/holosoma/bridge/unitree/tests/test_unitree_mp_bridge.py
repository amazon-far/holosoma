"""Tests for the multiprocess Unitree bridge (``UnitreeMpSdk2Bridge``) — pure, no simulator, no DDS.

Two concerns:
  1. Import isolation — importing the bridge modules must NOT pull in the ``unitree_interface`` C++
     binding (that is the whole point: keep CycloneDDS out of the rclpy process).
  2. End-to-end plumbing — with an on-disk fake ``unitree_interface`` injected, spawn the real child,
     round-trip publish_low_state / read_incoming_command / publish_wireless_controller, and confirm
     the inherited torque computation runs against the proxied command.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import pytest

from holosoma.config_types.simulator import BridgeConfig
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

pytestmark = pytest.mark.no_sim

_FAKE = Path(__file__).parent / "_fake_unitree_interface.py"


# ───────────────────────── 1. import isolation ──────────────────────────


def test_importing_bridge_modules_does_not_load_unitree_interface() -> None:
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
    """Reject parent native imports before a real binding can be loaded."""

    def __init__(self, forbidden: str) -> None:
        self.forbidden = forbidden
        self.tripped = False

    def find_spec(
        self,
        name: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> None:
        if name == self.forbidden:
            self.tripped = True
            raise AssertionError("unitree_interface must only be imported in the SDK child")


# ───────────────────────── 3. end-to-end via spawned child ──────────────


class _FakeSim:
    """Minimal simulator exposing exactly what the bridge reads off ``self.simulator``."""

    def __init__(self, num_motor: int, device: str = "cpu") -> None:
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

    def time(self) -> float:
        return 1.5

    def get_dof_forces(self, env_id: int) -> torch.Tensor:
        return torch.full((self._n,), 7.0)


def _robot_full_config(num_motor: int, robot_type: str = "g1_29dof", sdk_type: str = "unitree_mp") -> SimpleNamespace:
    from holosoma.config_types.robot import RobotBridgeConfig

    return SimpleNamespace(
        dof_effort_limit_list=[100.0] * num_motor,
        asset=SimpleNamespace(robot_type=robot_type),
        bridge=RobotBridgeConfig(sdk_type=sdk_type),
    )


@pytest.fixture
def fake_binding_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the fake as an importable ``unitree_interface`` for both the parent and the spawned child.

    ``multiprocessing`` spawn forwards ``sys.path``, but PYTHONPATH is set too so the fresh child
    interpreter resolves the fake regardless of platform. Returns the record-file path.
    """
    pkg_dir = tmp_path / "fake_pkg"
    pkg_dir.mkdir()
    binding_dir = pkg_dir / "unitree_interface"
    binding_dir.mkdir()
    shutil.copy(_FAKE, binding_dir / "__init__.py")
    record = tmp_path / "record.jsonl"

    monkeypatch.delenv("FAKE_UNITREE_API_VERSION", raising=False)
    monkeypatch.delenv("FAKE_UNITREE_FAILURE", raising=False)
    monkeypatch.syspath_prepend(str(pkg_dir))
    # Prepend for the spawned child interpreter (spawn forwards sys.path too, but be explicit).
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", str(pkg_dir) + (os.pathsep + existing if existing else ""))
    monkeypatch.setenv("FAKE_UNITREE_RECORD", str(record))
    return record


def _read_records(path: Path) -> list[dict[str, Any]]:
    import json

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_mp_bridge_end_to_end(fake_binding_on_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    num_motor = 2
    monkeypatch.setenv("FAKE_UNITREE_NUM_MOTOR", str(num_motor))
    record = fake_binding_on_path

    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import LowCommand, UnitreeMpSdk2Bridge

    sim = _FakeSim(num_motor)
    robot_cfg = _robot_full_config(num_motor)
    bridge_cfg = BridgeConfig(interface="lo")

    bridge = UnitreeMpSdk2Bridge(
        cast("BaseSimulator", sim),
        cast("RobotConfig", robot_cfg),
        bridge_cfg,
    )
    try:
        # Parent stand-ins are binding-free and provide an initial command before the first poll.
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
        cached_command = bridge.low_cmd

        # Inherited PD torque computation runs against the proxied command; result is clipped
        # to the effort limits and returned as a numpy array of the right length.
        torques = bridge.compute_torques()
        assert isinstance(torques, np.ndarray)
        assert cast("tuple[int, ...]", torques.shape) == (num_motor,)
        # tau_ff(1) + kp(2)*(q_target(4) - q_actual) + kd(3)*(dq_target(5) - 0), all within +-100.
        q_actual = sim.dof_pos[0].numpy()
        expected = 1.0 + 2.0 * (4.0 - q_actual) + 3.0 * (5.0 - 0.0)
        np.testing.assert_allclose(torques, np.clip(expected, -100.0, 100.0), rtol=1e-5)

        # No second transport poll: the same cached command is safe to reuse, while current
        # simulator q/dq changes the recomputed PD torque.
        sim.dof_pos += 0.5
        sim.dof_vel += 0.25
        cached_torques = bridge.compute_torques()
        assert bridge.low_cmd is cached_command
        q_actual = sim.dof_pos[0].numpy()
        dq_actual = sim.dof_vel[0].numpy()
        expected = 1.0 + 2.0 * (4.0 - q_actual) + 3.0 * (5.0 - dq_actual)
        np.testing.assert_allclose(cached_torques, np.clip(expected, -100.0, 100.0), rtol=1e-5)
    finally:
        bridge.close()

    # The child recorded a construction with the mapped enums and one publish_low_state.
    records = _read_records(record)
    kinds = [r["kind"] for r in records]
    assert "init" in kinds
    assert kinds.count("read_incoming_command") == 1
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


def test_mp_bridge_close_is_idempotent(fake_binding_on_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_UNITREE_NUM_MOTOR", "1")
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import UnitreeMpSdk2Bridge

    bridge = UnitreeMpSdk2Bridge(
        cast("BaseSimulator", _FakeSim(1)),
        cast("RobotConfig", _robot_full_config(1)),
        BridgeConfig(interface="lo"),
    )
    bridge.close()
    bridge.close()  # must not raise
    assert not bridge._proc.is_alive()


_XML = (
    ' \n<?xml version="1.0"?>\n<CycloneDDS><Domain Id="0"><Discovery><Peers>'
    '<Peer Address="sim.sdk.example"/><Peer Address="driver.sdk.example"/>'
    "</Peers></Discovery></Domain></CycloneDDS>\n"
)


def _construct_bridge(bridge_config: Any) -> Any:
    from holosoma.bridge import create_sdk2py_bridge

    return create_sdk2py_bridge(
        cast("BaseSimulator", _FakeSim(1)),
        cast("RobotConfig", _robot_full_config(1)),
        bridge_config,
    )


@pytest.mark.parametrize(
    "ros_uri",
    [
        None,
        "file:///ros/domain42.xml",
        '<CycloneDDS><Domain Id="42"/></CycloneDDS>',
        '<CycloneDDS><Domain Id="any"/></CycloneDDS>',
    ],
)
def test_explicit_config_stays_exact_and_child_only(
    fake_binding_on_path: Path, monkeypatch: pytest.MonkeyPatch, ros_uri: str | None
) -> None:
    monkeypatch.setenv("FAKE_UNITREE_API_VERSION", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    if ros_uri is None:
        monkeypatch.delenv("CYCLONEDDS_URI", raising=False)
    else:
        monkeypatch.setenv("CYCLONEDDS_URI", ros_uri)
    environment_before = dict(os.environ)
    guard = _ImportGuard("unitree_interface")
    sys.meta_path.insert(0, guard)
    try:
        bridge = _construct_bridge(BridgeConfig(interface="unused-nic", dds_config=_XML))
        bridge.close()
    finally:
        sys.meta_path.remove(guard)
    assert not guard.tripped
    assert dict(os.environ) == environment_before
    assert os.environ.get("CYCLONEDDS_URI") == ros_uri
    assert os.environ["ROS_DOMAIN_ID"] == "42"
    records = _read_records(fake_binding_on_path)
    assert [r["kind"] for r in records] == ["import", "dds_config", "init", "enable_motion_switcher_responder"]
    assert records[1]["payload"]["value"] == _XML
    assert records[2]["payload"]["interface"] == "unused-nic"
    assert all(r["pid"] != os.getpid() and r["pid"] == bridge._proc.pid for r in records)
    assert all(r["cyclonedds_uri"] == ros_uri and r["ros_domain_id"] == "42" for r in records)
    assert bridge._queues_closed and not bridge._proc.is_alive()


@pytest.mark.parametrize("interface", ["lo", None])
def test_no_config_keeps_old_signature_marker_and_environment(
    fake_binding_on_path: Path, monkeypatch: pytest.MonkeyPatch, interface: str | None
) -> None:
    ros_uri = "file:///inherited-ros.xml"
    monkeypatch.setenv("CYCLONEDDS_URI", ros_uri)
    cfg = BridgeConfig(interface=interface)
    bridge = _construct_bridge(cfg)
    bridge.close()
    records = _read_records(fake_binding_on_path)
    assert [r["kind"] for r in records] == ["import", "init", "enable_motion_switcher_responder"]
    assert records[1]["payload"]["interface"] == (interface or "eth0")
    assert all(r["cyclonedds_uri"] == ros_uri for r in records)
    assert os.environ["CYCLONEDDS_URI"] == ros_uri


@pytest.mark.parametrize(
    ("marker", "failure", "stage", "init_count", "responder_count"),
    [
        (None, None, "DDS_CONFIG_API_VERSION=1", 0, 0),
        ("0", None, "DDS_CONFIG_API_VERSION=1", 0, 0),
        ("2", None, "DDS_CONFIG_API_VERSION=1", 0, 0),
        ("1", "import", "SDK library preload/import", 0, 0),
        ("1", "preload", "SDK library preload/import", 0, 0),
        ("1", "init", "SDK construction.*RuntimeError", 1, 0),
        ("1", "type_error", "SDK construction.*TypeError", 1, 0),
        ("1", "unpicklable", "SDK construction.*NativeError.*exactly one Domain", 1, 0),
        ("1", "responder", "motion-switcher responder startup", 1, 1),
    ],
)
def test_explicit_startup_failures_propagate_without_fallback_or_leaked_child(
    fake_binding_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str | None,
    failure: str | None,
    stage: str,
    init_count: int,
    responder_count: int,
) -> None:
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import UnitreeMpSdk2Bridge

    if marker is not None:
        monkeypatch.setenv("FAKE_UNITREE_API_VERSION", marker)
    if failure is not None:
        monkeypatch.setenv("FAKE_UNITREE_FAILURE", failure)
    if failure == "preload":
        # Invalid text file: exercise the preload failure handler without loading any DDS library.
        lib = fake_binding_on_path.parent / "fake_pkg" / "unitree_interface" / "libddsc.so.0"
        lib.write_text("not an ELF library")
    monkeypatch.setenv("CYCLONEDDS_URI", "file:///parent-ros.xml")
    children_before = {child.pid for child in multiprocessing.active_children()}
    bridge = UnitreeMpSdk2Bridge.__new__(UnitreeMpSdk2Bridge)
    with pytest.raises(RuntimeError, match=stage) as error:
        UnitreeMpSdk2Bridge.__init__(
            bridge,
            cast("BaseSimulator", _FakeSim(1)),
            cast("RobotConfig", _robot_full_config(1)),
            BridgeConfig(interface="lo", dds_config=_XML),
        )
    assert _XML not in str(error.value)
    assert "<CycloneDDS>" not in str(error.value)
    assert "sim.sdk.example" not in str(error.value)
    assert "driver.sdk.example" not in str(error.value)
    assert type(error.value) is RuntimeError
    assert str(pickle.loads(pickle.dumps(error.value))) == str(error.value)
    if failure == "preload":
        assert "libddsc.so.0: file too short" in str(error.value)
    assert not bridge._proc.is_alive()
    assert bridge._proc.exitcode is not None
    assert bridge._queues_closed
    for closed_queue in (bridge._req_q, bridge._res_q):
        with pytest.raises(ValueError, match="closed"):
            closed_queue.get_nowait()
    bridge.close()
    assert {child.pid for child in multiprocessing.active_children()} == children_before
    assert os.environ["CYCLONEDDS_URI"] == "file:///parent-ros.xml"
    records = _read_records(fake_binding_on_path)
    assert sum(r["kind"] == "init" for r in records) == init_count
    assert sum(r["kind"] == "enable_motion_switcher_responder" for r in records) == responder_count
    assert all(r["cyclonedds_uri"] == "file:///parent-ros.xml" for r in records)


@pytest.mark.parametrize("response", ["timeout", "unexpected", "start_error"])
def test_startup_handshake_failure_cleans_partial_resources(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    import queue
    from unittest.mock import MagicMock

    from holosoma.bridge.unitree import unitree_sdk2py_bridge_mp as mp_bridge

    req_q, res_q = MagicMock(), MagicMock()
    if response == "timeout":
        res_q.get.side_effect = queue.Empty
    else:
        res_q.get.return_value = ("unexpected", None)
    proc = MagicMock()
    proc.is_alive.return_value = True
    if response == "start_error":
        proc.start.side_effect = RuntimeError("start failed")
        proc.pid = None
    ctx = MagicMock()
    ctx.Queue.side_effect = [req_q, res_q]
    ctx.Process.return_value = proc
    monkeypatch.setattr(mp_bridge.mp, "get_context", lambda _: ctx)
    with pytest.raises(RuntimeError):
        _construct_bridge(BridgeConfig(interface="lo"))
    req_q.close.assert_called_once_with()
    res_q.close.assert_called_once_with()
    if response != "start_error":
        proc.kill.assert_called_once_with()
        assert proc.join.call_count == 2


def test_spawn_seam_rejects_missing_dds_config_before_allocating_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from holosoma.bridge.unitree import unitree_sdk2py_bridge_mp as mp_bridge

    context = MagicMock()
    monkeypatch.setattr(mp_bridge.mp, "get_context", context)
    bridge = mp_bridge.UnitreeMpSdk2Bridge.__new__(mp_bridge.UnitreeMpSdk2Bridge)
    bridge.sdk_robot_type = "g1_29dof"
    bridge.bridge_config = cast("BridgeConfig", SimpleNamespace(interface="lo"))
    with pytest.raises(AttributeError, match="dds_config"):
        bridge._init_sdk_components()
    context.assert_not_called()


@pytest.mark.parametrize(
    "reason",
    [
        "dds_config must be nonempty inline CycloneDDS XML (no filename, URI or BOM)",
        'dds_config must be inline CycloneDDS XML with exactly one Domain Id="0" or Id="any"',
        "Could not allocate dds_config XML parser",
        "DDS initialization previously failed; start a fresh SDK process",
        "DDS already initialized with incompatible configuration; start a fresh SDK process",
        "Failed to create domain explicitly.",
        "Could not create DomainParticipant.",
        "libddsc.so.0: cannot open shared object file: No such file or directory",
        "libddscxx.so.0: invalid ELF header",
    ],
)
def test_safe_startup_reason_preserves_distinct_known_diagnostics(reason: str) -> None:
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import _safe_startup_reason

    # Fixed SDK 0.1.8 / CycloneDDS messages, not invented substitutes for native validation.
    assert _safe_startup_reason(RuntimeError(reason)) == reason


@pytest.mark.parametrize(
    "context",
    [
        _XML,
        repr(_XML),
        _XML.replace("<", "&lt;").replace(">", "&gt;"),
        '"<Peer Address="private-fragment',
        "'private-fragment'",
        "private-fragment",
        "\n\x00\x1b[31m\r\u202eprivate-fragment",
        "private-fragment" * 10000,
    ],
)
@pytest.mark.parametrize("reason", ["Failed to create domain explicitly.", "libddsc.so.0: file too short"])
def test_safe_startup_reason_never_copies_appended_context(reason: str, context: str) -> None:
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import _safe_startup_reason

    # No separator is intentional: a deceptive known-prefix suffix must not be copied either.
    result = _safe_startup_reason(RuntimeError(reason + context))
    assert result == reason
    assert len(result) <= 180
    assert all(char.isprintable() for char in result)


@pytest.mark.parametrize(
    "message",
    [
        "unrecognized native error: " + _XML,
        "unrecognized native error: " + repr(_XML),
        "private-fragment",
        "private-fragment.so: file too short",
        "\x00\x1b[31m\n" + "private-fragment" * 10000,
        "libddsc.so.0: unknown cause private-fragment",
    ],
)
def test_safe_startup_reason_withholds_unknown_or_uncertain_text(message: str) -> None:
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import _safe_startup_reason

    assert _safe_startup_reason(RuntimeError(message)) == "native reason withheld (unrecognized diagnostic)"


def test_safe_startup_reason_loader_path_and_wheel_hash_are_not_returned() -> None:
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import _safe_startup_reason

    error = OSError("/private-fragment/libddsc-4038630c.so: cannot open shared object file: No such file or directory")
    assert _safe_startup_reason(error) == "libddsc.so: cannot open shared object file: No such file or directory"


def test_safe_startup_reason_handles_unreadable_exception() -> None:
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import _safe_startup_reason

    class UnreadableError(Exception):
        def __str__(self) -> str:
            raise ValueError("private-fragment")

    assert _safe_startup_reason(UnreadableError()) == "native reason withheld (unreadable diagnostic)"


def test_safe_startup_reason_recognizes_released_native_initialization_wrapper() -> None:
    from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import _safe_startup_reason

    # Observed from SDK 0.1.8 in a network-none CPU container using a nonexistent NIC.
    error = RuntimeError(
        "Catch dds::core exception. Class:::dds::core::Error, Message:Error Error - "
        "Failed to create domain explicitly.\n"
        "===============================================================================\n"
        "Context     : org::eclipse::cyclonedds::domain::DomainWrap::DomainWrap\n"
        "Node        : private-fragment\n" + _XML
    )
    assert _safe_startup_reason(error) == "Failed to create domain explicitly."
