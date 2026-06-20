"""Unified policy service node (Variant B — single rclpy owner).

Runs *any* holosoma policy (locomotion, WBT, …) as a ROS2 service and supports
runtime policy swapping — the ``run_policy.py`` experience served over ROS2.

Variant B differs from Variant A in *who owns the ROS2 I/O*: here the service
node owns a single ``ServiceIONode`` that carries every subscription and
publisher — velocity Twist, state String, dense tracking target, and the
executed-cmd/heartbeat feedback — and is spun once. The locomotion policy does
not construct its own ``Ros2Input``; instead the node builds the velocity +
state providers and injects them via the ``"injected"`` input source.

    /cmd_vel + /…/state_input ─▶ (injected vel/state providers) ─┐
    /…/dense_tracking_command ─▶ (dense target, WBT only)        ├─▶ policy.run() ─▶ robot
                                  swap via SWITCH_MODE ───────────┘
                                  every tick ─▶ executed_cmd + heartbeat
                              ── all on ONE node, ONE spin thread ──

The policy is built like ``run_policy.py`` (``_select_policy_class`` /
``DualModePolicy``); injected providers are attached *before* ``__init__`` runs
(the base policy creates its providers during construction). In dual-mode only
the primary needs injection — the secondary reuses the primary's providers via
``_shared_hardware_source``.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import replace

import numpy as np
import rclpy
import tyro
from holosoma_msgs.msg import CmdDense, Heartbeat
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from holosoma_inference.config.config_values.inference import get_annotated_inference_config
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.dual_mode import DualModePolicy, _select_policy_class

from holosoma_service.policy_control.injected_inputs import (
    CMD_VEL_TOPIC,
    STATE_INPUT_TOPIC,
    InjectedRos2Input,
)

DENSE_TOPIC = "/holosoma/dense_tracking_command"
EXECUTED_CMD_TOPIC = "/holosoma/holosoma_executed_cmd"
HEARTBEAT_TOPIC = "/holosoma/heartbeat"
HEARTBEAT_EVERY = 10  # control ticks -> 5 Hz at a 50 Hz control loop

# This node loads rclpy in-process, which clashes with the Unitree SDK's bundled
# CycloneDDS (heap corruption / "free(): invalid pointer" at SDK init). The
# multiprocess interface ("unitree_mp", added by PR #124) runs the low-level
# binding in a spawned child so DDS never shares the parent's rclpy address
# space. Map the in-process SDK -> its isolated variant for any service run.
_MP_SDK_MAP = {"unitree": "unitree_mp"}


class ServiceIONode(Node):
    """Single node owning all ROS2 I/O for the policy service (Variant B).

    Holds the injected velocity/state providers, an optional dense target
    source (WBT only), and the executed-cmd/heartbeat feedback publishers — all
    on one node, spun once.
    """

    def __init__(self, num_dofs: int, vel_timeout: float = 1.0):
        super().__init__("policy_service")
        # --- Input: one combined vel+state provider (subscribes on THIS node) ---
        # A single object implements both protocols, mirroring Ros2Input, so the
        # base policy can share it for both roles when velocity_input ==
        # state_input == "injected".
        self.input = InjectedRos2Input(self, CMD_VEL_TOPIC, STATE_INPUT_TOPIC, vel_timeout=vel_timeout)

        # --- Input: dense tracking target (WBT only); attached on demand ---
        self._cmd = np.zeros((1, 2 * num_dofs), dtype=np.float32)  # held until first frame
        self._ref = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CmdDense, DENSE_TOPIC, self._dense_cb, qos)

        # --- Output: feedback publishers ---
        self.dof_names: list[str] = []
        self._cmd_pub = self.create_publisher(JointState, EXECUTED_CMD_TOPIC, 10)
        self._hb_pub = self.create_publisher(Heartbeat, HEARTBEAT_TOPIC, 10)
        self._tick = 0

    # --- dense TargetSource protocol (WBT) ---
    def _dense_cb(self, msg: CmdDense) -> None:
        self._cmd = np.concatenate([msg.q, msg.dq]).astype(np.float32).reshape(1, -1)
        r = msg.root_quat
        self._ref = np.array([r.x, r.y, r.z, r.w], dtype=np.float32)

    def get_target(self, num_dofs: int, rl_rate_hz: float, urdf_path: str | None):
        return self._cmd, self._ref

    # --- feedback, driven by the policy's per-tick hook ---
    def on_command_sent(self, policy, cmd_q) -> None:
        cmd = np.asarray(cmd_q, dtype=np.float64).reshape(-1)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if len(self.dof_names) == cmd.shape[0]:
            msg.name = self.dof_names
        msg.position = cmd.tolist()
        self._cmd_pub.publish(msg)

        if self._tick % HEARTBEAT_EVERY == 0:
            hb = Heartbeat()
            hb.header.stamp = msg.header.stamp
            hb.robot_connected = getattr(policy, "interface", None) is not None
            hb.control_mode = 0
            if getattr(policy, "use_policy_action", False):
                hb.status = "running"
            elif getattr(policy, "get_ready_state", False):
                hb.status = "get_ready"
            else:
                hb.status = "stiff_hold"
            self._hb_pub.publish(hb)
        self._tick += 1

    def make_hook(self, policy):
        def _hook(cmd_q, _state, _p=policy):
            self.on_command_sent(_p, cmd_q)

        return _hook


def _iter_policies(policy):
    """Yield the underlying policy instance(s): primary+secondary for dual-mode."""
    if isinstance(policy, DualModePolicy):
        yield policy.primary
        if policy.secondary is not None:
            yield policy.secondary
    else:
        yield policy


def _new_with_injected(cls, config, io: ServiceIONode):
    """Construct ``cls`` with injected providers attached before ``__init__``.

    The base policy builds its input providers during construction, so the
    ``_injected_*`` attributes must exist beforehand (see
    ``holosoma_inference.inputs.create_input`` 'injected' branch).
    """
    p = object.__new__(cls)
    p._injected_velocity_input = io.input
    p._injected_command_provider = io.input
    p.__init__(config=config)
    return p


def _build_policy(config, io: ServiceIONode):
    """Build the policy like run_policy.py, but with node-owned injected inputs.

    For dual-mode, only the primary takes injected providers; the secondary
    reuses them via ``_shared_hardware_source`` (matching DualModePolicy).
    """
    if config.secondary is not None:
        # Replicate DualModePolicy's construction with injection into primary.
        dm = object.__new__(DualModePolicy)
        primary_cls = _select_policy_class(config)
        secondary_cls = _select_policy_class(config.secondary)
        logger.info(f"Dual-mode: primary={primary_cls.__name__}, secondary={secondary_cls.__name__}")
        dm.primary = _new_with_injected(primary_cls, config, io)
        secondary = object.__new__(secondary_cls)
        secondary._shared_hardware_source = dm.primary
        secondary.__init__(config=config.secondary)
        dm.secondary = secondary
        dm.active = dm.primary
        dm.active_label = "primary"
        dm._setup_command_intercept()
        logger.info("Dual-mode ready. Publish 'switch_mode' to swap policies.")
        return dm

    policy_class = _select_policy_class(config)
    logger.info(f"Using {policy_class.__name__}")
    return _new_with_injected(policy_class, config, io)


def _attach_target_source(policy, io: ServiceIONode) -> None:
    """Attach the node as the target source of every WBT-style policy."""
    for p in _iter_policies(policy):
        if hasattr(p, "_target_source"):
            p._target_source = io
            logger.info(f"Attached dense target source to {type(p).__name__}")


def _wire_feedback(policy, io: ServiceIONode) -> None:
    """Wire the feedback publisher into each policy's per-tick hook."""
    for p in _iter_policies(policy):
        p._on_command_sent = io.make_hook(p)


def _dof_names(policy) -> list[str]:
    p = next(_iter_policies(policy))
    return list(getattr(p, "dof_names", []))


def _use_mp_sdk(config):
    """Force the rclpy-safe multiprocess SDK interface (DDS in a child process).

    Rewrites ``robot.sdk_type`` to its isolated variant (e.g. unitree ->
    unitree_mp) on the config and its secondary. A no-op for SDK types without
    a known isolated variant (e.g. already _mp, or booster), so an explicit
    override survives.
    """
    new_type = _MP_SDK_MAP.get(config.robot.sdk_type)
    if new_type is None:
        return config
    logger.info(f"Service node: using multiprocess SDK '{new_type}' (rclpy/DDS isolation)")
    config = replace(config, robot=replace(config.robot, sdk_type=new_type))
    if config.secondary is not None:
        config = replace(config, secondary=_use_mp_sdk(config.secondary))
    return config


def main() -> None:
    # Under `ros2 launch` / `ros2 run`, argv carries ROS args (e.g.
    # "--ros-args -r __node:=..."). tyro would reject those, so strip them
    # before parsing this node's own CLI; rclpy.init() consumes them separately.
    from rclpy.utilities import remove_ros_args

    sys.argv = remove_ros_args(args=sys.argv)

    config = tyro.cli(get_annotated_inference_config(), config=TYRO_CONFIG)
    rclpy.init()

    # Route the Unitree SDK through its multiprocess proxy so its CycloneDDS
    # never shares this process's rclpy (otherwise SDK init heap-corrupts).
    config = _use_mp_sdk(config)

    # One node owns every subscription + publisher.
    io = ServiceIONode(config.robot.num_joints, vel_timeout=config.task.ros_vel_timeout)

    # Build the policy with node-owned injected vel/state providers.
    policy = _build_policy(config, io)

    # WBT target source (no-op for loco) + policy-agnostic feedback.
    _attach_target_source(policy, io)
    io.dof_names = _dof_names(policy)
    _wire_feedback(policy, io)

    # Single spin thread for the one node.
    threading.Thread(target=rclpy.spin, args=(io,), daemon=True).start()

    logger.info("PolicyServiceNode (Variant B) ready — running policy.")
    try:
        policy.run()  # owns its RateLimiter + SDK I/O; blocks in this thread
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
