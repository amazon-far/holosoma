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

The policy is built via ``_select_policy_class``; injected providers are attached
*before* ``__init__`` runs (the base policy creates its providers during
construction). For N registered policies (``MultiModePolicy``) the index-0 owner
takes the injected providers and the peers reuse them via
``_shared_hardware_source``; each policy gets its OWN sensors. A client reads the
registry on ``/holosoma/policy_status`` and switches via ``/holosoma/select_policy``
(by name) or ``/holosoma/select_checkpoint`` (index within the active policy).

This is also the WBT entrypoint (it replaces the former ``holosoma_node``): a WBT
preset resolves to its policy class via ``config.task.policy_type`` →
``holosoma.policies.by_type`` (handled in ``_select_policy_class``), and the node
attaches itself as that policy's dense ``TargetSource`` (it already subscribes
``CmdDense`` and serves ``get_target()``). Drive it with ``--task.velocity-input
injected --task.state-input injected`` and feed ``CmdDense`` on the dense topic.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import replace

import numpy as np
import rclpy
from holosoma_msgs.msg import CmdDense, Heartbeat
from loguru import logger
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from holosoma_inference.config.config_values.inference import get_annotated_inference_config
from holosoma_inference.policies.dual_mode import MultiModePolicy, _select_policy_class
from holosoma_inference.sensors.base import Sensor
from holosoma_inference.utils.config_registry import parse_config
from holosoma_service.policy_control.injected_inputs import (
    CMD_VEL_TOPIC,
    STATE_INPUT_TOPIC,
    InjectedRos2Input,
)
from holosoma_service.policy_control.sensors import Ros2DepthConsumer

DENSE_TOPIC = "/holosoma/dense_tracking_command"
EXECUTED_CMD_TOPIC = "/holosoma/holosoma_executed_cmd"
HEARTBEAT_TOPIC = "/holosoma/heartbeat"
HEARTBEAT_EVERY = 10  # control ticks -> 5 Hz at a 50 Hz control loop

# Multi-policy interface: clients read the registry and switch by name / checkpoint.
POLICY_STATUS_TOPIC = "/holosoma/policy_status"
SELECT_POLICY_TOPIC = "/holosoma/select_policy"
SELECT_CHECKPOINT_TOPIC = "/holosoma/select_checkpoint"


def _basename(path: str) -> str:
    """Filename for status display; tolerant of empty/None."""
    return path.rsplit("/", 1)[-1] if path else ""


def _control_type(policy) -> str:
    """The command-contract category a consumer keys off: 'WBT' | 'LOCOMOTION'.

    Explicit ``policy.CONTROL_TYPE`` wins (extensions may declare it); else infer
    from the observation contract — a ``motion_command`` obs term means the policy
    consumes a dense per-DOF tracking target (WBT), otherwise it is velocity-driven
    locomotion.
    """
    declared = getattr(policy, "CONTROL_TYPE", None)
    if declared:
        return str(declared)
    obs = getattr(getattr(policy, "config", None), "observation", None)
    actor_obs = getattr(obs, "obs_dict", {}).get("actor_obs", []) if obs else []
    return "WBT" if "motion_command" in actor_obs else "LOCOMOTION"


def _interface_id(policy) -> str:
    """Semantic id of the command interface the policy speaks (D3 / interface_id).

    Prefer an explicit ``task.interface_id``; else fall back to ``task.policy_type``
    (the by_type key), which identifies the command contract in practice.
    """
    task = getattr(getattr(policy, "config", None), "task", None)
    return str(getattr(task, "interface_id", "") or getattr(task, "policy_type", "") or "")


def build_sensors(node: Node, task_config) -> dict[str, Sensor]:
    """Build the injected sensors for one policy from its own ``task`` config.

    Called per registered policy so each gets sensors sized from its own
    ``task.depth`` (policies may have different sensor configs). Empty
    ``task.depth.topics`` -> no sensors.
    """
    sensors: dict[str, Sensor] = {}
    depth_cfg = getattr(task_config, "depth", None) if task_config else None
    if depth_cfg is not None and depth_cfg.topics:
        sensors["depth"] = Ros2DepthConsumer(
            node,
            topics=list(depth_cfg.topics),
            resized_height=depth_cfg.resized_height,
            resized_width=depth_cfg.resized_width,
            near_clip=depth_cfg.near_clip,
            far_clip=depth_cfg.far_clip,
            frame_delay_ms=depth_cfg.frame_delay_ms,
        )
    return sensors

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

    def __init__(self, num_dofs: int, vel_timeout: float = 1.0, task_config=None):
        super().__init__("policy_service")
        # --- Input: one combined vel+state provider (subscribes on THIS node) ---
        # A single object implements both protocols, mirroring Ros2Input, so the
        # base policy can share it for both roles when velocity_input ==
        # state_input == "injected".
        self.input = InjectedRos2Input(self, CMD_VEL_TOPIC, STATE_INPUT_TOPIC, vel_timeout=vel_timeout)

        # Sensors for the owner (single-policy path). Multi-policy builds per-policy
        # sensors in _build_policy; each gets its own from its own task.depth.
        self.sensors: dict[str, Sensor] = build_sensors(self, task_config)

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

        # --- Multi-policy: status out + select-by-name / -checkpoint in ---
        # ``_switcher`` (a MultiModePolicy) is attached after the policy is built.
        # Select callbacks no-op until then, so an early client message can't crash.
        self._switcher = None
        from std_msgs.msg import String

        # Latched (transient_local, depth 1): a client connecting after boot and
        # before the next switch still gets the current registry on connect.
        status_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._policy_status_pub = self.create_publisher(String, POLICY_STATUS_TOPIC, status_qos)
        self.create_subscription(String, SELECT_POLICY_TOPIC, self._on_select_policy, 10)
        self.create_subscription(String, SELECT_CHECKPOINT_TOPIC, self._on_select_checkpoint, 10)

    def attach_switcher(self, switcher) -> None:
        """Wire the MultiModePolicy so select topics route to it + publish initial status."""
        self._switcher = switcher
        self.publish_policy_status()

    def _on_select_policy(self, msg) -> None:
        if self._switcher is not None:
            self._switcher.select(msg.data.strip())

    def _on_select_checkpoint(self, msg) -> None:
        if self._switcher is None:
            return
        try:
            index = int(msg.data.strip())
        except (TypeError, ValueError):
            logger.warning(f"select_checkpoint: expected an int index, got {msg.data!r}")
            return
        self._switcher.active._activate_policy(index)
        self.publish_policy_status()

    def publish_policy_status(self) -> None:
        """Publish the registry JSON: per-policy name, control_type, interface_id,
        checkpoints, active_checkpoint — plus the active policy name/index.

        ``control_type`` (WBT | LOCOMOTION) and ``interface_id`` describe the
        command CONTRACT a consumer (SlimNav/telemetry) keys off, independent of
        the policy name (D3): WBT consumes a dense motion command; locomotion
        consumes velocity. Derived per policy from its command contract.
        """
        import json

        from std_msgs.msg import String

        sw = self._switcher
        if sw is None:
            return
        policies = []
        for name, policy in zip(sw.names, sw.policies):
            groups = getattr(policy, "checkpoint_groups", [[getattr(policy, "active_model_path", "")]])
            policies.append(
                {
                    "name": name,
                    "control_type": _control_type(policy),
                    "interface_id": _interface_id(policy),
                    "checkpoints": [[_basename(f) for f in group] for group in groups],
                    "active_checkpoint": getattr(policy, "active_policy_index", 0),
                }
            )
        payload = {"policies": policies, "active": sw.active_label, "active_index": sw.active_index}
        self._policy_status_pub.publish(String(data=json.dumps(payload)))

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
    """Yield the underlying policy instance(s) (all registered, for multi-mode)."""
    if isinstance(policy, MultiModePolicy):
        yield from policy.policies
    else:
        yield policy


def _build_one_policy(cfg, io: ServiceIONode, owner):
    """Build one registered policy with node-owned injected inputs + its own sensors.

    ``owner`` is None for the hardware owner (index 0) or the owner policy for the
    shared-hardware peers. The owner takes the injected vel/state providers; peers
    reuse them via ``_shared_hardware_source``. Every policy gets its OWN sensors
    built from its own ``task`` (per-policy depth config).
    """
    cls = _select_policy_class(cfg)
    p = object.__new__(cls)
    if owner is None:
        p._injected_velocity_input = io.input
        p._injected_command_provider = io.input
    else:
        p._shared_hardware_source = owner
    sensors = build_sensors(io, cfg.task)
    if sensors:
        p._injected_sensors = sensors
    # Attach the dense target source before __init__ (WBT detects live-source mode).
    p._target_source = io
    p.__init__(config=cfg)
    return p


def _build_policy(config, io: ServiceIONode):
    """Build the policy (or MultiModePolicy for N registered policies) with injection.

    ``config.registered_policies()`` yields [owner, *extras] (normalizing the legacy
    ``secondary``). One policy -> build it directly; N -> owner owns hardware, peers
    share it, all wrapped in a MultiModePolicy that publishes status on switch.
    """
    registered = config.registered_policies()
    if len(registered) == 1:
        policy = _build_one_policy(registered[0], io, owner=None)
        logger.info(f"Using {type(policy).__name__}")
        return policy

    names = config.resolved_names()
    owner = _build_one_policy(registered[0], io, owner=None)
    peers = [_build_one_policy(cfg, io, owner=owner) for cfg in registered[1:]]
    switcher = MultiModePolicy([owner, *peers], names=names, on_switch=lambda i, n: io.publish_policy_status())
    io.attach_switcher(switcher)
    return switcher


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


def _map_registered(config, fn):
    """Apply ``fn`` to ``config`` and every registered extra (secondary + policies).

    ``fn`` transforms one config's own fields (not its extras); this threads it
    through the legacy ``secondary`` and the N-policy ``policies`` so every
    registered policy gets the same treatment.
    """
    config = fn(config)
    if config.secondary is not None:
        config = replace(config, secondary=fn(config.secondary))
    if config.policies:
        config = replace(config, policies=tuple(fn(p) for p in config.policies))
    return config


def _use_mp_sdk(config):
    """Rewrite ``robot.sdk_type`` to its rclpy-safe multiprocess variant on every
    registered policy (e.g. unitree -> unitree_mp). No-op for types without a
    known isolated variant, so an explicit override survives."""

    def _one(cfg):
        new_type = _MP_SDK_MAP.get(cfg.robot.sdk_type)
        if new_type is None:
            return cfg
        return replace(cfg, robot=replace(cfg.robot, sdk_type=new_type))

    out = _map_registered(config, _one)
    if _MP_SDK_MAP.get(config.robot.sdk_type):
        logger.info(f"Service node: using multiprocess SDK (rclpy/DDS isolation)")
    return out


def _apply_noninteractive_defaults(config):
    """Set ``skip_stiff_prompt`` on every registered policy (no TTY for the input()
    prompt in the service node)."""
    return _map_registered(config, lambda cfg: replace(cfg, task=replace(cfg.task, skip_stiff_prompt=True)))


def main() -> None:
    # Under `ros2 launch` / `ros2 run`, argv carries ROS args (e.g.
    # "--ros-args -r __node:=..."). tyro would reject those, so strip them
    # before parsing this node's own CLI; rclpy.init() consumes them separately.
    import argparse

    from rclpy.utilities import remove_ros_args

    sys.argv = remove_ros_args(args=sys.argv)

    # Handle the top-level secondary disable flag before parsing the config.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--secondary", default=None, help="Set to 'none' to disable dual-mode.")
    known, remaining = pre.parse_known_args()
    disable_secondary = known.secondary is not None and known.secondary.lower() == "none"
    sys.argv = [sys.argv[0]] + remaining

    config = parse_config(get_annotated_inference_config)

    if disable_secondary:
        config = replace(config, secondary=None)
    rclpy.init()

    # Route the Unitree SDK through its multiprocess proxy so its CycloneDDS
    # never shares this process's rclpy (otherwise SDK init heap-corrupts).
    config = _use_mp_sdk(config)

    # This node is non-interactive: skip the stiff-hold stdin prompt and quiet
    # the per-tick motion-timestep log.
    config = _apply_noninteractive_defaults(config)

    # One node owns every subscription + publisher.
    io = ServiceIONode(config.robot.num_joints, vel_timeout=config.task.ros_vel_timeout, task_config=config.task)

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
