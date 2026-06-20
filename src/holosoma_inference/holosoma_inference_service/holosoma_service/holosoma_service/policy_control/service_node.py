"""Unified policy service node (Variant A).

Runs *any* holosoma policy (locomotion, WBT, …) as a ROS2 service and supports
runtime policy swapping — the ``run_policy.py`` experience served over ROS2.

Design (Variant A — reuse ``Ros2Input`` via config):

    /cmd_vel + /…/state_input ─▶ (loco)  Ros2Input ─(pull)─┐
                                  [owned by the policy]      ├─▶ policy.run() ─▶ robot
    /…/dense_tracking_command ─▶ (wbt)  DenseTargetBridge ──┘
                                  swap via SWITCH_MODE ──────┘
                                  every tick ─▶ executed_cmd + heartbeat (policy-agnostic)

The policy is built exactly the way ``run_policy.py:main`` builds it
(``_select_policy_class`` / ``DualModePolicy``), so:

  * **Loco input** needs no code here — launch with
    ``--task.velocity-input ros2 --task.state-input ros2`` and each policy spins
    up its own ``Ros2Input`` (Twist + state String), including ``switch_mode``.
  * **WBT input** is injected: a ``DenseTargetBridge`` is attached as the
    policy's ``_target_source`` (only for policies that expose that attribute).
  * **Output** is policy-agnostic: a ``PolicyFeedbackPublisher`` is wired into
    each policy's per-tick ``_on_command_sent`` hook.

Unlike the original WBT-only ``holosoma_node``, this does not depend on the
extension-only ``config.task.policy_type`` / ``holosoma.policies.by_type`` entry
points — it uses the same observation/robot-type resolution as the CLI.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import rclpy
import tyro
from loguru import logger

from holosoma_inference.config.config_values.inference import get_annotated_inference_config
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.dual_mode import DualModePolicy, _select_policy_class

from holosoma_service.policy_control.feedback import PolicyFeedbackPublisher, spin_in_thread
from holosoma_service.policy_control.target_bridge import DenseTargetBridge

# This node loads rclpy in-process, which clashes with the Unitree SDK's bundled
# CycloneDDS (heap corruption / "free(): invalid pointer" at SDK init). The
# multiprocess interface ("unitree_mp", added by PR #124) runs the low-level
# binding in a spawned child so DDS never shares the parent's rclpy address
# space. Map the in-process SDK -> its isolated variant for any service run.
_MP_SDK_MAP = {"unitree": "unitree_mp"}


def _iter_policies(policy):
    """Yield the underlying policy instance(s): primary+secondary for dual-mode."""
    if isinstance(policy, DualModePolicy):
        yield policy.primary
        if policy.secondary is not None:
            yield policy.secondary
    else:
        yield policy


def _attach_target_source(policy, bridge: DenseTargetBridge) -> bool:
    """Attach the dense bridge as the target source of every WBT-style policy.

    Only policies that expose ``_target_source`` (WBT family) are touched;
    locomotion policies are left untouched. Returns True if any policy was
    attached (so the caller knows whether to spin the bridge).
    """
    attached = False
    for p in _iter_policies(policy):
        if hasattr(p, "_target_source"):
            p._target_source = bridge
            attached = True
            logger.info(f"Attached DenseTargetBridge to {type(p).__name__}")
    return attached


def _wire_feedback(policy, feedback: PolicyFeedbackPublisher) -> None:
    """Wire the feedback publisher into each policy's per-tick hook."""
    for p in _iter_policies(policy):
        p._on_command_sent = feedback.make_hook(p)


def _dof_names(policy) -> list[str]:
    """Joint names of the (primary) policy, for labeling executed-cmd msgs."""
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


def _build_policy(config):
    """Build the policy the same way run_policy.py does (dual-mode aware)."""
    if config.secondary is not None:
        return DualModePolicy(primary_config=config, secondary_config=config.secondary)
    policy_class = _select_policy_class(config)
    logger.info(f"Using {policy_class.__name__}")
    return policy_class(config=config)


def main() -> None:
    # Under `ros2 launch` / `ros2 run`, argv carries ROS args (e.g.
    # "--ros-args -r __node:=..."). tyro would reject those, so strip them
    # before parsing this node's own CLI; rclpy.init() consumes them separately.
    from rclpy.utilities import remove_ros_args

    cli_argv = remove_ros_args(args=sys.argv)
    sys.argv = cli_argv

    config = tyro.cli(get_annotated_inference_config(), config=TYRO_CONFIG)
    rclpy.init()

    # 0. Route the Unitree SDK through its multiprocess proxy so its CycloneDDS
    #    never shares this process's rclpy (otherwise SDK init heap-corrupts).
    config = _use_mp_sdk(config)

    # 1. Build the policy exactly like run_policy.py — swap support for free.
    policy = _build_policy(config)

    # 2. Loco input: nothing to wire (config velocity_input/state_input=ros2
    #    makes each policy own its Ros2Input — Twist + state String).
    # 3. WBT input: inject a CmdDense bridge as the target source, but only for
    #    policies that have one (WBT family). Spin it only if attached.
    bridge = DenseTargetBridge(config.robot.num_joints)
    if _attach_target_source(policy, bridge):
        spin_in_thread(bridge)

    # 4. Output: policy-agnostic feedback every tick, every state.
    feedback = PolicyFeedbackPublisher(dof_names=_dof_names(policy))
    _wire_feedback(policy, feedback)
    spin_in_thread(feedback)

    logger.info("PolicyServiceNode ready — running policy.")
    try:
        policy.run()  # owns its RateLimiter + SDK I/O; blocks in this thread
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
