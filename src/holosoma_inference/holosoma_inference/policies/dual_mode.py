"""Dual-mode policy with runtime switching between two policy instances."""

from __future__ import annotations

import itertools

from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig


def _select_policy_class(config: InferenceConfig):
    """Determine policy class from an explicit ``policy_type`` or the obs/robot heuristic.

    Resolution order:

    1. ``config.task.policy_type`` (when set) against the ``holosoma.policies.by_type``
       entry-point group. This lets extensions register custom policy classes keyed by an
       explicit type string -- the same mechanism the standalone ``run_policy`` path uses --
       so a service launched with an extension preset resolves to the same class. The field
       is optional (only some ``TaskConfig`` subclasses define it), hence ``getattr``; an
       unregistered ``policy_type`` falls through to the heuristic rather than hard-failing.
    2. The ``holosoma.policies.wbt`` / ``holosoma.policies.locomotion`` groups (keyed by
       ``robot_type``) plus a ``motion_command`` observation heuristic.

    All lookups are by string against entry-point groups, so core never imports or names
    extension-private policy classes.
    """
    from holosoma_inference.compat import entry_points
    from holosoma_inference.policies.locomotion import LocomotionPolicy
    from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

    # 1. Explicit policy_type -> by_type entry-point group.
    policy_type = getattr(config.task, "policy_type", None)
    if policy_type:
        for ep in entry_points(group="holosoma.policies.by_type"):
            if ep.name == policy_type:
                return ep.load()
        # policy_type set but unregistered: fall through to the heuristic below.

    # 2. robot_type-keyed groups + observation heuristic.
    robot_type = config.robot.robot_type
    actor_obs = config.observation.obs_dict.get("actor_obs", [])

    if "motion_command" in actor_obs:
        for ep in entry_points(group="holosoma.policies.wbt"):
            if ep.name == robot_type:
                return ep.load()
        return WholeBodyTrackingPolicy

    for ep in entry_points(group="holosoma.policies.locomotion"):
        if ep.name == robot_type:
            return ep.load()
    return LocomotionPolicy


class DualModePolicy:
    """Wraps two policy instances (potentially different classes) with X-button switching.

    The primary policy is fully initialized and owns the hardware (SDK, interface,
    input handlers). The secondary policy reuses the primary's hardware via the
    _shared_hardware_source guard pattern in BasePolicy.

    Press X (joystick) or x (keyboard) to switch between policies at runtime.
    The existing Select/1-9 multi-model switching still works within each policy.
    """

    def __init__(self, primary_config: InferenceConfig, secondary_config: InferenceConfig):
        primary_cls = _select_policy_class(primary_config)
        secondary_cls = _select_policy_class(secondary_config)

        logger.info(
            colored(f"Dual-mode: primary={primary_cls.__name__}, secondary={secondary_cls.__name__}", "magenta")
        )

        # Fully init primary (owns hardware)
        self.primary = primary_cls(config=primary_config)

        # Init secondary with shared hardware
        logger.info(colored("Initializing secondary policy (shared hardware)...", "magenta"))
        secondary = object.__new__(secondary_cls)
        secondary._shared_hardware_source = self.primary
        secondary.__init__(config=secondary_config)
        self.secondary = secondary

        self.active = self.primary
        self.active_label = "primary"

        self._setup_command_intercept()
        logger.info(colored("Dual-mode ready. Press X (joystick) or x (keyboard) to switch policies.", "magenta"))

    def _setup_command_intercept(self):
        """Inject SWITCH_MODE into mappings and patch dispatch for routing.

        Keyboard queue wiring is handled by the factory — the secondary's
        ``KeyboardInput`` gets its own subscriber queue from the shared
        ``_KeyboardListenerThread``.  Only ``_dispatch_command`` needs
        patching to intercept SWITCH_MODE.
        """
        from holosoma_inference.inputs.api.commands import StateCommand

        # Inject SWITCH_MODE into both command providers' key mappings (joystick X,
        # keyboard x). Only keyboard/joystick providers expose ``_mapping``; others
        # (e.g. injected ROS2 providers, which map the "switch_mode" string to
        # SWITCH_MODE natively) are intercepted purely via the dispatch patch below.
        for policy in (self.primary, self.secondary):
            mapping = getattr(policy._command_provider, "_mapping", None)
            if mapping is not None:
                mapping["X"] = StateCommand.SWITCH_MODE
                mapping["x"] = StateCommand.SWITCH_MODE

        # Patch _dispatch_command to intercept SWITCH_MODE
        self._orig_dispatch = {
            id(self.primary): self.primary._dispatch_command,
            id(self.secondary): self.secondary._dispatch_command,
        }

        def patched_dispatch(cmd):
            if cmd == StateCommand.SWITCH_MODE:
                self._handle_mode_switch()
            else:
                self._orig_dispatch[id(self.active)](cmd)

        self.primary._dispatch_command = patched_dispatch
        self.secondary._dispatch_command = patched_dispatch

    def _handle_mode_switch(self):
        """Switch from active to inactive policy."""
        self.active._handle_stop_policy()

        target = self.secondary if self.active is self.primary else self.primary
        target_label = "secondary" if target is self.secondary else "primary"

        # Update KP/KD on the shared interface for the target policy
        target._resolve_control_gains()

        # Carry over joystick key_states so edge detection doesn't see a false
        # rising edge on the X button (which is still physically held down).
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        active_dev = self.active._velocity_input
        target_dev = target._velocity_input
        if isinstance(active_dev, InterfaceInput) and isinstance(target_dev, InterfaceInput):
            target_dev.key_states = active_dev.key_states.copy()
            target_dev.last_key_states = active_dev.key_states.copy()

        self.active = target
        self.active_label = target_label

        # Re-initialize phase and activate
        self.active._init_phase_components()
        self.active._handle_start_policy()

        logger.info(
            colored(
                f"Switched to {self.active_label} policy ({type(self.active).__name__})",
                "magenta",
                attrs=["bold"],
            )
        )

    def run(self):
        """Main run loop — delegates to the active policy."""
        try:
            for it in itertools.count():
                self.active.latency_tracker.start_cycle()

                vc = self.active._velocity_input.poll_velocity()
                if vc is not None:
                    self.active._apply_velocity(vc)
                commands = self.active._command_provider.poll_commands()
                for cmd in commands:
                    self.active._dispatch_command(cmd)
                if commands:
                    self.active._print_control_status()
                if self.active.use_phase:
                    self.active.update_phase_time()

                self.active.policy_action()

                self.active.latency_tracker.end_cycle()

                if it % 50 == 0 and self.active.use_policy_action:
                    debug_str = (
                        f"[{self.active_label}] "
                        f"RL FPS: {self.active.latency_tracker.get_fps():.2f} | "
                        f"{self.active.latency_tracker.get_stats_str()}"
                    )
                    self.active.logger.info(debug_str, flush=True)

                self.active.rate.sleep()

        except KeyboardInterrupt:
            pass
