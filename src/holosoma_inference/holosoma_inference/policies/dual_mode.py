"""Dual-mode policy with runtime switching between two policy instances."""

from __future__ import annotations

import itertools

from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig


def _select_policy_class(config: InferenceConfig):
    """Determine policy class based on observation config and robot type.

    Checks entry point groups ``holosoma.policies.locomotion`` and
    ``holosoma.policies.wbt`` (keyed by ``robot_type``) so extensions can
    register custom policy classes without monkey-patching.
    """
    from importlib.metadata import entry_points

    from holosoma_inference.policies.locomotion import LocomotionPolicy
    from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

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

        self._patch_button_handlers()
        logger.info(colored("Dual-mode ready. Press X (joystick) or x (keyboard) to switch policies.", "magenta"))

    def _patch_button_handlers(self):
        """Intercept X (joystick) and x (keyboard) for mode switching.

        The keyboard/joystick listener thread belongs to the primary policy,
        so we only need to patch the primary. For non-switch keys, delegate
        to whichever policy is currently active.
        """
        # Store original (unpatched) handlers per policy for delegation
        self._orig_joy = {
            id(self.primary): self.primary.handle_joystick_button,
            id(self.secondary): self.secondary.handle_joystick_button,
        }
        self._orig_kb = {
            id(self.primary): self.primary.handle_keyboard_button,
            id(self.secondary): self.secondary.handle_keyboard_button,
        }

        def patched_joy(cur_key):
            if cur_key == "X":
                self._handle_mode_switch()
            else:
                self._orig_joy[id(self.active)](cur_key)

        def patched_kb(keycode):
            if keycode == "x":
                self._handle_mode_switch()
            else:
                self._orig_kb[id(self.active)](keycode)

        self.primary.handle_joystick_button = patched_joy
        self.primary.handle_keyboard_button = patched_kb
        self.secondary.handle_joystick_button = patched_joy
        self.secondary.handle_keyboard_button = patched_kb

    def _handle_mode_switch(self):
        """Switch from active to inactive policy."""
        self.active._handle_stop_policy()

        target = self.secondary if self.active is self.primary else self.primary
        target_label = "secondary" if target is self.secondary else "primary"

        # Update KP/KD on the shared interface for the target policy
        target._resolve_control_gains()

        # Carry over joystick key_states so edge detection doesn't see a false
        # rising edge on the X button (which is still physically held down).
        if hasattr(self.active, "key_states"):
            target.key_states = self.active.key_states.copy()
            target.last_key_states = self.active.key_states.copy()

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

                if self.active.use_joystick and self.active.interface.get_joystick_msg() is not None:
                    self.active.process_joystick_input()
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
