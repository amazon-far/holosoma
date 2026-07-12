"""Dual-mode policy with smooth interpolation switching between two policies."""

from __future__ import annotations

import itertools

import numpy as np
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig


def _select_policy_class(config: InferenceConfig):
    """Determine policy class based on observation config and robot type."""
    from holosoma_inference.compat import entry_points
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
    """Wraps two policy instances with smooth-interpolation switching.

    Loco → WBT
        1.5 s lerp from current pose to the first frame of the reference
        motion. Only upper-body DOFs (torso + arms) are overridden; the
        loco policy continues controlling the legs for balance.

    WBT → Loco
        0.5 s hold-current-pose, then switch to loco, then 1.5 s lerp to
        the default standing pose (upper body only).
    """

    # H1 upper body DOF indices (torso + arms).  Legs 0..9 stay under
    # loco control during transitions.
    _UPPER_BODY_INDICES = list(range(10, 19))

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def __init__(self, primary_config: InferenceConfig, secondary_config: InferenceConfig):
        primary_cls = _select_policy_class(primary_config)
        secondary_cls = _select_policy_class(secondary_config)

        logger.info(
            colored(f"Dual-mode: primary={primary_cls.__name__}, secondary={secondary_cls.__name__}", "magenta")
        )

        self.primary = primary_cls(config=primary_config)
        self._num_dofs = self.primary.num_dofs

        logger.info(colored("Initializing secondary policy (shared hardware)...", "magenta"))
        secondary = object.__new__(secondary_cls)
        secondary._shared_hardware_source = self.primary
        secondary.__init__(config=secondary_config)
        self.secondary = secondary

        self.active = self.primary
        self.active_label = "primary"
        self.secondary._dual_mode_active = True

        self._setup_command_intercept()
        logger.info(
            colored(
                "Dual-mode ready.  x = switch policy  |  m = start WBT motion  |  o = stop",
                "magenta",
            )
        )

    # ------------------------------------------------------------------
    #  Interpolation durations (seconds)
    # ------------------------------------------------------------------

    @property
    def _switch_durations(self):
        return {
            "loco_to_wbt": (0.0, 1.5),
            "wbt_to_loco": (0.5, 1.5),
        }

    # ------------------------------------------------------------------
    #  Command-intercept wiring
    # ------------------------------------------------------------------

    def _setup_command_intercept(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        for policy in (self.primary, self.secondary):
            policy._command_provider._mapping["X"] = StateCommand.SWITCH_MODE
            policy._command_provider._mapping["x"] = StateCommand.SWITCH_MODE
            policy._command_provider._mapping["m"] = StateCommand.START_MOTION_CLIP
            policy._command_provider._mapping["o"] = StateCommand.STOP

        self._orig_dispatch = {
            id(self.primary): self.primary._dispatch_command,
            id(self.secondary): self.secondary._dispatch_command,
        }

        def patched_dispatch(cmd):
            # m / START_MOTION_CLIP
            if cmd == StateCommand.START_MOTION_CLIP:
                if self.active is self.primary and self._is_wbt(self.secondary):
                    # Switch to WBT AND start motion
                    self._switch_to_wbt(auto_start_motion=True)
                elif self.active is self.secondary and self._is_wbt(self.secondary):
                    self.secondary._handle_start_motion_clip()
            # o / STOP on WBT → return to loco
            elif cmd == StateCommand.STOP and self.active is self.secondary and self._is_wbt(self.secondary):
                self._return_to_primary()
            # x / SWITCH_MODE
            elif cmd == StateCommand.SWITCH_MODE:
                self._handle_mode_switch()
            else:
                self._orig_dispatch[id(self.active)](cmd)

        self.primary._dispatch_command = patched_dispatch
        self.secondary._dispatch_command = patched_dispatch

    @staticmethod
    def _is_wbt(policy) -> bool:
        return hasattr(policy, "_handle_start_motion_clip")

    # ------------------------------------------------------------------
    #  Policy activation helpers
    # ------------------------------------------------------------------

    def _copy_interface_key_state(self, target) -> None:
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        active_dev = self.active._velocity_input
        target_dev = target._velocity_input
        if isinstance(active_dev, InterfaceInput) and isinstance(target_dev, InterfaceInput):
            target_dev.key_states = active_dev.key_states.copy()
            target_dev.last_key_states = active_dev.key_states.copy()

    def _activate_target(self, target, target_label: str) -> None:
        target._resolve_control_gains()
        self._copy_interface_key_state(target)
        self.active = target
        self.active_label = target_label
        self.active._init_phase_components()

    # ------------------------------------------------------------------
    #  Loco → WBT  (switch with optional auto-start of motion)
    # ------------------------------------------------------------------

    def _switch_to_wbt(self, auto_start_motion: bool = False) -> None:
        """Activate WBT, optionally with smooth upper-body interpolation.

        *auto_start_motion=True* (``m`` key) switches immediately and starts
        the motion clip — WBT's own PD control pulls the robot to the
        reference pose.

        *auto_start_motion=False* (``x`` key) lerps the upper body over
        1.5 s before handing control to WBT in stiff-hold.
        """
        if auto_start_motion:
            logger.info(
                colored("Loco → WBT (motion): immediate switch…", "magenta", attrs=["bold"])
            )
            self.active._cancel_policy_interp()
            self._activate_target(self.secondary, "secondary")
            self.secondary._handle_start_motion_clip()
            logger.info(colored("WBT policy now active – motion started.", "magenta"))
            return

        # --- manual switch (x key) with smooth interpolation ---
        logger.info(
            colored("Loco → WBT (switch): smooth interpolation (upper body only)…",
                    "magenta", attrs=["bold"])
        )
        delay_s, interp_s = self._switch_durations["loco_to_wbt"]
        target_q = self.secondary.motion_command_0[:, : self._num_dofs].copy()

        def _on_done():
            self.active._cancel_policy_interp()
            self._activate_target(self.secondary, "secondary")
            self.secondary._stiff_hold_confirmed = True
            self.secondary._stiff_hold_transition_pending = True
            logger.info(colored("WBT policy now active.", "magenta"))

        self.active._start_policy_interp(
            target_q=target_q,
            duration_s=interp_s,
            delay_s=delay_s,
            on_done=_on_done,
            override_indices=self._upper_body_indices,
        )

    # ------------------------------------------------------------------
    #  WBT → Loco  (stop / motion end)
    # ------------------------------------------------------------------

    def _return_to_primary(self) -> None:
        """WBT → Loco: switch immediately, lerp upper body to default stand.

        Legs stay under loco control — the locomotion policy naturally
        converges to a stable stance without being overridden.
        """
        logger.info(
            colored(
                "WBT → Loco: immediate switch + upper-body lerp to default stand…",
                "magenta",
                attrs=["bold"],
            )
        )

        target_q = self.primary.default_dof_angles.copy().reshape(1, -1)
        interp_s = 1.5  # seconds

        if self.active is self.secondary and self._is_wbt(self.secondary):
            self.secondary.use_policy_action = False
            self.secondary.motion_clip_progressing = False

        self.active._cancel_policy_interp()
        self._activate_target(self.primary, "primary")

        # Reset _last_sent_q to the robot's *actual* joint positions so the
        # _begin_target_transition inside _handle_start_policy blends from
        # where the robot physically is, not from WBT's last dance command.
        robot_state = self.active.interface.get_low_state()
        self.primary._last_sent_q = robot_state[:, 7 : 7 + self._num_dofs].copy()

        self.primary._handle_start_policy()
        self.primary._start_policy_interp(
            target_q=target_q,
            duration_s=interp_s,
            delay_s=0.0,
            on_done=lambda: logger.info(colored("WBT → Loco complete.", "magenta")),
            override_indices=self._upper_body_indices,
        )

    # ------------------------------------------------------------------
    #  Manual mode switch (x key) — just switch, don't start motion
    # ------------------------------------------------------------------

    def _handle_mode_switch(self):
        """x-key: switch between loco and WBT without auto-starting motion."""
        if self.active is self.primary and self._is_wbt(self.secondary):
            self._switch_to_wbt(auto_start_motion=False)
        else:
            self._return_to_primary()

    # ------------------------------------------------------------------
    #  Upper body indices (configurable per robot)
    # ------------------------------------------------------------------

    @property
    def _upper_body_indices(self) -> list[int]:
        """DOF indices to override during interpolation.

        H1: indices 10-18 = torso + left arm 4 + right arm 4.
        Legs 0-9 stay under the active policy's control for balance.
        """
        if hasattr(self.primary.robot_config, "dof_names_upper_body"):
            upper_names = self.primary.robot_config.dof_names_upper_body
            dof_names = self.primary.robot_config.dof_names
            return [dof_names.index(n) for n in upper_names if n in dof_names]
        return self._UPPER_BODY_INDICES

    # ------------------------------------------------------------------
    #  Main run loop
    # ------------------------------------------------------------------

    def run(self):
        """Main run loop — delegates to the active policy."""
        try:
            for it in itertools.count():
                cycle_policy = self.active
                cycle_policy.latency_tracker.start_cycle()

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

                # Warm up inactive policy's observation history
                inactive = self.secondary if self.active is self.primary else self.primary
                robot_state = self.active.interface.get_low_state()
                inactive.prepare_obs_for_rl(robot_state)

                self.active.policy_action()

                # Auto-return: WBT stopped or motion ended → back to loco.
                # Only fire once — skip while a transition is already in flight.
                if (
                    self.active is self.secondary
                    and self._is_wbt(self.secondary)
                    and not self.secondary.use_policy_action
                    and not self.secondary.motion_clip_progressing
                    and self.active._interp_state == "idle"
                ):
                    self._return_to_primary()

                cycle_policy.latency_tracker.end_cycle()

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
