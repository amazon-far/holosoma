"""Multi-mode policy with runtime switching between N policy instances.

``MultiModePolicy`` wraps N already-constructed policy instances (potentially
different classes) that share one set of hardware — index 0 owns the hardware
(SDK, interface, input providers) and the rest were built with
``_shared_hardware_source = policies[0]`` (see ``BasePolicy``). Exactly one is
``active`` at a time; its control loop runs while the others stay dormant.

Switching:
  * ``switch_mode`` (keyboard ``x`` / joystick ``X`` / the injected ROS2 string)
    advances to the NEXT policy in the ring.
  * ``select(name)`` / ``select(index)`` jumps to a specific registered policy —
    this is what a client (e.g. far-pi) drives over ``holosoma/select_policy``.
Per-policy CHECKPOINT switching (``SWITCH_POLICY_1..9`` over ``model_path``) is
unchanged and still handled inside each policy.

``DualModePolicy`` is kept as a thin 2-policy compatibility shim that builds its
two policies and delegates to ``MultiModePolicy``.
"""

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
    from holosoma_inference.policies.locomotion import LocomotionPolicy
    from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
    from holosoma_inference.pycompat import entry_points

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


class MultiModePolicy:
    """Runtime switching across N pre-built policy instances sharing one HW set.

    Construct with the already-built policies (index 0 = hardware owner) and their
    switch-key ``names`` (same order). The service node builds each policy with the
    right injected inputs/sensors and hands them here; this class owns only the
    switching + run loop, not construction — so per-policy sensor injection stays
    in the node.

    An optional ``on_switch`` callback (``fn(index, name) -> None``) fires after
    every activation (initial + each switch) so the node can publish policy status.

    Run-status safety today is coarse: a switch never auto-starts the target (D5)
    and ``select`` refuses to switch while the active policy is running. This is a
    safety interlock, not a formal FSM. TODO(fsm): a proper run-status FSM should
    live PER POLICY TYPE — each policy variant declaring/validating its own legal
    transitions (STIFF_HOLD → GET_READY → RUNNING, and what a switch/start/stop is
    allowed to do from each) — with this wrapper deferring to that contract rather
    than reading raw ``use_policy_action``. Owner: coordinate with the run-status
    FSM author (jtomasle) since it touches the shared base command layer.
    """

    def __init__(self, policies, names, active_index: int = 0, on_switch=None):
        if not policies:
            raise ValueError("MultiModePolicy requires at least one policy")
        if len(policies) != len(names):
            raise ValueError(f"policies ({len(policies)}) and names ({len(names)}) length mismatch")
        if len(set(names)) != len(names):
            raise ValueError(f"policy names must be unique, got {names}")
        if not (0 <= active_index < len(policies)):
            raise ValueError(f"active_index {active_index} out of range for {len(policies)} policies")

        self.policies = list(policies)
        self.names = list(names)
        self._name_to_index = {n: i for i, n in enumerate(self.names)}
        self.active_index = active_index
        self._on_switch = on_switch

        self.active = self.policies[self.active_index]
        self.active_label = self.names[self.active_index]

        self._setup_command_intercept()
        if len(self.policies) > 1:
            logger.info(
                colored(
                    f"Multi-mode ready with {len(self.policies)} policies {self.names}; "
                    f"active={self.active_label}. Publish 'switch_mode' or select by name to swap.",
                    "magenta",
                )
            )
        if self._on_switch is not None:
            self._on_switch(self.active_index, self.active_label)

    def _setup_command_intercept(self):
        """Route SWITCH_MODE to the ring-advance handler across all policies.

        Every policy's ``_dispatch_command`` is patched so that whichever one is
        active, a SWITCH_MODE advances the ring; all other commands go to the
        active policy's original dispatch. Keyboard/joystick providers also get
        ``x``/``X`` mapped to SWITCH_MODE (injected ROS2 providers map the
        ``"switch_mode"`` string natively, so they need no mapping edit).
        """
        from holosoma_inference.inputs.api.commands import StateCommand

        for policy in self.policies:
            mapping = getattr(policy._command_provider, "_mapping", None)
            if mapping is not None:
                mapping["X"] = StateCommand.SWITCH_MODE
                mapping["x"] = StateCommand.SWITCH_MODE

        self._orig_dispatch = {id(p): p._dispatch_command for p in self.policies}

        def patched_dispatch(cmd):
            if cmd == StateCommand.SWITCH_MODE:
                self.switch_to_next()
            else:
                self._orig_dispatch[id(self.active)](cmd)

        for policy in self.policies:
            policy._dispatch_command = patched_dispatch

    def switch_to_next(self):
        """Advance to the next policy in the ring (the ``switch_mode`` toggle).

        Routes through ``select`` so the running-policy guard applies here too:
        the toggle is refused while the active policy is running (stop first).
        """
        self.select((self.active_index + 1) % len(self.policies))

    def select(self, target, force: bool = False) -> bool:
        """Switch to a policy by ``name`` (str) or ``index`` (int).

        Returns True if a switch occurred, False otherwise (unknown/out-of-range
        target, already active, or refused because the active policy is running).
        Never raises on a bad target — a stray select message from a client must
        not crash the running policy.

        Guard: a switch is REFUSED while the active policy is running
        (``use_policy_action``), unless ``force=True``. Yanking the active policy
        out from under a moving robot is unsafe; the operator must stop first, then
        switch, then explicitly start the target (D5). This is a coarse safety
        interlock, not a full run-status FSM — see the class note.
        """
        if isinstance(target, str):
            index = self._name_to_index.get(target)
            if index is None:
                logger.warning(f"select: unknown policy name {target!r}; known={self.names}")
                return False
        else:
            index = int(target)
            if not (0 <= index < len(self.policies)):
                logger.warning(f"select: index {index} out of range for {len(self.policies)} policies")
                return False
        if index == self.active_index:
            return False
        if not force and getattr(self.active, "use_policy_action", False):
            logger.warning(
                f"select: refused switch to {self.names[index]!r} — active policy "
                f"{self.active_label!r} is running; stop it first (or select(force=True))."
            )
            return False
        self._activate(index)
        return True

    def _activate(self, index: int):
        """Make ``policies[index]`` the active policy, STOPPED and awaiting an
        explicit start.

        A switch does NOT auto-start the target: over ROS2 the caller can't see the
        robot's run status, so force-starting would be a surprise (potential sudden
        motion). We stop the outgoing policy, hand the shared hardware to the target
        in a stopped state (gains resolved, phase re-init), and leave it to the
        operator to issue an explicit ``start`` — mirroring the boot-in-stiff-hold
        contract. Run status therefore does not carry across a switch.
        """
        self.active._handle_stop_policy()

        target = self.policies[index]

        # Re-resolve control gains for the target on the shared interface.
        target._resolve_control_gains()

        # Carry over joystick key_states so edge detection doesn't see a false
        # rising edge on the button that is still physically held during the swap.
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        active_dev = self.active._velocity_input
        target_dev = target._velocity_input
        if isinstance(active_dev, InterfaceInput) and isinstance(target_dev, InterfaceInput):
            target_dev.key_states = active_dev.key_states.copy()
            target_dev.last_key_states = active_dev.key_states.copy()

        self.active_index = index
        self.active = target
        self.active_label = self.names[index]

        # Re-init phase, then leave the target STOPPED (no auto-start; explicit
        # ``start`` required). Stop is idempotent and puts it in a defined state.
        self.active._init_phase_components()
        self.active._handle_stop_policy()

        logger.info(
            colored(
                f"Activated policy [{index + 1}/{len(self.policies)}] "
                f"'{self.active_label}' ({type(self.active).__name__}) — STOPPED, awaiting 'start'.",
                "magenta",
                attrs=["bold"],
            )
        )
        if self._on_switch is not None:
            self._on_switch(self.active_index, self.active_label)

    def run(self):
        """Main run loop — delegates to the active policy each cycle."""
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


class DualModePolicy(MultiModePolicy):
    """Back-compat 2-policy shim: builds primary + secondary, delegates to MultiModePolicy.

    Retained for the original standalone dual-mode entry (primary owns hardware,
    secondary shares it). New code should build N policies and use MultiModePolicy
    directly (the service node does). ``self.primary`` / ``self.secondary`` are
    exposed for callers that referenced them.
    """

    def __init__(self, primary_config: InferenceConfig, secondary_config: InferenceConfig):
        primary_cls = _select_policy_class(primary_config)
        secondary_cls = _select_policy_class(secondary_config)
        logger.info(
            colored(f"Dual-mode: primary={primary_cls.__name__}, secondary={secondary_cls.__name__}", "magenta")
        )

        # Fully init primary (owns hardware).
        primary = primary_cls(config=primary_config)
        # Init secondary with shared hardware.
        logger.info(colored("Initializing secondary policy (shared hardware)...", "magenta"))
        secondary = object.__new__(secondary_cls)
        secondary._shared_hardware_source = primary
        secondary.__init__(config=secondary_config)

        self.primary = primary
        self.secondary = secondary
        super().__init__([primary, secondary], names=["primary", "secondary"], active_index=0)
