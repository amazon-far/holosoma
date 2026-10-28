"""Python SDK/input transport around the shared native control engine."""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from holosoma_inference import _native
from holosoma_inference.inputs.api.commands import StateCommand
from holosoma_inference.native import NativePolicy
from holosoma_inference.policies.transport import RunnerTransport


def _is_valid_state(state) -> bool:
    """A finite state whose orientation can be normalised; zeros mean no message yet."""
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    return bool(np.isfinite(values).all() and np.dot(values[3:7], values[3:7]) >= 1e-16)


class NativeRunnerPolicy(RunnerTransport):
    """Retain Python SDK plugins and input providers; execute policy logic in C++.

    Explicit injection is supported without pre-initialization attribute patches:
    interface owns I/O, velocity_input and command_provider own operator inputs.
    """

    is_wbt = False

    # Hooks of the retired Python policy classes. The native runner never calls
    # them, so a subclass overriding one would silently run the native policy.
    _RETIRED_HOOKS = (
        "setup_policy",
        "rl_inference",
        "get_current_obs_buffer_dict",
        "_get_manual_command",
        "get_init_target",
        "_target_source",
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        overridden = [name for name in cls._RETIRED_HOOKS if name in vars(cls)]
        if overridden:
            raise TypeError(
                f"{cls.__qualname__} overrides {', '.join(overridden)}, which the native runner never calls: "
                "observation building and inference run in the native engine. Adapt final joint commands "
                "with transform_command, or drive a NativePolicy directly."
            )

    def __init__(
        self, config, *, interface=None, velocity_input=None, command_provider=None, shared_hardware=None, clock=None
    ):
        self.config = config
        self._init_robot_config(config.robot)
        self.native = NativePolicy(config, autostart=False)
        self.model_paths = (
            list(config.task.model_path)
            if isinstance(config.task.model_path, (list, tuple))
            else [config.task.model_path]
        )
        self._init_rate_handler()
        self._init_latency_tracking()
        self.use_phase = config.task.use_phase
        self.use_keyboard = False
        self.use_joystick = bool({"interface", "joystick"} & {config.task.velocity_input, config.task.state_input})
        self.cmd_q = np.zeros(self.num_dofs)
        self._state_ready = False
        self._state_wait_started = None
        self._start_pending = False
        self._commands_sent = 0
        self._clock = clock
        self._owns_clock = False
        self._owned_inputs = []
        if shared_hardware is not None:
            interface = shared_hardware.interface
            velocity_input = shared_hardware._velocity_input
            command_provider = shared_hardware._command_provider
        if interface is None:
            self._init_sdk_components()
            self._init_communication_components()
        else:
            self.interface = interface
            self.sdk_type = config.robot.sdk_type
        self._resolve_control_gains()
        if velocity_input is not None and command_provider is not None:
            self._velocity_input = velocity_input
            self._command_provider = command_provider
        elif velocity_input is not None or command_provider is not None:
            raise ValueError("Supply both velocity_input and command_provider")
        else:
            self._init_input_device()
        if self.is_wbt and config.task.use_sim_time and self._clock is None:
            if shared_hardware is not None and getattr(shared_hardware, "_clock", None) is not None:
                self._clock = shared_hardware._clock
            else:
                from holosoma_inference.utils.clock import ClockSub

                self._clock = ClockSub()
                self._clock.start()
                self._owns_clock = True
        if self.is_wbt and shared_hardware is None and not config.task.skip_stiff_prompt and sys.stdin.isatty():
            try:
                input("Ready to enter stiff hold mode. Press Enter to continue...")
            except EOFError:
                # isatty() can be true while stdin is already closed (some launchers).
                self.logger.warning("No stdin to confirm stiff hold; continuing without the prompt")

    @property
    def use_policy_action(self):
        return self.native._engine.active

    @use_policy_action.setter
    def use_policy_action(self, value):
        if not getattr(self, "_state_ready", False):
            # Headless auto-start: engage only once the first valid state has arrived.
            self._start_pending = bool(value)
            return
        self.native.command("START" if value else "STOP")

    @property
    def get_ready_state(self):
        return self.native._engine.ready

    @property
    def active_policy_index(self):
        return self.native._engine.active_index

    @property
    def active_model_path(self):
        return self.model_paths[self.active_policy_index]

    def _resolve_control_gains(self):
        # Both native commands and Python SDK overrides are joint-ordered.
        # SDK transport owns motor permutation and its operator gain scale.
        self._kp = np.asarray(self.native._engine.joint_kp)
        self._kd = np.asarray(self.native._engine.joint_kd)

    def _tick_ms(self):
        return self._clock.get_clock() if self._clock is not None else 0

    def _apply_velocity(self, command):
        self.native.velocity(float(command.lin_vel[0]), float(command.lin_vel[1]), float(command.ang_vel))

    def update_phase_time(self):
        if self._state_ready:
            self.native._engine.update_phase()

    def _init_phase_components(self):
        self.native.reset()

    def _dispatch_command(self, command):
        if command == StateCommand.KILL:
            raise SystemExit(0)
        if command in {
            StateCommand.KP_UP,
            StateCommand.KP_DOWN,
            StateCommand.KP_UP_FINE,
            StateCommand.KP_DOWN_FINE,
            StateCommand.KP_RESET,
        }:
            increments = {
                StateCommand.KP_UP: 0.1,
                StateCommand.KP_DOWN: -0.1,
                StateCommand.KP_UP_FINE: 0.01,
                StateCommand.KP_DOWN_FINE: -0.01,
            }
            self.interface.kp_level = (
                1.0 if command == StateCommand.KP_RESET else max(0.0, self.interface.kp_level + increments[command])
            )
            return
        previous_index = self.active_policy_index
        if command == StateCommand.START:
            self._handle_start_policy()
        elif command == StateCommand.STOP:
            self._handle_stop_policy()
        elif command == StateCommand.INIT:
            self._handle_init_state()
        else:
            if self.is_wbt and self._state_ready:
                self.native.set_state(self.interface.get_low_state(), tick_ms=self._tick_ms())
            zero_requests = self.native._engine.velocity_zero_requests
            self.native.command(command)
            # The engine's own velocity input is a stub; the operator input is ours. Only the
            # locomotion engine handles ZERO_VELOCITY, so zero it here for every policy.
            if command == StateCommand.ZERO_VELOCITY or self.native._engine.velocity_zero_requests != zero_requests:
                self._velocity_input.zero()
        self._resolve_control_gains()
        if self.active_policy_index != previous_index:
            self._on_policy_switched(self.active_model_path)

    def _on_policy_switched(self, model_path):
        """Observe a native model switch without replacing inference."""

    def _await_valid_state(self, state) -> bool:
        """Return True once a valid robot state has arrived; fail after the startup timeout.

        Only the first valid state is waited for. Afterwards every state goes to the
        native engine, which rejects an invalid one.
        """
        if self._state_ready:
            return True
        now = time.monotonic()
        if _is_valid_state(state):
            waited = 0.0 if self._state_wait_started is None else now - self._state_wait_started
            values = np.asarray(state, dtype=np.float64).reshape(-1)
            self.logger.info(
                "Received first valid robot state after {:.2f}s: quat_wxyz={} with {} finite joint positions",
                waited,
                np.round(values[3:7], 4).tolist(),
                self.num_dofs,
            )
            self._state_ready = True
            if self._start_pending:
                self._start_pending = False
                self._handle_start_policy()
            return True
        timeout = self.config.task.startup_state_timeout
        if self._state_wait_started is None:
            self._state_wait_started = now
            self.logger.info("Waiting up to {:.1f}s for a valid robot state...", timeout)
        elif now - self._state_wait_started > timeout:
            raise TimeoutError(
                f"No valid robot state within {timeout:g}s (--task.startup-state-timeout): "
                "check that the robot or simulator is publishing low state on the configured interface"
            )
        return False

    def _handle_start_policy(self):
        # A dual-mode secondary never steps while inactive, so it has not seen a
        # state yet when it is switched to; read the shared interface now.
        if not self._state_ready and not _is_valid_state(self.interface.get_low_state()):
            self.logger.warning("Ignoring start: no valid robot state received yet")
            return
        self._state_ready = True
        self.native.set_state(self.interface.get_low_state(), tick_ms=self._tick_ms())
        self.native.command("START")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_stop_policy(self):
        self._start_pending = False
        self.native.command("STOP")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0 if self.is_wbt else 1

    def _handle_init_state(self):
        self.native.command("INIT")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _step_native(self, state):
        """One native step; subclasses whose engine takes extra per-tick input override this."""
        return self.native.step_state(state, tick_ms=self._tick_ms(), advance_phase=False)

    def policy_action(self):
        state = self.interface.get_low_state()
        if not self._await_valid_state(state):
            return False
        command = self._step_native(state)
        transformed = self.transform_command(command, state) or {}
        q = np.asarray(transformed.get("q", command.q), dtype=float).reshape(-1)
        kp = np.asarray(transformed.get("kp", command.kp), dtype=float).reshape(-1)
        kd = np.asarray(transformed.get("kd", command.kd), dtype=float).reshape(-1)
        if any(value.shape != (self.num_dofs,) or not np.isfinite(value).all() for value in (q, kp, kd)):
            raise ValueError("Command transform must return finite joint-ordered q/kp/kd arrays")
        if (kp < 0).any() or (kd < 0).any():
            raise ValueError("Command transform gains must be nonnegative")
        self.cmd_q[:] = q
        self.interface.send_low_command(
            self.cmd_q,
            np.asarray(command.dq),
            np.asarray(command.tau),
            state[0, 7 : 7 + self.num_dofs],
            kp_override=kp,
            kd_override=kd,
        )
        self._on_command_sent(self.cmd_q, state)
        self._commands_sent += 1
        if self._commands_sent == 1:
            self.logger.info("Sent first native command: finite q/kp/kd for {} joints", self.num_dofs)
        return True

    def transform_command(self, command, robot_state_data):
        """Optional robot-specific output adaptation; inference remains native.

        Return overrides for final, calibrated joint-order q/kp/kd, or None.
        The SDK still owns motor permutation and transmission.
        """
        return

    def rl_inference(self, robot_state_data):
        """Compatibility warm-up hook: compute offsets, publish nothing."""
        return self.native.infer_offsets(robot_state_data, tick_ms=self._tick_ms())

    def close(self):
        if self._owns_clock and self._clock is not None:
            self._clock.close()
            self._clock = None
        owned, self._owned_inputs = self._owned_inputs, []
        for provider in owned:
            stop = getattr(provider, "stop", None)
            if callable(stop):
                stop()

    def run(self):
        try:
            super().run()
        finally:
            self.close()


class NativeLocomotionRunner(NativeRunnerPolicy):
    """Native locomotion with Python-owned transport."""


class NativeWbtRunner(NativeRunnerPolicy):
    """Native embedded-motion tracking with Python-owned transport."""

    is_wbt = True


class NativeContextualRunner(NativeRunnerPolicy):
    """Depth-conditioned locomotion; the native engine needs one ContextualInput per tick.

    Callers push frames with ``set_depth`` from whatever transport they own, and the freshest one
    is what the policy gets. Nothing here delays depth: a real pipeline is already about as stale
    as training modelled it, so adding delay on a robot would double the staleness. Simulated
    depth is instantaneous, so emulating that latency is the sim harness's job and it should delay
    frames before pushing them.

    ``set_depth`` may run on any thread and only stores the frame. Engagement, and the stop when
    the newest frame is older than ``contextual.max_depth_age_s``, happen on the policy thread at
    the start of its next tick.
    """

    def __init__(self, config, **kwargs):
        self._depth_lock = threading.Lock()
        self._frame = None
        self._frame_stamp = None
        self._frame_received = None
        self._max_depth_age_s = config.contextual.max_depth_age_s
        self._monotonic = time.monotonic
        self._monotonic_base = self._monotonic()
        super().__init__(config, **kwargs)

    def set_depth(self, frame, stamp_s=None):
        """Store the newest depth frame; returns False for a repeated or out-of-order stamp."""
        frame = np.asarray(frame, dtype=np.float32)
        if not np.isfinite(frame).all():
            raise ValueError("depth frame is not finite; the producer must clip and fill")
        with self._depth_lock:
            if stamp_s is not None and self._frame_stamp is not None and stamp_s <= self._frame_stamp:
                return False
            self._frame, self._frame_stamp, self._frame_received = frame, stamp_s, self._monotonic()
        return True

    def _latest_depth(self):
        """The newest frame and its age in seconds, or (None, None) before the first frame."""
        with self._depth_lock:
            frame, received = self._frame, self._frame_received
        if frame is None:
            return None, None
        return frame, self._monotonic() - received

    def _step_native(self, state):
        frame, age = self._latest_depth()
        fresh = age is not None and age <= self._max_depth_age_s
        if self.use_policy_action and not fresh:
            self.logger.error(
                "Stopping the contextual policy: {} (max age {:.3f}s); check the depth camera, then start again",
                "no depth frame" if age is None else f"newest depth frame is {age:.3f}s old",
                self._max_depth_age_s,
            )
            self._handle_stop_policy()
        elif self._start_pending and fresh:
            self._start_pending = False
            super()._handle_start_policy()
        inputs = _native.ContextualInput()
        if frame is not None:
            inputs.depth = frame.reshape(-1)
        inputs.monotonic_time = self._monotonic() - self._monotonic_base
        return self.native.step_contextual(state, inputs, tick_ms=self._tick_ms())

    def _handle_start_policy(self):
        # Deferred, not refused, before the first frame: start arrives before the camera does, so
        # refusing here would drop the engage with nothing to retry it. A zero frame reads as
        # ground at the clip midpoint rather than as missing data, which is not something to engage
        # on. Once frames have arrived, a stale one means the camera has gone, and a deferred start
        # would engage by itself whenever it came back, so that start is refused instead.
        frame, age = self._latest_depth()
        if frame is None:
            self._start_pending = True
            self.logger.info("Deferring start until the first depth frame arrives")
            return
        if age > self._max_depth_age_s:
            self._start_pending = False
            self.logger.error(
                "Refusing start: newest depth frame is {:.3f}s old (max {:.3f}s)", age, self._max_depth_age_s
            )
            return
        super()._handle_start_policy()
