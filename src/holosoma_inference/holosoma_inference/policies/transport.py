"""Python SDK/operator I/O. Policy computation lives in the native engine."""

from __future__ import annotations

import itertools
import sys

import numpy as np
from loguru import logger

from holosoma_inference.inputs import create_input
from holosoma_inference.sdk import create_interface
from holosoma_inference.utils.latency import LatencyTracker
from holosoma_inference.utils.rate import RateLimiter


class RunnerTransport:
    def _init_robot_config(self, config):
        self.robot_config = config
        self.num_dofs = config.num_joints
        self.default_dof_angles = np.asarray(config.default_dof_angles)
        self.joint_offsets = (
            np.deg2rad(config.joint_offsets_deg) if config.joint_offsets_deg is not None else np.zeros(self.num_dofs)
        )
        self.dof_names = config.dof_names

    def _init_sdk_components(self):
        self.sdk_type = self.robot_config.sdk_type
        if self.sdk_type == "booster":
            import netifaces
            from booster_robotics_sdk import ChannelFactory

            ip = netifaces.ifaddresses(self.config.task.interface)[netifaces.AF_INET][0]["addr"]
            ChannelFactory.Instance().Init(self.config.task.domain_id, ip)

    def _init_communication_components(self):
        task = self.config.task
        need_joystick = "interface" in {task.velocity_input, task.state_input}
        self.interface = create_interface(self.robot_config, task.domain_id, task.interface, need_joystick)

    def _init_rate_handler(self):
        self.rl_rate = self.config.task.rl_rate
        self.logger = logger
        self.rate = RateLimiter(self.rl_rate)

    def _init_latency_tracking(self):
        self.latency_tracker = LatencyTracker(window_size=int(self.rl_rate))

    def _init_input_device(self):
        task = self.config.task
        sources = {task.velocity_input, task.state_input}
        self.use_joystick = bool({"interface", "joystick"} & sources)
        if self.use_joystick and sys.platform == "darwin":
            self.logger.warning("Joystick is not supported on Mac; falling back to keyboard for the joystick channel")
            self.use_joystick = False
        self._create_input_providers()

    def _setup_keyboard_listener(self):
        task = self.config.task
        sources = {task.velocity_input, task.state_input}
        if not self.use_joystick and {"interface", "joystick"} & sources:
            sources.add("keyboard")  # create_input falls back to the keyboard
        if "keyboard" in sources:
            from holosoma_inference.inputs.impl.keyboard import get_keyboard_listener

            self.use_keyboard = get_keyboard_listener().start()
            if not self.use_keyboard:
                self.use_policy_action = True

    def _create_input_providers(self):
        self._setup_keyboard_listener()
        task = self.config.task
        self._velocity_input = create_input(self, task.velocity_input, "velocity")
        self._command_provider = (
            self._velocity_input
            if task.velocity_input == task.state_input
            else create_input(self, task.state_input, "command")
        )
        # An injected provider belongs to whoever attached it; close() stops only ours.
        self._owned_inputs = []
        if task.velocity_input != "injected":
            self._owned_inputs.append(self._velocity_input)
        if task.state_input != "injected" and self._command_provider is not self._velocity_input:
            self._owned_inputs.append(self._command_provider)
        self._velocity_input.start()
        if self._command_provider is not self._velocity_input:
            self._command_provider.start()

    def _print_control_status(self):
        self.logger.info("Active model: {} | engaged: {}", self.active_model_path, self.use_policy_action)

    def _on_command_sent(self, cmd_q, robot_state_data):
        """Observe final commands without replacing policy computation."""

    def run(self):
        try:
            for tick in itertools.count():
                self.latency_tracker.start_cycle()
                velocity = self._velocity_input.poll_velocity()
                if velocity is not None:
                    self._apply_velocity(velocity)
                commands = self._command_provider.poll_commands()
                for command in commands:
                    self._dispatch_command(command)
                if commands:
                    self._print_control_status()
                if self.use_phase:
                    self.update_phase_time()
                sent = self.policy_action()
                self.latency_tracker.end_cycle()
                # An idle loop is not inference: report FPS only for ticks that sent a command.
                if tick % 50 == 0 and self.use_policy_action and sent is not False:
                    self.logger.info("RL FPS: {:.2f}", self.latency_tracker.get_fps())
                self.rate.sleep()
        except KeyboardInterrupt:
            pass
