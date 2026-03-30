import numpy as np
from termcolor import colored

from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.inputs.impl.keyboard import KEYBOARD_VELOCITY_LOCOMOTION

from .base import BasePolicy


class LocomotionPolicy(BasePolicy):
    _keyboard_velocity_mapping = KEYBOARD_VELOCITY_LOCOMOTION

    def __init__(self, config):
        super().__init__(config)
        self.is_standing = False

    def _apply_velocity(self, vc: VelCmd) -> None:
        """Gate velocity by stand_command — zero when standing."""
        s = self.stand_command[0, 0]
        self.lin_vel_command[0] = (vc.lin_vel[0] * s, vc.lin_vel[1] * s)
        self.ang_vel_command[0, 0] = vc.ang_vel * s

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_buffer_dict["actions"] = self.last_policy_action
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_stand"] = self.stand_command

        # Add phase observations only if they are configured
        if "sin_phase" in self.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["sin_phase"] = self._get_obs_sin_phase()
        if "cos_phase" in self.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["cos_phase"] = self._get_obs_cos_phase()

        return current_obs_buffer_dict

    def _get_obs_sin_phase(self):
        """Calculate sin phase for gait."""
        return np.array([np.sin(self.phase[0, :])])

    def _get_obs_cos_phase(self):
        """Calculate cos phase for gait."""
        return np.array([np.cos(self.phase[0, :])])

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(self.lin_vel_command[0]) < 0.01 and np.linalg.norm(self.ang_vel_command[0]) < 0.01:
            # Robot should stand still - set both feet to same phase
            self.phase[0, :] = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            # When the robot starts to move, reset the phase to initial state
            self.phase = np.array([[0.0, np.pi]])
            self.is_standing = False

    def _dispatch_command(self, cmd):
        if cmd == StateCommand.STAND_TOGGLE:
            self._handle_stand_command()
        elif cmd == StateCommand.ZERO_VELOCITY:
            self._handle_zero_velocity()
        elif cmd == StateCommand.WALK:
            self.stand_command[0, 0] = 1
            self.base_height_command[0, 0] = self.desired_base_height
            self.logger.info("ROS2 command: walk")
        elif cmd == StateCommand.STAND:
            self.stand_command[0, 0] = 0
            self.logger.info("ROS2 command: stand")
        else:
            super()._dispatch_command(cmd)

    def _handle_stand_command(self):
        """Handle stand command toggle."""
        self.stand_command[0, 0] = 1 - self.stand_command[0, 0]
        if self.stand_command[0, 0] == 0:
            self._velocity_input.zero()
            self.ang_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
            self.logger.info(colored("Stance command", "blue"))
        else:
            self.base_height_command[0, 0] = self.desired_base_height
            self.logger.info(colored("Walk command", "blue"))

    def _handle_zero_velocity(self):
        """Handle zero velocity command."""
        self._velocity_input.zero()
        self.ang_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.logger.info(colored("Velocities set to zero", "blue"))

    def _print_control_status(self):
        """Print current control status."""
        super()._print_control_status()

        # Extract values for better formatting
        lin_vel_x = self.lin_vel_command[0, 0]
        lin_vel_y = self.lin_vel_command[0, 1]
        ang_vel_z = self.ang_vel_command[0, 0]
        is_walking = self.stand_command[0, 0] == 1

        # Print with clear labels and units
        mode = "Walking" if is_walking else "Standing"
        status = "✓ applied" if is_walking else "✗ not applied"
        print(f"Linear velocity: x={lin_vel_x:+.2f} m/s, y={lin_vel_y:+.2f} m/s")
        print(f"Angular velocity: {ang_vel_z:+.2f} rad/s")
        print(f"Mode: {mode} ({status})")
        print("💡 Terminal keys: W/A/S/D (lin) | Q/E (ang) | = (toggle mode)")
        print("🎬 MuJoCo keys (in simulator only): 7/8 (band) | 9 (toggle) | BACKSPACE (reset)")
