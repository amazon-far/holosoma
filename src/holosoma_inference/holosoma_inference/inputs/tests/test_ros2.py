"""Tests for ROS2 input providers (impl API)."""

from types import SimpleNamespace

import numpy as np


class TestRos2VelCmdProvider:
    def _make(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        return Ros2Input("cmd_vel", "holosoma/state_input")

    def test_callback_stores_velocity(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.5, y=-0.3),
                angular=SimpleNamespace(z=0.8),
            )
        )
        prov._vel_callback(msg)
        vc = prov.poll_velocity()

        np.testing.assert_almost_equal(vc.lin_vel[0], 0.5)
        np.testing.assert_almost_equal(vc.lin_vel[1], -0.3)
        np.testing.assert_almost_equal(vc.ang_vel, 0.8)

    def test_callback_clamps_to_range(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=5.0, y=-5.0),
                angular=SimpleNamespace(z=99.0),
            )
        )
        prov._vel_callback(msg)
        vc = prov.poll_velocity()

        assert vc.lin_vel[0] == 1.0
        assert vc.lin_vel[1] == -1.0
        assert vc.ang_vel == 1.0

    def test_callback_clamps_negative_angular(self):
        prov = self._make()
        msg = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0),
                angular=SimpleNamespace(z=-99.0),
            )
        )
        prov._vel_callback(msg)
        vc = prov.poll_velocity()

        assert vc.ang_vel == -1.0

    def test_zero_resets_velocity(self):
        prov = self._make()
        prov._lin_vel[0, 0] = 0.5
        prov._ang_vel[0, 0] = 0.3
        prov.zero()
        vc = prov.poll_velocity()

        assert vc.lin_vel == (0.0, 0.0)
        assert vc.ang_vel == 0.0

    def test_poll_velocity_returns_velcmd(self):
        from holosoma_inference.inputs.api.commands import VelCmd

        prov = self._make()
        vc = prov.poll_velocity()
        assert isinstance(vc, VelCmd)


class TestRos2StateCommandProvider:
    def _make(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        return Ros2Input("cmd_vel", "holosoma/state_input")

    def test_known_commands_queued(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        prov = self._make()
        for cmd_str in ("start", "stop", "init", "walk", "stand"):
            prov._cmd_callback(SimpleNamespace(data=cmd_str))
        commands = prov.poll_commands()
        assert StateCommand.START in commands
        assert StateCommand.STOP in commands
        assert StateCommand.INIT in commands
        assert StateCommand.WALK in commands
        assert StateCommand.STAND in commands

    def test_case_insensitive(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="  WALK  "))
        assert prov.poll_commands() == [StateCommand.WALK]

    def test_unknown_command_warns(self):
        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="bogus"))
        # Unknown commands are logged via loguru (warning emitted) and not queued
        assert prov.poll_commands() == []

    def test_empty_string_warns(self):
        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="   "))
        # Empty/whitespace-only strings are logged via loguru and not queued
        assert prov.poll_commands() == []

    def test_poll_commands_drains_queue(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="start"))
        prov._cmd_callback(SimpleNamespace(data="stop"))
        assert prov.poll_commands() == [StateCommand.START, StateCommand.STOP]
        assert prov.poll_commands() == []
