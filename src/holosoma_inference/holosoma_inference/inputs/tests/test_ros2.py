"""Tests for ROS2 input providers (impl API)."""

from types import SimpleNamespace


class TestRos2VelCmdProvider:
    def _make(self):
        from holosoma_inference.inputs.impl.ros2 import Ros2Input

        return Ros2Input("cmd_vel", "holosoma/state_input")

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

    def test_poll_commands_drains_queue(self):
        from holosoma_inference.inputs.api.commands import StateCommand

        prov = self._make()
        prov._cmd_callback(SimpleNamespace(data="start"))
        prov._cmd_callback(SimpleNamespace(data="stop"))
        assert prov.poll_commands() == [StateCommand.START, StateCommand.STOP]
        assert prov.poll_commands() == []
