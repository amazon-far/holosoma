"""Unit tests for ``NativeContextualRunner``'s per-tick depth input and engage gate.

Drives a real native engine built from the redistributable contextual graph, with the injected
interface/input doubles the other runner tests use, so the assertions are on engine state rather
than on recorded calls.
"""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import holosoma_inference as inference
from holosoma_inference.config.config_types.contextual import ContextualConfig
from holosoma_inference.config.config_types.observation import ObservationConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.inputs.api.commands import StateCommand
from holosoma_inference.policies.native_runner import NativeContextualRunner

# The generated graph takes depth_image [-1, 2, 2, 3]: history 2, then 2x3 frames.
SHAPE = (2, 3)
# Its climb head is ReduceMean(depth_image), and the kernel enters climb at >= 0.1.
CLIMB = np.ones(SHAPE, dtype=np.float32)
FLAT = np.zeros(SHAPE, dtype=np.float32)


class _Interface:
    kp_level = kd_level = 1.0

    def get_low_state(self):
        state = np.zeros((1, 19))
        state[0, 3] = 1
        return state

    def send_low_command(self, *args, **kwargs):
        self.sent = kwargs


class _Input:
    zeroed = 0

    def poll_velocity(self):
        return None

    def poll_commands(self):
        return []

    def zero(self):
        self.zeroed += 1


def _config(tmp_path):
    generator = Path(__file__).parents[3] / "holosoma_cpp/tests/gen_contextual_models.py"
    path = tmp_path / "contextual.onnx"
    runpy.run_path(str(generator))["write_model"](path)
    base = inference.get_config("g1-29dof-loco")
    robot = RobotConfig(
        robot_type="test",
        robot="test",
        num_joints=3,
        num_motors=3,
        dof_names=("hip", "knee", "ankle"),
        dof_names_upper_body=(),
        dof_names_lower_body=("hip", "knee", "ankle"),
        default_dof_angles=(0.1, 0.2, 0.3),
        default_motor_angles=(0.1, 0.2, 0.3),
        motor2joint=(0, 1, 2),
        joint2motor=(0, 1, 2),
    )
    return replace(
        base,
        robot=robot,
        observation=ObservationConfig({"actor_obs": []}, {}, {}, {}),
        task=replace(base.task, model_path=str(path), use_phase=False),
        contextual=ContextualConfig(depth_height=SHAPE[0], depth_width=SHAPE[1], gravity_frame_joints=()),
        secondary=None,
    )


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _runner(tmp_path):
    inputs = _Input()
    runner = NativeContextualRunner(
        _config(tmp_path), interface=_Interface(), velocity_input=inputs, command_provider=inputs
    )
    runner._monotonic = _Clock()
    runner._monotonic_base = runner._monotonic()
    return runner


def _tick(runner):
    return runner._step_native(runner.interface.get_low_state())


def test_depth_reaches_the_graph(tmp_path):
    """The graph's climb head is a mean over depth, so the frame must move the command column.

    Driven forward first: from stand, ``StandBlocked`` stops the climb head having any effect, so
    a stand-only comparison would pass with depth going nowhere.
    """
    columns = {}
    for name, frame in (("flat", FLAT), ("climb", CLIMB)):
        runner = _runner(tmp_path)
        runner._handle_start_policy()
        runner.native.velocity(0.5, 0.0, 0.0)
        for _ in range(2):
            runner.set_depth(frame)
            runner._step_native(runner.interface.get_low_state())
        columns[name] = int(np.argmax(np.asarray(runner.native.last_observation)[-15:]))
    assert columns["flat"] != columns["climb"], columns
    assert columns["climb"] == 14, columns


def test_a_missing_frame_still_supplies_an_input(tmp_path):
    """The kernel raises on a tick with no ContextualInput, so one must go in regardless."""
    runner = _runner(tmp_path)
    command = runner._step_native(runner.interface.get_low_state())
    assert np.isfinite(np.asarray(command.q, float)).all()


def test_engage_is_deferred_until_the_first_frame(tmp_path):
    """Start arrives before any frame, so it must be held and retried, not dropped."""
    runner = _runner(tmp_path)
    runner._handle_start_policy()
    assert runner._start_pending
    assert not runner.use_policy_action
    _tick(runner)
    assert not runner.use_policy_action
    runner.set_depth(FLAT)
    _tick(runner)
    assert runner.use_policy_action
    assert not runner._start_pending


def test_set_depth_only_stores_and_never_engages(tmp_path):
    """set_depth runs on the depth executor thread; engaging there races the policy thread.

    The pending start must survive the frame and be taken by the next policy tick instead.
    """
    runner = _runner(tmp_path)
    runner._handle_start_policy()
    engaged = []
    runner._handle_start_policy = lambda: engaged.append("start")
    runner.native.command = lambda command: engaged.append(command)
    runner.set_depth(FLAT)
    assert engaged == []
    assert runner._start_pending
    assert not runner.use_policy_action


def test_a_stale_frame_stops_an_engaged_policy(tmp_path):
    """If the camera dies mid-walk the policy must not keep walking on the last image."""
    runner = _runner(tmp_path)
    limit = runner.config.contextual.max_depth_age_s
    runner.set_depth(FLAT)
    runner._handle_start_policy()
    assert runner.use_policy_action
    runner._monotonic.now += limit * 0.9
    _tick(runner)
    assert runner.use_policy_action, "a frame inside the limit must keep the policy engaged"
    runner._monotonic.now += limit * 0.2
    command = _tick(runner)
    assert not runner.use_policy_action
    # The stopped tick holds the measured pose, the same as an operator stop.
    np.testing.assert_allclose(command.q, runner.interface.get_low_state()[0, 7:10])
    # Frames coming back do not re-engage on their own; the operator starts again.
    runner.set_depth(CLIMB)
    _tick(runner)
    assert not runner.use_policy_action
    runner._handle_start_policy()
    assert runner.use_policy_action


def test_start_is_refused_not_deferred_once_frames_have_gone_stale(tmp_path):
    """A deferred start would engage by itself whenever the camera came back."""
    runner = _runner(tmp_path)
    runner.set_depth(FLAT)
    runner._monotonic.now += runner.config.contextual.max_depth_age_s * 2
    runner._handle_start_policy()
    assert not runner.use_policy_action
    assert not runner._start_pending
    runner.set_depth(CLIMB)
    _tick(runner)
    assert not runner.use_policy_action


def test_frame_age_uses_receive_time_not_the_publisher_stamp(tmp_path):
    """Publisher stamps may be sim or another host's clock; only local receipt measures staleness."""
    runner = _runner(tmp_path)
    runner.set_depth(FLAT, stamp_s=1.0)  # Ancient by any wall clock, but just received.
    runner._handle_start_policy()
    assert runner.use_policy_action
    _tick(runner)
    assert runner.use_policy_action


def test_set_depth_honours_a_publisher_stamp(tmp_path):
    runner = _runner(tmp_path)
    assert runner.set_depth(FLAT, stamp_s=12.5)
    assert not runner.set_depth(FLAT, stamp_s=12.5)


def test_set_depth_rejects_a_non_finite_frame(tmp_path):
    runner = _runner(tmp_path)
    bad = FLAT.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="must clip and fill"):
        runner.set_depth(bad)


def test_zero_velocity_zeroes_the_operator_input(tmp_path):
    """The contextual engine ignores ZERO_VELOCITY, so the runner must zero the operator input itself."""
    runner = _runner(tmp_path)
    runner._dispatch_command(StateCommand.ZERO_VELOCITY)
    assert runner._velocity_input.zeroed == 1
