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


def _config(tmp_path, **model_kwargs):
    generator = Path(__file__).parents[3] / "holosoma_cpp/tests/gen_contextual_models.py"
    path = tmp_path / "contextual.onnx"
    runpy.run_path(str(generator))["write_model"](path, **model_kwargs)
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
        contextual=ContextualConfig(
            depth_height=SHAPE[0],
            depth_width=SHAPE[1],
            require_climb_head=model_kwargs.get("head", True),
        ),
        secondary=None,
    )


def _runner(tmp_path, **model_kwargs):
    inputs = _Input()
    return NativeContextualRunner(
        _config(tmp_path, **model_kwargs),
        interface=_Interface(),
        velocity_input=inputs,
        command_provider=inputs,
    )


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
    runner.set_depth(FLAT)
    assert runner.use_policy_action
    assert not runner._start_pending


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


def test_a_raw_depth_graph_takes_the_camera_frame_untouched(tmp_path):
    """Preprocessing inside the graph makes NaN expected input, so no guard may reject it."""
    runner = _runner(tmp_path, raw=True, head=False)
    assert runner.expects_raw_depth
    frame = FLAT.copy()
    frame[0, 0] = np.nan
    assert runner.set_depth(frame)
    command = runner._step_native(runner.interface.get_low_state())
    assert np.isfinite(np.asarray(command.q, float)).all()
