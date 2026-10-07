"""Tests for the depth-distillation policy's observation and command contracts.

The student checkpoint's input layout is a wire format: 93 proprioceptive dims in
training's sorted term order, then a 15-dim one-hot direction command, then a
32-dim depth latent. Reordering any of that silently produces a plausible-looking
but wrong action, so these tests pin the layout rather than just checking it runs.

No ONNX files are needed: the sessions are stubbed, so the tests assert the
policy's own assembly logic.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections import deque
from unittest.mock import patch

import numpy as np
import pytest

from holosoma_inference.config.config_values.inference import INFERENCE_REGISTRY
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.inputs.impl.keyboard import KEYBOARD_VELOCITY_LOCOMOTION, KeyboardInput
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.depth_distillation import DepthDistillationPolicy
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.sensors.depth_heartbeat import STATE_FAILED, STATE_OK, STATE_RECONNECTING, DepthLiveness

NUM_DOFS = 29
LATENT_DIM = 32
DEPTH_H, DEPTH_W = 58, 87
# 29 actions + 3 ang_vel + 29 dof_pos + 29 dof_vel + 3 gravity, in training's sorted term order.
PROPRIO_DIM = 93
ONE_HOT_DIM = 15
# Slices of the proprioceptive block, derived from that sorted order.
ANG_VEL_SLICE = slice(NUM_DOFS, NUM_DOFS + 3)
GRAVITY_SLICE = slice(PROPRIO_DIM - 3, PROPRIO_DIM)


class _FakeInterface:
    kp_level = 1.0
    no_action = 0

    def get_low_state(self):
        return None

    def send_low_command(self, *args, **kwargs):
        pass

    def update_config(self, config):
        pass


class _FakeSession:
    """Minimal onnxruntime.InferenceSession stand-in."""

    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self):
        return [dataclasses.replace(spec) for spec in self._inputs]

    def get_outputs(self):
        return [dataclasses.replace(spec) for spec in self._outputs]

    def run(self, output_names, input_feed):
        # Echo the flattened input so tests can inspect what the policy fed in.
        if "depth_image" in input_feed:
            return [np.full((1, LATENT_DIM), 0.5, dtype=np.float32)]
        return [np.zeros((1, NUM_DOFS), dtype=np.float32) for _ in output_names]


@dataclasses.dataclass
class _Spec:
    name: str
    shape: list


def _make_policy(
    *,
    action_scale: str = "1.0",
    model_dof_names: list[str] | None = None,
    **task_overrides,
):
    """Build a DepthDistillationPolicy with stubbed ONNX sessions and hardware."""
    config = INFERENCE_REGISTRY["g1-wbt-distillation"]
    task = dataclasses.replace(
        config.task,
        model_path=["backbone.onnx", "student.onnx"],
        depth_shm=dataclasses.replace(config.task.depth_shm, required=False),
        **task_overrides,
    )
    config = dataclasses.replace(config, task=task)

    backbone = _FakeSession([_Spec("depth_image", [1, DEPTH_H, DEPTH_W])], [_Spec("depth_latent", [1, LATENT_DIM])])
    student = _FakeSession(
        [_Spec("obs", [1, PROPRIO_DIM + ONE_HOT_DIM + LATENT_DIM]), _Spec("time_step", [1, 1])],
        [_Spec("actions", [1, NUM_DOFS])],
    )
    sessions = iter([backbone, student])

    # Metadata mirrors the real export: gains under joint_stiffness/joint_damping,
    # joint_names in the robot's own order (so reordering is a no-op here).
    dof_names = ",".join(model_dof_names or config.robot.dof_names)

    class _FakeOnnxModel:
        metadata_props = [
            _Spec("joint_names", []),
        ]

    def fake_onnx_load(path):
        model = _FakeOnnxModel()
        model.metadata_props = [
            type("P", (), {"key": "joint_names", "value": dof_names})(),
            type("P", (), {"key": "action_scale", "value": action_scale})(),
        ]
        return model

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("holosoma_inference.policies.base.create_interface", return_value=_FakeInterface()))
        stack.enter_context(patch("holosoma_inference.policies.base.create_input"))
        stack.enter_context(patch("onnxruntime.InferenceSession", side_effect=lambda *_a, **_k: next(sessions)))
        stack.enter_context(
            patch("holosoma_inference.policies.depth_distillation.onnx.load", side_effect=fake_onnx_load)
        )
        return DepthDistillationPolicy(config)


def _robot_state(dof_pos=None, dof_vel=None, ang_vel=(0.0, 0.0, 0.0)):
    """Build a synthetic upright robot-state array."""
    state = np.zeros((1, 7 + NUM_DOFS + 6 + NUM_DOFS), dtype=np.float32)
    state[0, 3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quat -> upright
    if dof_pos is not None:
        state[0, 7 : 7 + NUM_DOFS] = dof_pos
    state[0, 7 + NUM_DOFS + 3 : 7 + NUM_DOFS + 6] = ang_vel
    if dof_vel is not None:
        state[0, 7 + NUM_DOFS + 6 : 7 + NUM_DOFS + 6 + NUM_DOFS] = dof_vel
    return state


def test_observation_is_proprio_then_command_then_latent():
    """The student's 140-dim input must be assembled in exactly this order.

    Training concatenates the group over ``sorted(term_names)``, so the proprioceptive block is
    [actions(29) | base_ang_vel(3) | dof_pos(29) | dof_vel(29) | projected_gravity(3)]."""
    policy = _make_policy()
    obs = policy.prepare_obs_for_rl(_robot_state(ang_vel=(0.11, 0.22, 0.33)))["obs"]

    assert obs.shape == (1, PROPRIO_DIM + ONE_HOT_DIM + LATENT_DIM)
    np.testing.assert_allclose(obs[0, ANG_VEL_SLICE], [0.11, 0.22, 0.33], atol=1e-6)
    # Upright with zero waist angles -> gravity points straight down.
    np.testing.assert_allclose(obs[0, GRAVITY_SLICE], [0.0, 0.0, -1.0], atol=1e-6)
    # One-hot block sums to 1 (stand is active at startup).
    assert obs[0, PROPRIO_DIM : PROPRIO_DIM + ONE_HOT_DIM].sum() == pytest.approx(1.0)
    # Latent block is the backbone's output, last.
    np.testing.assert_allclose(obs[0, PROPRIO_DIM + ONE_HOT_DIM :], 0.5, atol=1e-6)


def test_actor_obs_terms_are_in_training_sorted_order():
    """The wire order is training's ``sorted()`` order, used verbatim.

    Not the ONNX ``observation_names`` metadata order (which the exporter writes as the training
    config's *declaration* order, gravity first) — trusting that would transpose gravity and
    actions and silently degrade tracking."""
    policy = _make_policy()
    assert policy.obs_terms_sorted["actor_obs"] == [
        "actions",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "projected_gravity",
    ]
    # The list is literally sorted, which is the property that makes it match training.
    assert policy.obs_terms_sorted["actor_obs"] == sorted(policy.obs_terms_sorted["actor_obs"])


def test_two_model_paths_required():
    """A single path cannot satisfy the backbone+student pair."""
    config = INFERENCE_REGISTRY["g1-wbt-distillation"]
    config = dataclasses.replace(config, task=dataclasses.replace(config.task, model_path="only_one.onnx"))
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("holosoma_inference.policies.base.create_interface", return_value=_FakeInterface()))
        stack.enter_context(patch("holosoma_inference.policies.base.create_input"))
        stack.enter_context(pytest.raises(ValueError, match="exactly 2 model paths"))
        DepthDistillationPolicy(config)


def test_model_paths_collapse_to_single_switchable_slot():
    """The pair is one composite policy, so digit-key switching sees one slot."""
    policy = _make_policy()
    assert len(policy.model_paths) == 1
    assert len(policy._policy_states) == 1


def test_vector_action_scale_is_reordered_to_robot_joint_order():
    config = INFERENCE_REGISTRY["g1-wbt-distillation"]
    model_dof_names = list(reversed(config.robot.dof_names))
    model_scale = np.arange(1, NUM_DOFS + 1, dtype=np.float32)

    policy = _make_policy(
        action_scale=str(model_scale.tolist()),
        model_dof_names=model_dof_names,
    )

    np.testing.assert_array_equal(policy.policy_action_scale, model_scale[::-1])


def test_vector_action_scale_requires_one_value_per_joint():
    with pytest.raises(ValueError, match=f"must be scalar or have {NUM_DOFS} entries"):
        _make_policy(action_scale="[0.1, 0.2]")


def test_scalar_action_scale_remains_supported():
    policy = _make_policy(action_scale="0.25")

    assert policy.policy_action_scale == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("w", "forward"),
        ("s", "back"),
        ("a", "left_45"),
        ("d", "right_45"),
        ("q", "left_90"),
        ("e", "right_90"),
    ],
)
def test_single_keypress_selects_direction(key, expected):
    """One press must select the heading outright.

    Regression: these keys used to feed the shared velocity accumulator (0.1 per
    press), so reversing from forward to back took six presses to cross zero and
    the policy appeared to ignore keys.
    """
    policy = _make_policy()
    queue = deque()
    keyboard = KeyboardInput(
        queue,
        velocity_keys=KEYBOARD_VELOCITY_LOCOMOTION,
        direction_keys=policy.KEYBOARD_DIRECTION_KEYS,
    )
    # Direction keys win, so the accumulator is inert for this policy.
    assert keyboard._velocity_keys is None

    queue.append(key)
    for cmd in keyboard.poll_commands():
        policy._dispatch_command(cmd)
    assert policy.active_velocity_command_idx == policy.CMD_CODES[0][expected]


def test_direction_reversal_is_immediate():
    """forward -> back must take one press, not several."""
    policy = _make_policy()
    queue = deque()
    keyboard = KeyboardInput(queue, direction_keys=policy.KEYBOARD_DIRECTION_KEYS)

    for key, expected in [("w", "forward"), ("s", "back"), ("w", "forward"), ("e", "right_90")]:
        queue.append(key)
        for cmd in keyboard.poll_commands():
            policy._dispatch_command(cmd)
        assert policy.active_velocity_command_idx == policy.CMD_CODES[0][expected], f"key {key!r}"


def test_locomotion_keys_still_use_the_velocity_accumulator():
    """Policies without KEYBOARD_DIRECTION_KEYS keep the incremental behavior."""
    assert not hasattr(LocomotionPolicy, "KEYBOARD_DIRECTION_KEYS")

    queue = deque(["w", "w", "w", "e"])
    keyboard = KeyboardInput(queue, velocity_keys=KEYBOARD_VELOCITY_LOCOMOTION)
    vel = keyboard.poll_velocity()
    assert vel.lin_vel[0] == pytest.approx(0.3)
    assert vel.ang_vel == pytest.approx(0.1)
    # Velocity keys must not also dispatch as commands.
    assert keyboard.poll_commands() == []


@pytest.mark.parametrize(
    ("lin_vel", "ang_vel", "expected"),
    [
        ((0.5, 0.0), 0.0, "forward"),
        ((-0.5, 0.0), 0.0, "back"),
        ((0.5, 0.4), 0.0, "left_45"),
        ((0.5, -0.4), 0.0, "right_45"),
        ((0.0, 0.5), 0.0, "left_90"),
        ((0.0, 0.0), 0.5, "left_90"),
        ((0.0, 0.0), -0.5, "right_90"),
        ((0.0, 0.0), 0.0, "stand"),
    ],
)
def test_continuous_velocity_quantizes_to_direction(lin_vel, ang_vel, expected):
    """Continuous velocity input maps onto the trained discrete direction classes."""
    policy = _make_policy()
    policy._apply_velocity(VelCmd(lin_vel=lin_vel, ang_vel=ang_vel))
    assert policy.active_velocity_command_idx == policy.CMD_CODES[0][expected]
    assert policy.velocity_command.sum() == pytest.approx(1.0)


def test_speed_mode_cycles_and_remaps_commands():
    """STAND_TOGGLE cycles speed, which selects a different command-code table."""
    policy = _make_policy()
    assert policy.speed_mode == 0
    policy._dispatch_command(StateCommand.STAND_TOGGLE)
    assert policy.speed_mode == 1
    policy._apply_velocity(VelCmd(lin_vel=(0.5, 0.0), ang_vel=0.0))
    assert policy.active_velocity_command_idx == policy.CMD_CODES[1]["forward"]

    # Two modes, so the cycle wraps on the next toggle.
    policy._dispatch_command(StateCommand.STAND_TOGGLE)
    assert policy.speed_mode == 0


def test_state_machine_transitions():
    """Boots into stiff hold; stop enters damping rather than zeroing gains."""
    policy = _make_policy()
    # Stiff hold is the startup mode. (use_policy_action is not asserted here:
    # with no TTY the base class auto-enables it, since a headless run has no
    # way to press start.)
    assert policy._stiff_hold_active

    policy._handle_start_policy()
    assert policy.use_policy_action and not policy._stiff_hold_active

    policy._handle_stop_policy()
    assert policy._damping_mode_active and not policy.use_policy_action

    policy._handle_init_state()
    assert policy._stiff_hold_active


def test_stiff_hold_ramps_gains_from_zero():
    """Gains ease in so enabling stiffness never snaps the joints."""
    policy = _make_policy()
    state = _robot_state()

    first = policy._get_manual_command(state)
    assert first["kp"].max() == pytest.approx(0.0)

    for _ in range(policy.STIFF_BLEND_TICKS):
        last = policy._get_manual_command(state)
    np.testing.assert_allclose(last["kp"], policy._stiff_hold_kp, rtol=1e-6)
    np.testing.assert_allclose(last["q"], policy._stiff_hold_q, rtol=1e-5)


def test_damping_mode_holds_position_with_zero_kp():
    """Damping resists motion without tracking a target pose."""
    policy = _make_policy()
    policy._handle_stop_policy()
    command = policy._get_manual_command(_robot_state(dof_pos=np.arange(NUM_DOFS) * 0.01))

    np.testing.assert_allclose(command["kp"], 0.0)
    np.testing.assert_allclose(command["kd"], policy._stiff_hold_kd)


def test_recording_writes_quat_first_clip(tmp_path, monkeypatch):
    """Recorded clips use the quat-first qpos layout the clip consumer expects."""
    monkeypatch.chdir(tmp_path)
    policy = _make_policy(record_dir="stair", record_label="stair")

    state = _robot_state(dof_pos=np.arange(NUM_DOFS) * 0.01)
    state[0, 0:3] = [1.5, 2.5, 0.78]  # root position

    policy._dispatch_command(StateCommand.TOGGLE_RECORDING)
    for _ in range(50):
        policy.prepare_obs_for_rl(state)
    policy._dispatch_command(StateCommand.TOGGLE_RECORDING)
    policy.prepare_obs_for_rl(state)

    clips = list((tmp_path / "recorded_motion" / "stair").glob("*.npz"))
    assert len(clips) == 1
    assert clips[0].name == "stair_duration1.0s_motion.npz"

    data = np.load(clips[0])
    assert data["qpos"].shape == (50, 7 + NUM_DOFS)
    assert data["vel_cmd"].shape == (50, ONE_HOT_DIM)
    assert int(data["fps"]) == 50
    # Quaternion occupies [0:4] (unit norm), root position [4:7].
    assert np.linalg.norm(data["qpos"][0, 0:4]) == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(data["qpos"][0, 4:7], [1.5, 2.5, 0.78], atol=1e-5)


def test_recording_autosaves_when_policy_stops():
    """An in-flight recording is flushed instead of being silently dropped."""
    policy = _make_policy()
    policy._dispatch_command(StateCommand.TOGGLE_RECORDING)
    policy.prepare_obs_for_rl(_robot_state())
    assert policy._recording

    with patch.object(policy, "_stop_and_save_recording") as save:
        policy._handle_stop_policy()
    save.assert_called_once()


#########################################################################################################
## Depth liveness: a stalled producer stops the policy instead of feeding it a frozen frame
#########################################################################################################
class _LiveSensor:
    """Depth sensor stand-in whose heartbeat status each test sets directly."""

    def __init__(self) -> None:
        self.status: DepthLiveness | None = None

    def get_latest(self) -> np.ndarray:
        return np.zeros((1, 1, DEPTH_H, DEPTH_W), dtype=np.float32)

    def liveness(self) -> DepthLiveness | None:
        return self.status


def _status(stale_s: float, state: str = STATE_OK, checked: bool = True) -> DepthLiveness:
    return DepthLiveness(checked=checked, state=state, stale_s=stale_s, seq=1, restarted=False)


def _running_policy_with_live_sensor() -> tuple[DepthDistillationPolicy, _LiveSensor]:
    policy = _make_policy()
    sensor = _LiveSensor()
    policy._depth_sensor = sensor
    policy._handle_start_policy()
    assert policy.use_policy_action
    return policy, sensor


def test_stale_depth_enters_stiff_hold():
    policy, sensor = _running_policy_with_live_sensor()
    sensor.status = _status(stale_s=policy.config.task.depth_shm.stale_stop_s + 0.01, state=STATE_RECONNECTING)

    policy._check_depth_liveness()

    assert not policy.use_policy_action and policy._stiff_hold_active


def test_failed_producer_enters_stiff_hold_immediately():
    policy, sensor = _running_policy_with_live_sensor()
    sensor.status = _status(stale_s=0.0, state=STATE_FAILED)

    policy._check_depth_liveness()

    assert not policy.use_policy_action and policy._stiff_hold_active


def test_brief_staleness_only_warns():
    policy, sensor = _running_policy_with_live_sensor()
    cfg = policy.config.task.depth_shm
    sensor.status = _status(stale_s=(cfg.stale_warn_s + cfg.stale_stop_s) / 2)

    policy._check_depth_liveness()

    assert policy.use_policy_action and policy._depth_stale_warned
    sensor.status = _status(stale_s=0.0)
    policy._check_depth_liveness()
    assert not policy._depth_stale_warned


@pytest.mark.parametrize("status", [None, _status(stale_s=99.0, checked=False)])
def test_producer_without_a_heartbeat_is_left_alone(status):
    """Producers that predate the heartbeat must keep working exactly as before."""
    policy, sensor = _running_policy_with_live_sensor()
    sensor.status = status

    policy._check_depth_liveness()

    assert policy.use_policy_action


def test_liveness_is_checked_before_the_action_snapshot(monkeypatch):
    """The stalled tick itself must not send a policy action."""
    policy, sensor = _running_policy_with_live_sensor()
    sensor.status = _status(stale_s=0.0, state=STATE_FAILED)
    seen = {}
    monkeypatch.setattr(BasePolicy, "policy_action", lambda self: seen.setdefault("use_policy", self.use_policy_action))

    policy.policy_action()

    assert seen["use_policy"] is False
