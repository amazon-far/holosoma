"""Exercise the installed native library with a real ONNX model, no robot SDK."""

import gc
import json
import runpy
import subprocess
import sys
import time
import weakref
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from pydantic.dataclasses import dataclass

import holosoma_inference as inference
from holosoma_inference.config.config_types.robot import RobotConfig


@pytest.fixture
def native_config(tmp_path):
    config = inference.get_config("g1-29dof-loco")
    obs = config.observation
    width = sum(obs.obs_dims[name] for name in obs.obs_dict["actor_obs"]) * obs.history_length_dict.get("actor_obs", 1)
    weights = np.zeros((width, 29), dtype=np.float32)
    weights[:29] = np.eye(29, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["actor_obs", "weights"], ["actions"])],
        "native-api-fixture",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 29])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    helper.set_model_props(model, {"kp": json.dumps([40.0] * 29), "kd": json.dumps([2.0] * 29)})
    path = tmp_path / "policy.onnx"
    onnx.save(model, path)
    return replace(config, task=replace(config.task, model_path=str(path)), secondary=None), width


def test_library_import_has_no_robot_side_effects():
    code = (
        "import sys; import holosoma_inference; from holosoma_inference import _native; "
        "assert not any(n in sys.modules for n in "
        "('unitree_interface','rclpy','pinocchio','holosoma_inference.sdk'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_contextual_config_and_inputs_use_the_same_native_policy(tmp_path):
    from holosoma_inference.config.config_types.contextual import ContextualConfig
    from holosoma_inference.config.config_types.observation import ObservationConfig

    generator = Path(__file__).parents[1] / "holosoma_cpp/tests/gen_contextual_models.py"
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
    config = replace(
        base,
        robot=robot,
        observation=ObservationConfig({"actor_obs": []}, {}, {}, {}),
        task=replace(base.task, model_path=str(path), use_phase=False),
        contextual=ContextualConfig(depth_height=2, depth_width=3),
        secondary=None,
    )
    policy = inference.get_policy_from_preset(config)
    assert isinstance(policy._engine, inference.Policy)
    inputs = inference.ContextualInput()
    inputs.monotonic_time = 1
    state = np.zeros(19)
    state[3] = 1
    state[7:10] = robot.default_dof_angles
    first = policy.step_contextual(state, inputs)
    np.testing.assert_allclose(first.q, [15.1, -9.8, 40.3])
    np.testing.assert_array_equal(first.kp, [10, 20, 30])
    inputs.monotonic_time = 2
    second = policy.step_contextual(state, inputs)
    np.testing.assert_array_equal(policy.last_observation[12:15], [20, -20, 5])
    np.testing.assert_allclose(second.q, [60.1, -9.8, 40.3])
    with pytest.raises(ValueError, match="ContextualInput"):
        policy.step_state(state)
    policy.reset()
    np.testing.assert_allclose(policy.step_contextual(state, inputs).q, first.q)


def test_packed_observation_uses_native_graph_scaling_and_defaults(native_config):
    config, width = native_config
    policy = inference.LocomotionPolicy(config)
    obs = np.arange(width, dtype=np.float32)[None, :]
    expected = np.minimum(obs[:, :29], 100) * np.float32(config.task.policy_action_scale) + np.asarray(
        config.robot.default_dof_angles
    )
    np.testing.assert_allclose(policy.step(obs), expected, rtol=1e-6, atol=1e-7)
    policy.reset()
    np.testing.assert_allclose(policy.step(obs), expected, rtol=1e-6, atol=1e-7)


def test_native_rejects_invalid_shapes_and_nonfinite_input(native_config):
    config, width = native_config
    policy = inference.get_policy_from_preset(config)
    for obs in (np.zeros((2, width)), np.zeros((1, width - 1)), np.full((1, width), np.nan)):
        with pytest.raises(ValueError, match="observation"):
            policy.step(obs)
    with pytest.raises(ValueError, match="orientation"):
        policy.step_state(np.zeros((1, 71)))  # invalid all-zero orientation
    assert np.isfinite(policy.step(np.zeros((1, width)))).all()


def test_state_api_preserves_hold_and_explicit_engagement(native_config):
    config, _ = native_config
    policy = inference.get_policy_from_preset(config, autostart=False)
    state = np.zeros((1, 71))
    state[0, 3] = 1
    state[0, 7:36] = np.linspace(-0.2, 0.2, 29)
    held = policy.step_state(state)
    np.testing.assert_allclose(held.q, state[0, 7:36])
    assert policy.last_observation.size == 0
    policy.command("START")
    active = policy.step_state(state)
    assert policy.last_observation.size > 0
    assert not np.allclose(active.q, held.q)
    policy.command("STOP")
    np.testing.assert_allclose(policy.step_state(state).q, held.q)


def test_state_api_preserves_explicit_corrected_gravity(native_config):
    config, _ = native_config
    policy = inference.get_policy_from_preset(config)
    state = np.zeros((1, 74))
    state[0, 3] = 1
    state[0, -3:] = [0.2, -0.1, -0.97]
    policy.step_state(state)
    terms = sorted(config.observation.obs_dict["actor_obs"])
    history = config.observation.history_length_dict.get("actor_obs", 1)
    start = sum(config.observation.obs_dims[t] * history for t in terms[: terms.index("projected_gravity")])
    latest = start + (history - 1) * 3
    np.testing.assert_allclose(policy.last_observation[latest : latest + 3], state[0, -3:], atol=1e-6)


def test_state_api_advances_locomotion_phase_once_per_tick(native_config):
    config, _ = native_config
    config = replace(config, task=replace(config.task, auto_walk_on_vel_cmd=True, use_phase=True))
    policy = inference.get_policy_from_preset(config)
    state = np.zeros((1, 71))
    state[0, 3] = 1
    policy.velocity(0.2, 0.0, 0.0)
    policy.step_state(state)
    first = policy.last_observation.copy()
    policy.step_state(state)
    second = policy.last_observation.copy()
    terms = sorted(config.observation.obs_dict["actor_obs"])
    start = sum(config.observation.obs_dims[t] for t in terms[: terms.index("sin_phase")])
    dt = 2 * np.pi / (config.task.rl_rate * config.task.gait_period)
    np.testing.assert_allclose(first[start : start + 2], [np.sin(dt), -np.sin(dt)], atol=1e-6)
    np.testing.assert_allclose(second[start : start + 2], [np.sin(2 * dt), -np.sin(2 * dt)], atol=1e-6)


def test_preset_lookup_does_not_share_mutable_observation_maps():
    config = inference.get_config("g1-29dof-loco")
    config.observation.obs_dims["native_test_only"] = 9
    assert "native_test_only" not in inference.get_config("g1-29dof-loco").observation.obs_dims


def test_python_factory_uses_the_public_policy_interface(native_config):
    config, _ = native_config
    policy = inference.get_policy_from_preset(config)
    assert isinstance(policy._engine, inference.Policy)
    state = inference.RobotState()
    state.dof_pos = [0.0] * 29
    state.dof_vel = [0.0] * 29
    command = policy._engine.step(state)
    assert isinstance(command, inference.JointCommand)
    assert len(command.q) == policy._engine.num_dofs() == len(policy._engine.dof_names) == 29
    with pytest.raises(RuntimeError, match="own reference"):
        policy._engine.set_reference(inference.ReferenceWindow())


def test_public_policy_borrows_one_reference_and_returns_command_snapshots(tmp_path):
    generator = Path(__file__).parents[1] / "holosoma_cpp/tests/gen_live_stub_models.py"
    write_model = runpy.run_path(str(generator))["_write_model"]
    checkpoint = tmp_path / "live.onnx"
    write_model(checkpoint, history=9, lookahead_layout="flat_0p2s")
    # Zero padding makes the command change while history fills, which the snapshot check needs.
    policy = inference.Policy.from_live_model(str(checkpoint), history_initialization="zero_pad")
    assert policy.lookahead_offsets == [0, 2, 4, 6, 8]
    state = inference.RobotState()
    state.dof_pos = state.dof_vel = [0.0] * 29
    with pytest.raises(RuntimeError, match="set_reference"):
        policy.step(state)

    def window():
        reference = inference.ReferenceWindow()
        reference.q = reference.dq = [0.0] * (5 * 29)
        reference.quat = [[1, 0, 0, 0]] * 5
        return reference

    first = window()
    retained = weakref.ref(first)
    policy.set_reference(first)
    del first
    gc.collect()
    assert retained() is not None
    command = policy.step(state)
    saved = list(command.q)
    for _ in range(9):
        policy.step(state)
    assert command.q == saved
    assert policy.step(state).q != saved
    assert len(policy.last_observation) == 2267
    policy.set_reference(window())
    gc.collect()
    assert retained() is None
    assert len(policy.step(state).q) == 29
    policy.set_reference(None)
    with pytest.raises(RuntimeError, match="set_reference"):
        policy.step(state)


def test_library_accepts_robot_config_with_host_transport_fields(native_config):
    from dataclasses import asdict

    @dataclass(frozen=True)
    class TransportRobot(RobotConfig):
        command_topic: str = "/example/commands"

    config, width = native_config
    robot = TransportRobot(**asdict(config.robot))
    config = replace(config, robot=robot)
    policy = inference.get_policy_from_preset(config)
    assert policy.config.robot.command_topic == "/example/commands"
    assert np.isfinite(policy.step(np.zeros((1, width)))).all()


def test_embedded_motion_resumes_progress_after_a_library_reset():
    config = inference.get_config("g1-29dof-wbt")
    path = files("holosoma_inference").joinpath("models/wbt/fastsac_g1_29dof_dancing.onnx")
    config = replace(config, task=replace(config.task, model_path=str(path), use_sim_time=False), secondary=None)
    policy = inference.get_policy_from_preset(config)
    state = np.zeros((1, 71))
    state[0, 3] = 1
    state[0, 7:36] = config.robot.default_dof_angles
    policy.step_state(state)
    policy.step_state(state)
    assert policy.motion_timestep == 2
    policy.reset()
    policy.step_state(state)
    policy.step_state(state)
    assert policy.motion_timestep == 2


def test_library_autostart_captures_the_first_robot_heading():
    config = inference.get_config("g1-29dof-wbt")
    path = files("holosoma_inference").joinpath("models/wbt/fastsac_g1_29dof_dancing.onnx")
    config = replace(config, task=replace(config.task, model_path=str(path), use_sim_time=False), secondary=None)
    observations = []
    for yaw in (0.0, 1.0):
        policy = inference.get_policy_from_preset(config)
        state = np.zeros((1, 71))
        state[0, 3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        state[0, 7:36] = config.robot.default_dof_angles
        policy.step_state(state)
        observations.append(policy.last_observation)
    np.testing.assert_allclose(observations[0], observations[1], atol=2e-5)


def test_python_runner_passes_joint_ordered_gains_to_the_sdk(native_config):
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    config, _ = native_config
    permutation = tuple(reversed(range(29)))
    config = replace(
        config,
        robot=replace(
            config.robot,
            joint2motor=permutation,
            motor2joint=permutation,
            motor_kp=tuple(range(10, 39)),
            motor_kd=tuple(range(1, 30)),
        ),
    )

    class Interface:
        kp_level = kd_level = 1.0

        def get_low_state(self):
            state = np.zeros((1, 71))
            state[0, 3] = 1
            return state

        def send_low_command(self, *args, **kwargs):
            self.sent = kwargs

    class Input:
        def poll_velocity(self):
            return None

        def poll_commands(self):
            return []

    interface, inputs = Interface(), Input()
    runner = NativeLocomotionRunner(config, interface=interface, velocity_input=inputs, command_provider=inputs)
    runner._handle_start_policy()
    runner.policy_action()
    np.testing.assert_allclose(interface.sent["kp_override"], np.asarray(config.robot.motor_kp)[list(permutation)])


def test_injected_runner_can_warm_up_without_publishing_or_engaging(native_config):
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    config, width = native_config

    class Interface:
        def get_low_state(self):
            state = np.zeros((1, 71))
            state[0, 3] = 1
            return state

        def send_low_command(self, *_args, **_kwargs):
            pytest.fail("warm-up published a command")

    class Input:
        def poll_velocity(self):
            return None

        def poll_commands(self):
            return []

    interface, provider = Interface(), Input()
    runner = NativeLocomotionRunner(config, interface=interface, velocity_input=provider, command_provider=provider)
    result = runner.rl_inference(interface.get_low_state())
    assert result.shape == (1, 29) and np.isfinite(result).all()
    assert runner.native.last_observation.size == width
    assert not runner.use_policy_action


@pytest.mark.parametrize(
    ("override", "message"),
    [({"q": [np.nan] * 29}, "finite"), ({"q": [0.0]}, "finite"), ({"kp": [-1.0] * 29}, "nonnegative")],
)
def test_invalid_robot_output_transform_never_reaches_transport(native_config, override, message):
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    class Runner(NativeLocomotionRunner):
        def transform_command(self, _command, _state):
            return override

    class Interface:
        def get_low_state(self):
            state = np.zeros((1, 71))
            state[0, 3] = 1
            return state

        def send_low_command(self, *_args, **_kwargs):
            pytest.fail("invalid transformed command reached the SDK")

    class Inputs:
        def poll_velocity(self):
            return None

        def poll_commands(self):
            return []

    config, _ = native_config
    inputs = Inputs()
    runner = Runner(config, interface=Interface(), velocity_input=inputs, command_provider=inputs)
    with pytest.raises(ValueError, match=message):
        runner.policy_action()


class _ScriptedInterface:
    """Return scripted low states (the last one repeats) and record published commands."""

    kp_level = kd_level = 1.0

    def __init__(self, *states):
        self.states = list(states)
        self.sent = []

    def get_low_state(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def send_low_command(self, *args, **kwargs):
        self.sent.append(kwargs)


class _NoInput:
    def poll_velocity(self):
        return None

    def poll_commands(self):
        return []


def _low_state(quat=(1.0, 0.0, 0.0, 0.0), **values):
    state = np.zeros((1, 71))
    state[0, 3:7] = quat
    for index, value in values.items():
        state[0, int(index.removeprefix("i"))] = value
    return state


def _runner(config, interface, **task):
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    config = replace(config, task=replace(config.task, **task))
    inputs = _NoInput()
    return NativeLocomotionRunner(config, interface=interface, velocity_input=inputs, command_provider=inputs)


@pytest.mark.parametrize(
    "waiting_state",
    [_low_state(quat=(0.0, 0.0, 0.0, 0.0)), _low_state(i7=np.nan), _low_state(quat=(np.inf, 0.0, 0.0, 0.0))],
    ids=["no-message-yet", "nonfinite-joint", "nonfinite-orientation"],
)
def test_runner_waits_for_first_valid_state_without_inferring_or_publishing(native_config, waiting_state):
    config, _ = native_config
    interface = _ScriptedInterface(waiting_state, waiting_state, _low_state())
    runner = _runner(config, interface)
    runner.use_policy_action = True
    assert runner.policy_action() is False
    assert runner.policy_action() is False
    assert interface.sent == []
    assert runner.native.last_observation.size == 0
    assert not runner.use_policy_action
    assert runner.policy_action() is True
    assert len(interface.sent) == 1
    assert runner.use_policy_action


def test_runner_fails_when_no_valid_state_arrives_within_the_startup_timeout(native_config):
    config, _ = native_config
    interface = _ScriptedInterface(_low_state(quat=(0.0, 0.0, 0.0, 0.0)))
    runner = _runner(config, interface, startup_state_timeout=0.05)
    assert runner.policy_action() is False
    time.sleep(0.1)
    with pytest.raises(TimeoutError, match="startup-state-timeout"):
        runner.policy_action()
    assert interface.sent == []


@pytest.mark.parametrize(
    ("invalid_state", "message"),
    [(_low_state(quat=(0.0, 0.0, 0.0, 0.0)), "orientation must be nonzero"), (_low_state(i7=np.nan), "finite")],
    ids=["zero-orientation", "nonfinite-joint"],
)
def test_runner_rejects_an_invalid_state_once_running(native_config, invalid_state, message):
    config, _ = native_config
    interface = _ScriptedInterface(_low_state(), invalid_state)
    runner = _runner(config, interface)
    assert runner.policy_action() is True
    with pytest.raises(ValueError, match=message):
        runner.policy_action()
    assert len(interface.sent) == 1


def test_headless_auto_start_waits_for_the_first_valid_state(native_config):
    config, _ = native_config
    interface = _ScriptedInterface(_low_state(quat=(0.0, 0.0, 0.0, 0.0)), _low_state())
    runner = _runner(config, interface)
    runner.use_policy_action = True
    assert not runner.native._engine.active
    assert runner.policy_action() is False
    assert not runner.native._engine.active
    assert runner.policy_action() is True
    assert runner.native._engine.active


def test_stop_while_waiting_cancels_a_pending_auto_start(native_config):
    config, _ = native_config
    interface = _ScriptedInterface(_low_state(quat=(0.0, 0.0, 0.0, 0.0)), _low_state())
    runner = _runner(config, interface)
    runner.use_policy_action = True
    runner._handle_stop_policy()
    assert runner.policy_action() is False
    assert runner.policy_action() is True
    assert not runner.native._engine.active


def test_runner_does_not_advance_phase_while_waiting(native_config):
    config, _ = native_config
    interface = _ScriptedInterface(_low_state(quat=(0.0, 0.0, 0.0, 0.0)), _low_state())
    runner = _runner(config, interface)

    class PhaseSpy:
        def __init__(self, engine):
            self.engine, self.updates = engine, 0

        def update_phase(self):
            self.updates += 1
            self.engine.update_phase()

        def __getattr__(self, name):
            return getattr(self.engine, name)

    spy = runner.native._engine = PhaseSpy(runner.native._engine)
    runner.update_phase_time()
    assert runner.policy_action() is False
    runner.update_phase_time()
    assert spy.updates == 0
    assert runner.policy_action() is True
    runner.update_phase_time()
    assert spy.updates == 1


def test_initially_valid_state_takes_the_fast_path(native_config):
    config, _ = native_config
    interface = _ScriptedInterface(_low_state())
    runner = _runner(config, interface)
    assert runner.policy_action() is True
    assert runner._state_wait_started is None


def test_runner_ignores_start_until_a_valid_state_arrives(native_config):
    config, _ = native_config
    interface = _ScriptedInterface(_low_state(quat=(0.0, 0.0, 0.0, 0.0)))
    runner = _runner(config, interface)
    runner._handle_start_policy()
    assert not runner.use_policy_action


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_startup_state_timeout_must_be_a_positive_finite_duration(native_config, timeout):
    config, _ = native_config
    with pytest.raises(ValueError, match="startup-state-timeout"):
        replace(config.task, startup_state_timeout=timeout)


def _write_live_model(tmp_path, name="live.onnx", **kwargs):
    generator = Path(__file__).parents[1] / "holosoma_cpp/tests/gen_live_stub_models.py"
    write_model = runpy.run_path(str(generator))["_write_model"]
    checkpoint = tmp_path / name
    write_model(checkpoint, **kwargs)
    return str(checkpoint)


def _split_window(marker):
    reference = inference.ReferenceWindow()
    reference.q = [marker] * (10 * 29)
    reference.dq = [0.0] * (10 * 29)
    reference.quat = [[1, 0, 0, 0]] * 10
    return reference


def _oldest_and_newest_history_slot(policy, marker):
    state = inference.RobotState()
    state.dof_pos = state.dof_vel = [0.0] * 29
    window = _split_window(marker)
    policy.set_reference(window)
    policy.step(state)
    observation = np.asarray(policy.last_observation)
    return observation[0], observation[2 * 58]


@pytest.mark.parametrize(("mode", "padded"), [("repeat_first_valid_obs", 1.0), ("zero_pad", 0.0)])
def test_live_model_history_initialization_reaches_the_native_core(tmp_path, mode, padded):
    checkpoint = _write_live_model(tmp_path, history=3, lookahead_layout="split")
    policy = inference.Policy.from_live_model(checkpoint, history_initialization=mode)
    oldest, newest = _oldest_and_newest_history_slot(policy, 0.5)
    assert newest == pytest.approx(0.5)
    assert oldest == pytest.approx(padded * 0.5)


def test_live_model_defaults_to_repeat_first_valid_obs_and_rejects_other_spellings(tmp_path):
    checkpoint = _write_live_model(tmp_path, history=3, lookahead_layout="split", history_initialization="zero")
    oldest, newest = _oldest_and_newest_history_slot(inference.Policy.from_live_model(checkpoint), 0.5)
    assert oldest == newest == pytest.approx(0.5)
    with pytest.raises(Exception, match="wbt_history_initialization"):
        inference.Policy.from_live_model(checkpoint, history_initialization="zero")


def test_python_config_forwards_history_initialization_to_the_metadata_policy(tmp_path):
    from holosoma_inference.native import GeneralWBTPolicy

    checkpoint = _write_live_model(tmp_path, history=3, lookahead_layout="split")
    config = inference.get_config("g1-29dof-loco")
    observation = replace(config.observation, obs_dict={"actor_obs": [], "actor_targets": []})
    results = {}
    for mode in ("repeat_first_valid_obs", "zero_pad"):
        task = replace(config.task, model_path=checkpoint, use_phase=False, wbt_history_initialization=mode)
        policy = GeneralWBTPolicy(replace(config, observation=observation, task=task, secondary=None))
        window = _split_window(0.5)
        policy.set_reference(window)
        state = np.zeros((1, 71))
        state[0, 3] = 1
        policy.step_state(state)
        results[mode] = (policy.last_observation[0], policy.last_observation[2 * 58])
    assert results["repeat_first_valid_obs"] == pytest.approx((0.5, 0.5))
    assert results["zero_pad"] == pytest.approx((0.0, 0.5))


def test_task_config_rejects_an_unknown_history_initialization(native_config):
    config, _ = native_config
    with pytest.raises(ValueError, match="wbt_history_initialization"):
        replace(config.task, wbt_history_initialization="zero")


@pytest.mark.parametrize(
    ("control", "raw", "clip"),
    [(None, 50.0, 20.0), ({"action_clip_value": 10.0}, 50.0, 10.0), ({"clip_actions": False}, 150.0, 20.0)],
    ids=["missing-uses-ceiling", "tighter-declared", "training-clip-disabled"],
)
def test_live_wbt_policy_reports_and_applies_the_effective_action_clip(tmp_path, control, raw, clip):
    checkpoint = _write_live_model(tmp_path, control=control, constant_action=raw, action_scale=0.25)
    policy = inference.LiveWbtPolicy(checkpoint, history_initialization="zero_pad")
    assert policy.history_initialization == "zero_pad"
    assert policy.action_clip == pytest.approx(clip)
    observation = inference.LiveRobotObservation()
    reference = inference.ReferenceSequence()
    reference.lookahead = [inference.ReferenceFrame() for _ in range(10)]
    # The stub checkpoint's default pose is 0.01 rad per joint index.
    targets = np.asarray(policy.step(observation, reference)).reshape(-1) - 0.01 * np.arange(29)
    expected = np.where(np.arange(29) % 2, -clip, clip) * 0.25
    np.testing.assert_allclose(targets, expected, rtol=1e-6)
    np.testing.assert_allclose(policy.last_raw_action, np.where(np.arange(29) % 2, -raw, raw))
    np.testing.assert_allclose(policy.last_clipped_action, np.where(np.arange(29) % 2, -clip, clip))


def test_live_wbt_policy_rejects_an_unknown_history_initialization(tmp_path):
    with pytest.raises(ValueError, match="repeat_first_valid_obs or zero_pad"):
        inference.LiveWbtPolicy(_write_live_model(tmp_path), history_initialization="zero")
