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


class _RecordingVelocity(_NoInput):
    def __init__(self):
        self.zeroed = 0
        self.stopped = 0

    def zero(self):
        self.zeroed += 1

    def start(self):
        pass

    def stop(self):
        self.stopped += 1


def test_dual_mode_secondary_engages_when_switched_to_before_it_has_stepped(native_config):
    from holosoma_inference.policies.dual_mode import DualModePolicy
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    config, _ = native_config
    interface, inputs = _ScriptedInterface(_low_state()), _NoInput()

    class Runner(NativeLocomotionRunner):
        def __init__(self, config, shared_hardware=None):
            if shared_hardware is not None:
                super().__init__(config, shared_hardware=shared_hardware)
            else:
                super().__init__(config, interface=interface, velocity_input=inputs, command_provider=inputs)

    dual = DualModePolicy(config, config, policy_selector=lambda _config: Runner)
    dual.primary._handle_start_policy()
    assert dual.primary.policy_action() is True and dual.primary.use_policy_action
    dual._handle_mode_switch()
    assert dual.active is dual.secondary
    assert not dual.primary.use_policy_action
    assert dual.secondary.use_policy_action


@pytest.mark.parametrize("shortcut", ["use_joystick", "use_usb_joystick", "use_keyboard"])
def test_input_shortcut_survives_the_native_model_path_rewrite(native_config, shortcut):
    from holosoma_inference.native import NativePolicy

    config, _ = native_config
    config = replace(config, task=replace(config.task, **{shortcut: True}))
    NativePolicy(config, autostart=False)
    resolved = config.task.velocity_input
    assert replace(config.task, model_path="other.onnx").velocity_input == resolved
    with pytest.raises(ValueError, match="Cannot combine"):
        replace(config.task, velocity_input="ros2")


@pytest.mark.parametrize(
    ("command", "zeroes"),
    [("ZERO_VELOCITY", [1]), ("STAND_TOGGLE", [0, 1]), ("WALK", [0])],
    ids=["zero-velocity", "stand-toggle-walk-then-stance", "walk"],
)
def test_runner_zeroes_the_operator_velocity_input_when_the_engine_does(native_config, command, zeroes):
    from holosoma_inference.inputs.api.commands import StateCommand
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    config, _ = native_config
    velocity = _RecordingVelocity()
    runner = NativeLocomotionRunner(
        config, interface=_ScriptedInterface(_low_state()), velocity_input=velocity, command_provider=_NoInput()
    )
    for expected in zeroes:
        runner._dispatch_command(StateCommand[command])
        assert velocity.zeroed == expected


def test_runner_rejects_subclasses_overriding_retired_python_policy_hooks():
    from holosoma_inference.policies import BasePolicy, LocomotionPolicy, WholeBodyTrackingPolicy

    for alias in (BasePolicy, LocomotionPolicy, WholeBodyTrackingPolicy):
        with pytest.raises(TypeError, match="get_current_obs_buffer_dict, rl_inference|rl_inference"):

            class _Teleop(alias):
                def rl_inference(self, robot_state_data):
                    return robot_state_data

        with pytest.raises(TypeError, match="get_current_obs_buffer_dict"):

            class _Obs(alias):
                def get_current_obs_buffer_dict(self, robot_state_data):
                    return {}

        class _Supported(alias):
            def transform_command(self, command, robot_state_data):
                return None


def test_close_stops_only_the_providers_the_runner_created(native_config, monkeypatch):
    from holosoma_inference.policies import transport
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    config, _ = native_config
    injected, created = _RecordingVelocity(), _RecordingVelocity()
    monkeypatch.setattr(
        transport, "create_input", lambda _policy, source, _role: injected if source == "injected" else created
    )
    for state_input, created_stops in (("injected", 0), ("ros2", 1)):
        task = replace(config.task, velocity_input="injected", state_input=state_input)
        runner = object.__new__(NativeLocomotionRunner)
        runner._injected_velocity_input = runner._injected_command_provider = injected
        runner.__init__(replace(config, task=task), interface=_ScriptedInterface(_low_state()))
        runner.close()
        assert injected.stopped == 0
        assert created.stopped == created_stops


def test_stiff_hold_prompt_survives_a_closed_stdin(native_config, monkeypatch):
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    class Runner(NativeLocomotionRunner):
        is_wbt = True

    prompts = []

    def closed_stdin(prompt):
        prompts.append(prompt)
        raise EOFError

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", closed_stdin)
    config, _ = native_config
    inputs = _NoInput()
    Runner(config, interface=_ScriptedInterface(_low_state()), velocity_input=inputs, command_provider=inputs)
    assert len(prompts) == 1


@pytest.mark.parametrize(("platform", "expected"), [("darwin", "KeyboardInput"), ("linux", "UsbJoystickInput")])
def test_joystick_input_falls_back_to_keyboard_on_macos(native_config, monkeypatch, platform, expected):
    from holosoma_inference.inputs.impl import usb_joystick
    from holosoma_inference.policies import transport
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    class _FakeUsbJoystick(_RecordingVelocity):
        def __init__(self, device_index=0):
            super().__init__()

        def start(self):
            pass

    _FakeUsbJoystick.__name__ = "UsbJoystickInput"
    monkeypatch.setattr(usb_joystick, "UsbJoystickInput", _FakeUsbJoystick)
    monkeypatch.setattr(transport.sys, "platform", platform)
    config, _ = native_config
    task = replace(config.task, velocity_input="joystick", state_input="joystick")
    runner = NativeLocomotionRunner(replace(config, task=task), interface=_ScriptedInterface(_low_state()))
    try:
        assert type(runner._velocity_input).__name__ == expected
        assert runner.use_joystick == (platform != "darwin")
    finally:
        runner.close()


def test_booster_command_sender_uses_joint_gains_when_the_config_has_none():
    from types import SimpleNamespace

    from holosoma_inference.config.config_values.robot import t1_29dof
    from holosoma_inference.sdk.booster.command_sender.base.basic_command_sender import BasicCommandSender

    class Sender(BasicCommandSender):
        def _init_sdk_components(self):
            pass

        def send_command(self, *args, **kwargs):
            pass

    config = replace(t1_29dof, motor_kp=None, motor_kd=None)
    sender = Sender(config)
    motors = [SimpleNamespace() for _ in range(config.num_motors)]
    n = config.num_joints
    kp, kd = np.arange(n) + 10.0, np.arange(n) + 1.0
    sender._fill_motor_commands(motors, np.zeros(n), np.zeros(n), np.zeros(n), kp_override=kp, kd_override=kd)
    for motor_id, joint_id in enumerate(config.motor2joint):
        if joint_id != -1:
            assert (motors[motor_id].kp, motors[motor_id].kd) == (kp[joint_id], kd[joint_id])
    with pytest.raises(ValueError, match="No KP gain"):
        sender._fill_motor_commands(motors, np.zeros(n), np.zeros(n), np.zeros(n))


def _live_wbt_command(tmp_path, gravity=None, **debug):
    from holosoma_inference.native import GeneralWBTPolicy

    checkpoint = _write_live_model(tmp_path, lookahead_layout="split", constant_action=1.0)
    config = inference.get_config("g1-29dof-loco")
    observation = replace(config.observation, obs_dict={"actor_obs": [], "actor_targets": []})
    task = replace(config.task, model_path=checkpoint, use_phase=False, debug=replace(config.task.debug, **debug))
    policy = GeneralWBTPolicy(replace(config, observation=observation, task=task, secondary=None))
    policy.set_reference(_split_window(0.0))
    state = np.zeros(71)
    state[3:7] = (np.cos(0.2), np.sin(0.2), 0.0, 0.0)  # rolled, so derived gravity is not straight down
    state[39:42] = (0.1, 0.2, 0.3)  # base_ang_vel
    if gravity is not None:
        state = np.concatenate([state, gravity])
    command = policy.step_state(state)
    # Stub frame layout: motion_command (58), motion_ref_ori_b (6), projected_gravity (3), base_ang_vel (3).
    obs = policy.last_observation
    return np.asarray(command.q), obs[64:67], obs[67:70]


def test_live_wbt_honours_debug_overrides_and_corrected_gravity(tmp_path):
    defaults = 0.01 * np.arange(29)  # the stub checkpoint's default pose
    q, gravity, ang_vel = _live_wbt_command(tmp_path)
    assert not np.allclose(q, defaults)
    assert not np.allclose(gravity, (0.0, 0.0, -1.0))
    np.testing.assert_allclose(ang_vel, (0.1, 0.2, 0.3), rtol=1e-6)

    q, _, _ = _live_wbt_command(tmp_path, force_zero_action=True)
    np.testing.assert_allclose(q, defaults, atol=1e-6)
    _, _, ang_vel = _live_wbt_command(tmp_path, force_zero_angular_velocity=True)
    np.testing.assert_allclose(ang_vel, 0.0)
    _, gravity, _ = _live_wbt_command(tmp_path, force_upright_imu=True)
    np.testing.assert_allclose(gravity, (0.0, 0.0, -1.0))
    _, gravity, _ = _live_wbt_command(tmp_path, gravity=(0.1, -0.2, -0.97))
    np.testing.assert_allclose(gravity, (0.1, -0.2, -0.97), rtol=1e-6)
