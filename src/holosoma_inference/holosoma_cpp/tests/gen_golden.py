#!/usr/bin/env python3
"""Record Python reference runs for the C++ parity tests.

Each scenario builds a config with the real holosoma_inference CLI parser,
runs the real policy classes (LocomotionPolicy, WholeBodyTrackingPolicy,
DualModePolicy) tick by tick against a scripted robot, and records what the
robot receives: the joint targets and gains of every LowCmd, plus the actor
observation of every inference. The C++ test replays the same robot states and
operator inputs through holosoma_run_policy's controller and compares.

The robot side is a fake ``unitree_interface`` binding, so the real
``UnitreeInterface`` (joint/motor mapping, kp level, overrides) and the real
keyboard/joystick input providers are exercised too.

Usage (from the repository root, in an environment with holosoma_inference):
    python src/holosoma_inference/holosoma_cpp/tests/gen_golden.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import zmq

os.environ.setdefault("LOGURU_LEVEL", "WARNING")

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "golden"
MODELS = "src/holosoma_inference/holosoma_inference/models"
LOCO_PPO = f"{MODELS}/loco/g1_29dof/ppo_g1_29dof.onnx"
LOCO_FASTSAC = f"{MODELS}/loco/g1_29dof/fastsac_g1_29dof.onnx"
WBT_PPO = f"{MODELS}/wbt/ppo_g1_29dof_dancing.onnx"
WBT_FASTSAC = f"{MODELS}/wbt/fastsac_g1_29dof_dancing.onnx"


def f32(x) -> float:
    """float32 value as the float64 with the shortest repr that round-trips to it (compact JSON)."""
    return float(np.format_float_positional(np.float32(x), unique=True, trim="0"))


def f32_list(values) -> list[float]:
    return [f32(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


# ---------------------------------------------------------------------------
# Fake far-unitree-sdk binding
# ---------------------------------------------------------------------------


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeRobot:
    """Stands in for unitree_interface.UnitreeInterface (the C++ binding)."""

    def __init__(self, num_motors: int):
        self.num_motors = num_motors
        self.state = _Namespace(
            imu=_Namespace(quat=[1.0, 0.0, 0.0, 0.0], omega=[0.0] * 3, rpy=[0.0] * 3, accel=[0.0] * 3),
            motor=_Namespace(q=[0.0] * num_motors, dq=[0.0] * num_motors),
        )
        self.tick_ms = 0
        self.wireless = _Namespace(lx=0.0, ly=0.0, rx=0.0, ry=0.0, keys=0)
        self.last_command = None

    def set_control_mode(self, mode):
        pass

    def read_low_state(self):
        return self.state

    def read_wireless_controller(self):
        return self.wireless

    def create_zero_command(self):
        n = self.num_motors
        return _Namespace(q_target=[0.0] * n, dq_target=[0.0] * n, tau_ff=[0.0] * n, kp=[0.0] * n, kd=[0.0] * n)

    def write_low_command(self, cmd):
        # The binding stores std::vector<float>.
        self.last_command = {k: f32_list(getattr(cmd, k)) for k in ("q_target", "kp", "kd")}
        self.last_command["q"] = self.last_command.pop("q_target")


FAKE_ROBOTS: list[FakeRobot] = []


def install_fake_binding():
    module = types.ModuleType("unitree_interface")
    module.RobotType = _Namespace(G1="G1", H1="H1", H1_2="H1_2", GO2="GO2")
    module.MessageType = _Namespace(HG="HG", GO2="GO2")
    module.ControlMode = _Namespace(PR="PR", AB="AB")

    def create_robot(interface, robot_type, message_type):
        robot = FakeRobot(29)
        FAKE_ROBOTS.append(robot)
        return robot

    module.create_robot = create_robot
    sys.modules["unitree_interface"] = module


# ---------------------------------------------------------------------------
# Scripted robot states
# ---------------------------------------------------------------------------


def make_states(default_q, num_ticks: int, seed: int, ticks_ms: list[int]):
    rng = np.random.default_rng(seed)
    n = len(default_q)
    amp = rng.uniform(0.02, 0.15, n)
    freq = rng.uniform(0.3, 1.5, n)
    phase = rng.uniform(0, 2 * np.pi, n)
    states = []
    for t in range(num_ticks):
        time = t * 0.02
        q = np.asarray(default_q) + amp * np.sin(2 * np.pi * freq * time + phase)
        dq = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * time + phase)
        roll, pitch, yaw = 0.05 * np.sin(1.3 * time), 0.04 * np.cos(0.9 * time), 0.3 + 0.2 * np.sin(0.5 * time)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        quat = np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
        omega = np.array([0.3 * np.sin(2.1 * time), -0.2 * np.cos(1.7 * time), 0.1 * np.sin(0.7 * time)])
        states.append(
            {
                "q": f32_list(np.round(q, 4)),
                "dq": f32_list(np.round(dq, 4)),
                "quat": f32_list(np.round(quat / np.linalg.norm(quat), 6)),
                "omega": f32_list(np.round(omega, 4)),
                "tick": int(ticks_ms[t]),
            }
        )
    return states


def apply_state(robot: FakeRobot, state: dict):
    robot.state.motor.q = list(state["q"])
    robot.state.motor.dq = list(state["dq"])
    robot.state.imu.quat = list(state["quat"])
    robot.state.imu.omega = list(state["omega"])
    robot.tick_ms = state["tick"]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

# Each scenario: CLI args shared by both runtimes (C++ runs them through parse_cli),
# extra Python-only args, the input device, and per-tick operator input.
SCENARIOS = [
    {
        "name": "loco_keyboard_multi_model",
        "args": ["inference:g1-29dof-loco", "--task.model-path", LOCO_PPO, LOCO_FASTSAC, "--secondary", "none"],
        "input": "keyboard",
        "ticks": 90,
        "events": {
            5: "i",
            25: "]",
            30: "=",
            35: "ww",
            40: "a",
            45: "e",
            50: "g2",
            55: "z",
            58: "w",
            60: "=",
            65: "fv",
            70: "1",
            75: "o",
            80: "]",
            84: "rb",
        },
    },
    {
        "name": "wbt_dual_mode_keyboard",
        "args": ["inference:g1-29dof-wbt", "--task.model-path", WBT_FASTSAC],
        "input": "keyboard",
        "ticks": 110,
        "events": {5: "i", 20: "]", 25: "m", 60: "x", 62: "=", 65: "ww", 80: "x", 85: "m", 100: "o", 105: "]"},
    },
    {
        "name": "wbt_sim_time_options",
        "args": [
            "inference:g1-29dof-wbt",
            "--task.model-path",
            WBT_PPO,
            "--task.use-sim-time",
            "--task.motion-start-timestep",
            "10",
            "--task.motion-end-timestep",
            "60",
            "--robot.interp-gain-scale",
            "1.5",
            "--robot.joint-offsets-deg",
            *[str(v) for v in np.round(np.linspace(-2.0, 2.0, 29), 3)],
            "--secondary",
            "none",
        ],
        "input": "keyboard",
        "ticks": 80,
        "events": {2: "i", 10: "]", 12: "m", 70: "o", 74: "]"},
        # Simulator clock: 20 ms per tick, a forward jump when the clip starts and a reset (backward jump).
        "clock_jumps": {13: 500, 40: -300},
    },
    {
        "name": "loco_joystick_kill",
        "args": [
            "inference:g1-29dof-loco",
            "--task.model-path",
            LOCO_FASTSAC,
            LOCO_PPO,
            "--task.use-joystick",
            "--secondary",
            "none",
        ],
        "input": "interface",
        "ticks": 70,
        # (keys, lx, ly, rx) held from the given tick on.
        "wireless": {
            0: (0, 0.0, 0.0, 0.0),
            5: (256, 0.0, 0.0, 0.0),
            6: (0, 0.0, 0.0, 0.0),
            10: (4, 0.0, 0.0, 0.0),
            11: (0, 0.0, 0.0, 0.0),
            15: (0, 0.05, 0.6, -0.4),
            25: (264, 0.3, 0.6, 0.0),
            26: (0, -0.5, 0.2, 0.3),
            35: (8, 0.0, 0.0, 0.0),
            36: (0, 0.0, 0.4, 0.0),
            40: (4096, 0.0, 0.0, 0.0),
            41: (0, 0.0, 0.4, 0.0),
            45: (32, 0.0, 0.0, 0.0),
            46: (0, 0.0, 0.0, 0.0),
            55: (512, 0.0, 0.0, 0.0),
            56: (0, 0.0, 0.0, 0.0),
            58: (2048, 0.0, 0.0, 0.0),
            59: (0, 0.0, 0.0, 0.0),
            62: (256, 0.0, 0.0, 0.0),
            63: (0, 0.0, 0.0, 0.0),
            68: (3, 0.0, 0.0, 0.0),
        },
    },
]


def build_config(args: list[str], python_only: list[str]):
    from holosoma_inference.config.config_values.inference import get_annotated_inference_config
    from holosoma_inference.utils.config_registry import parse_config

    args = list(args)
    disable_secondary = False
    if "--secondary" in args:
        i = args.index("--secondary")
        assert args[i + 1] == "none"
        del args[i : i + 2]
        disable_secondary = True
    # Model paths relative to the repository root, as the C++ test runs from there.
    args = [str(REPO_ROOT / a) if a.startswith(MODELS) else a for a in args]
    # tyro 1.0.x cannot build a parser for multi-value flags on these configs
    # ("dict[str, Any] with struct-type values requires a default value"), so
    # list-valued overrides are applied with dataclasses.replace instead.
    single, multi, i = [], {}, 0
    while i < len(args):
        j = i + 1
        while j < len(args) and not args[j].startswith("--"):
            j += 1
        if args[i].startswith("--") and j - i > 2:
            section, field = args[i][2:].split(".")
            multi[(section, field.replace("-", "_"))] = args[i + 1 : j]
        else:
            single.extend(args[i:j])
        i = j
    if "--task.use-joystick" in single:
        single.remove("--task.use-joystick")
        single += ["--task.velocity-input", "interface", "--task.state-input", "interface"]
    config = parse_config(get_annotated_inference_config, args=single + python_only)
    for (section, field), values in multi.items():
        value = values if field == "model_path" else tuple(float(v) for v in values)
        config = replace(config, **{section: replace(getattr(config, section), **{field: value})})
    config = replace(config, task=replace(config.task, skip_stiff_prompt=True))
    if disable_secondary:
        config = replace(config, secondary=None)
    return config


def run_scenario(spec: dict) -> dict:
    import holosoma_inference.policies.base as base_module
    from holosoma_inference.inputs.impl.interface import InterfaceInput
    from holosoma_inference.inputs.impl.keyboard import KEYBOARD_VELOCITY_LOCOMOTION, KeyboardInput
    from holosoma_inference.policies.dual_mode import DualModePolicy, _select_policy_class
    from holosoma_inference.sdk.unitree.unitree_interface import UnitreeInterface

    python_only = ["--task.velocity-input", "injected", "--task.state-input", "injected"]
    if spec["input"] == "interface":
        python_only = []
    config = build_config(spec["args"], python_only)

    interface = UnitreeInterface(config.robot, 0, "lo", True)
    robot = FAKE_ROBOTS[-1]
    key_queue: deque[str] = deque()
    if spec["input"] == "keyboard":
        provider = KeyboardInput(key_queue, KEYBOARD_VELOCITY_LOCOMOTION)
    else:
        provider = InterfaceInput(interface)

    base_module.create_interface = lambda *_args, **_kwargs: interface
    base_module.create_input = lambda *_args: provider

    num_ticks = spec["ticks"]
    ticks_ms, now = [], 1000
    for t in range(num_ticks):
        now += 20 + spec.get("clock_jumps", {}).get(t, 0)
        ticks_ms.append(now)
    states = make_states(config.robot.default_dof_angles, num_ticks, seed=len(spec["name"]), ticks_ms=ticks_ms)
    apply_state(robot, states[0])

    if config.secondary is not None:
        controller = DualModePolicy(primary_config=config, secondary_config=config.secondary)
        policies = [controller.primary, controller.secondary]
    else:
        controller = _select_policy_class(config)(config=config)
        policies = [controller]

    captured: list[np.ndarray] = []
    for p in policies:
        original = p.prepare_obs_for_rl

        def wrapped(robot_state_data, _orig=original):
            out = _orig(robot_state_data)
            captured.append(out["actor_obs"].copy())
            return out

        p.prepare_obs_for_rl = wrapped
        # Drive the WBT simulator clock from LowState.tick, as the C++ runtime does.
        timestep_util = getattr(p, "timestep_util", None)
        if timestep_util is not None:
            clock_sub = timestep_util._clock._clock_sub
            clock_sub._drain_messages = lambda _cs=clock_sub: setattr(_cs, "last_clock", robot.tick_ms)

    wireless = spec.get("wireless", {})
    events = spec.get("events", {})
    ticks = []
    last_gains: dict[str, list[float]] = {}
    for t in range(num_ticks):
        apply_state(robot, states[t])
        if t in wireless:
            keys, lx, ly, rx = wireless[t]
            robot.wireless = _Namespace(lx=f32(lx), ly=f32(ly), rx=f32(rx), ry=0.0, keys=keys)
        keys_pressed = events.get(t, "")
        key_queue.extend(keys_pressed)

        record = {"state": states[t], "keys": keys_pressed}
        if spec["input"] == "interface":
            w = robot.wireless
            record["wireless"] = {"keys": w.keys, "lx": w.lx, "ly": w.ly, "rx": w.rx}
        captured.clear()
        robot.last_command = None
        active = controller.active if isinstance(controller, DualModePolicy) else controller
        try:
            # One iteration of BasePolicy.run / DualModePolicy.run (without sleeping).
            vc = active._velocity_input.poll_velocity()
            if vc is not None:
                active._apply_velocity(vc)
            commands = active._command_provider.poll_commands()
            for cmd in commands:
                active._dispatch_command(cmd)
            if isinstance(controller, DualModePolicy):
                active = controller.active
            if active.use_phase:
                active.update_phase_time()
            active.policy_action()
        except SystemExit:
            record["killed"] = True
            ticks.append(record)
            break
        command = dict(robot.last_command)
        for gain in ("kp", "kd"):
            if command[gain] == last_gains.get(gain):
                del command[gain]  # unchanged since the previous tick
            else:
                last_gains[gain] = command[gain]
        record["command"] = command
        if hasattr(active, "curr_motion_timestep"):
            record["motion_timestep"] = int(active.curr_motion_timestep)
        if captured and t % 4 == 0:
            record["obs"] = f32_list(captured[-1])
        ticks.append(record)

    # WBT policies leave a ZMQ ClockSub socket open; an open socket makes a later
    # garbage-collected zmq.Context.term() block forever.
    for p in policies:
        clock_sub = getattr(getattr(getattr(p, "timestep_util", None), "_clock", None), "_clock_sub", None)
        if clock_sub is not None and clock_sub.socket is not None:
            clock_sub.socket.setsockopt(zmq.LINGER, 0)
            clock_sub.close()

    return {
        "name": spec["name"],
        "args": spec["args"],
        "input": spec["input"],
        "ticks": ticks,
    }


def kinematics_golden(num_samples: int = 40) -> dict:
    """Torso orientation from Pinocchio for random base orientations and joint angles."""
    import onnx

    from holosoma_inference.config.config_values.robot import g1_29dof
    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    model = onnx.load(str(REPO_ROOT / WBT_PPO), load_external_data=False)
    urdf = json.loads(next(p.value for p in model.metadata_props if p.key == "robot_urdf"))
    robot = PinocchioRobot(replace(g1_29dof, motion={"body_name_ref": ["torso_link"]}), urdf)
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(num_samples):
        quat_wxyz = rng.normal(size=4)
        quat_wxyz /= np.linalg.norm(quat_wxyz)
        q = rng.uniform(-1.0, 1.0, 29)
        configuration = np.concatenate([np.zeros(3), quat_wxyz[[1, 2, 3, 0]], q[robot.real2pinocchio_index]])
        torso_xyzw = robot.fk_and_get_ref_body_orientation_in_world(configuration)[0]
        samples.append(
            {"base_quat_wxyz": quat_wxyz.tolist(), "dof_pos": q.tolist(), "torso_quat_xyzw": torso_xyzw.tolist()}
        )
    return {"model": WBT_PPO, "body": "torso_link", "samples": samples}


def math_golden(num_samples: int = 20) -> dict:
    from holosoma_inference.utils.math import quat as Q

    rng = np.random.default_rng(3)
    out = []
    for _ in range(num_samples):
        a = rng.normal(size=(1, 4))
        b = rng.normal(size=(1, 4))
        a /= np.linalg.norm(a)
        v = rng.normal(size=(1, 3))
        rpy = rng.uniform(-1.5, 1.5, 3)
        out.append(
            {
                "a": a[0].tolist(),
                "b": b[0].tolist(),
                "v": v[0].tolist(),
                "rpy": rpy.tolist(),
                "quat_mul": Q.quat_mul(a, b)[0].tolist(),
                "quat_rotate_inverse": Q.quat_rotate_inverse(a, v)[0].tolist(),
                "matrix_from_quat": Q.matrix_from_quat(b)[0].reshape(-1).tolist(),
                "rpy_to_quat": Q.rpy_to_quat(rpy).tolist(),
                "quat_to_rpy": list(Q.quat_to_rpy(a[0])),
                "subtract_frame_transforms": Q.subtract_frame_transforms(a, b)[0].tolist(),
            }
        )
    return {"samples": out}


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    install_fake_binding()
    os.chdir(REPO_ROOT)
    write_json(args.output_dir / "math.json", math_golden())
    write_json(args.output_dir / "kinematics.json", kinematics_golden())
    for spec in SCENARIOS:
        write_json(args.output_dir / f"{spec['name']}.json", run_scenario(spec))


if __name__ == "__main__":
    main()
