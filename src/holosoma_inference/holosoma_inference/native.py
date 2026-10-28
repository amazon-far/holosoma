"""In-process native policy API, without robot SDK or input-device initialization."""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import asdict, fields, replace

import numpy as np

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig


def get_config(name: str) -> InferenceConfig:
    """Return an independent registered preset, including extension presets."""
    from holosoma_inference.config.config_values.inference import INFERENCE_REGISTRY
    from holosoma_inference.utils.config_registry import load_plugins

    load_plugins()
    return copy.deepcopy(INFERENCE_REGISTRY[name.removeprefix("inference:")])


def _native_config(config: InferenceConfig, *, use_gpu=False) -> str:
    document = asdict(config)
    # Robot subclasses can carry SDK transport settings. The original config
    # remains available to the transport owner; only the common robot contract
    # crosses the native inference boundary.
    document["robot"] = {field.name: document["robot"][field.name] for field in fields(RobotConfig)}
    # These transports are serviced by the Python runner, not by the native
    # standalone executable. Preserve the original dataclass for those callers.
    document["secondary"] = None
    task = document["task"]
    task["velocity_input"] = task["state_input"] = "keyboard"
    for shortcut in ("use_keyboard", "use_joystick", "use_usb_joystick"):
        task[shortcut] = False
    task.get("debug", {})["record_rosbag"] = False
    document["runtime"] = {"use_gpu": use_gpu}
    return json.dumps(document)


class NativePolicy:
    """A stateful C++ policy. step accepts one packed actor observation.

    Returns absolute joint targets with shape (1, num_joints). Use step_state
    to build observations/history from robot state inside the same native engine.
    Construction and stepping never initialize a robot SDK or send motor I/O.
    """

    def __init__(self, config: InferenceConfig, *, use_gpu=False, autostart=True):
        from holosoma_inference import _native

        self.config = config
        self._lock = threading.RLock()
        paths = config.task.model_path
        if isinstance(paths, str):
            paths = [paths]
        resolved = []
        for path in paths or []:
            resolved_path = str(path)
            if resolved_path.startswith(("wandb://", "https://")):
                from holosoma_inference.utils.wandb import load_checkpoint

                resolved_path = str(load_checkpoint(None, resolved_path))
            resolved.append(resolved_path)
        native_config = replace(config, task=replace(config.task, model_path=resolved))
        self._engine = _native.Policy.from_config(_native_config(native_config, use_gpu=use_gpu), autostart)

    def reset(self):
        with self._lock:
            self._engine.reset()

    def set_reference(self, window):
        """Replace the reference buffer without resetting state; call reset() for a new clip."""
        with self._lock:
            self._engine.set_reference(window)

    def step(self, obs):
        with self._lock:
            return self._engine.step(np.asarray(obs, dtype=np.float32))

    def _state(self, values, tick_ms=0):
        from holosoma_inference import _native

        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        n = self.config.robot.num_joints
        width = 13 + 2 * n
        if values.shape not in {(width,), (width + 3,)}:
            raise ValueError(f"Expected one robot state with {width} values, optionally followed by corrected gravity")
        state = _native.RobotState()
        state.base_pos = values[:3]
        state.base_quat_wxyz = values[3:7]
        state.dof_pos = values[7 : 7 + n]
        state.base_lin_vel = values[7 + n : 10 + n]
        state.base_ang_vel = values[10 + n : 13 + n]
        state.dof_vel = values[13 + n : width]
        if len(values) == width + 3:
            state.projected_gravity = values[width:]
        state.tick_ms = tick_ms
        return state

    def step_state(self, state, *, tick_ms=0, advance_phase=True):
        with self._lock:
            return self._engine.step_state(self._state(state, tick_ms), advance_phase)

    def step_contextual(self, state, inputs, *, tick_ms=0):
        """Forward typed depth/command input to the shared C++ policy."""
        with self._lock:
            return self._engine.step(self._state(state, tick_ms), inputs)

    def set_state(self, state, *, tick_ms=0):
        with self._lock:
            self._engine.set_state(self._state(state, tick_ms))

    def infer_offsets(self, state, *, tick_ms=0):
        """Warm/evaluate native inference without engagement or command publication."""
        with self._lock:
            return self._engine.infer_offsets(self._state(state, tick_ms))

    def command(self, command):
        with self._lock:
            self._engine.command(getattr(command, "name", command))

    def velocity(self, x, y, yaw):
        with self._lock:
            self._engine.velocity(x, y, yaw)

    @property
    def last_observation(self):
        with self._lock:
            return np.asarray(self._engine.last_observation, dtype=np.float32)

    @property
    def motion_timestep(self):
        with self._lock:
            return self._engine.motion_timestep

    @property
    def initial_motion(self):
        with self._lock:
            return self._engine.initial_motion

    @property
    def depth_input(self):
        """Name of the graph's depth input, or "" when the policy is not contextual."""
        with self._lock:
            return self._engine.depth_input


class LocomotionPolicy(NativePolicy):
    """Configurable native locomotion inference without a hardware interface."""


class WholeBodyTrackingPolicy(NativePolicy):
    """Embedded-motion native WBT inference without a hardware interface."""


class GeneralWBTPolicy(NativePolicy):
    """Metadata-driven WBT; the host supplies a ReferenceWindow before stepping."""


def get_policy_from_preset(config: InferenceConfig | str, **kwargs) -> NativePolicy:
    if isinstance(config, str):
        config = get_config(config)
    if config.contextual is not None:
        return NativePolicy(config, **kwargs)
    if "actor_targets" in config.observation.obs_dict:
        return GeneralWBTPolicy(config, **kwargs)
    cls = (
        WholeBodyTrackingPolicy
        if "motion_command" in config.observation.obs_dict.get("actor_obs", [])
        else LocomotionPolicy
    )
    return cls(config, **kwargs)
