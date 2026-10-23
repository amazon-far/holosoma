"""Build a policy and step it, without a robot.

``run_policy`` resolves a preset, opens a transport, starts input providers and runs its own
loop. Everything that is not the loop is useful on its own -- in a test, a replay, an eval
harness, or a host that owns its own tick -- so this exposes the same resolution and the same
policy classes with the transport left out:

    config = inference.get_config("alpha-29dof-loco")
    policy = inference.get_default_policy(config)
    policy.reset()
    cmd = policy.step(obs)

Nothing here is new behaviour. ``_shared_hardware_source`` and ``_injected_*`` are the seams
dual mode and the FAR-Draco node already use; this fills them with doubles instead of hardware.
"""

from __future__ import annotations

import dataclasses as dc

import numpy as np
from loguru import logger

from holosoma_inference.command import JointCommand
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from holosoma_inference.utils.rate import RateLimiter

__all__ = ["JointCommand", "NullInterface", "get_config", "get_default_policy"]


class NullInterface:
    """A transport that reads nothing and publishes nowhere.

    ``_resolve_control_gains`` calls ``update_config`` when the gains come from ONNX metadata,
    which is the only interface method reached while building a policy. The read/write methods
    raise rather than return zeros: a caller that reaches them wanted hardware and should hear
    about it at the call site, not receive a plausible pose.
    """

    def __init__(self, robot_config):
        self.robot_config = robot_config

    def update_config(self, robot_config) -> None:
        self.robot_config = robot_config

    def get_low_state(self):
        raise RuntimeError("get_default_policy built this policy without a transport; use step(obs)")

    def send_low_command(self, *args, **kwargs):
        raise RuntimeError("get_default_policy built this policy without a transport; step(obs) returns the command")

    def poll(self) -> None:
        pass

    def close(self) -> None:
        pass


class _NoInput:
    """An input channel that never produces anything (task.*_input == "injected")."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def poll_velocity(self) -> VelCmd | None:
        return None

    def poll_commands(self) -> list[StateCommand]:
        return []


class _HeadlessHardware:
    """Stands in for the primary policy that dual mode shares its hardware from."""

    def __init__(self, config: InferenceConfig):
        self.interface = NullInterface(config.robot)
        # Carried rather than re-derived: _init_sdk_components reads it off the source, and
        # "booster" would otherwise open a DDS channel off a network interface name.
        self.sdk_type = config.robot.sdk_type
        self.logger = logger
        self.rl_rate = config.task.rl_rate
        self.rate = RateLimiter(config.task.rl_rate)
        self.use_joystick = False
        self.use_keyboard = False
        self._velocity_input = _NoInput()
        self._command_provider = self._velocity_input


def get_config(name: str) -> InferenceConfig:
    """Resolve a preset name to its ``InferenceConfig``.

    The same menu ``run_policy`` offers: core presets plus any registered by an installed
    extension through the ``holosoma.config.inference`` entry point, which is how the alpha
    presets arrive.
    """
    from holosoma_inference.config.config_values.inference import INFERENCE_REGISTRY
    from holosoma_inference.utils.config_registry import load_entrypoint_presets

    load_entrypoint_presets(INFERENCE_REGISTRY)
    try:
        return INFERENCE_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(INFERENCE_REGISTRY))
        raise KeyError(
            f"unknown inference preset {name!r}. Known presets: {known}. "
            "A preset from an extension needs that package installed."
        ) from None


def get_default_policy(config: InferenceConfig, *, model_path: str | None = None):
    """Build the policy class this config selects, with no transport and no input devices.

    The class is resolved exactly as ``run_policy`` resolves it, so a WBT preset gets the WBT
    policy and an extension preset gets the extension's class off its entry point.

    ``secondary`` is dropped: dual mode exists to hand a falling robot to a fallback policy,
    which needs the hardware this policy does not have. Pass the secondary preset to a second
    call if you want to step both.
    """
    from holosoma_inference.policies.dual_mode import _select_policy_class

    task_overrides = {"velocity_input": "injected", "state_input": "injected", "skip_stiff_prompt": True}
    if model_path is not None:
        task_overrides["model_path"] = model_path
    declared = {f.name for f in dc.fields(config.task)}
    missing = sorted(set(task_overrides) - declared)
    if missing:
        # pydantic dataclasses drop an undeclared name in replace() instead of raising, which
        # would silently leave an input channel pointed at real hardware.
        raise ValueError(f"{type(config.task).__name__} has no field(s) {', '.join(missing)}")

    if config.secondary is not None:
        logger.info("get_default_policy: dropping secondary -- dual mode needs a shared transport")
    headless = dc.replace(config, task=dc.replace(config.task, **task_overrides), secondary=None)
    if not headless.task.model_path:
        raise ValueError("this preset ships no model_path; pass get_default_policy(config, model_path=...)")

    policy_class = _select_policy_class(headless)
    policy = object.__new__(policy_class)
    # Attached before __init__, which is when the base class reads them.
    policy._shared_hardware_source = _HeadlessHardware(headless)
    policy._injected_velocity_input = policy._shared_hardware_source._velocity_input
    policy._injected_command_provider = policy._shared_hardware_source._command_provider
    policy.__init__(headless)
    logger.info(f"built {policy_class.__name__} headless ({headless.task.model_path})")
    policy.reset()
    return policy


def zeros_for(policy) -> np.ndarray:
    """A correctly shaped zero observation -- the width the policy's model actually takes."""
    return np.zeros_like(policy.obs_buf_dict["actor_obs"])
