"""Holosoma's device-free inference API.

Robot transport is initialized only by the explicit run_policy entry point.
"""

from holosoma_inference.native import (
    InferenceConfig,
    LocomotionPolicy,
    WholeBodyTrackingPolicy,
    get_config,
    get_policy_from_preset,
)

__all__ = [
    "ContextualInput",
    "InferenceConfig",
    "JointCommand",
    "LiveRobotObservation",
    "LiveWbtPolicy",
    "LocomotionPolicy",
    "Policy",
    "PolicyReferenceFrame",
    "ReferenceFrame",
    "ReferenceSequence",
    "ReferenceWindow",
    "RobotState",
    "WholeBodyTrackingPolicy",
    "get_config",
    "get_policy_from_preset",
]

_LIVE_TYPES = {
    "ContextualInput",
    "LiveWbtPolicy",
    "LiveRobotObservation",
    "ReferenceFrame",
    "ReferenceSequence",
    "Policy",
    "PolicyReferenceFrame",
    "RobotState",
    "JointCommand",
    "ReferenceWindow",
}


def __getattr__(name):
    if name in _LIVE_TYPES:
        from holosoma_inference import _native

        return getattr(_native, name)
    raise AttributeError(name)
