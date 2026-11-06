"""Compatibility import for native whole-body tracking with Python transport."""

from holosoma_inference.policies.native_runner import NativeWbtRunner as WholeBodyTrackingPolicy
from holosoma_inference.policies.wbt_utils import PinocchioRobot

__all__ = ["PinocchioRobot", "WholeBodyTrackingPolicy"]
