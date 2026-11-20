"""Arm tracking presets for G1 robot."""

from __future__ import annotations

from typing import List

from holosoma.config_values.arm_tracking.g1.randomization import (
    g1_arm_tracking_randomization,
    g1_arm_tracking_dynamics_randomization,
)

__all__: List[str] = [
    "g1_arm_tracking_randomization",
    "g1_arm_tracking_dynamics_randomization",
]
