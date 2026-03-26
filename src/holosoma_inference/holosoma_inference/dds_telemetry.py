"""Lightweight DDS message for policy observation telemetry.

Published at the policy rate (typically 50 Hz) on ``rt/policy_obs``.
The ``data`` field carries the flat observation vector whose layout
depends on the active observation config.
"""

from __future__ import annotations

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence


@dataclass
class PolicyTelemetry(IdlStruct):
    """Timestamped flat float vector with monotonic sequence number."""

    seq: int = 0
    timestamp: float = 0.0
    data: sequence[float] = ()  # type: ignore[assignment]
