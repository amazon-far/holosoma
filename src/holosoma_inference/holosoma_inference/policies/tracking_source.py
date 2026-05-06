"""Real-time tracking input channel for WholeBodyTrackingPolicy.

Contract for an external ROS integration to push SMPLH tracking
targets into a running policy. The channel is transport-agnostic:
holosoma core defines the payload schema and the injection protocol;
the concrete transport (Unix socket, shared memory, ...) lives on the
consumer side.

Short version:

* The policy does NOT call the transport. It holds a ``TrackingSource``
  protocol object and calls ``get_latest()`` once per tick from
  ``rl_inference``. The call is non-blocking; implementations return
  ``None`` when no new payload is available since the last poll.
* Default behavior when no source is injected: ``NullTrackingSource``
  always returns ``None``, and the policy falls back to its ONNX-clip
  ``motion_command_t`` path — identical to today's behavior.
* The payload is SMPLH-native (body + optional hand joints + gripper
  intent + mode bits). Converting SMPLH into the policy's internal
  ``motion_command_t`` (shape ``(1, 58)`` = 29 DOF joint_pos + 29 DOF
  joint_vel) is retargeting and is a follow-up. Until that bridge
  lands, ``rl_inference`` logs the received payload but does not
  substitute it into ``motion_command_t``. This gives the ROS
  integration something to integration-test against (serialize → log
  round-trip) without forcing a half-designed retargeter into this
  change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class TrackingPayload:
    """One tick of external tracking input.

    Fields mirror the union of the three ROS topics the downstream
    integration publishes: body tracking state, gripper command, and
    control. Keep this schema stable; consumers on both sides of the
    service boundary are serializing to it.

    Body joints use SMPLH naming. Hand joints (optional) use MANO
    naming. Gripper intent is device-normalized (``0.0 = closed,
    1.0 = open``), consumed by downstream hand-pipeline logic inside
    the service. The mode field is one of (0=STOP, 1=TELEOP,
    2=INFERENCE, 3=IDLE).
    """

    # Body joint tracking (SMPLH body joints).
    joint_names: list[str] = field(default_factory=list)
    # (N, 7) — pos (x, y, z) + quat (x, y, z, w) per joint, flattened.
    joint_transforms: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    joint_confidences: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # Hand joint tracking (MANO joints, optional — may be empty).
    hand_joint_names: list[str] = field(default_factory=list)
    hand_joint_transforms: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    hand_joint_confidences: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # Device-normalized gripper intent.
    left_gripper_value: float = 0.0
    right_gripper_value: float = 0.0
    left_gripper_confidence: float = 0.0
    right_gripper_confidence: float = 0.0

    # Tracking source metadata.
    device_type: str = ""
    tracking_quality: int = 0

    # Control fields — carried per-tick so the service can respond
    # to mode transitions and HITL gating without a second channel.
    mode: int = 3  # IDLE by default
    forward_gripper_commands: bool = False
    control_reason: str = ""


@runtime_checkable
class TrackingSource(Protocol):
    """Non-blocking poll protocol for external tracking input.

    Implementations MUST:
        * return the latest payload since the previous call, or ``None``
          if nothing new has arrived;
        * never block on I/O — the policy's control loop calls this once
          per tick (50-60 Hz) and cannot tolerate jitter;
        * be safe to call from the same thread as ``rl_inference``
          (no thread-synchronization requirement on the policy side).

    The concrete transport (Unix socket, shared memory, ...) is the
    implementation's concern, left to the consumer.
    """

    def get_latest(self) -> TrackingPayload | None: ...


class NullTrackingSource:
    """Default source — always returns ``None``.

    With this source injected (the constructor default for
    ``WholeBodyTrackingPolicy``), ``rl_inference`` falls back to the
    ONNX-clip ``motion_command_t`` path and behavior is byte-identical
    to the pre-change policy.
    """

    def get_latest(self) -> TrackingPayload | None:
        return None
