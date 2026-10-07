"""Command types for the input system.

``StateCommand`` enums represent discrete user *intent* (e.g. "start the
policy") decoupled from the physical input that triggered it.

``VelCmd`` is a value object carrying absolute velocity state
produced by ``VelCmdProvider`` providers each cycle.

Device-to-command mappings live in their respective impl modules
(``keyboard.py``, ``joystick.py``, ``ros2.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class VelCmd:
    """Absolute velocity state emitted by VelCmdProvider providers."""

    lin_vel: tuple[float, float]  # linear x, y (m/s)
    ang_vel: float  # angular z (rad/s)


class StateCommand(Enum):
    """All discrete commands dispatched through the input system."""

    # --- Common ---
    START = auto()
    STOP = auto()
    INIT = auto()
    NEXT_POLICY = auto()
    KILL = auto()
    KP_UP = auto()
    KP_DOWN = auto()
    KP_UP_FINE = auto()
    KP_DOWN_FINE = auto()
    KP_RESET = auto()
    SWITCH_POLICY_1 = auto()
    SWITCH_POLICY_2 = auto()
    SWITCH_POLICY_3 = auto()
    SWITCH_POLICY_4 = auto()
    SWITCH_POLICY_5 = auto()
    SWITCH_POLICY_6 = auto()
    SWITCH_POLICY_7 = auto()
    SWITCH_POLICY_8 = auto()
    SWITCH_POLICY_9 = auto()

    # --- Locomotion ---
    STAND_TOGGLE = auto()
    ZERO_VELOCITY = auto()
    WALK = auto()  # ROS2 only
    STAND = auto()  # ROS2 only

    # --- Discrete direction ---
    # For policies commanded by a direction *class* rather than a velocity
    # vector. Each command is an absolute heading, not an increment.
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    MOVE_LEFT_45 = auto()
    MOVE_RIGHT_45 = auto()
    MOVE_LEFT_90 = auto()
    MOVE_RIGHT_90 = auto()

    # --- Whole-body tracking ---
    START_MOTION_CLIP = auto()

    # --- Motion recording ---
    TOGGLE_RECORDING = auto()

    # --- Dual mode ---
    SWITCH_MODE = auto()  # Injected by DualModePolicy at runtime
