"""Keys of the steps a task runs as hooks, for ordering other hooks against them."""

from __future__ import annotations

from enum import Enum, auto

TASK_SLOT = "task"
"""Slot the task publishes its step hooks under; order against one with ``HookRef(TASK_SLOT, TaskHook.X)``."""


class TaskHook(Enum):
    """A step the task registers as a hook, named ``task.<member>`` in lower case.

    Each phase's steps run in the order listed here, chained with ``after``. Steps marked policy run only
    on policy tasks.
    """

    # PRE_STEP: first in every physics substep.
    APPLY_ACTIONS = auto()

    # FRAME_END: after the phase's other hooks.
    COUNT_FRAME = auto()
    TERMINATION = auto()
    OBSERVATION_FRAME = auto()  # policy
    PRE_COMPUTE_OBSERVATIONS = auto()  # policy
    REWARD = auto()  # policy
    UPDATE_LOG_DICT = auto()  # policy
    TERMINAL_OBSERVATIONS = auto()  # policy
    RESET = auto()
    COMMAND_STEP = auto()  # policy
    RANDOMIZATION_STEP = auto()
    METRICS = auto()  # policy
    OBSERVATIONS = auto()  # policy

    # RESET: steps that write state skip a frame's empty batch; manager resets always run.
    PREPARE_RESET = auto()
    WRITE_STATE = auto()
    COUNT_EPISODE = auto()
    REWARD_RESET = auto()  # policy
    RANDOMIZATION_RESET = auto()
    TERMINATION_RESET = auto()
    ACTION_RESET = auto()  # policy
    COMMAND_RESET = auto()  # policy
    MANAGERS_RESET = auto()
    REFRESH = auto()
    AFTER_RESET_SYNC = auto()
