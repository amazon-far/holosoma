"""The command a policy produces for one control step.

The flat tuple ``send_low_command`` takes, named. ``q``/``dq``/``tau`` are in joint order
(``robot.dof_names``, length ``num_dofs``); ``kp``/``kd`` are in motor order
(length ``num_motors``) because that is the order the driver applies them in, and
``None`` means "leave the gains the driver already holds", which is what the policy
phase does today.

A struct rather than a bare array so a field can be added without breaking callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointCommand:
    q: np.ndarray
    dq: np.ndarray | None = None
    tau: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None

    def __post_init__(self):
        for name in ("q", "dq", "tau", "kp", "kd"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(f"JointCommand.{name} must be a finite 1-D array, got {array.shape}")
            object.__setattr__(self, name, array)
        if self.kp is not None and (self.kp < 0).any():
            raise ValueError("JointCommand.kp must be non-negative")
        if self.kd is not None and (self.kd < 0).any():
            raise ValueError("JointCommand.kd must be non-negative")
