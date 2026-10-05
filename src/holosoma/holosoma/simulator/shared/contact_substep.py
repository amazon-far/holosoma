from __future__ import annotations

from typing import Callable

from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.utils.safe_torch_import import torch


class ContactSubstepRecorder:
    """Contact forces at each physics substep of the current control step.

    Owns its own hooks: ``POST_STEP`` appends ``current_forces()``, ``FRAME_BEGIN`` re-anchors the
    next sample to slot 0. A loop that emits those phases therefore needs no contact-specific code,
    but one that steps physics without emitting ``POST_STEP`` records nothing.

    Slot i holds the forces *after* substep i, unlike the ``*_substep`` buffers in joint_control,
    which record the state entering the substep. Read through :attr:`recorded_forces`.
    """

    def __init__(
        self,
        hooks: HookRegistry,
        current_forces: Callable[[], torch.Tensor],
        num_envs: int,
        decimation: int,
        num_bodies: int,
        device: str,
    ) -> None:
        self.buffer = torch.zeros(num_envs, decimation, num_bodies, 3, device=device)
        self._current_forces = current_forces
        self._decimation = decimation
        self._substep_idx = 0
        hooks.add(Phase.FRAME_BEGIN, self.begin_frame, name="contact.begin_frame")
        hooks.add(Phase.POST_STEP, self._record_current, name="contact.record")

    @property
    def recorded_forces(self) -> torch.Tensor:
        """Substeps recorded so far this control step, [num_envs, n, num_bodies, 3], oldest first
        as long as FRAME_BEGIN anchors the step (see :meth:`record`).

        ``n`` grows from 0 to the decimation across the step, so a mid-step reader never sees a slot
        left over from the previous one. A consumer that reduces over dim 1 must read after the
        substep loop, where ``n`` is always the full decimation — it is 0 before the first sample.
        """
        return self.buffer[:, : min(self._substep_idx, self._decimation)]

    @property
    def latest_forces(self) -> torch.Tensor:
        """Most recent substep, [num_envs, num_bodies, 3]; zeros before the first record.

        Between FRAME_BEGIN and the first POST_STEP the index is 0 and this wraps to the last slot,
        which is the previous step's final substep — still the most recent sample taken.
        """
        return self.buffer[:, (self._substep_idx - 1) % self._decimation]

    def begin_frame(self) -> None:
        self._substep_idx = 0

    def record(self, frame: torch.Tensor) -> None:
        """Store one substep's contact forces [num_envs, num_bodies, 3]."""
        # Slot order is only meaningful when FRAME_BEGIN anchors it; a caller that emits POST_STEP
        # without it must still not index past the end.
        self.buffer[:, self._substep_idx % self._decimation] = frame
        self._substep_idx += 1

    def _record_current(self) -> None:
        self.record(self._current_forces())
