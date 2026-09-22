from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


class ContactSubstepRecorder:
    """Contact forces at each physics substep of the current control step.

    ``buffer`` is [num_envs, decimation, num_bodies, 3]; slot i holds the forces after substep i
    (unlike the ``*_substep`` buffers in joint_control, which record the state entering the
    substep). Fully rewritten every control step, so it needs no clearing on reset.
    """

    def __init__(self, num_envs: int, decimation: int, num_bodies: int, device: str) -> None:
        self.buffer = torch.zeros(num_envs, decimation, num_bodies, 3, device=device)
        self._decimation = decimation
        self._substep_idx = 0

    def begin_frame(self) -> None:
        self._substep_idx = 0

    def record(self, frame: torch.Tensor) -> None:
        """Store one substep's contact forces [num_envs, num_bodies, 3]."""
        # Slot order is only meaningful when FRAME_BEGIN anchors it; a caller that skips it (the
        # test harnesses step raw physics) must still not index past the end.
        self.buffer[:, self._substep_idx % self._decimation] = frame
        self._substep_idx += 1
