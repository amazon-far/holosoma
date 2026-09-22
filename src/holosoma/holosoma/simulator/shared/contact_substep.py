"""Per-control-step contact-force samples, one per physics substep."""

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
        self._idx = 0

    def begin_frame(self) -> None:
        self._idx = 0

    def record(self, frame: torch.Tensor) -> None:
        """Store one substep's contact forces [num_envs, num_bodies, 3]."""
        # The standalone sim harnesses step physics with no control-step boundary (no FRAME_BEGIN),
        # so there is no meaningful slot order to preserve; wrap rather than run off the end.
        self.buffer[:, self._idx % self._decimation] = frame
        self._idx += 1
