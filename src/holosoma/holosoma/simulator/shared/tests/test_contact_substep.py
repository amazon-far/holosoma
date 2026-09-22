"""Unit tests for the per-substep contact-force recorder (pure torch, no simulator).

Pins the properties the reward terms rely on: every slot of a control step holds a real sample
(no permanently-zero tail at any decimation), slots are in substep order, and each control step
fully replaces the previous one's samples so no reset-time clearing is needed.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.shared.contact_substep import ContactSubstepRecorder
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 2
NUM_BODIES = 3


def _recorder(decimation):
    return ContactSubstepRecorder(NUM_ENVS, decimation, NUM_BODIES, device="cpu")


def _frame(value):
    return torch.full((NUM_ENVS, NUM_BODIES, 3), float(value))


@pytest.mark.parametrize("decimation", [1, 4, 8])
def test_every_slot_holds_its_substep(decimation):
    recorder = _recorder(decimation)
    recorder.begin_frame()
    for substep in range(decimation):
        recorder.record(_frame(substep + 1))

    for substep in range(decimation):
        assert torch.equal(recorder.buffer[:, substep], _frame(substep + 1))


def test_next_control_step_replaces_the_previous_one():
    recorder = _recorder(4)
    recorder.begin_frame()
    for substep in range(4):
        recorder.record(_frame(substep + 1))

    recorder.begin_frame()
    for substep in range(4):
        recorder.record(_frame(100 + substep))

    for substep in range(4):
        assert torch.equal(recorder.buffer[:, substep], _frame(100 + substep))


def test_recorded_frame_is_a_copy():
    """IsaacGym records a live gymtorch view that the next substep mutates in place."""
    recorder = _recorder(4)
    frame = _frame(1)
    recorder.record(frame)
    frame[:] = 99.0

    assert torch.equal(recorder.buffer[:, 0], _frame(1))


def test_recording_past_the_decimation_wraps():
    """The standalone sim harnesses step physics with no control-step boundary."""
    recorder = _recorder(4)
    for substep in range(6):
        recorder.record(_frame(substep))

    assert torch.equal(recorder.buffer[:, 0], _frame(4))
    assert torch.equal(recorder.buffer[:, 1], _frame(5))
    assert torch.equal(recorder.buffer[:, 2], _frame(2))


def test_buffer_is_not_reallocated():
    """Consumers read simulator.contact_forces_substep, so the buffer keeps its identity."""
    recorder = _recorder(4)
    ptr = recorder.buffer.data_ptr()
    recorder.record(_frame(1))
    assert recorder.buffer.data_ptr() == ptr
