"""Unit tests for the per-substep contact-force recorder (pure torch, no simulator).

Pins the properties the reward terms rely on: every slot of a control step holds a real sample
(no permanently-zero tail at any decimation), slots are in substep order, each control step fully
replaces the previous one's samples so no reset-time clearing is needed, and a mid-step reader
never sees the previous step's slots.

Driven through a real ``HookRegistry``, so the constructor's own hook registration is under test
rather than assumed.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.contact_substep import ContactSubstepRecorder
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 2
NUM_BODIES = 3


def _frame(value):
    return torch.full((NUM_ENVS, NUM_BODIES, 3), float(value))


class _Driver:
    """A recorder plus the source tensor its getter reads, stepped through the hook phases."""

    def __init__(self, decimation):
        self.hooks = HookRegistry()
        self.source = _frame(0)
        self.recorder = ContactSubstepRecorder(
            self.hooks, lambda: self.source, NUM_ENVS, decimation, NUM_BODIES, device="cpu"
        )

    def begin_frame(self):
        self.hooks.emit(Phase.FRAME_BEGIN)

    def step(self, value):
        self.source[:] = float(value)
        self.hooks.emit(Phase.POST_STEP)


@pytest.mark.parametrize("decimation", [1, 4, 8])
def test_every_slot_holds_its_substep(decimation):
    driver = _Driver(decimation)
    driver.begin_frame()
    for substep in range(decimation):
        driver.step(substep + 1)

    for substep in range(decimation):
        assert torch.equal(driver.recorder.buffer[:, substep], _frame(substep + 1))


def test_next_control_step_replaces_the_previous_one():
    driver = _Driver(4)
    driver.begin_frame()
    for substep in range(4):
        driver.step(substep + 1)

    driver.begin_frame()
    for substep in range(4):
        driver.step(100 + substep)

    for substep in range(4):
        assert torch.equal(driver.recorder.buffer[:, substep], _frame(100 + substep))


def test_frame_begin_restarts_the_slot_index():
    """A short control step must not leave the next one's slot 0 off by the shortfall."""
    driver = _Driver(4)
    driver.step(1)
    driver.step(2)
    driver.begin_frame()
    driver.step(3)

    assert torch.equal(driver.recorder.buffer[:, 0], _frame(3))
    assert torch.equal(driver.recorder.buffer[:, 1], _frame(2))


def test_recorded_frame_is_a_copy():
    """IsaacGym's getter returns a live gymtorch view that the next substep mutates in place."""
    driver = _Driver(4)
    driver.step(1)
    driver.source[:] = 99.0

    assert torch.equal(driver.recorder.buffer[:, 0], _frame(1))


def test_recording_past_the_decimation_wraps():
    """A loop that emits POST_STEP without a control-step boundary must not index past the end."""
    driver = _Driver(4)
    for substep in range(6):
        driver.step(substep)

    assert torch.equal(driver.recorder.buffer[:, 0], _frame(4))
    assert torch.equal(driver.recorder.buffer[:, 1], _frame(5))
    assert torch.equal(driver.recorder.buffer[:, 2], _frame(2))


def test_recorded_forces_covers_only_this_frames_samples():
    driver = _Driver(4)
    assert driver.recorder.recorded_forces.shape[1] == 0

    driver.begin_frame()
    for substep in range(4):
        driver.step(substep + 1)
        assert driver.recorder.recorded_forces.shape[1] == substep + 1

    driver.step(5)  # past the decimation: the width caps instead of growing
    assert driver.recorder.recorded_forces.shape[1] == 4

    driver.begin_frame()
    assert driver.recorder.recorded_forces.shape[1] == 0


def test_latest_forces_is_the_most_recent_sample():
    driver = _Driver(4)
    assert torch.equal(driver.recorder.latest_forces, _frame(0))  # zeros before the first record

    for substep in range(6):
        driver.step(substep + 1)
        assert torch.equal(driver.recorder.latest_forces, _frame(substep + 1))


def test_buffer_is_not_reallocated():
    """The live harness binds the buffer once, so it has to keep its identity."""
    driver = _Driver(4)
    ptr = driver.recorder.buffer.data_ptr()
    driver.step(1)
    assert driver.recorder.buffer.data_ptr() == ptr
