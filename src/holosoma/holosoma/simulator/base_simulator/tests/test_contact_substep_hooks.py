"""Unit test for BaseSimulator's contact-substep plumbing (no simulator).

Pins the one piece that is easy to break by refactor and invisible when broken: the FRAME_BEGIN
hook re-anchoring the slot index, so slot 0 is always the first substep of the control step.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.contact_substep import ContactSubstepRecorder
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def test_frame_begin_hook_restarts_the_slot_index():
    # Only the contact-substep state is set; a full BaseSimulator needs a whole tyro config.
    sim = BaseSimulator.__new__(BaseSimulator)
    sim.contact_recorder = ContactSubstepRecorder(1, 4, 2, device="cpu")
    hooks = HookRegistry()
    hooks.add(Phase.FRAME_BEGIN, sim._begin_contact_substep, name="contact.begin_frame")

    sim.record_contact_substep(torch.full((1, 2, 3), 1.0))
    sim.record_contact_substep(torch.full((1, 2, 3), 2.0))
    hooks.emit(Phase.FRAME_BEGIN)
    sim.record_contact_substep(torch.full((1, 2, 3), 3.0))

    assert torch.equal(sim.contact_forces_substep[:, 0], torch.full((1, 2, 3), 3.0))
    assert torch.equal(sim.contact_forces_substep[:, 1], torch.full((1, 2, 3), 2.0))
