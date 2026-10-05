"""Unit tests for the ``UndesiredContacts`` WBT reward term (no sim).

The term is the only consumer of ``contact_recorder.recorded_forces``, so a stub simulator holding
a real recorder exercises the full term -> buffer path without a backend. Pins:
- the max over substeps: a contact present in one substep and gone by the last still counts;
- the count is over selected bodies, not substeps or contacts;
- bodies outside the name pattern are ignored;
- the threshold is strict.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.reward.terms.wbt import UndesiredContacts
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.contact_substep import ContactSubstepRecorder
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

BODY_NAMES = ["pelvis", "left_knee", "right_knee"]
DECIMATION = 4


class _ContactSim:
    """Stub simulator: body names plus a real recorder, driven through its own hooks."""

    def __init__(self, num_envs: int = 1):
        self.body_names = BODY_NAMES
        self.hooks = HookRegistry()
        self.contact_forces = torch.zeros(num_envs, len(BODY_NAMES), 3)
        self.contact_recorder = ContactSubstepRecorder(
            self.hooks, lambda: self.contact_forces, num_envs, DECIMATION, len(BODY_NAMES), "cpu"
        )

    def control_step(self, substep_frames) -> None:
        self.hooks.emit(Phase.FRAME_BEGIN)
        for frame in substep_frames:
            self.contact_forces[:] = frame
            self.hooks.emit(Phase.POST_STEP)


class _Env:
    def __init__(self, simulator: _ContactSim):
        self.simulator = simulator
        self.device = "cpu"


def _term(env: _Env, pattern: str, threshold: float = 1.0) -> UndesiredContacts:
    cfg = RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
        params={"undesired_contacts_body_names": pattern, "threshold": threshold},
    )
    # The term only reaches env.device and env.simulator, so the stub satisfies it structurally.
    return UndesiredContacts(cfg, env)  # type: ignore[arg-type]


def _frame(num_envs: int = 1, **body_force):
    """A [num_envs, num_bodies, 3] frame with a vertical force on each named body."""
    frame = torch.zeros(num_envs, len(BODY_NAMES), 3)
    for name, force in body_force.items():
        frame[:, BODY_NAMES.index(name), 2] = force
    return frame


def test_contact_in_an_early_substep_still_counts():
    """Reading only the last substep would miss a touch that ends before the control step does."""
    sim = _ContactSim()
    env = _Env(sim)
    term = _term(env, "left_knee")
    sim.control_step([_frame(left_knee=2.0), _frame(), _frame(), _frame()])

    assert term(env).tolist() == [1]


def test_repeated_contact_on_one_body_counts_once():
    sim = _ContactSim()
    env = _Env(sim)
    term = _term(env, "left_knee")
    sim.control_step([_frame(left_knee=2.0)] * DECIMATION)

    assert term(env).tolist() == [1]


def test_each_matching_body_in_contact_adds_one():
    sim = _ContactSim()
    env = _Env(sim)
    term = _term(env, ".*_knee")
    sim.control_step([_frame(left_knee=2.0, right_knee=3.0)] + [_frame()] * 3)

    assert term(env).tolist() == [2]


def test_bodies_outside_the_pattern_are_ignored():
    sim = _ContactSim()
    env = _Env(sim)
    term = _term(env, ".*_knee")
    sim.control_step([_frame(pelvis=500.0)] * DECIMATION)

    assert term(env).tolist() == [0]


def test_threshold_is_strict():
    sim = _ContactSim()
    env = _Env(sim)
    term = _term(env, "left_knee", threshold=1.0)
    sim.control_step([_frame(left_knee=1.0)] * DECIMATION)

    assert term(env).tolist() == [0]


def test_envs_are_counted_independently():
    sim = _ContactSim(num_envs=2)
    env = _Env(sim)
    term = _term(env, ".*_knee")
    frame = _frame(num_envs=2, left_knee=2.0)
    frame[1] = 0.0
    sim.control_step([frame] + [_frame(num_envs=2)] * 3)

    assert term(env).tolist() == [1, 0]
