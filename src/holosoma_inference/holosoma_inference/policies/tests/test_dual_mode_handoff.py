"""Command continuity at Loco -> WBT handoff, without DDS or robot access."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.dual_mode import DualModePolicy


class TestWbtHandoff(unittest.TestCase):
    def make_dual(self, sent_q, offsets=None, duration_steps=50):
        n = 19
        measured = np.linspace(-0.15, 0.15, n)
        state = np.zeros((1, 51))
        state[0, 7:26] = measured
        primary = SimpleNamespace(
            _last_sent_q=sent_q,
            interface=SimpleNamespace(get_low_state=Mock(return_value=state)),
            _cancel_policy_interp=Mock(),
        )
        secondary = object.__new__(BasePolicy)
        secondary.num_dofs = n
        secondary.joint_offsets = np.zeros(n) if offsets is None else offsets
        secondary._last_sent_q = np.full(n, 9.0)  # Stale previous dance command.
        secondary._target_transition = None

        def start_motion():
            secondary._begin_target_transition("policy_start", duration_steps)

        secondary._handle_start_motion_clip = Mock(side_effect=start_motion)
        dual = object.__new__(DualModePolicy)
        dual.primary = primary
        dual.secondary = secondary
        dual.active = primary
        dual.active_label = "primary"
        dual._num_dofs = n

        def activate(target, label):
            dual.active = target
            dual.active_label = label

        dual._activate_target = Mock(side_effect=activate)
        return dual, state

    def test_first_cycle_retains_outgoing_pd_support_error(self):
        sent = np.linspace(-0.3, 0.6, 19)
        dual, state = self.make_dual(sent)
        dual._switch_to_wbt(auto_start_motion=True)
        target = np.linspace(0.2, -0.2, 19)[None]
        blended = dual.secondary._apply_target_transition(target, state)
        a = (1/50)**2 * (3 - 2/50)
        np.testing.assert_allclose(blended, sent[None] + a*(target-sent[None]))
        # With matching PD gains and dq/tau commands, the torque change must
        # be only the deliberately blended target change, not a support drop.
        kp = np.full(19, 200.0)
        old_tau = kp*(sent-state[0, 7:26])
        new_tau = kp*(blended[0]-state[0, 7:26])
        np.testing.assert_allclose(new_tau-old_tau, kp*a*(target[0]-sent), atol=1e-12)
        self.assertIs(dual.active, dual.secondary)

    def test_incoming_offsets_are_not_applied_twice(self):
        sent = np.linspace(-0.3, 0.6, 19)
        offsets = np.full(19, 0.03)
        dual, state = self.make_dual(sent, offsets)
        dual._switch_to_wbt(auto_start_motion=True)
        np.testing.assert_allclose(dual.secondary._target_transition["start_q"] + offsets, sent[None])

    def test_outgoing_command_snapshot_does_not_alias_primary_buffer(self):
        sent = np.linspace(-0.3, 0.6, 19)
        expected = sent.copy()
        dual, _ = self.make_dual(sent)
        dual._switch_to_wbt(auto_start_motion=True)
        sent[:] = 8.0
        np.testing.assert_allclose(dual.secondary._target_transition["start_q"], expected[None])

    def test_missing_outgoing_command_falls_back_to_current_pose(self):
        dual, state = self.make_dual(None)
        dual._switch_to_wbt(auto_start_motion=True)
        np.testing.assert_array_equal(dual.secondary._target_transition["start_q"], state[:, 7:26])

    def test_zero_duration_keeps_immediate_switch_behavior(self):
        dual, _ = self.make_dual(np.zeros(19), duration_steps=0)
        dual._switch_to_wbt(auto_start_motion=True)
        self.assertIsNone(dual.secondary._target_transition)


if __name__ == "__main__":
    unittest.main()
