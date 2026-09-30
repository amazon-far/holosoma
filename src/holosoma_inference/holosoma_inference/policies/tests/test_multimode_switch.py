"""Unit tests for MultiModePolicy switching logic.

These drive the switch/ring/select logic with FAKE policy instances — no rclpy,
no hardware, no real InferenceConfig — so they run anywhere (host or bazel). They
cover: initial activation, ring advance (switch_mode), select-by-name,
select-by-index, unknown/duplicate handling, and that a switch stops the old
policy and leaves the new one STOPPED (no auto-start — explicit start required, D5).
"""

from __future__ import annotations

import pytest

from holosoma_inference.policies.dual_mode import MultiModePolicy


class _FakeProvider:
    """Command provider stub with the ``_mapping`` MultiModePolicy patches."""

    def __init__(self):
        self._mapping = {}


class _FakePolicy:
    """Records the lifecycle calls MultiModePolicy makes during a switch."""

    def __init__(self, name):
        self.name = name
        self._command_provider = _FakeProvider()
        self._velocity_input = object()  # not an InterfaceInput -> key_state copy skipped
        self.calls: list[str] = []
        self._orig_dispatch_calls: list = []
        # Run-status flag the switch guard reads; start/stop flip it like the real policy.
        self.use_policy_action = False

    # lifecycle hooks exercised by _activate
    def _handle_stop_policy(self):
        self.calls.append("stop")
        self.use_policy_action = False

    def _resolve_control_gains(self):
        self.calls.append("gains")

    def _init_phase_components(self):
        self.calls.append("phase")

    def _handle_start_policy(self):
        self.calls.append("start")
        self.use_policy_action = True

    # the original dispatch MultiModePolicy captures + wraps
    def _dispatch_command(self, cmd):
        self._orig_dispatch_calls.append(cmd)


def _build(n=3):
    policies = [_FakePolicy(f"p{i}") for i in range(n)]
    names = [f"mode{i}" for i in range(n)]
    switches: list[tuple[int, str]] = []
    mm = MultiModePolicy(policies, names, on_switch=lambda i, nm: switches.append((i, nm)))
    return mm, policies, names, switches


def test_initial_active_and_on_switch_fires():
    mm, policies, names, switches = _build()
    assert mm.active is policies[0]
    assert mm.active_label == "mode0"
    assert mm.active_index == 0
    assert switches == [(0, "mode0")]  # on_switch fired for initial activation


def test_switch_to_next_rings():
    mm, policies, names, switches = _build(3)
    mm.switch_to_next()
    assert mm.active_index == 1 and mm.active_label == "mode1"
    mm.switch_to_next()
    assert mm.active_index == 2
    mm.switch_to_next()  # wraps
    assert mm.active_index == 0
    assert switches[-1] == (0, "mode0")


def test_switch_stops_old_and_leaves_new_stopped_not_started():
    # D5: a switch never auto-starts the target. The outgoing policy is stopped;
    # the target gets gains + phase re-init then is left STOPPED (explicit start
    # required). "start" must NOT appear.
    mm, policies, names, _ = _build(2)
    mm.switch_to_next()  # 0 -> 1
    assert policies[0].calls == ["stop"]  # old policy stopped
    assert policies[1].calls == ["gains", "phase", "stop"]  # activated but NOT started
    assert "start" not in policies[1].calls


def test_select_by_name():
    mm, policies, names, _ = _build(3)
    assert mm.select("mode2") is True
    assert mm.active_index == 2 and mm.active_label == "mode2"


def test_select_by_index():
    mm, policies, names, _ = _build(3)
    assert mm.select(1) is True
    assert mm.active_index == 1


def test_select_unknown_name_is_noop():
    mm, policies, names, _ = _build(3)
    assert mm.select("does-not-exist") is False
    assert mm.active_index == 0  # unchanged


def test_select_out_of_range_index_is_noop():
    mm, policies, names, _ = _build(3)
    assert mm.select(99) is False
    assert mm.active_index == 0


def test_select_already_active_is_noop():
    mm, policies, names, switches = _build(3)
    n_before = len(switches)
    assert mm.select("mode0") is False  # already active
    assert len(switches) == n_before  # no extra on_switch


def test_select_refused_while_active_policy_running():
    # Safety guard: cannot switch out from under a running policy — stop first.
    mm, policies, names, _ = _build(3)
    policies[0].use_policy_action = True  # active policy is running
    assert mm.select("mode1") is False
    assert mm.active_index == 0  # unchanged


def test_select_force_overrides_running_guard():
    mm, policies, names, _ = _build(3)
    policies[0].use_policy_action = True
    assert mm.select("mode1", force=True) is True
    assert mm.active_index == 1


def test_switch_to_next_refused_while_running():
    # The switch_mode toggle routes through select, so it is guarded too.
    mm, policies, names, _ = _build(3)
    policies[0].use_policy_action = True
    mm.switch_to_next()
    assert mm.active_index == 0  # refused


def test_select_allowed_after_stop():
    mm, policies, names, _ = _build(3)
    policies[0].use_policy_action = True
    policies[0]._handle_stop_policy()  # operator stops first
    assert mm.select("mode1") is True
    assert mm.active_index == 1


def test_switch_mode_command_routes_through_patched_dispatch():
    from holosoma_inference.inputs.api.commands import StateCommand

    mm, policies, names, _ = _build(2)
    # The active policy's dispatch was patched; sending SWITCH_MODE advances ring.
    policies[0]._dispatch_command(StateCommand.SWITCH_MODE)
    assert mm.active_index == 1


def test_non_switch_command_goes_to_active_original_dispatch():
    from holosoma_inference.inputs.api.commands import StateCommand

    mm, policies, names, _ = _build(2)
    policies[0]._dispatch_command(StateCommand.START)
    # routed to policy 0's ORIGINAL dispatch (captured before patching)
    assert policies[0]._orig_dispatch_calls == [StateCommand.START]


def test_keyboard_mapping_gets_switch_mode():
    from holosoma_inference.inputs.api.commands import StateCommand

    mm, policies, names, _ = _build(2)
    for p in policies:
        assert p._command_provider._mapping["x"] == StateCommand.SWITCH_MODE
        assert p._command_provider._mapping["X"] == StateCommand.SWITCH_MODE


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        MultiModePolicy([_FakePolicy("a"), _FakePolicy("b")], names=["dup", "dup"])


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        MultiModePolicy([_FakePolicy("a")], names=["one", "two"])


def test_empty_rejected():
    with pytest.raises(ValueError, match="at least one"):
        MultiModePolicy([], names=[])
