"""Unit tests for InferenceConfig multi-policy registry helpers.

``registered_policies`` / ``resolved_names`` are pure (no rclpy / onnx), so these
run anywhere. They use lightweight stand-ins for the nested configs — the helpers
only touch ``secondary`` / ``policies`` / ``name`` / ``task.policy_type``, never the
full robot/observation schema — built via ``object.__new__`` to skip pydantic
validation of unrelated required fields.
"""

from __future__ import annotations

import pytest

from holosoma_inference.config.config_types.inference import InferenceConfig


class _Task:
    def __init__(self, policy_type=""):
        self.policy_type = policy_type


def _cfg(name="", policy_type="", secondary=None, policies=()):
    """Build an InferenceConfig without running pydantic __init__ (only the
    registry-relevant fields are set)."""
    c = object.__new__(InferenceConfig)
    object.__setattr__(c, "name", name)
    object.__setattr__(c, "task", _Task(policy_type))
    object.__setattr__(c, "secondary", secondary)
    object.__setattr__(c, "policies", tuple(policies))
    return c


# --- registered_policies ------------------------------------------------------


def test_single_policy_is_just_self():
    c = _cfg(name="loco")
    assert c.registered_policies() == [c]


def test_secondary_normalized_to_list():
    sec = _cfg(name="wbt")
    c = _cfg(name="loco", secondary=sec)
    assert c.registered_policies() == [c, sec]


def test_policies_tuple_prepends_owner():
    p1, p2 = _cfg(name="wbt"), _cfg(name="stair")
    c = _cfg(name="loco", policies=(p1, p2))
    assert c.registered_policies() == [c, p1, p2]


def test_secondary_and_policies_both_set_rejected():
    c = _cfg(name="loco", secondary=_cfg(name="wbt"), policies=(_cfg(name="stair"),))
    with pytest.raises(ValueError, match="not both"):
        c.registered_policies()


# --- resolved_names -----------------------------------------------------------


def test_names_from_explicit_name():
    c = _cfg(name="loco", policies=(_cfg(name="contextual_loco"),))
    assert c.resolved_names() == ["loco", "contextual_loco"]


def test_names_fall_back_to_policy_type():
    c = _cfg(name="", policy_type="locomotion", policies=(_cfg(name="", policy_type="depth_distillation"),))
    assert c.resolved_names() == ["locomotion", "depth_distillation"]


def test_names_fall_back_to_policy_literal():
    c = _cfg(name="", policy_type="")
    assert c.resolved_names() == ["policy"]


def test_duplicate_names_deduped_with_suffix():
    # Two WBT policies both resolving to "wbt" -> wbt, wbt#2.
    c = _cfg(name="wbt", policies=(_cfg(name="wbt"), _cfg(name="wbt")))
    assert c.resolved_names() == ["wbt", "wbt#2", "wbt#3"]


def test_names_order_matches_registered_policies():
    c = _cfg(name="loco", policies=(_cfg(name="wbt"), _cfg(name="stair")))
    assert c.resolved_names() == ["loco", "wbt", "stair"]
    assert len(c.resolved_names()) == len(c.registered_policies())
