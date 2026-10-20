"""Pure tests for immutable action-config composition helpers."""

from __future__ import annotations

import pytest

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg, JointControllerCfg, replace_action_term

pytestmark = pytest.mark.no_sim


def _action() -> ActionManagerCfg:
    return ActionManagerCfg(
        terms={
            "base": ActionTermCfg(func="example:Base", scale=0.25, clip=(-1.0, 1.0)),
            "arm": ActionTermCfg(func="example:Arm", scale=0.5),
        }
    )


def test_replace_action_term_changes_only_the_named_term() -> None:
    action = _action()
    controller = JointControllerCfg(control_type="T", clip_torques=False)

    updated = replace_action_term(action, "arm", scale=0.75, clip=None, controller=controller)

    assert list(updated.terms) == ["base", "arm"]
    assert updated.terms["base"] is action.terms["base"]
    assert updated.terms["arm"].scale == 0.75
    assert updated.terms["arm"].clip is None
    assert updated.terms["arm"].controller is controller
    assert action.terms["arm"].scale == 0.5
    assert action.terms["arm"].controller is None


def test_replace_action_term_rejects_a_missing_term() -> None:
    with pytest.raises(KeyError, match=r"'missing'.*available terms.*base.*arm"):
        replace_action_term(_action(), "missing", scale=1.0)


def test_replace_action_term_rejects_unknown_term_fields() -> None:
    with pytest.raises(TypeError, match=r"Unknown ActionTermCfg field.*not_a_field"):
        replace_action_term(_action(), "arm", not_a_field=1)
