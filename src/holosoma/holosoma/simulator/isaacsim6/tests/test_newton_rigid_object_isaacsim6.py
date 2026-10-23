"""Resolving which Newton articulation owns a rigid body's prim.

Why this is a unit test
-----------------------
``_owning_articulation_expression`` is pure string arithmetic over Newton's label lists, and it is
the fix for a real mismatch: ``isaaclab_newton``'s ``RigidObject`` resolves the prim carrying
``UsdPhysics.RigidBodyAPI`` and then queries the *articulation* namespace with it. For holosoma's
converted assets those differ -- ``.../freebox/Geometry/baseLink`` is a body label while
``.../freebox`` is the articulation label -- so the lookup raised ``KeyError`` even though the
model was complete.

The first implementation inferred one global trim depth from the label set and got it wrong,
because assets sit at different depths in the same scene: the robot's articulation root is
``Robot/Geometry/pelvis``, four segments deeper than ``freebox``. That is the case
``test_trim_is_measured_per_asset`` pins, and it needs no simulator to express.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.isaacsim6.newton_rigid_object import _owning_articulation_expression

pytestmark = pytest.mark.no_sim

_ENV = "/World/envs/env_0"
_EXPR_NS = "/World/envs/env_[^/]+"


class _Model:
    """Just the ``articulation_label`` list the resolver reads."""

    def __init__(self, *labels: str) -> None:
        self.articulation_label = list(labels)


def test_body_below_the_articulation_root_is_trimmed_to_it() -> None:
    """The case that raised: the body prim is two segments below the articulation label."""
    model = _Model(f"{_ENV}/freebox", f"{_ENV}/pillar")
    result = _owning_articulation_expression(
        model, f"{_ENV}/freebox/Geometry/baseLink", f"{_EXPR_NS}/freebox/Geometry/baseLink"
    )
    assert result == f"{_EXPR_NS}/freebox"


def test_a_body_that_is_already_the_articulation_root_is_left_alone() -> None:
    """Isaac Lab's own primitive assets put ``RigidBodyAPI`` on the configured prim.

    That is why upstream never hits the mismatch, and this lane must not "fix" it for them --
    trimming here would address a non-existent parent.
    """
    model = _Model("/World/Cube")
    assert _owning_articulation_expression(model, "/World/Cube", "/World/Cube") == "/World/Cube"


def test_trim_is_measured_per_asset_not_from_a_global_depth() -> None:
    """Assets sit at different depths in one scene, so a single inferred depth mis-trims.

    With the robot's articulation root four segments deeper than ``freebox``, a global-depth rule
    produced ``.../freebox/Geometry`` -- an articulation that does not exist -- and raised a
    second, differently-worded KeyError.
    """
    model = _Model(f"{_ENV}/Robot/Geometry/pelvis", f"{_ENV}/freebox")
    assert (
        _owning_articulation_expression(
            model, f"{_ENV}/freebox/Geometry/baseLink", f"{_EXPR_NS}/freebox/Geometry/baseLink"
        )
        == f"{_EXPR_NS}/freebox"
    )
    assert (
        _owning_articulation_expression(model, f"{_ENV}/Robot/Geometry/pelvis", f"{_EXPR_NS}/Robot/Geometry/pelvis")
        == f"{_EXPR_NS}/Robot/Geometry/pelvis"
    )


def test_the_longest_matching_ancestor_wins() -> None:
    """Nested articulations must resolve to the nearest one, not the outermost."""
    model = _Model(f"{_ENV}", f"{_ENV}/freebox")
    result = _owning_articulation_expression(
        model, f"{_ENV}/freebox/Geometry/baseLink", f"{_EXPR_NS}/freebox/Geometry/baseLink"
    )
    assert result == f"{_EXPR_NS}/freebox"


def test_a_name_prefix_is_not_an_ancestor() -> None:
    """``/freebox`` must not be treated as the owner of ``/freebox_lid/...``.

    A plain ``str.startswith`` would match, and the view would then be built over the wrong
    object -- reading and randomizing a different rigid body with no error anywhere.
    """
    model = _Model(f"{_ENV}/freebox")
    expr = f"{_EXPR_NS}/freebox_lid/Geometry/baseLink"
    assert _owning_articulation_expression(model, f"{_ENV}/freebox_lid/Geometry/baseLink", expr) == expr


def test_no_matching_articulation_returns_the_expression_unchanged() -> None:
    """Leave Isaac Lab to raise its own error rather than inventing a different one.

    If nothing owns the body, this resolver has no better answer than upstream's, and a
    differently-worded failure from here would send the reader to the wrong module.
    """
    model = _Model("/World/envs/env_0/other")
    expr = f"{_EXPR_NS}/freebox/Geometry/baseLink"
    assert _owning_articulation_expression(model, f"{_ENV}/freebox/Geometry/baseLink", expr) == expr


def test_an_empty_label_list_returns_the_expression_unchanged() -> None:
    """A model with no articulations yet must not crash the resolver."""
    expr = f"{_EXPR_NS}/freebox/Geometry/baseLink"
    assert _owning_articulation_expression(_Model(), f"{_ENV}/freebox/Geometry/baseLink", expr) == expr
