"""The shipped g1/t1 loco termination and mass-DR presets name robot-specific link sets by tag.

Any robot that declares ``root``, ``hip_links``, ``knee_links`` (and ``shoulder_links`` for g1's termination,
``ankle_links`` + ``feet`` for t1's DR) can reuse them via ``replace(<experiment>, robot=...)``; the resolved
bodies equal the literal lists the presets carried before.
"""

from __future__ import annotations

import pytest

from holosoma.config_values.loco.g1.randomization import g1_29dof_randomization
from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.t1.randomization import t1_29dof_randomization
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.config_values.robot import g1_29dof, t1_29dof_waist_wrist
from holosoma.utils.name_tags import expand_name_spec

pytestmark = pytest.mark.no_sim

_G1_HIPS = [f"{side}_hip_{axis}_link" for side in ("left", "right") for axis in ("pitch", "roll", "yaw")]
_G1_SHOULDERS = [f"{side}_shoulder_{axis}_link" for side in ("left", "right") for axis in ("pitch", "roll", "yaw")]
_T1_HIPS = [f"Hip_{axis}_{side}" for side in ("Left", "Right") for axis in ("Pitch", "Roll", "Yaw")]


def _bodies(spec: list[str], robot) -> list[str]:  # type: ignore[no-untyped-def]
    return expand_name_spec(spec, robot.body_tags, robot.body_names, kind="body")


def test_g1_termination_bodies_resolve_to_the_pre_tag_list() -> None:
    spec = g1_29dof_termination.terms["contact"].params["bodies"]
    assert spec == ["#root", "#shoulder_links", "#hip_links"]
    assert _bodies(spec, g1_29dof) == ["pelvis", *_G1_SHOULDERS, *_G1_HIPS]


def test_g1_mass_dr_links_resolve_to_the_pre_tag_set() -> None:
    spec = g1_29dof_randomization.setup_terms["mass_randomizer"].params["link_bodies"]
    assert spec == ["#root", "#hip_links", "#knee_links"]
    assert set(_bodies(spec, g1_29dof)) == {"pelvis", *_G1_HIPS, "left_knee_link", "right_knee_link"}


def test_t1_termination_bodies_resolve_to_the_pre_tag_set() -> None:
    spec = t1_29dof_termination.terms["contact"].params["bodies"]
    assert set(_bodies(spec, t1_29dof_waist_wrist)) == {"Trunk", "Waist", *_T1_HIPS}


def test_t1_mass_dr_links_resolve_to_the_pre_tag_set() -> None:
    spec = t1_29dof_randomization.setup_terms["mass_randomizer"].params["link_bodies"]
    assert set(_bodies(spec, t1_29dof_waist_wrist)) == {
        "Trunk",
        *_T1_HIPS,
        "Shank_Left",
        "Shank_Right",
        "Ankle_Cross_Left",
        "Ankle_Cross_Right",
        "left_foot_link",
        "right_foot_link",
    }


def test_root_tag_is_the_first_body() -> None:
    for robot in (g1_29dof, t1_29dof_waist_wrist):
        assert robot.body_tags["#root"] == [robot.body_names[0]]
