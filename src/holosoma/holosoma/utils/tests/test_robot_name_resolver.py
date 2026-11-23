"""RobotNameResolver owns canonical indices, per-spec caches, and returned name lists."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from holosoma.utils.robot_name_resolver import RobotNameResolver
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig

pytestmark = pytest.mark.no_sim

BODY_NAMES = ["pelvis", "left_hip_link", "right_hip_link", "left_foot", "right_foot", "torso"]
DOF_NAMES = ["left_hip", "left_knee", "right_hip", "right_knee", "waist"]

BODY_TAGS = {
    "#feet": ["left_foot", "right_foot"],
    "#hips": ["left_hip_link", "right_hip_link"],
    "#legs": ["#hips", "#feet"],
}
JOINT_TAGS = {
    "#left_leg": ["left_hip", "left_knee"],
    "#right_leg": ["right_hip", "right_knee"],
    "#legs": ["#left_leg", "#right_leg"],
}


def _resolver(
    body_tags: dict[str, list[str]] | None = None, joint_tags: dict[str, list[str]] | None = None
) -> RobotNameResolver:
    robot = SimpleNamespace(
        body_names=BODY_NAMES,
        dof_names=DOF_NAMES,
        body_tags=BODY_TAGS if body_tags is None else body_tags,
        joint_tags=JOINT_TAGS if joint_tags is None else joint_tags,
    )
    return RobotNameResolver(cast("RobotConfig", robot), "cpu")


def test_resolve_body_indices_use_canonical_observation_order() -> None:
    resolver = _resolver()

    idx = resolver.resolve_body_indices(["pelvis", "#feet"])

    assert idx.dtype == torch.long
    assert idx.device.type == "cpu"
    assert idx.tolist() == [BODY_NAMES.index(n) for n in ("pelvis", "left_foot", "right_foot")]


def test_resolve_body_indices_cache_hit_returns_same_tensor() -> None:
    resolver = _resolver()

    first = resolver.resolve_body_indices(["#feet"])
    second = resolver.resolve_body_indices(("#feet",))

    assert second is first


def test_resolve_body_names_composes_tags_and_dedupes() -> None:
    resolver = _resolver()

    names = resolver.resolve_body_names(["#legs", "left_foot", "pelvis"])

    assert names == ["left_hip_link", "right_hip_link", "left_foot", "right_foot", "pelvis"]


def test_resolve_body_names_returns_a_copy() -> None:
    resolver = _resolver()

    resolver.resolve_body_names(["#feet"]).append("torso")

    assert resolver.resolve_body_names(["#feet"]) == ["left_foot", "right_foot"]


def test_resolve_joint_indices_are_positions_in_dof_names() -> None:
    resolver = _resolver()

    idx = resolver.resolve_joint_indices(["#legs", "waist"])

    assert idx.dtype == torch.long
    assert idx.tolist() == [0, 1, 2, 3, 4]
    assert resolver.resolve_joint_indices(["#legs", "waist"]) is idx


def test_body_and_joint_caches_do_not_collide() -> None:
    resolver = _resolver(body_tags={"#left": ["left_foot"]}, joint_tags={"#left": ["left_hip", "left_knee"]})

    assert resolver.resolve_body_indices(["#left"]).tolist() == [BODY_NAMES.index("left_foot")]
    assert resolver.resolve_joint_indices(["#left"]).tolist() == [0, 1]


def test_empty_spec_gives_empty_long_tensor() -> None:
    resolver = _resolver()

    assert resolver.resolve_body_indices([]).shape == (0,)
    assert resolver.resolve_body_indices([]).dtype == torch.long
    assert resolver.resolve_joint_indices([]).shape == (0,)
