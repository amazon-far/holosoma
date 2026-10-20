"""ObjectRegistry ``is_static`` flag: band defaults, the fixed-base robot, and ``static_names`` order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from holosoma.simulator.shared.object_registry import ObjectRegistry, ObjectType

pytestmark = pytest.mark.no_sim


def _pose(num_envs: int) -> torch.Tensor:
    pose = torch.zeros(num_envs, 7)
    pose[:, 6] = 1.0  # qw
    return pose


def _registry(*, robot_is_static: bool | None, num_envs: int = 2) -> ObjectRegistry:
    """robot @ ROBOT/0, wall + table @ SCENE/0,1, box @ INDIVIDUAL/0; registered in that order."""
    reg = ObjectRegistry(device="cpu")
    reg.setup_ranges(num_envs, robot_count=1, scene_count=2, individual_count=1)
    reg.register_object("robot", ObjectType.ROBOT, 0, _pose(num_envs), is_static=robot_is_static)
    reg.register_object("wall", ObjectType.SCENE, 0, _pose(num_envs))
    reg.register_object("table", ObjectType.SCENE, 1, _pose(num_envs))
    reg.register_object("box", ObjectType.INDIVIDUAL, 0, _pose(num_envs))
    reg.finalize_registration()
    return reg


def test_is_static_defaults_by_band() -> None:
    reg = _registry(robot_is_static=None)
    flags = {obj.name: obj.is_static for obj in reg.objects}
    assert flags == {"robot": False, "wall": True, "table": True, "box": False}


def test_robot_registered_static_when_asked_keeps_band_and_slot() -> None:
    reg = _registry(robot_is_static=True)
    robot = reg.objects[reg.name_to_index["robot"]]
    assert robot.is_static is True
    assert (robot.type, robot.position_in_type) == (ObjectType.ROBOT.value, 0)
    # Band/index computation is untouched: the robot still leads every env block.
    assert reg.get_object_indices("robot", torch.tensor([0, 1])).tolist() == [0, 4]
    assert reg.get_names_by_type(ObjectType.ROBOT) == ["robot"]
    assert reg.get_names_by_type(ObjectType.SCENE) == ["wall", "table"]


def test_explicit_is_static_overrides_band_default() -> None:
    reg = ObjectRegistry(device="cpu")
    reg.setup_ranges(1, robot_count=1, scene_count=1, individual_count=0)
    reg.register_object("robot", ObjectType.ROBOT, 0, _pose(1), is_static=False)
    reg.register_object("wall", ObjectType.SCENE, 0, _pose(1), is_static=False)
    reg.finalize_registration()
    assert reg.static_names() == []


@pytest.mark.parametrize(
    ("robot_is_static", "expected"),
    [(None, ["wall", "table"]), (False, ["wall", "table"]), (True, ["robot", "wall", "table"])],
)
def test_static_names_registration_order(robot_is_static: bool | None, expected: list[str]) -> None:
    assert _registry(robot_is_static=robot_is_static).static_names() == expected


def test_finalize_preserves_is_static() -> None:
    reg = _registry(robot_is_static=True)
    # finalize_registration rebuilds each record via _replace(indices=...); the flag must survive.
    assert [obj.is_static for obj in reg.objects] == [True, True, True, False]
    assert reg.objects[0].indices.tolist() == [0, 1]
