from unittest.mock import Mock

from holosoma.simulator.isaacsim.isaacsim import IsaacSim


def _simulator(*, body_names, body_ids, matches):
    simulator = object.__new__(IsaacSim)
    simulator.body_names = body_names
    simulator.body_ids = body_ids
    simulator._robot = Mock()
    simulator._robot.find_bodies.return_value = matches
    return simulator


def test_exact_name_returns_tracked_body_index_without_regex_search() -> None:
    simulator = _simulator(
        body_names=["pelvis", "torso", "left_ankle"],
        body_ids=[2, 5, 9],
        matches=([5], ["torso"]),
    )

    assert simulator.find_rigid_body_indice("torso") == 1
    simulator._robot.find_bodies.assert_not_called()


def test_regex_fallback_filters_untracked_robot_bodies() -> None:
    simulator = _simulator(
        body_names=["pelvis", "tracked_ankle"],
        body_ids=[2, 9],
        matches=([4, 9], ["untracked_ankle", "tracked_ankle"]),
    )

    assert simulator.find_rigid_body_indice(".*ankle") == 1


def test_regex_fallback_maps_all_full_model_matches_to_tracked_indices() -> None:
    simulator = _simulator(
        body_names=["left_ankle", "right_ankle"],
        body_ids=[8, 12],
        matches=([12, 8], ["right_ankle", "left_ankle"]),
    )

    assert simulator.find_rigid_body_indice(".*ankle") == [1, 0]


def test_regex_fallback_returns_none_when_only_untracked_bodies_match() -> None:
    simulator = _simulator(
        body_names=["pelvis"],
        body_ids=[2],
        matches=([7], ["decorative_torso"]),
    )

    assert simulator.find_rigid_body_indice("torso") is None
