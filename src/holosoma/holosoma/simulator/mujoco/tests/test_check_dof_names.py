"""``check_dof_names``: the robot config must name exactly the model's DOF joints, in any order."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from holosoma.simulator.mujoco.robot_addressing import check_dof_names

pytestmark = pytest.mark.no_sim

_MODEL = ["j0", "j1", "j2"]


def test_reordered_config_is_accepted() -> None:
    check_dof_names(["j2", "j0", "j1"], _MODEL)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (["j0", "j1"], r"Missing from the config: \['j2'\]"),
        (["j0", "j1", "j2", "j3"], r"Missing from the model: \['j3'\]"),
        (["j0", "j1", "j1"], "duplicate"),
    ],
)
def test_mismatched_config_raises(config: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_dof_names(config, _MODEL)
