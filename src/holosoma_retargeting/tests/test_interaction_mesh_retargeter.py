"""Unit tests for retargeting constraint activation."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
from holosoma_retargeting.config_types.retargeter import FootLockConfig
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


def _retargeter_with_non_penetration(enabled: bool) -> InteractionMeshRetargeter:
    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter.activate_obj_non_penetration = enabled
    retargeter.q_a_indices = np.array([0, 1], dtype=int)
    retargeter.penetration_tolerance = 1e-3
    return retargeter


def test_environment_non_penetration_constraints_are_disabled_by_config() -> None:
    retargeter = _retargeter_with_non_penetration(False)

    constraints = retargeter._environment_non_penetration_constraints(
        cp.Variable(2),
        {"ground-contact": np.zeros(2)},
        {"ground-contact": -0.1},
    )

    assert constraints == []


def test_environment_non_penetration_constraints_are_enabled_by_default() -> None:
    retargeter = _retargeter_with_non_penetration(True)

    constraints = retargeter._environment_non_penetration_constraints(
        cp.Variable(2),
        {"ground-contact": np.zeros(2)},
        {"ground-contact": -0.1},
    )

    assert len(constraints) == 1


def test_foot_lock_windows_return_per_window_floor_height() -> None:
    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter._init_foot_lock(
        FootLockConfig(
            enable=True,
            windows={
                "L_Toe": [(10, 20, 0.12), (30, 40)],
                "R_Toe": [(15, 25, 0.56)],
            },
            z_floor=0.34,
        )
    )

    assert retargeter._is_foot_locked_in_window("left_ankle_link", 15) == pytest.approx(0.12)
    assert retargeter._is_foot_locked_in_window("left_ankle_link", 35) == pytest.approx(0.34)
    assert retargeter._is_foot_locked_in_window("right_ankle_link", 20) == pytest.approx(0.56)
    assert retargeter._is_foot_locked_in_window("left_ankle_link", 25) is None
    assert retargeter._is_foot_locked_in_window("torso_link", 15) is None
