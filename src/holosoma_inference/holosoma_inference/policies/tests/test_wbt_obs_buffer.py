"""Invariants for ``WholeBodyTrackingPolicy.get_current_obs_buffer_dict``.

This override builds the per-tick observation buffer for the WBT policy.
The dense-tracker preset (``wbt-dense``) expects ``projected_gravity`` as
one of the actor_obs terms; a regression where the override doesn't
populate it surfaces at runtime as

    KeyError: "Observation term 'projected_gravity' missing from current
    observation buffer."

and the policy loop tears down — so guard that here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

NUM_DOFS = 29


def _bare_policy() -> WholeBodyTrackingPolicy:
    """Hand-populate just the attributes ``get_current_obs_buffer_dict`` reads."""
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.num_dofs = NUM_DOFS
    policy.default_dof_angles = np.zeros(NUM_DOFS)
    policy.last_policy_action = np.zeros((1, NUM_DOFS), dtype=np.float32)
    policy.motion_command_t = np.full((1, 58), 0.42, dtype=np.float32)
    # ref_quat_xyzw_t is consumed by the motion_ref_ori_b branch — give it
    # an identity so matrix_from_quat / subtract_frame_transforms return
    # well-defined values rather than NaNs.
    policy.ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    policy.motion_yaw_offset = 0.0
    policy.robot_yaw_offset = 0.0
    policy.config = SimpleNamespace(task=SimpleNamespace(debug=SimpleNamespace(force_upright_imu=False)))
    # Real ``_get_ref_body_orientation_in_world`` requires a pinocchio robot;
    # stub to a deterministic wxyz identity so the frame-subtract path works.
    policy._get_ref_body_orientation_in_world = lambda _state: np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    policy._remove_yaw_offset = lambda quat_wxyz, _offset: quat_wxyz
    return policy


def _fake_state_without_gravity() -> np.ndarray:
    """Robot state without the optional trailing projected_gravity (3) block.

    Shape: ``base_pos(3) + quat(4) + dof_pos(29) + base_lin_vel(3) + base_ang_vel(3) + dof_vel(29)``.
    """
    width = 7 + NUM_DOFS + 6 + NUM_DOFS
    state = np.zeros((1, width), dtype=np.float32)
    state[0, 3] = 1.0  # identity quat (wxyz -> w=1 so projected_gravity=[0,0,-1])
    return state


def _fake_state_with_gravity(gravity: tuple[float, float, float]) -> np.ndarray:
    """Robot state including the 3-element projected_gravity trailer."""
    base = _fake_state_without_gravity()
    extra = np.array([gravity], dtype=np.float32)
    return np.concatenate([base, extra], axis=1)


class TestProjectedGravityPopulated:
    """Regression: the dense preset fails hard if this key is missing."""

    def test_projected_gravity_key_present(self) -> None:
        policy = _bare_policy()
        buf = policy.get_current_obs_buffer_dict(_fake_state_without_gravity())
        assert "projected_gravity" in buf

    def test_projected_gravity_shape(self) -> None:
        policy = _bare_policy()
        buf = policy.get_current_obs_buffer_dict(_fake_state_without_gravity())
        assert buf["projected_gravity"].shape == (1, 3)

    def test_force_upright_imu_produces_minus_z(self) -> None:
        policy = _bare_policy()
        policy.config.task.debug.force_upright_imu = True
        buf = policy.get_current_obs_buffer_dict(_fake_state_without_gravity())
        np.testing.assert_allclose(buf["projected_gravity"], [[0.0, 0.0, -1.0]])

    def test_interface_provided_gravity_passthrough(self) -> None:
        policy = _bare_policy()
        gravity = (0.1, -0.2, -0.974679)
        buf = policy.get_current_obs_buffer_dict(_fake_state_with_gravity(gravity))
        np.testing.assert_allclose(buf["projected_gravity"], [list(gravity)], rtol=1e-6)

    def test_identity_quat_computes_minus_z(self) -> None:
        """Upright robot (identity orientation) → gravity along -z in body frame."""
        policy = _bare_policy()
        buf = policy.get_current_obs_buffer_dict(_fake_state_without_gravity())
        np.testing.assert_allclose(buf["projected_gravity"], [[0.0, 0.0, -1.0]], atol=1e-6)


class TestStockObsKeysStillPresent:
    """Belt-and-suspenders: the six original WBT obs terms must still populate."""

    def test_all_stock_terms_present(self) -> None:
        policy = _bare_policy()
        buf = policy.get_current_obs_buffer_dict(_fake_state_without_gravity())
        for key in (
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ):
            assert key in buf, f"stock WBT term {key!r} missing from obs buffer"
