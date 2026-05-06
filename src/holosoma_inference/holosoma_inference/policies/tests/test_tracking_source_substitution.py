"""Unit tests for SMPLH-to-motion_command_t substitution in WBT policy.

The full ``WholeBodyTrackingPolicy`` constructor loads ONNX models, the
SDK, a clock subscriber, and the policy's input handlers — far heavier
than this unit test needs. These tests exercise the
``_retarget_payload_to_motion_command`` helper and the
``rl_inference``-side branching in isolation by constructing the policy
with ``object.__new__`` and injecting only the attributes the covered
paths read.

Covers three cases per the service-mode contract:

1. Null tracking source → ``rl_inference`` calls the ONNX-clip path
   (``self.policy``) and keeps behavior byte-identical to today.
2. Custom source delivering a valid ``TrackingPayload`` → the retargeter
   produces ``motion_command_t`` and the policy conditions on it.
3. Custom source whose retargeter raises → the policy logs once and
   falls through to the ONNX-clip path; it does NOT crash, preserving
   the driver's sticky-fault contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from holosoma_inference.policies.tracking_source import NullTrackingSource, TrackingPayload
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


class _OneShotSource:
    """Serve a single payload, then ``None`` on subsequent polls."""

    def __init__(self, payload: TrackingPayload | None) -> None:
        self._payload = payload
        self._served = False

    def get_latest(self) -> TrackingPayload | None:
        if self._served or self._payload is None:
            return None
        self._served = True
        return self._payload


class _RaisingSource:
    """Source that yields a valid-looking payload every poll."""

    def __init__(self, payload: TrackingPayload) -> None:
        self._payload = payload

    def get_latest(self) -> TrackingPayload | None:
        return self._payload


def _make_payload() -> TrackingPayload:
    """24-joint SMPL body payload with non-degenerate quaternions."""
    transforms = np.zeros(24 * 7, dtype=np.float32)
    # Give each joint a identity quat (xyzw = 0,0,0,1) so the retargeter
    # does not immediately reject them as zero-norm.
    reshaped = transforms.reshape(24, 7)
    reshaped[:, 6] = 1.0  # qw
    return TrackingPayload(
        joint_names=[f"j{i}" for i in range(24)],
        joint_transforms=transforms,
        device_type="pico",
        tracking_quality=2,
        mode=1,
    )


def _bare_policy(tracking_source) -> WholeBodyTrackingPolicy:
    """Build a policy without running the real ``__init__``.

    The tests below exercise only the substitution helper and the
    relevant branch of ``rl_inference``; we hand-populate the attributes
    those code paths touch, and leave the rest unset.
    """
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._tracking_source = tracking_source
    policy._retargeter = None
    policy._retargeter_init_failed = False
    policy._retargeter_runtime_warned = False
    policy._retargeter_runtime_err_count = 0
    policy._retargeter_runtime_last_warn_count = 0
    policy._retargeter_runtime_rewarn_every = 500
    policy.num_dofs = 29
    policy.config = SimpleNamespace(
        robot=SimpleNamespace(urdf_path="/tmp/fake-g1.xml"),
        task=SimpleNamespace(rl_rate=50.0, motion_start_timestep=0, print_observations=False),
    )
    # Pre-seed motion_command_t so the ONNX-clip path has something to
    # echo when the retargeter does not fire.
    policy.motion_command_t = np.full((1, 58), 0.42, dtype=np.float32)
    policy.ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    policy.motion_clip_progressing = False
    policy.curr_motion_timestep = 0
    policy.timestep_util = MagicMock()
    policy.timestep_util.timestep = 0
    policy.last_policy_action = np.zeros((1, 29), dtype=np.float32)
    policy.per_joint_policy_action_scale = None
    policy.policy_action_scale = 1.0
    policy.scaled_policy_action = np.zeros((1, 29), dtype=np.float32)
    return policy


# ----------------------------------------------------------------------
# Tests: _retarget_payload_to_motion_command
# ----------------------------------------------------------------------


class TestRetargetHelper:
    def test_missing_urdf_returns_none(self) -> None:
        policy = _bare_policy(NullTrackingSource())
        policy.config.robot = SimpleNamespace(urdf_path=None)
        result = policy._retarget_payload_to_motion_command(_make_payload())
        assert result is None
        assert policy._retargeter_init_failed is True

    def test_retargeter_output_shape(self) -> None:
        """Retargeter output is repackaged into (motion_command, ref_quat_xyzw)."""
        policy = _bare_policy(NullTrackingSource())
        # Bypass lazy construction by planting a fake retargeter.
        fake_q = np.linspace(0.0, 1.0, 29, dtype=np.float32)
        fake_dq = np.linspace(1.0, 2.0, 29, dtype=np.float32)
        # Retargeter returns wxyz; helper converts to xyzw for the caller.
        fake_root_wxyz = np.array([0.7071, 0.0, 0.0, 0.7071], dtype=np.float32)
        fake_retargeter = MagicMock()
        fake_retargeter.retarget.return_value = (fake_q, fake_dq, fake_root_wxyz)
        policy._retargeter = fake_retargeter

        result = policy._retarget_payload_to_motion_command(_make_payload())
        assert result is not None
        motion_command, ref_quat_xyzw = result
        assert motion_command.shape == (1, 58)
        np.testing.assert_allclose(motion_command[0, :29], fake_q)
        np.testing.assert_allclose(motion_command[0, 29:], fake_dq)
        # xyzw ordering: (x, y, z, w). Input wxyz was (0.7071, 0, 0, 0.7071),
        # so expected xyzw is (0, 0, 0.7071, 0.7071).
        assert ref_quat_xyzw.shape == (1, 4)
        np.testing.assert_allclose(ref_quat_xyzw[0], [0.0, 0.0, 0.7071, 0.7071], atol=1e-6)

    def test_retargeter_invalid_root_quat_returns_none_quat(self) -> None:
        """Non-finite root_orn_wxyz from the retargeter falls through to
        ref_quat=None while still returning a valid motion_command."""
        policy = _bare_policy(NullTrackingSource())
        fake_q = np.zeros(29, dtype=np.float32)
        fake_dq = np.zeros(29, dtype=np.float32)
        fake_retargeter = MagicMock()
        fake_retargeter.retarget.return_value = (
            fake_q,
            fake_dq,
            np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        policy._retargeter = fake_retargeter

        result = policy._retarget_payload_to_motion_command(_make_payload())
        assert result is not None
        motion_command, ref_quat_xyzw = result
        assert motion_command.shape == (1, 58)
        assert ref_quat_xyzw is None

    def test_retargeter_exception_is_swallowed(self) -> None:
        policy = _bare_policy(NullTrackingSource())
        fake_retargeter = MagicMock()
        fake_retargeter.retarget.side_effect = RuntimeError("IK diverged")
        policy._retargeter = fake_retargeter

        result = policy._retarget_payload_to_motion_command(_make_payload())
        assert result is None
        # Second call should also be safe and logged silently.
        result2 = policy._retarget_payload_to_motion_command(_make_payload())
        assert result2 is None

    def test_bad_reshape_returns_none(self) -> None:
        policy = _bare_policy(NullTrackingSource())
        policy._retargeter = MagicMock()
        payload = TrackingPayload(joint_transforms=np.zeros(17, dtype=np.float32))
        assert policy._retarget_payload_to_motion_command(payload) is None

    def test_construction_exception_is_swallowed(self) -> None:
        """Importing or constructing the retargeter fails → fall through cleanly."""
        policy = _bare_policy(NullTrackingSource())

        # Inject a fake ``holosoma_retargeting.src.realtime_smpl_retargeter``
        # module into ``sys.modules`` so the lazy ``from ... import`` inside
        # ``_retarget_payload_to_motion_command`` resolves to our stub,
        # whose ``SMPLRetargeter`` raises on construction. This lets the
        # test run without the full retargeting package installed.
        import sys
        import types

        fake_module = types.ModuleType("holosoma_retargeting.src.realtime_smpl_retargeter")

        def _raising_ctor(*_args, **_kwargs):
            raise RuntimeError("mujoco load fail")

        fake_module.SMPLRetargeter = _raising_ctor  # type: ignore[attr-defined]
        sys.modules["holosoma_retargeting"] = types.ModuleType("holosoma_retargeting")
        sys.modules["holosoma_retargeting.src"] = types.ModuleType("holosoma_retargeting.src")
        sys.modules["holosoma_retargeting.src.realtime_smpl_retargeter"] = fake_module
        try:
            result = policy._retarget_payload_to_motion_command(_make_payload())
        finally:
            for name in (
                "holosoma_retargeting.src.realtime_smpl_retargeter",
                "holosoma_retargeting.src",
                "holosoma_retargeting",
            ):
                sys.modules.pop(name, None)

        assert result is None
        assert policy._retargeter_init_failed is True


# ----------------------------------------------------------------------
# Tests: rl_inference branching
# ----------------------------------------------------------------------


def _stub_rl_inference_deps(policy: WholeBodyTrackingPolicy) -> MagicMock:
    """Wire minimal stubs so ``rl_inference`` runs to completion.

    Returns the MagicMock standing in for ``self.policy`` (the ONNX
    session wrapper), so tests can assert on its motion_command output.
    """
    policy.prepare_obs_for_rl = MagicMock(return_value={"actor_obs": np.zeros((1, 1), dtype=np.float32)})
    onnx_motion = np.full((1, 58), 9.99, dtype=np.float32)
    onnx_action = np.zeros((1, 29), dtype=np.float32)
    onnx_ref_quat = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    policy.policy = MagicMock(return_value=(onnx_action, onnx_motion, onnx_ref_quat))
    policy.logger = MagicMock()
    policy._set_motion_timestep = MagicMock()
    return policy.policy


class TestRlInferenceBranching:
    def test_null_source_uses_onnx_clip_path(self) -> None:
        policy = _bare_policy(NullTrackingSource())
        onnx_mock = _stub_rl_inference_deps(policy)

        policy.rl_inference(robot_state_data=np.zeros((1, 100), dtype=np.float32))

        # ONNX path ran, motion_command_t comes from ONNX output.
        assert onnx_mock.call_count == 1
        assert policy.motion_command_t[0, 0] == pytest.approx(9.99)

    def test_custom_source_with_valid_payload_uses_retargeter(self) -> None:
        payload = _make_payload()
        policy = _bare_policy(_OneShotSource(payload))

        # Plant a fake retargeter — we are NOT testing the mink pipeline
        # here, only that rl_inference routes the payload through it and
        # that the produced motion_command_t is visible to prepare_obs_for_rl.
        fake_q = np.full(29, 0.11, dtype=np.float32)
        fake_dq = np.full(29, 0.22, dtype=np.float32)
        fake_retargeter = MagicMock()
        fake_retargeter.retarget.return_value = (fake_q, fake_dq, np.array([1.0, 0.0, 0.0, 0.0]))
        policy._retargeter = fake_retargeter

        onnx_mock = _stub_rl_inference_deps(policy)

        # Capture motion_command_t at the moment prepare_obs_for_rl runs,
        # since the ONNX call afterward overwrites it.
        seen_motion: list[np.ndarray] = []

        def _capture_obs(_state):
            seen_motion.append(policy.motion_command_t.copy())
            return {"actor_obs": np.zeros((1, 1), dtype=np.float32)}

        policy.prepare_obs_for_rl = _capture_obs

        policy.rl_inference(robot_state_data=np.zeros((1, 100), dtype=np.float32))

        fake_retargeter.retarget.assert_called_once()
        assert onnx_mock.call_count == 1
        # motion_command_t at obs time is the retargeter output, not the 0.42 seed.
        assert len(seen_motion) == 1
        np.testing.assert_allclose(seen_motion[0][0, :29], fake_q)
        np.testing.assert_allclose(seen_motion[0][0, 29:], fake_dq)

    def test_custom_source_substitutes_ref_quat_xyzw_t(self) -> None:
        """Regression: live teleop must also replace the reference orientation,
        not just the joint targets. Code review #5 (2026-05-05):
        leaving ref_quat_xyzw_t at the ONNX-clip baseline while joint targets
        track live teleop caused arm commands to under-reach on-robot."""
        payload = _make_payload()
        policy = _bare_policy(_OneShotSource(payload))

        fake_q = np.zeros(29, dtype=np.float32)
        fake_dq = np.zeros(29, dtype=np.float32)
        # Retargeter returns wxyz; helper converts to xyzw before the swap.
        fake_root_wxyz = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        fake_retargeter = MagicMock()
        fake_retargeter.retarget.return_value = (fake_q, fake_dq, fake_root_wxyz)
        policy._retargeter = fake_retargeter

        onnx_mock = _stub_rl_inference_deps(policy)

        # Seed a clearly distinguishable pre-teleop ref quat so we can tell
        # a substitution happened (vs the ONNX call overwriting it later).
        policy.ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        seen_ref_quat: list[np.ndarray] = []

        def _capture_obs(_state):
            seen_ref_quat.append(policy.ref_quat_xyzw_t.copy())
            return {"actor_obs": np.zeros((1, 1), dtype=np.float32)}

        policy.prepare_obs_for_rl = _capture_obs
        policy.rl_inference(robot_state_data=np.zeros((1, 100), dtype=np.float32))

        fake_retargeter.retarget.assert_called_once()
        assert onnx_mock.call_count == 1
        # ref_quat_xyzw_t seen by prepare_obs_for_rl is the retargeter's
        # output converted to xyzw: wxyz (0.5, 0.5, 0.5, 0.5) -> xyzw (0.5, 0.5, 0.5, 0.5).
        assert len(seen_ref_quat) == 1
        np.testing.assert_allclose(seen_ref_quat[0][0], [0.5, 0.5, 0.5, 0.5], atol=1e-6)

    def test_custom_source_that_raises_falls_through(self) -> None:
        payload = _make_payload()
        policy = _bare_policy(_RaisingSource(payload))

        fake_retargeter = MagicMock()
        fake_retargeter.retarget.side_effect = RuntimeError("IK divergence")
        policy._retargeter = fake_retargeter

        onnx_mock = _stub_rl_inference_deps(policy)

        # Capture what prepare_obs_for_rl sees.
        seen_motion: list[np.ndarray] = []

        def _capture_obs(_state):
            seen_motion.append(policy.motion_command_t.copy())
            return {"actor_obs": np.zeros((1, 1), dtype=np.float32)}

        policy.prepare_obs_for_rl = _capture_obs

        # Must not raise — sticky-fault contract.
        policy.rl_inference(robot_state_data=np.zeros((1, 100), dtype=np.float32))

        # Retargeter was attempted and the ONNX clip path still ran.
        fake_retargeter.retarget.assert_called_once()
        assert onnx_mock.call_count == 1
        # motion_command_t at obs time is the pre-seeded ONNX-clip value
        # (0.42), NOT retargeter output.
        assert len(seen_motion) == 1
        assert seen_motion[0][0, 0] == pytest.approx(0.42)
