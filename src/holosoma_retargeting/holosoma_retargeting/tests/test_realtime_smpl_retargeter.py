"""Shape invariants for ``SMPLRetargeter.retarget``.

The G1 29-DOF MuJoCo XML uses a ``<freejoint/>`` root, so
``mj_model.nq == 36`` (xyz + wxyz + 29 joints). A previous revision of
``retarget()`` returned the full 36-element qpos without stripping the
freejoint prefix, which caused the downstream WBT policy to fall
through to ONNX-clip mode with the warning

    retarget_payload_to_motion_command: retargeter returned unexpected
    dim (q=36, dq=36, expected=29); falling through to ONNX-clip

Guard the contract: the retargeter returns 29-DOF joint angles + 29-DOF
velocities, regardless of ``nq`` on the underlying MuJoCo model.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

# Skip gracefully when the retargeter's heavy deps (mink, mujoco, scipy) are
# not installed in the running environment. That matches the wider
# holosoma_inference test pattern for optional deps.
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mink")


from holosoma_retargeting.src.realtime_smpl_retargeter import (  # noqa: E402
    NUM_JOINTS,
    SMPLRetargeter,
)

_MODEL_PATH = pathlib.Path(__file__).resolve().parents[1] / "models" / "g1" / "g1_29dof.xml"


def _make_smpl_frame() -> np.ndarray:
    """24-joint SMPL frame with identity orientations and a neutral stance."""
    frame = np.zeros((NUM_JOINTS, 7), dtype=np.float64)
    # Stack joints vertically so the pelvis→ankle (idx 0 → idx 7) baseline
    # has a non-zero length; the height_ratio computation divides by it.
    frame[0, :3] = [0.0, 0.0, 1.0]  # pelvis
    frame[7, :3] = [0.0, 0.0, 0.2]  # left ankle below pelvis
    frame[:, 6] = 1.0  # qw = 1 → identity orientations
    return frame


@pytest.fixture(scope="module")
def retargeter() -> SMPLRetargeter:
    if not _MODEL_PATH.exists():
        pytest.skip(f"retargeter MuJoCo model not found at {_MODEL_PATH}")
    return SMPLRetargeter(urdf_path=str(_MODEL_PATH), dt=0.02)


class TestRetargetShape:
    def test_q_joints_is_29_dof(self, retargeter: SMPLRetargeter) -> None:
        q, _dq, _root = retargeter.retarget(_make_smpl_frame())
        assert q.shape == (29,), (
            f"retarget() must return 29 joint angles (stripping the freejoint's 7 qpos entries). Got shape {q.shape}."
        )

    def test_dq_joints_is_29_dof(self, retargeter: SMPLRetargeter) -> None:
        _q, dq, _root = retargeter.retarget(_make_smpl_frame())
        assert dq.shape == (29,)

    def test_root_orientation_is_wxyz_quat(self, retargeter: SMPLRetargeter) -> None:
        _q, _dq, root = retargeter.retarget(_make_smpl_frame())
        assert root.shape == (4,)

    def test_dq_is_zero_on_first_call(self, retargeter: SMPLRetargeter) -> None:
        # First call seeds _prev_q_joints; the finite-difference path uses
        # np.zeros_like on initialization — assert the invariant.
        retargeter.reset()
        _q, dq, _root = retargeter.retarget(_make_smpl_frame())
        np.testing.assert_allclose(dq, np.zeros(29), atol=1e-9)

    def test_consecutive_calls_produce_finite_output(self, retargeter: SMPLRetargeter) -> None:
        retargeter.reset()
        q1, _, _ = retargeter.retarget(_make_smpl_frame())
        q2, dq2, _ = retargeter.retarget(_make_smpl_frame())
        assert np.all(np.isfinite(q1))
        assert np.all(np.isfinite(q2))
        assert np.all(np.isfinite(dq2))


class TestRetargeterStateShapes:
    """The internal state buffers must also use the articulated-joint count."""

    def test_prev_q_joints_matches_articulated_count(self, retargeter: SMPLRetargeter) -> None:
        assert retargeter._prev_q_joints.shape == (retargeter._num_joints,)
        assert retargeter._num_joints == 29

    def test_prev_q_visualizer_blob(self, retargeter: SMPLRetargeter) -> None:
        # Pinocchio-compatible layout: pos(3) + quat_xyzw(4) + joints(N)
        assert retargeter._prev_q.shape == (3 + 4 + retargeter._num_joints,)
