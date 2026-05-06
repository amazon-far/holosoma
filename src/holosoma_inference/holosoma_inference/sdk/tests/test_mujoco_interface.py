"""Unit tests for the MuJoCo virtual-robot backend.

Only runs when both the `mujoco` package and the shipped G1 MJCF at
``holosoma_retargeting/models/g1/g1_29dof.xml`` are importable. Skips cleanly
otherwise so the dampening tests remain runnable on every platform.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

try:
    from holosoma_inference.config.config_values.robot import g1_29dof as G1_CONFIG
except Exception:
    G1_CONFIG = None  # tests will skip below


def _can_locate_mjcf() -> bool:
    try:
        import holosoma_retargeting  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _can_locate_mjcf() or G1_CONFIG is None,
    reason="mujoco + holosoma_retargeting + g1_29dof config required",
)


@pytest.fixture
def iface(monkeypatch):
    from holosoma_inference.sdk.mujoco.mujoco_interface import MujocoInterface

    # Disable real-time integration so tests are deterministic.
    monkeypatch.setenv("HOLOSOMA_MUJOCO_REAL_TIME", "0")
    # Supply motor_kp/kd if missing in G1 config (newer configs load from ONNX).
    cfg = G1_CONFIG
    if cfg.motor_kp is None or cfg.motor_kd is None:
        from dataclasses import replace as _replace

        cfg = _replace(
            cfg,
            motor_kp=tuple([100.0] * cfg.num_motors),
            motor_kd=tuple([5.0] * cfg.num_motors),
        )
    return MujocoInterface(cfg, use_joystick=False)


def test_construct_and_default_pose(iface):
    state = iface.get_low_state()
    # Shape: 3 + 4 + 29 + 3 + 3 + 29 = 71
    assert state.shape == (1, 3 + 4 + 29 + 3 + 3 + 29)
    joint_pos = state[0, 7 : 7 + 29]
    defaults = np.array(G1_CONFIG.default_dof_angles)
    assert np.allclose(joint_pos, defaults, atol=1e-6)


def test_zero_gains_write_zero_qfrc(iface):
    # With zero kp, zero kd, zero tau_ff, the PD evaluator must write zero
    # into qfrc_applied regardless of robot state. Pure unit check of the
    # torque computation; decoupled from mj_step integration noise.
    iface.send_low_command(
        cmd_q=np.zeros(29),
        cmd_dq=np.zeros(29),
        cmd_tau=np.zeros(29),
        kp_override=np.zeros(29),
        kd_override=np.zeros(29),
    )
    # Call _apply_pd_torque directly so we sample qfrc before mj_step zeroes it.
    iface._apply_pd_torque(iface._latest_cmd)
    assert np.allclose(iface.data.qfrc_applied, 0.0)


def test_pd_torque_expected_value(iface):
    # tau = kp*(q_tgt - q) + kd*(dq_tgt - dq) + tau_ff
    knee_idx = G1_CONFIG.dof_names.index("left_knee_joint")
    target = np.array(G1_CONFIG.default_dof_angles, dtype=np.float64)
    target[knee_idx] += 0.1  # delta = 0.1
    kp = np.zeros(29)
    kp[knee_idx] = 50.0
    kd = np.zeros(29)
    iface.send_low_command(
        cmd_q=target,
        cmd_dq=np.zeros(29),
        cmd_tau=np.zeros(29),
        kp_override=kp,
        kd_override=kd,
    )
    iface._apply_pd_torque(iface._latest_cmd)
    knee_dofadr = iface._dof_qvel_idx[knee_idx]
    expected = 50.0 * 0.1  # = 5 Nm, well below knee saturation ±139
    assert abs(iface.data.qfrc_applied[knee_dofadr] - expected) < 1e-6


def test_nonzero_pd_drives_toward_target(iface):
    # Zero gravity + pelvis aloft so the legs don't flop mid-test; we only
    # want to verify the PD loop converges.
    iface.model.opt.gravity[:] = 0.0
    iface.data.qpos[2] = 5.0
    target = np.array(G1_CONFIG.default_dof_angles, dtype=np.float64)
    shoulder_idx = G1_CONFIG.dof_names.index("left_shoulder_pitch_joint")
    target[shoulder_idx] = 1.0
    kp = np.ones(29) * 200.0
    kd = np.ones(29) * 5.0
    for _ in range(2000):
        iface.send_low_command(
            cmd_q=target,
            cmd_dq=np.zeros(29),
            cmd_tau=np.zeros(29),
            kp_override=kp,
            kd_override=kd,
        )
        iface.step(1)
    final_q = iface.get_low_state()[0, 7 : 7 + 29]
    assert abs(final_q[shoulder_idx] - 1.0) < 0.15, f"shoulder={final_q[shoulder_idx]}"


def test_joint_limits_captured_from_mjcf(iface):
    lo = iface._joint_limits_lo
    hi = iface._joint_limits_hi
    knee_idx = G1_CONFIG.dof_names.index("left_knee_joint")
    # From MJCF: range="-0.087267 2.8798".
    assert abs(lo[knee_idx] - (-0.087267)) < 1e-3
    assert abs(hi[knee_idx] - 2.8798) < 1e-3


def test_dampening_q_limit_clip_engages(iface, monkeypatch):
    monkeypatch.setenv("HOLOSOMA_Q_LIMIT_SCALE", "1.0")
    # Send a command outside knee range — must get clipped to hi.
    target = np.array(G1_CONFIG.default_dof_angles, dtype=np.float64)
    knee_idx = G1_CONFIG.dof_names.index("left_knee_joint")
    target[knee_idx] = -3.94  # the exact value from the 05-02 hardware run
    kp = np.zeros(29)
    kd = np.zeros(29)
    iface.send_low_command(target, np.zeros(29), np.zeros(29), kp_override=kp, kd_override=kd)
    # The dampener's prev_q_out is the clipped target.
    assert iface._dampener._prev_q_out is not None
    assert iface._dampener._prev_q_out[knee_idx] >= -0.088  # within MJCF lo


def test_kp_level_property_round_trip(iface):
    iface.kp_level = 0.5
    assert iface.kp_level == 0.5
    iface.kd_level = 0.25
    assert iface.kd_level == 0.25


def test_joystick_stubs_return_empty(iface):
    msg = iface.get_joystick_msg()
    assert msg is not None
    assert msg.keys == 0
    assert iface.get_joystick_key() is None


def test_init_pelvis_freejoint_placed_feet_on_floor(iface):
    # Regression: earlier init left qpos[0:7] at MuJoCo's zero default
    # (pos=0, quat=0,0,0,0), putting legs through the floor and producing
    # an invalid base_quat the policy then amplified into runaway commands.
    # Fix: write identity quat and measure pelvis height so both feet sit
    # on z=0. This test asserts the init post-conditions.
    pelvis_z = float(iface.data.qpos[2])
    assert 0.4 < pelvis_z < 1.2, f"pelvis z={pelvis_z} outside standing range"
    quat = iface.data.qpos[3:7].copy()
    assert abs(np.linalg.norm(quat) - 1.0) < 1e-6, f"non-unit init quat {quat}"
    # At least one foot within 2 cm of the floor (the FK-computed offset
    # nails the lower foot; the other can be up to 2 cm higher on the
    # default pose since left/right legs aren't perfectly symmetric once
    # the shipped default_dof_angles are applied).
    foot_zs = []
    for foot_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        bid = iface._mujoco.mj_name2id(iface.model, iface._mujoco.mjtObj.mjOBJ_BODY, foot_name)
        assert bid >= 0, f"missing body {foot_name} in MJCF"
        foot_zs.append(float(iface.data.xpos[bid, 2]))
    assert min(foot_zs) < 0.02, f"neither foot near floor; foot_zs={foot_zs}"


def test_actfrcrange_clip_and_cache(iface):
    """Code review #3: actuator-force saturation must clip
    a large commanded torque down to the MJCF's jnt_actfrcrange, and
    the per-joint (lo, hi) cache must be populated at construction
    time (no mj_name2id calls in the hot PD loop).
    """
    # Cache is populated and matches dof_names order.
    assert iface._actfrc_lo.shape == (29,)
    assert iface._actfrc_hi.shape == (29,)
    # G1 MJCF declares actuatorfrcrange on every joint; every entry
    # should have a finite, non-degenerate range.
    assert iface._actfrc_has_range.all(), (
        "expected all 29 G1 joints to have actuatorfrcrange declared; "
        f"missing: {(~iface._actfrc_has_range).nonzero()[0].tolist()}"
    )
    assert (iface._actfrc_hi > iface._actfrc_lo).all()

    # Force a grossly over-range torque command via very high kp +
    # off-default q target, then step once and verify the actually
    # written qfrc_applied was clipped at the per-joint range.
    defaults = np.asarray(G1_CONFIG.default_dof_angles, dtype=np.float64)
    cmd_q = defaults + 5.0  # ~5 rad error on every joint
    cmd_dq = np.zeros(29)
    cmd_tau = np.zeros(29)
    kp = np.full(29, 10_000.0)  # would produce |tau| >> any declared range
    kd = np.zeros(29)
    iface.send_low_command(
        cmd_q=cmd_q,
        cmd_dq=cmd_dq,
        cmd_tau=cmd_tau,
        kp_override=kp,
        kd_override=kd,
    )
    iface.step(1)
    # Read back what was written into qfrc_applied at the cached dof indices.
    written = iface.data.qfrc_applied[iface._dof_qvel_idx]
    assert np.all(written >= iface._actfrc_lo - 1e-6), (
        f"some joints below actfrc_lo: diffs={(written - iface._actfrc_lo)[written < iface._actfrc_lo].tolist()}"
    )
    assert np.all(written <= iface._actfrc_hi + 1e-6), (
        f"some joints above actfrc_hi: diffs={(iface._actfrc_hi - written)[written > iface._actfrc_hi].tolist()}"
    )
    # Sanity: at least one joint was actually saturated (otherwise the
    # test isn't exercising the clip path).
    saturated = np.isclose(written, iface._actfrc_lo) | np.isclose(written, iface._actfrc_hi)
    assert saturated.any(), (
        "expected at least one joint to saturate at actfrcrange, but none did; "
        "test is no longer exercising the clip path"
    )


def test_init_gravity_hold_stays_upright(iface):
    # With gravity on and zero kp/kd (no control), a correctly-initialized
    # robot should not fall catastrophically within a short window. If the
    # pelvis/feet are mis-placed the robot accelerates through the floor
    # and joints blow past their range. Asserts a bounded fall.
    # Low gains so there's no PD driving the sim; only gravity + contacts.
    kp = np.zeros(29)
    kd = np.zeros(29)
    target = np.array(G1_CONFIG.default_dof_angles, dtype=np.float64)
    pelvis_z_start = float(iface.data.qpos[2])
    for _ in range(100):
        iface.send_low_command(
            cmd_q=target,
            cmd_dq=np.zeros(29),
            cmd_tau=np.zeros(29),
            kp_override=kp,
            kd_override=kd,
        )
        iface.step(1)
    pelvis_z_end = float(iface.data.qpos[2])
    # Accept some settling (a few cm) but reject "robot through floor".
    assert pelvis_z_end > pelvis_z_start - 0.15, (
        f"pelvis fell from {pelvis_z_start:.3f} to {pelvis_z_end:.3f} — initialization may be misplaced"
    )
    # And no joint has blown through MJCF range
    joint_pos = iface.get_low_state()[0, 7 : 7 + 29]
    lo = iface._joint_limits_lo
    hi = iface._joint_limits_hi
    violations = np.where((joint_pos < lo - 0.1) | (joint_pos > hi + 0.1))[0]
    assert len(violations) == 0, (
        f"joints blew past range: indices={violations.tolist()}, values={joint_pos[violations].tolist()}"
    )
