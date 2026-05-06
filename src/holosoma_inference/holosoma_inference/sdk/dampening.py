"""Command dampening shim applied inside every interface backend.

The 2026-05-02 G1 run showed the robot receiving q_target values far outside
the mechanical joint range (e.g. left knee cmd = -3.94 rad vs MJCF range
[-0.087, 2.88]). We need a shared, env-configurable layer that every backend
(unitree binding, booster sdk2py, mujoco sim) runs commands through before
they leave the process, so we can dampen policy output while debugging.

All knobs default to pass-through (identity transform) so existing behavior
is preserved when no env vars are set.

Env vars
--------
HOLOSOMA_KP_LEVEL        float, default 1.0.  Multiplicative scale on kp.
HOLOSOMA_KD_LEVEL        float, default 1.0.  Multiplicative scale on kd.
HOLOSOMA_Q_SLEW_PER_TICK float, default unset (= off).  Max |Δq| per call,
                         applied per-joint against the previous q_target
                         that left this shim.
HOLOSOMA_Q_LIMIT_SCALE   float, default unset (= off).  Scale factor in
                         [0, 1] against the robot's per-joint hard limits
                         when clipping q_target.  1.0 = clip to hard limits,
                         0.5 = clip to the midpoint, etc.
HOLOSOMA_BLEND_ALPHA     float, default 1.0.  alpha in
                         q_send = alpha*q_tgt + (1-alpha)*q_current.
                         alpha=1 is pass-through; alpha=0 freezes output
                         at the current measured q.  Requires the caller
                         to pass dof_pos_latest.

All knobs are read from the environment at every ``apply()`` call so they can
be toggled at runtime without restarting the driver.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class DampeningKnobs:
    kp_level: float
    kd_level: float
    q_slew_per_tick: float | None
    q_limit_scale: float | None
    blend_alpha: float

    @classmethod
    def from_env(cls) -> DampeningKnobs:
        kp = _env_float("HOLOSOMA_KP_LEVEL", 1.0)
        kd = _env_float("HOLOSOMA_KD_LEVEL", 1.0)
        blend = _env_float("HOLOSOMA_BLEND_ALPHA", 1.0)
        return cls(
            kp_level=1.0 if kp is None else kp,
            kd_level=1.0 if kd is None else kd,
            q_slew_per_tick=_env_float("HOLOSOMA_Q_SLEW_PER_TICK", None),
            q_limit_scale=_env_float("HOLOSOMA_Q_LIMIT_SCALE", None),
            blend_alpha=1.0 if blend is None else blend,
        )


class Dampener:
    """Stateful per-interface dampening shim.

    One instance lives on each interface (UnitreeInterface, MujocoInterface,
    ...) and is called via :meth:`apply` just before the command hits the
    wire / the simulator.

    State kept:
        * previous post-shim q_target (for slew clamp)
    """

    def __init__(
        self,
        joint_limits_lo: Sequence[float] | None = None,
        joint_limits_hi: Sequence[float] | None = None,
    ):
        self._prev_q_out: np.ndarray | None = None
        self._joint_limits_lo = np.asarray(joint_limits_lo, dtype=np.float64) if joint_limits_lo is not None else None
        self._joint_limits_hi = np.asarray(joint_limits_hi, dtype=np.float64) if joint_limits_hi is not None else None

    def set_joint_limits(self, lo: Sequence[float], hi: Sequence[float]) -> None:
        self._joint_limits_lo = np.asarray(lo, dtype=np.float64)
        self._joint_limits_hi = np.asarray(hi, dtype=np.float64)

    def reset(self) -> None:
        """Forget the previous q_target. Call on engage/disengage transitions."""
        self._prev_q_out = None

    def apply(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        dof_pos_latest: np.ndarray | None,
        knobs: DampeningKnobs | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return dampened (q, dq, tau, kp, kd)."""
        if knobs is None:
            knobs = DampeningKnobs.from_env()

        q = np.asarray(cmd_q, dtype=np.float64).copy()
        dq = np.asarray(cmd_dq, dtype=np.float64).copy()
        tau = np.asarray(cmd_tau, dtype=np.float64).copy()
        kp_out = np.asarray(kp, dtype=np.float64) * float(knobs.kp_level)
        kd_out = np.asarray(kd, dtype=np.float64) * float(knobs.kd_level)

        # (1) Blend toward measured q. Applied FIRST so slew + clip operate on
        # the already-blended target.
        if knobs.blend_alpha != 1.0 and dof_pos_latest is not None:
            alpha = float(knobs.blend_alpha)
            q_cur = np.asarray(dof_pos_latest, dtype=np.float64).reshape(q.shape)
            q = alpha * q + (1.0 - alpha) * q_cur

        # (2) Hard joint-limit clip. Skip joints that are unlimited (+/-inf):
        # 0.5 * (-inf + +inf) is NaN, and clipping against NaN silently
        # turns a live target into NaN. In practice every G1 joint has a
        # range, but guard so a future MJCF without one doesn't trip a
        # silent failure.
        if knobs.q_limit_scale is not None and self._joint_limits_lo is not None and self._joint_limits_hi is not None:
            scale = float(knobs.q_limit_scale)
            limited = np.isfinite(self._joint_limits_lo) & np.isfinite(self._joint_limits_hi)
            if limited.any():
                lo_fin = self._joint_limits_lo[limited]
                hi_fin = self._joint_limits_hi[limited]
                mid = 0.5 * (lo_fin + hi_fin)
                half = 0.5 * (hi_fin - lo_fin) * scale
                q[limited] = np.clip(q[limited], mid - half, mid + half)

        # (3) Slew clamp against previous post-shim target.
        if knobs.q_slew_per_tick is not None and self._prev_q_out is not None:
            cap = float(knobs.q_slew_per_tick)
            delta = np.clip(q - self._prev_q_out, -cap, cap)
            q = self._prev_q_out + delta

        self._prev_q_out = q.copy()
        return q, dq, tau, kp_out, kd_out
