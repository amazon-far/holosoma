"""MuJoCo-backed virtual Unitree driver.

Implements :class:`~holosoma_inference.sdk.base.base_interface.BaseInterface`
against an in-process MuJoCo simulation so the full WBT policy stack can
run without a real robot or CycloneDDS.

Why a pure-Python driver instead of a DDS-mocking node:
    * Deterministic and fast — one process, one thread, no participant
      discovery.
    * Usable from unit tests (no network, no D-Bus, no container).
    * Identical policy / driver / retargeter code path as the real robot;
      only the SDK backend changes.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface
from holosoma_inference.sdk.dampening import Dampener
from holosoma_inference.sdk.send_log import SendLogger


def _resolve_mjcf_path(robot_config: RobotConfig) -> Path:
    """Locate the MJCF for this robot.

    Priority:
        1. Env var ``HOLOSOMA_MUJOCO_MJCF`` (absolute path override).
        2. ``robot_config.urdf_path`` if it ends in ``.xml``.
        3. Shipped MJCF at
           ``holosoma_retargeting/models/<robot>/<robot_type>.xml``.
    """
    override = os.environ.get("HOLOSOMA_MUJOCO_MJCF")
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise FileNotFoundError(f"HOLOSOMA_MUJOCO_MJCF={override!r} is not a file")

    urdf = getattr(robot_config, "urdf_path", None)
    if urdf and urdf.endswith(".xml") and Path(urdf).is_file():
        return Path(urdf)

    # Fallback: resolve against the holosoma_retargeting package.
    try:
        import holosoma_retargeting  # type: ignore[import-not-found]

        pkg_root = Path(holosoma_retargeting.__file__).parent
    except Exception as exc:
        raise FileNotFoundError(
            "Could not locate MJCF for MuJoCo backend; set HOLOSOMA_MUJOCO_MJCF or "
            "populate robot_config.urdf_path with a .xml path"
        ) from exc

    candidate = pkg_root / "models" / robot_config.robot / f"{robot_config.robot_type}.xml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No MJCF found for {robot_config.robot_type} at {candidate}")


class _MjJoystickStub:
    """Empty joystick message so policies expecting .keys/.lx/.ly don't crash."""

    keys = 0
    lx = 0.0
    ly = 0.0
    rx = 0.0
    ry = 0.0


class MujocoInterface(BaseInterface):
    """BaseInterface implementation backed by a MuJoCo simulation.

    The interface keeps its own :class:`mujoco.MjModel` / :class:`mujoco.MjData`
    and advances the simulation forward by exactly the amount of wall-time
    elapsed since the last command (so the policy's rate limiter sets the
    sim step implicitly). On explicit ``step`` calls (used in tests) the
    simulation advances by ``model.opt.timestep``.

    Joint order (the robot_config's ``dof_names``) is assumed to match the
    MJCF's joint order. For the shipped G1 29dof MJCF this is true by
    construction; the interface verifies on init and raises if it drifts.
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        domain_id: int = 0,
        interface_str: str | None = None,
        use_joystick: bool = True,
    ):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)

        import mujoco  # local so import failure is easy to diagnose

        self._mujoco = mujoco

        mjcf_path = _resolve_mjcf_path(robot_config)
        self._mjcf_path = mjcf_path
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)

        # ------------------------------------------------------------------
        # Build dof_name → (qpos_idx, qvel_idx) using MjModel joint metadata.
        # The pelvis freejoint contributes 7 qpos / 6 qvel before any hinge
        # joint starts, so index lookups must be done via model.jnt_qposadr.
        # ------------------------------------------------------------------
        self._dof_qpos_idx = np.zeros(robot_config.num_joints, dtype=np.int32)
        self._dof_qvel_idx = np.zeros(robot_config.num_joints, dtype=np.int32)
        for j_id, name in enumerate(robot_config.dof_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(
                    f"MJCF at {mjcf_path} has no joint named {name!r}; robot_config.dof_names does not match the model."
                )
            self._dof_qpos_idx[j_id] = int(self.model.jnt_qposadr[jid])
            self._dof_qvel_idx[j_id] = int(self.model.jnt_dofadr[jid])

        # ------------------------------------------------------------------
        # Capture joint limits from the MJCF for the dampening layer.
        # Use jnt_range where limited; else ±inf.
        # ------------------------------------------------------------------
        lo = np.full(robot_config.num_joints, -np.inf)
        hi = np.full(robot_config.num_joints, np.inf)
        for j_id, name in enumerate(robot_config.dof_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if bool(self.model.jnt_limited[jid]):
                lo[j_id] = float(self.model.jnt_range[jid][0])
                hi[j_id] = float(self.model.jnt_range[jid][1])
        self._joint_limits_lo = lo
        self._joint_limits_hi = hi

        # Cache per-joint actuator-force range in dof_names order so the
        # PD loop doesn't call mj_name2id 29x per tick. The MJCF marks a
        # joint's force range as "declared" via jnt_actfrcrange with a
        # non-degenerate (lo < hi) interval; for joints without a range
        # we store (-inf, +inf) and skip clipping in _apply_pd_torque.
        #
        # This distinguishes "no range declared" from "range is exactly
        # (0, 0)": under the prior hi > lo check the latter silently
        # no-op'd, which is indistinguishable from the former and would
        # mask a corrupt-MJCF situation.
        actfrc_lo = np.full(robot_config.num_joints, -np.inf)
        actfrc_hi = np.full(robot_config.num_joints, np.inf)
        if hasattr(self.model, "jnt_actfrcrange") and self.model.jnt_actfrcrange is not None:
            for j_id, name in enumerate(robot_config.dof_names):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                rng = self.model.jnt_actfrcrange[jid]
                jlo, jhi = float(rng[0]), float(rng[1])
                if jhi > jlo:
                    actfrc_lo[j_id] = jlo
                    actfrc_hi[j_id] = jhi
        self._actfrc_lo = actfrc_lo
        self._actfrc_hi = actfrc_hi
        self._actfrc_has_range = np.isfinite(actfrc_lo) & np.isfinite(actfrc_hi)

        self._kp_level = 1.0
        self._kd_level = 1.0

        # Initialize qpos to default_dof_angles so the sim starts in the
        # configured default pose (same pose the policy expects on hardware
        # after stiff-startup).
        defaults = np.asarray(robot_config.default_dof_angles, dtype=np.float64)
        for j_id in range(robot_config.num_joints):
            self.data.qpos[self._dof_qpos_idx[j_id]] = defaults[j_id]
        # Initialize the pelvis freejoint. MuJoCo's qpos default for an
        # unwritten freejoint is (pos=0, quat=0,0,0,0) which is both
        # underground and a zero quaternion. Set identity-quat, then FK
        # with a throwaway tall pelvis to measure the foot-z when the
        # legs are in their default bent-knee pose, and drop the pelvis
        # by that offset so the feet rest on the floor at t=0.
        self.data.qpos[0:3] = [0.0, 0.0, 5.0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        foot_z_values = []
        for foot_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
            if bid >= 0:
                foot_z_values.append(float(self.data.xpos[bid, 2]))
        if foot_z_values:
            pelvis_z = 5.0 - min(foot_z_values)
        else:
            pelvis_z = 0.793
        self.data.qpos[2] = pelvis_z
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._dampener = Dampener(joint_limits_lo=lo, joint_limits_hi=hi)
        self._send_logger = SendLogger()
        self._last_cmd_time = time.monotonic()
        self._lock = threading.Lock()

        # Runtime knobs read from env (can be overridden per-call).
        self._real_time = os.environ.get("HOLOSOMA_MUJOCO_REAL_TIME", "1") != "0"

    # ----------------------------------------------------------------------
    # BaseInterface API
    # ----------------------------------------------------------------------

    def get_low_state(self) -> np.ndarray:
        """Return the same (1, 3+4+N+3+3+N) vector UnitreeInterface produces."""
        with self._lock:
            base_pos = self.data.qpos[0:3].copy()
            # MuJoCo quat is (w, x, y, z); Unitree SDK also reports (w, x, y, z).
            quat = self.data.qpos[3:7].copy()
            joint_pos = self.data.qpos[self._dof_qpos_idx].copy()
            base_lin_vel = self.data.qvel[0:3].copy()
            base_ang_vel = self.data.qvel[3:6].copy()
            joint_vel = self.data.qvel[self._dof_qvel_idx].copy()

        return np.concatenate([base_pos, quat, joint_pos, base_lin_vel, base_ang_vel, joint_vel]).reshape(1, -1)

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ) -> None:
        """Apply a PD control command to the MuJoCo sim for one rollout step.

        The integration window is ``wall_time_since_last_call``, capped at
        100 ms to prevent runaway forward-integration if the policy stalls.
        In test mode, call :meth:`step` directly to control integration.
        """
        motor_kp = np.asarray(
            kp_override if kp_override is not None else self.robot_config.motor_kp,
            dtype=np.float64,
        )
        motor_kd = np.asarray(
            kd_override if kd_override is not None else self.robot_config.motor_kd,
            dtype=np.float64,
        )

        # Use BaseInterface's joint-indexed PD space (not motor-indexed),
        # because MuJoCo joint order already matches dof_names.
        q_d, dq_d, tau_d, kp_d, kd_d = self._dampener.apply(
            cmd_q=np.asarray(cmd_q, dtype=np.float64),
            cmd_dq=np.asarray(cmd_dq, dtype=np.float64),
            cmd_tau=np.asarray(cmd_tau, dtype=np.float64),
            kp=motor_kp,
            kd=motor_kd,
            dof_pos_latest=dof_pos_latest,
        )

        self._latest_cmd = {
            "q": q_d,
            "dq": dq_d,
            "tau": tau_d,
            "kp": kp_d * self._kp_level,
            "kd": kd_d * self._kd_level,
        }

        self._send_logger.maybe_log(
            q_target=q_d,
            kp=kp_d * self._kp_level,
            kd=kd_d * self._kd_level,
        )

        if self._real_time:
            now = time.monotonic()
            dt = min(now - self._last_cmd_time, 0.1)
            self._last_cmd_time = now
            self._advance(dt)

    def step(self, n: int = 1) -> None:
        """Advance the sim by ``n`` mj_step calls (test helper)."""
        for _ in range(n):
            self._advance(self.model.opt.timestep)

    # ----------------------------------------------------------------------
    # Joystick stubs
    # ----------------------------------------------------------------------

    def get_joystick_msg(self):
        return _MjJoystickStub()

    def get_joystick_key(self, wc_msg=None):
        return None

    @property
    def kp_level(self) -> float:
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value: float) -> None:
        self._kp_level = float(value)

    @property
    def kd_level(self) -> float:
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value: float) -> None:
        self._kd_level = float(value)

    # ----------------------------------------------------------------------
    # Internal: PD evaluation + mj_step
    # ----------------------------------------------------------------------

    def _advance(self, duration: float) -> None:
        if duration <= 0:
            return
        latest = getattr(self, "_latest_cmd", None)
        ts = float(self.model.opt.timestep)
        n_steps = max(1, round(duration / ts))
        for _ in range(n_steps):
            if latest is not None:
                self._apply_pd_torque(latest)
            self._mujoco.mj_step(self.model, self.data)

    def _apply_pd_torque(self, cmd: dict) -> None:
        q = self.data.qpos[self._dof_qpos_idx]
        dq = self.data.qvel[self._dof_qvel_idx]
        tau = cmd["kp"] * (cmd["q"] - q) + cmd["kd"] * (cmd["dq"] - dq) + cmd["tau"]
        # Saturate joints with a declared actuator force range. Uses the
        # cached (lo, hi) vectors from __init__ so this path is allocation-
        # and name-lookup-free at 50 Hz * substep.
        if self._actfrc_has_range.any():
            np.clip(tau, self._actfrc_lo, self._actfrc_hi, out=tau)
        # Write into qfrc_applied at the per-joint dofadr; this bypasses
        # MuJoCo's actuator graph (we don't need it — we're doing PD outside
        # MuJoCo's motor model).
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self._dof_qvel_idx] = tau
