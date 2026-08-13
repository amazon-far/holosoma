"""Pico/SMPL 24-joint to G1 29-DOF online retargeting via mink differential IK.

Pipeline (following GR00T/GMR approach):
  1. Coordinate transform: Pico Y-up → robot Z-up: (x,y,z) → (x,-z,y)
  2. Scale positions relative to root with per-body factors
  3. Apply per-joint rotation offsets to align human → robot joint frames
  4. Ground anchoring: shift skeleton so feet are at ground level
  5. mink differential IK to solve for G1 joint angles
  6. Finite-difference velocity from previous frame

Input: (24, 7) array — per-joint (x, y, z, qx, qy, qz, qw) in Pico/Unity frame.
Output: (29,) G1 joint angles in URDF/Mujoco order, velocity, root orientation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import mink
import mujoco
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation


@runtime_checkable
class Retargeter(Protocol):
    """Embodiment-agnostic online retargeter interface.

    A retargeter maps a single frame of body-tracker joint poses to a robot's
    joint-space target. Implementations are embodiment-specific (e.g.
    :class:`G1SmplRetargeter`); consumers like ``RetargeterNode`` depend on this
    Protocol so new embodiments can be registered without subclassing.
    """

    def retarget(self, joint_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map one frame of tracker joint poses to (joint angles, velocities, root quat)."""
        ...

    def reset(self) -> None:
        """Reset any per-episode retargeting state."""
        ...


# =============================================================================
# Pico/SMPL Joint Indices (24-joint SMPL ordering)
# =============================================================================

JOINT_NAMES = [
    "pelvis",  # 0
    "left_hip",  # 1
    "right_hip",  # 2
    "spine1",  # 3
    "left_knee",  # 4
    "right_knee",  # 5
    "spine2",  # 6
    "left_ankle",  # 7
    "right_ankle",  # 8
    "spine3",  # 9
    "left_foot",  # 10
    "right_foot",  # 11
    "neck",  # 12
    "left_collar",  # 13
    "right_collar",  # 14
    "head",  # 15
    "left_shoulder",  # 16
    "right_shoulder",  # 17
    "left_elbow",  # 18
    "right_elbow",  # 19
    "left_wrist",  # 20
    "right_wrist",  # 21
    "left_hand",  # 22
    "right_hand",  # 23
]

NUM_JOINTS = 24

# =============================================================================
# Coordinate Transform: Pico/Unity Y-up → Robot Z-up
# (x, y, z) → (x, -z, y)
# =============================================================================

_COORD_ROT_MAT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_COORD_ROT_QUAT_WXYZ = Rotation.from_matrix(_COORD_ROT_MAT).as_quat(scalar_first=True)
_COORD_ROT_SCIPY = Rotation.from_matrix(_COORD_ROT_MAT)

# =============================================================================
# Per-joint rotation offsets: align Pico joint frames → G1 joint frames
# From GR00T xrobot_to_g1 config (wxyz quaternions).
# =============================================================================

# fmt: off
_ROT_OFFSETS_WXYZ: dict[int, tuple[float, float, float, float]] = {
    # Lower body + torso: 120° rotation aligning post-transform Pico to G1 MJCF
    0:  (-0.5,  0.5, -0.5, -0.5),  # pelvis
    1:  (-0.5,  0.5, -0.5, -0.5),  # left_hip
    2:  (-0.5,  0.5, -0.5, -0.5),  # right_hip
    3:  (-0.5,  0.5, -0.5, -0.5),  # spine1
    4:  (-0.5,  0.5, -0.5, -0.5),  # left_knee
    5:  (-0.5,  0.5, -0.5, -0.5),  # right_knee
    6:  (-0.5,  0.5, -0.5, -0.5),  # spine2
    7:  (-0.5,  0.5, -0.5, -0.5),  # left_ankle
    8:  (-0.5,  0.5, -0.5, -0.5),  # right_ankle
    9:  (-0.5,  0.5, -0.5, -0.5),  # spine3 / torso
    10: (-0.5,  0.5, -0.5, -0.5),  # left_foot
    11: (-0.5,  0.5, -0.5, -0.5),  # right_foot
    # Left arm
    16: ( 0.707, 0.0, 0.707, 0.0),  # left_shoulder
    18: ( 0.0,   0.0, 1.0,   0.0),  # left_elbow
    20: ( 0.0,   0.0, 1.0,   0.0),  # left_wrist
    # Right arm
    17: ( 0.0,  0.707, 0.0, -0.707),  # right_shoulder
    19: ( 0.0,  1.0,   0.0,  0.0),    # right_elbow
    21: ( 0.0,  1.0,   0.0,  0.0),    # right_wrist
}
# fmt: on

# Pre-compute scipy Rotation objects for offsets
_ROT_OFFSET_SCIPY: dict[int, Rotation] = {}
for _idx, _wxyz in _ROT_OFFSETS_WXYZ.items():
    # Convert wxyz → xyzw for scipy
    _ROT_OFFSET_SCIPY[_idx] = Rotation.from_quat([_wxyz[1], _wxyz[2], _wxyz[3], _wxyz[0]])

# =============================================================================
# Per-body scale factors (from GMR human_scale_table)
# =============================================================================

_SCALE_FACTORS: dict[int, float] = {
    0: 0.9,  # pelvis
    1: 0.9,
    2: 0.9,  # hips
    3: 0.9,
    6: 0.9,
    9: 0.9,  # spine
    4: 0.9,
    5: 0.9,  # knees
    7: 0.9,
    8: 0.9,  # ankles
    10: 0.9,
    11: 0.9,  # feet
    12: 0.9,
    15: 0.9,  # neck, head
    13: 0.8,
    14: 0.8,  # collars
    16: 0.8,
    17: 0.8,  # shoulders
    18: 0.8,
    19: 0.8,  # elbows
    20: 0.8,
    21: 0.8,  # wrists
    22: 0.8,
    23: 0.8,  # hands
}

# =============================================================================
# IK targets: (mujoco_body_name, tracker_joint_index, position_cost, orientation_cost)
# =============================================================================

_IK_TARGETS: list[tuple[str, int, float, float]] = [
    # Torso: orientation only — waist chain is too short for position targets
    ("torso_link", 9, 0.0, 10.0),
    # Feet: high position cost for standing stability
    ("left_ankle_roll_link", 7, 100.0, 10.0),
    ("right_ankle_roll_link", 8, 100.0, 10.0),
    # Knees: orientation only — position is constrained by hip-ankle chain
    ("left_knee_link", 4, 0.0, 5.0),
    ("right_knee_link", 5, 0.0, 5.0),
    # Arms: moderate position + orientation
    ("left_elbow_link", 18, 5.0, 10.0),
    ("right_elbow_link", 19, 5.0, 10.0),
    ("left_wrist_yaw_link", 20, 5.0, 5.0),
    ("right_wrist_yaw_link", 21, 5.0, 5.0),
]

# Ground foot height target (meters)
_GROUND_FOOT_HEIGHT = 0.05


def _strip_freejoint(xml_path: str) -> str | None:
    """If the MJCF at ``xml_path`` has a ``<freejoint/>`` element, return
    the XML with that element removed (so the model loads fixed-base).
    Returns ``None`` if the file has no freejoint and can be loaded
    directly via ``from_xml_path``.

    Used by the retargeter to accept the shipped freejoint MJCFs (which
    the policy's MuJoCo backend needs for its own sim) without inheriting
    the freejoint DOF into the IK.
    """
    import re as _re
    from pathlib import Path as _Path

    text = _Path(xml_path).read_text(encoding="utf-8")
    if "<freejoint" not in text:
        return None
    # Remove any line containing <freejoint .../> or <freejoint></freejoint>.
    stripped = _re.sub(r"\s*<freejoint[^>]*/>\s*", "\n", text)
    return _re.sub(r"\s*<freejoint[^>]*>.*?</freejoint>\s*", "\n", stripped, flags=_re.DOTALL)


_ASSET_CACHE: dict[str, dict[str, bytes]] = {}


def _asset_dir_for(xml_path: str) -> dict:
    """Return a single-entry asset dict mapping ``assets/<name>`` relative
    paths (the way the shipped MJCFs reference mesh files) to their bytes,
    so ``mujoco.MjModel.from_xml_string`` can find them without an
    on-disk ``meshdir`` resolving relative to cwd.

    Cache the manifest keyed on xml_path so repeated G1SmplRetargeter construction
    (e.g. WBT policy re-init) doesn't re-read every OBJ/STL from disk every time.
    mujoco does not cache from_xml_string assets across calls, so without this every
    retargeter instance paid ~60 file reads before first retarget().
    """
    import os as _os

    base = _os.path.dirname(_os.path.abspath(xml_path))
    cache_key = _os.path.abspath(xml_path)
    cached = _ASSET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    assets: dict[str, bytes] = {}
    for root, _dirs, files in _os.walk(base):
        for fn in files:
            if not fn.lower().endswith((".obj", ".stl", ".mtl", ".png", ".jpg")):
                continue
            p = _os.path.join(root, fn)
            rel = _os.path.relpath(p, base)
            with open(p, "rb") as f:
                assets[rel] = f.read()
    _ASSET_CACHE[cache_key] = assets
    return assets


class G1SmplRetargeter:
    """Online retargeting from Pico body tracker to G1 29-DOF via mink differential IK.

    Implements the :class:`Retargeter` Protocol for the Unitree G1 (29-DOF).
    """

    def __init__(self, urdf_path: str, dt: float = 0.02, *, max_ik_iters: int = 20):
        # -- MuJoCo model (fixed base, 29 DOF) --
        # The retargeter's IK is designed for a fixed-base model — all
        # ``_IK_TARGETS`` are root-relative and the solver must not have
        # freejoint DOF to absorb the tasks into. The shipped ``g1_29dof.xml``
        # carries a ``<freejoint/>`` (needed by the policy's MuJoCo backend);
        # if we load it as-is the IK bakes up to 7 DOF of body rotation into
        # the root, yielding joint angles that don't match the SMPL pose. So
        # strip the freejoint here and load fixed-base.
        model_xml = _strip_freejoint(urdf_path)
        if model_xml is not None:
            self._mj_model = mujoco.MjModel.from_xml_string(model_xml, _asset_dir_for(urdf_path))
        else:
            self._mj_model = mujoco.MjModel.from_xml_path(urdf_path)
        self._mj_data = mujoco.MjData(self._mj_model)

        self._dt = dt
        self._max_ik_iters = max_ik_iters

        # -- mink Configuration --
        self._config = mink.Configuration(self._mj_model)
        self._config.update()

        # -- Create FrameTask objects --
        self._frame_tasks: list[tuple[mink.FrameTask, int]] = []
        for body_name, joint_idx, pos_cost, ori_cost in _IK_TARGETS:
            task = mink.FrameTask(
                frame_name=body_name,
                frame_type="body",
                position_cost=pos_cost,
                orientation_cost=ori_cost,
                lm_damping=1e-3,
            )
            self._frame_tasks.append((task, joint_idx))

        # -- Posture regularization (keep near neutral when underconstrained) --
        self._posture_task = mink.PostureTask(model=self._mj_model, cost=1e-2)

        # -- Joint limits --
        self._config_limit = mink.ConfigurationLimit(self._mj_model)

        # -- G1 neutral leg length for height ratio --
        mujoco.mj_forward(self._mj_model, self._mj_data)
        ankle_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self._g1_leg = float(np.linalg.norm(self._mj_data.xpos[ankle_id]))
        if self._g1_leg < 0.1:
            self._g1_leg = 0.75

        self._height_ratio: float | None = None

        # -- State for velocity computation and warm-starting --
        self._prev_q_joints = np.zeros(self._mj_model.nq)
        self._prev_root_pos = np.zeros(3)
        self._prev_root_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self._initialized = False

        # Pinocchio-compatible state for visualizer: [pos(3), quat_xyzw(4), joints(29)]
        self._prev_q = np.zeros(3 + 4 + self._mj_model.nq)
        self._prev_q[6] = 1.0  # identity quaternion w component at index 6 (xyzw)

    # ------------------------------------------------------------------
    # Transform Pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _coordinate_transform(positions: np.ndarray, quats_xyzw: np.ndarray):
        """Pico Y-up → robot Z-up: (x,y,z) → (x,-z,y), plus rotation."""
        positions_robot = positions @ _COORD_ROT_MAT.T
        rots_pico = Rotation.from_quat(quats_xyzw)
        rots_robot = _COORD_ROT_SCIPY * rots_pico
        return positions_robot, rots_robot

    @staticmethod
    def _apply_rotation_offsets(rotations: Rotation) -> Rotation:
        """Apply per-joint rotation offsets to align human → G1 joint frames."""
        quats = rotations.as_quat()  # (24, 4) xyzw
        for idx, offset in _ROT_OFFSET_SCIPY.items():
            joint_rot = Rotation.from_quat(quats[idx])
            quats[idx] = (joint_rot * offset).as_quat()
        return Rotation.from_quat(quats)

    def _scale_positions(self, positions: np.ndarray) -> np.ndarray:
        """Scale positions relative to root with per-body factors and height ratio."""
        root_pos = positions[0].copy()
        scaled = positions.copy()
        assert self._height_ratio is not None
        for i in range(NUM_JOINTS):
            factor = _SCALE_FACTORS.get(i, 0.9) * self._height_ratio
            scaled[i] = (positions[i] - root_pos) * factor + root_pos
        return scaled

    @staticmethod
    def _anchor_to_ground(positions: np.ndarray) -> np.ndarray:
        """Shift entire skeleton so lowest foot is at ground level."""
        foot_indices = [7, 8, 10, 11]  # ankles and feet
        min_z = min(positions[i][2] for i in foot_indices)
        z_shift = _GROUND_FOOT_HEIGHT - min_z
        positions = positions.copy()
        positions[:, 2] += z_shift
        return positions

    # ------------------------------------------------------------------
    # mink Differential IK
    # ------------------------------------------------------------------

    def _solve_ik_mink(self, positions: np.ndarray, rotations: Rotation) -> np.ndarray:
        """Solve IK using mink differential IK.

        Targets are expressed relative to root (pelvis) frame since the
        MuJoCo model has a fixed base (pelvis = world body at origin).
        """
        root_pos = positions[0]
        root_rot = rotations[0].as_matrix()
        root_rot_inv = root_rot.T

        # Set SE3 targets for each frame task (root-relative)
        for task, joint_idx in self._frame_tasks:
            pos_rel = root_rot_inv @ (positions[joint_idx] - root_pos)
            rot_rel = root_rot_inv @ rotations[joint_idx].as_matrix()
            quat_wxyz = Rotation.from_matrix(rot_rel).as_quat(scalar_first=True)
            wxyz_xyz = np.concatenate([quat_wxyz, pos_rel])
            task.set_target(mink.SE3(wxyz_xyz))

        # Posture regularization toward current pose (smooth warm-start)
        self._posture_task.set_target(self._config.q.copy())

        tasks = [t for t, _ in self._frame_tasks] + [self._posture_task]
        limits = [self._config_limit]

        for _ in range(self._max_ik_iters):
            vel = mink.solve_ik(self._config, tasks, dt=0.1, solver="daqp", damping=1e-4, limits=limits)
            self._config.integrate_inplace(vel, 0.1)
            if np.linalg.norm(vel) < 1e-4:
                break

        return self._config.q.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retarget(self, joint_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retarget a single frame of Pico body tracker data to G1 29-DOF.

        Args:
            joint_poses: (24, 7) per-joint (x, y, z, qx, qy, qz, qw)
                in Pico/Unity Y-up frame. Quaternions in xyzw convention.

        Returns:
            q_joints: (29,) G1 joint angles in URDF/Mujoco order.
            dq_joints: (29,) joint velocities via finite difference.
            root_orn_wxyz: (4,) root orientation quaternion (wxyz).
        """
        joint_poses = np.asarray(joint_poses, dtype=np.float64).reshape(NUM_JOINTS, 7)

        positions = joint_poses[:, :3].copy()
        quats_xyzw = joint_poses[:, 3:7].copy()

        # Fix zero-norm quaternions (untracked joints)
        norms = np.linalg.norm(quats_xyzw, axis=1)
        quats_xyzw[norms < 1e-8] = [0.0, 0.0, 0.0, 1.0]

        # 1. Coordinate transform: Pico Y-up → robot Z-up
        positions, rotations = self._coordinate_transform(positions, quats_xyzw)

        # 2. Compute height ratio on first frame
        if self._height_ratio is None:
            tracker_leg = float(np.linalg.norm(positions[7] - positions[0]))
            self._height_ratio = self._g1_leg / max(tracker_leg, 1e-6)
            logger.info(
                "G1SmplRetargeter: height_ratio={:.4f} (G1 leg={:.3f}m, tracker leg={:.3f}m)",
                self._height_ratio,
                self._g1_leg,
                tracker_leg,
            )

        # 3. Scale positions relative to root
        positions = self._scale_positions(positions)

        # Body orientation for TML: conjugate out the coordinate rotation's frame
        # offset so that standing upright → identity, turns → yaw around Z.
        # rotations[0] = coord_rot * pico_pelvis; we want coord_rot * pico * coord_rot⁻¹
        root_body_rot = rotations[0] * _COORD_ROT_SCIPY.inv()
        root_body_quat_xyzw = root_body_rot.as_quat()  # xyzw

        # 4. Apply per-joint rotation offsets (for IK targets only)
        rotations = self._apply_rotation_offsets(rotations)

        # 5. Ground anchoring
        positions = self._anchor_to_ground(positions)

        # 6. Solve IK via mink
        q_joints = self._solve_ik_mink(positions, rotations)

        # 7. Finite-difference velocity
        if self._initialized:
            dq_joints = (q_joints - self._prev_q_joints) / self._dt
        else:
            dq_joints = np.zeros_like(q_joints)
            self._initialized = True

        # 8. Store state
        self._prev_q_joints = q_joints.copy()
        self._prev_root_pos = positions[0].copy()
        self._prev_root_quat_xyzw = root_body_quat_xyzw.copy()

        # Pinocchio-compatible _prev_q for visualizer (pos[3] + quat_xyzw[4] + joints[29])
        self._prev_q = np.concatenate([positions[0], root_body_quat_xyzw, q_joints])

        # 9. Root orientation as wxyz (pre-offset = actual body orientation)
        root_orn_wxyz = np.array(
            [root_body_quat_xyzw[3], root_body_quat_xyzw[0], root_body_quat_xyzw[1], root_body_quat_xyzw[2]]
        )

        return q_joints, dq_joints, root_orn_wxyz

    def reset(self):
        """Reset IK state."""
        self._config.update(np.zeros(self._mj_model.nq))
        self._prev_q_joints = np.zeros(self._mj_model.nq)
        self._prev_root_pos = np.zeros(3)
        self._prev_root_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self._prev_q = np.zeros(3 + 4 + self._mj_model.nq)
        self._prev_q[6] = 1.0  # identity quaternion w
        self._initialized = False
        self._height_ratio = None
