"""Dedicated Xsens T-pose calibration IK for G1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np
from scipy.optimize import least_squares, minimize_scalar  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from holosoma_retargeting.config_types.data_type import XSENS_DEMO_JOINTS
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XsensHdf5Tpose,
    load_xsens_hdf5_tpose,
    resolve_xsens_hdf5_path,
)
from holosoma_retargeting.src.mujoco_utils import (
    MujocoFrameRef,
    mujoco_frame_pose,
    resolve_mujoco_frame,
    resolve_mujoco_frames,
)
from holosoma_retargeting.transformation_utils import (
    rotation_matrices_as_wxyz,
    rotation_matrices_from_wxyz,
)
from holosoma_retargeting.xsens.orientation_tracking import build_xsens_axis_calibration_metadata

CALIBRATION_POSITION_MAPPING = {
    "Pelvis": "pelvis_contour_link",
    "L5": "torso_link",
    "Head": "head_link",
    "Left Shoulder": "left_shoulder_pitch_link",
    "Right Shoulder": "right_shoulder_pitch_link",
    "Left Upper Arm": "left_shoulder_yaw_link",
    "Right Upper Arm": "right_shoulder_yaw_link",
    "Left Forearm": "left_elbow_link",
    "Right Forearm": "right_elbow_link",
    "Left Hand": "left_wrist_yaw_link",
    "Right Hand": "right_wrist_yaw_link",
    "Left Upper Leg": "left_hip_yaw_link",
    "Right Upper Leg": "right_hip_yaw_link",
    "Left Lower Leg": "left_knee_link",
    "Right Lower Leg": "right_knee_link",
    "Left Foot": "left_ankle_roll_link",
    "Right Foot": "right_ankle_roll_link",
    "Left Toe": "left_ankle_roll_metatarsal_site",
    "Right Toe": "right_ankle_roll_metatarsal_site",
}

CANDIDATE_ORIENTATION_MAPPING = {
    "L5": "torso_link",
    "Left Foot": "left_ankle_roll_link",
    "Right Foot": "right_ankle_roll_link",
    "Left Hand": "left_rubber_hand_link",
    "Right Hand": "right_rubber_hand_link",
}


@dataclass(frozen=True)
class CalibrationAxisTarget:
    """One T-pose direction target represented by two frames or a body-fixed axis."""

    source_start: str
    source_end: str
    robot_start: str
    robot_end: str | None
    name: str
    robot_local_axis: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        has_end_frame = self.robot_end is not None
        has_local_axis = self.robot_local_axis is not None
        if has_end_frame == has_local_axis:
            raise ValueError("A calibration axis must define exactly one robot end frame or body-fixed local axis")


LIMB_AXIS_TARGETS = (
    CalibrationAxisTarget(
        "Left Upper Arm", "Left Forearm", "left_shoulder_yaw_link", "left_elbow_link", "left_upper_arm"
    ),
    CalibrationAxisTarget(
        "Right Upper Arm", "Right Forearm", "right_shoulder_yaw_link", "right_elbow_link", "right_upper_arm"
    ),
    CalibrationAxisTarget("Left Forearm", "Left Hand", "left_elbow_link", "left_wrist_yaw_link", "left_forearm"),
    CalibrationAxisTarget("Right Forearm", "Right Hand", "right_elbow_link", "right_wrist_yaw_link", "right_forearm"),
    # Match the runtime whole-hip direction; the Upper Leg point itself is
    # calibrated against the distal hip-yaw origin above.
    CalibrationAxisTarget("Left Upper Leg", "Left Lower Leg", "left_hip_pitch_link", "left_knee_link", "left_thigh"),
    CalibrationAxisTarget(
        "Right Upper Leg", "Right Lower Leg", "right_hip_pitch_link", "right_knee_link", "right_thigh"
    ),
    CalibrationAxisTarget("Left Lower Leg", "Left Foot", "left_knee_link", "left_ankle_pitch_link", "left_shank"),
    CalibrationAxisTarget("Right Lower Leg", "Right Foot", "right_knee_link", "right_ankle_pitch_link", "right_shank"),
    CalibrationAxisTarget("Left Foot", "Left Toe", "left_ankle_roll_link", None, "left_foot", (1.0, 0.0, 0.0)),
    CalibrationAxisTarget("Right Foot", "Right Toe", "right_ankle_roll_link", None, "right_foot", (1.0, 0.0, 0.0)),
    CalibrationAxisTarget("Pelvis", "L5", "pelvis_contour_link", "torso_link", "torso"),
)

SYMMETRY_LINK_PAIRS = (
    ("left_shoulder_yaw_link", "right_shoulder_yaw_link", "shoulder"),
    ("left_elbow_link", "right_elbow_link", "elbow"),
    ("left_wrist_yaw_link", "right_wrist_yaw_link", "hand"),
    ("left_hip_yaw_link", "right_hip_yaw_link", "hip"),
    ("left_knee_link", "right_knee_link", "knee"),
    ("left_ankle_pitch_link", "right_ankle_pitch_link", "ankle"),
    ("left_ankle_roll_metatarsal_site", "right_ankle_roll_metatarsal_site", "toe"),
)

ARM_COLLINEARITY_TARGETS = (
    ("left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_arm"),
    ("right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_arm"),
)

FOOT_CONTACT_GEOMS = {
    side: tuple(f"{side}_ankle_roll_sphere_{index}_link" for index in range(1, 6)) for side in ("left", "right")
}

LEG_ALIGNMENT_TARGETS = (
    ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left"),
    ("right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right"),
)

HAND_ORIENTATION_TARGETS = (
    # In the G1 rubber-hand frames, local +Z points from the pinky toward the
    # thumb.  Xsens defines a T-pose with both thumbs pointing character-forward.
    ("left_rubber_hand_link", np.array([0.0, 0.0, 1.0]), 0, 1.0, "left_thumb_forward"),
    ("right_rubber_hand_link", np.array([0.0, 0.0, 1.0]), 0, 1.0, "right_thumb_forward"),
    # The hand plane is local X-Z, so its local Y normal must be vertical.  The
    # mirrored right-hand frame has the opposite normal in the same T-pose.
    ("left_rubber_hand_link", np.array([0.0, 1.0, 0.0]), 2, 1.0, "left_hand_horizontal"),
    ("right_rubber_hand_link", np.array([0.0, 1.0, 0.0]), 2, -1.0, "right_hand_horizontal"),
)

MIRRORED_JOINT_PAIRS = (
    ("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
    ("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
    ("left_hip_yaw_joint", "right_hip_yaw_joint", -1.0),
    ("left_knee_joint", "right_knee_joint", 1.0),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1.0),
    ("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0),
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1.0),
    ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1.0),
    ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1.0),
    ("left_elbow_joint", "right_elbow_joint", 1.0),
    ("left_wrist_roll_joint", "right_wrist_roll_joint", -1.0),
    ("left_wrist_pitch_joint", "right_wrist_pitch_joint", 1.0),
    ("left_wrist_yaw_joint", "right_wrist_yaw_joint", -1.0),
)


@dataclass(frozen=True)
class XsensTposeCalibrationConfig:
    """Configuration for dedicated Xsens T-pose calibration IK."""

    robot_type: str = "g1"
    variant: str = "Tpose"
    robot_urdf_file: str | None = None
    default_human_height: float = 1.78
    fps: int = 30
    max_nfev: int = 400
    verbose: int = 1
    base_xy_radius_m: float = 0.5
    base_z_margin_m: float = 0.35
    position_weight: float = 100.0
    head_position_weight: float = 100.0
    hand_position_weight: float = 100.0
    axis_weight: float = 4.0
    symmetry_weight: float = 7.0
    straightness_weight: float = 100.0
    base_heading_weight: float = 100000.0
    torso_upright_weight: float = 2000.0
    torso_heading_weight: float = 2000.0
    foot_flat_weight: float = 100000.0
    foot_heading_weight: float = 200000.0
    foot_ground_weight: float = 500000.0
    knee_alignment_weight: float = 500000.0
    foot_stance_weight: float = 500000.0
    leg_collinearity_weight: float = 30000.0
    leg_verticality_weight: float = 30000.0
    ankle_roll_neutral_weight: float = 50000.0
    nominal_weight: float = 0.5
    wrist_neutral_weight: float = 0.5
    arm_collinearity_weight: float = 5000.0
    arm_horizontal_weight: float = 5000.0
    hand_orientation_weight: float = 2000.0
    mirror_joint_weight: float = 12.0
    active_position_error_threshold_m: float = 0.15
    head_axis_error_threshold_deg: float = 15.0
    head_orientation_offset_threshold_deg: float = 140.0
    candidate_orientation_mapping: dict[str, str] = field(default_factory=lambda: dict(CANDIDATE_ORIENTATION_MAPPING))


@dataclass(frozen=True)
class XsensTposeCalibrationResult:
    """Result of Xsens T-pose calibration IK."""

    qpos: np.ndarray
    xsens_tpose_positions_m: np.ndarray
    xsens_tpose_quaternions_wijk: np.ndarray
    candidate_orientation_mapping_names: list[str]
    candidate_robot_link_names: list[str]
    active_orientation_mapping_names: list[str]
    robot_link_names: list[str]
    orientation_offsets_wijk: np.ndarray
    axis_names: list[str]
    axis_xsens_segment_names: list[str]
    axis_local_tpose_xyz: np.ndarray
    axis_robot_start_link_names: list[str]
    axis_robot_end_link_names: list[str]
    axis_robot_local_vectors: np.ndarray
    axis_weights: np.ndarray
    position_error_names: list[str]
    position_errors_m: np.ndarray
    position_offsets_robot_minus_xsens_m: np.ndarray
    axis_error_names: list[str]
    axis_errors_deg: np.ndarray
    head_candidate_status: str
    solver_cost: float
    solver_success: bool
    variant: str
    scale_factor: float


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_model_path(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.is_file():
        return model_path
    package_relative = _package_root() / model_path
    if package_relative.is_file():
        return package_relative
    raise FileNotFoundError(f"Robot model not found: {path}")


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is None:
        fallback = np.zeros_like(vector)
    return np.asarray(fallback, dtype=float)


def _segment_index(name: str) -> int:
    return XSENS_DEMO_JOINTS.index(name)


def compute_canonical_axes(positions_m: np.ndarray) -> np.ndarray:
    """Compute right-handed `[forward, left, up]` axes from Xsens T-pose geometry."""
    world_up = np.array([0.0, 0.0, 1.0])
    left = positions_m[_segment_index("Left Upper Leg")] - positions_m[_segment_index("Right Upper Leg")]
    left[2] = 0.0

    left_foot = positions_m[_segment_index("Left Toe")] - positions_m[_segment_index("Left Foot")]
    right_foot = positions_m[_segment_index("Right Toe")] - positions_m[_segment_index("Right Foot")]
    forward = 0.5 * (left_foot + right_foot)
    forward[2] = 0.0

    forward = _normalize(forward, np.array([1.0, 0.0, 0.0]))
    left = left - np.dot(left, forward) * forward
    left = _normalize(left, np.array([0.0, 1.0, 0.0]))
    up = _normalize(np.cross(forward, left), world_up)
    if float(np.dot(up, world_up)) < 0.0:
        left = -left
        up = -up
    left = _normalize(np.cross(up, forward), np.array([0.0, 1.0, 0.0]))
    return np.vstack([forward, left, up])


def align_and_scale_tpose_targets(
    positions_m: np.ndarray,
    *,
    scale_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale, ground-align, and pelvis-center Xsens T-pose positions."""
    positions = np.asarray(positions_m, dtype=float).copy() * float(scale_factor)
    foot_indices = [
        _segment_index("Left Foot"),
        _segment_index("Right Foot"),
        _segment_index("Left Toe"),
        _segment_index("Right Toe"),
    ]
    positions[:, 2] -= float(np.min(positions[foot_indices, 2]))
    pelvis_xy = positions[_segment_index("Pelvis"), :2].copy()
    positions[:, :2] -= pelvis_xy
    axes = compute_canonical_axes(positions)
    return positions, axes


def limb_axis_error_deg(
    source_proximal: np.ndarray,
    source_distal: np.ndarray,
    target_proximal: np.ndarray,
    target_distal: np.ndarray,
) -> float:
    source_axis = _normalize(np.asarray(source_distal) - np.asarray(source_proximal), np.array([1.0, 0.0, 0.0]))
    target_axis = _normalize(np.asarray(target_distal) - np.asarray(target_proximal), np.array([1.0, 0.0, 0.0]))
    dot = float(np.clip(np.dot(source_axis, target_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def elbow_bend_angle_deg(shoulder_point: np.ndarray, elbow_point: np.ndarray, wrist_point: np.ndarray) -> float:
    """Return the bend angle between shoulder->elbow and elbow->wrist segments."""
    upper_axis = _normalize(np.asarray(elbow_point) - np.asarray(shoulder_point), np.array([1.0, 0.0, 0.0]))
    forearm_axis = _normalize(np.asarray(wrist_point) - np.asarray(elbow_point), upper_axis)
    dot = float(np.clip(np.dot(upper_axis, forearm_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def axis_alignment_error_deg(axis: np.ndarray, target_axis: np.ndarray) -> float:
    """Return the angle between two axes in degrees."""
    axis = _normalize(np.asarray(axis), np.array([1.0, 0.0, 0.0]))
    target_axis = _normalize(np.asarray(target_axis), np.array([1.0, 0.0, 0.0]))
    dot = float(np.clip(np.dot(axis, target_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def symmetry_residual(left_point: np.ndarray, right_point: np.ndarray) -> np.ndarray:
    """Residual for mirror symmetry across the X-Z sagittal plane."""
    left_point = np.asarray(left_point, dtype=float)
    right_point = np.asarray(right_point, dtype=float)
    return np.array(
        [
            left_point[0] - right_point[0],
            left_point[1] + right_point[1],
            left_point[2] - right_point[2],
        ],
        dtype=float,
    )


class _G1CalibrationProblem:
    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        target_positions: np.ndarray,
        target_axes: np.ndarray,
        config: XsensTposeCalibrationConfig,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.target_positions = target_positions
        self.target_axes = target_axes
        self.config = config
        self.actuated_joint_names = [
            model.joint(i).name for i in range(model.njnt) if model.joint(i).name and model.jnt_qposadr[i] >= 7
        ]
        self.joint_name_to_qadr = {
            model.joint(i).name: int(model.jnt_qposadr[i])
            for i in range(model.njnt)
            if model.joint(i).name and model.jnt_qposadr[i] >= 7
        }
        self.joint_name_to_id = {model.joint(i).name: i for i in range(model.njnt) if model.joint(i).name}
        self.body_name_to_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
        self.geom_name_to_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
        frame_names = list(CALIBRATION_POSITION_MAPPING.values())
        frame_names.extend(config.candidate_orientation_mapping.values())
        for target in LIMB_AXIS_TARGETS:
            frame_names.append(target.robot_start)
            if target.robot_end is not None:
                frame_names.append(target.robot_end)
        frame_names.extend(frame for pair in SYMMETRY_LINK_PAIRS for frame in pair[:2])
        self.frame_refs = resolve_mujoco_frames(model, frame_names)
        self.nominal_half_hip_width = self._compute_nominal_half_hip_width()
        self.straight_elbow_qpos = self._compute_straight_elbow_qpos()
        self.nominal_qpos = self._initial_qpos()

    def _joint_qpos_index(self, joint_name: str) -> int:
        return self.joint_name_to_qadr[joint_name]

    def _joint_position(self, joint_name: str) -> np.ndarray:
        return self.data.xanchor[self.joint_name_to_id[joint_name]].copy()

    def _foot_contact_positions(self, side: str) -> np.ndarray:
        return np.asarray(
            [self.data.geom_xpos[self.geom_name_to_id[name]] for name in FOOT_CONTACT_GEOMS[side]],
            dtype=float,
        )

    def _foot_sole_axes(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        contacts = self._foot_contact_positions(side)
        heel_center = np.mean(contacts[:2], axis=0)
        toe_center = np.mean(contacts[2:], axis=0)
        positive_lateral = np.mean(contacts[[0, 2]], axis=0)
        negative_lateral = np.mean(contacts[[1, 3]], axis=0)
        forward = _normalize(toe_center - heel_center, self.target_axes[0])
        lateral = _normalize(positive_lateral - negative_lateral, self.target_axes[1])
        normal = _normalize(np.cross(forward, lateral), self.target_axes[2])
        if float(np.dot(normal, self.target_axes[2])) < 0.0:
            normal = -normal
        return forward, normal

    def _compute_nominal_half_hip_width(self) -> float:
        """Return the fixed G1 pelvis-to-hip lateral offset in its neutral pose."""
        reference_qpos = np.zeros(self.model.nq, dtype=float)
        reference_qpos[3] = 1.0
        self.set_qpos(reference_qpos)
        left_hip = self._joint_position("left_hip_pitch_joint")
        right_hip = self._joint_position("right_hip_pitch_joint")
        return 0.5 * float(np.linalg.norm(left_hip - right_hip))

    def _compute_straight_elbow_qpos(self) -> dict[str, float]:
        """Find each elbow angle that best aligns its adjacent joint-center segments."""
        reference_qpos = np.zeros(self.model.nq, dtype=float)
        reference_qpos[3] = 1.0
        straight_qpos: dict[str, float] = {}

        for shoulder_joint, elbow_joint, wrist_joint, _name in ARM_COLLINEARITY_TARGETS:
            elbow_id = self.joint_name_to_id[elbow_joint]
            elbow_qadr = self._joint_qpos_index(elbow_joint)
            lower, upper = self.model.jnt_range[elbow_id]

            def collinearity_error(
                elbow_angle: float,
                *,
                elbow_qadr: int = elbow_qadr,
                shoulder_joint: str = shoulder_joint,
                elbow_joint: str = elbow_joint,
                wrist_joint: str = wrist_joint,
            ) -> float:
                reference_qpos[elbow_qadr] = elbow_angle
                self.set_qpos(reference_qpos)
                upper_axis = _normalize(
                    self._joint_position(elbow_joint) - self._joint_position(shoulder_joint),
                    np.array([1.0, 0.0, 0.0]),
                )
                forearm_axis = _normalize(
                    self._joint_position(wrist_joint) - self._joint_position(elbow_joint),
                    upper_axis,
                )
                return float(np.dot(forearm_axis - upper_axis, forearm_axis - upper_axis))

            result = minimize_scalar(
                collinearity_error,
                bounds=(float(lower), float(upper)),
                method="bounded",
                options={"xatol": 1e-12},
            )
            if not result.success:
                raise RuntimeError(f"Could not derive a straight reference for {elbow_joint}: {result.message}")
            straight_qpos[elbow_joint] = float(result.x)

        return straight_qpos

    def _frame_ref(self, frame_name: str) -> MujocoFrameRef:
        if frame_name not in self.frame_refs:
            self.frame_refs[frame_name] = resolve_mujoco_frame(self.model, frame_name)
        return self.frame_refs[frame_name]

    def _frame_position(self, frame_name: str) -> np.ndarray:
        return mujoco_frame_pose(self.model, self.data, self._frame_ref(frame_name))[0]

    def _frame_rotation(self, frame_name: str) -> np.ndarray:
        return mujoco_frame_pose(self.model, self.data, self._frame_ref(frame_name))[1]

    def _robot_axis(self, target: CalibrationAxisTarget, fallback: np.ndarray) -> np.ndarray:
        if target.robot_end is not None:
            return _normalize(
                self._frame_position(target.robot_end) - self._frame_position(target.robot_start),
                fallback,
            )
        if target.robot_local_axis is None:
            raise ValueError(f"Body-fixed calibration axis '{target.name}' has no local vector")
        return _normalize(
            self._frame_rotation(target.robot_start) @ np.asarray(target.robot_local_axis, dtype=float),
            fallback,
        )

    def _initial_qpos(self) -> np.ndarray:
        qpos = np.zeros(self.model.nq, dtype=float)
        yaw = float(np.arctan2(self.target_axes[0, 1], self.target_axes[0, 0]))
        qpos[:3] = self.target_positions[_segment_index("Pelvis")]
        qpos[3:7] = _yaw_quat_wxyz(yaw)
        for joint_name, value in {
            "left_shoulder_roll_joint": 1.3,
            "right_shoulder_roll_joint": -1.3,
            "left_elbow_joint": self.straight_elbow_qpos["left_elbow_joint"],
            "right_elbow_joint": self.straight_elbow_qpos["right_elbow_joint"],
        }.items():
            if joint_name in self.joint_name_to_qadr:
                qpos[self._joint_qpos_index(joint_name)] = value
        return qpos

    def x_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        yaw = Rotation.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_euler("zyx")[0]
        return np.concatenate([qpos[:3], [yaw], qpos[7:]])

    def qpos_from_x(self, x: np.ndarray) -> np.ndarray:
        qpos = np.zeros(self.model.nq, dtype=float)
        qpos[:3] = x[:3]
        qpos[3:7] = _yaw_quat_wxyz(float(x[3]))
        qpos[7:] = x[4:]
        return qpos

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        q0 = self.nominal_qpos
        lower = np.empty(4 + len(self.actuated_joint_names), dtype=float)
        upper = np.empty_like(lower)
        lower[:2] = q0[:2] - self.config.base_xy_radius_m
        upper[:2] = q0[:2] + self.config.base_xy_radius_m
        lower[2] = max(0.2, q0[2] - self.config.base_z_margin_m)
        upper[2] = q0[2] + self.config.base_z_margin_m
        lower[3] = -np.pi
        upper[3] = np.pi

        joint_lowers = []
        joint_uppers = []
        for joint_name in self.actuated_joint_names:
            qadr = self._joint_qpos_index(joint_name)
            joint_id = self.model.jnt_qposadr.tolist().index(qadr)
            joint_lowers.append(float(self.model.jnt_range[joint_id, 0]))
            joint_uppers.append(float(self.model.jnt_range[joint_id, 1]))
        lower[4:] = joint_lowers
        upper[4:] = joint_uppers

        robot_config = RobotConfig(robot_type=self.config.robot_type, robot_urdf_file=self.config.robot_urdf_file)
        for key, value in robot_config.MANUAL_LB.items():
            qadr = int(key)
            if qadr >= 7:
                lower[4 + qadr - 7] = max(lower[4 + qadr - 7], float(value))
        for key, value in robot_config.MANUAL_UB.items():
            qadr = int(key)
            if qadr >= 7:
                upper[4 + qadr - 7] = min(upper[4 + qadr - 7], float(value))
        return lower, upper

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        qpos = self.qpos_from_x(x)
        self.set_qpos(qpos)
        residuals: list[np.ndarray] = []

        self._append_position_residuals(residuals)
        self._append_ground_contact_residuals(residuals)
        self._append_leg_alignment_residuals(residuals)
        self._append_axis_residuals(residuals)
        self._append_arm_collinearity_residuals(residuals)
        self._append_arm_horizontal_residuals(residuals)
        self._append_symmetry_residuals(residuals)
        self._append_joint_priors(residuals, qpos)
        self._append_orientation_priors(residuals)
        return np.concatenate(residuals)

    def _append_position_residuals(self, residuals: list[np.ndarray]) -> None:
        for xsens_name, link_name in CALIBRATION_POSITION_MAPPING.items():
            if xsens_name == "Head":
                weight = self.config.head_position_weight
            elif xsens_name in ("Left Hand", "Right Hand"):
                weight = self.config.hand_position_weight
            else:
                weight = self.config.position_weight
            target = self.target_positions[_segment_index(xsens_name)]
            current = self._frame_position(link_name)
            residuals.append(np.sqrt(weight) * (current - target))

    def _append_ground_contact_residuals(self, residuals: list[np.ndarray]) -> None:
        for side, geom_names in FOOT_CONTACT_GEOMS.items():
            contact_positions = self._foot_contact_positions(side)
            contact_radii = np.asarray(
                [self.model.geom_size[self.geom_name_to_id[name], 0] for name in geom_names],
                dtype=float,
            )
            contact_bottom_heights = contact_positions[:, 2] - contact_radii
            residuals.append(np.sqrt(self.config.foot_ground_weight) * contact_bottom_heights)

    def _append_leg_alignment_residuals(self, residuals: list[np.ndarray]) -> None:
        lateral = self.target_axes[1]
        base_lateral = float(np.dot(self.data.qpos[:3], lateral))
        down = -self.target_axes[2]
        for hip_joint, knee_joint, ankle_joint, side in LEG_ALIGNMENT_TARGETS:
            hip_position = self._joint_position(hip_joint)
            knee_position = self._joint_position(knee_joint)
            ankle_position = self._joint_position(ankle_joint)
            foot_center = np.mean(self._foot_contact_positions(side), axis=0)
            side_sign = 1.0 if side == "left" else -1.0
            target_lateral = base_lateral + side_sign * self.nominal_half_hip_width
            knee_offset = float(np.dot(knee_position, lateral) - target_lateral)
            foot_offset = float(np.dot(foot_center, lateral) - target_lateral)
            residuals.append(np.array([np.sqrt(self.config.knee_alignment_weight) * knee_offset]))
            residuals.append(np.array([np.sqrt(self.config.foot_stance_weight) * foot_offset]))
            upper_leg_axis = _normalize(knee_position - hip_position, down)
            lower_leg_axis = _normalize(ankle_position - knee_position, down)
            full_leg_axis = _normalize(ankle_position - hip_position, down)
            residuals.append(np.sqrt(self.config.leg_collinearity_weight) * (lower_leg_axis - upper_leg_axis))
            residuals.append(np.sqrt(self.config.leg_verticality_weight) * (full_leg_axis - down))

    def _append_axis_residuals(self, residuals: list[np.ndarray]) -> None:
        for target in LIMB_AXIS_TARGETS:
            target_axis = _normalize(
                self.target_positions[_segment_index(target.source_end)]
                - self.target_positions[_segment_index(target.source_start)],
                np.array([1.0, 0.0, 0.0]),
            )
            robot_axis = self._robot_axis(target, target_axis)
            residuals.append(np.sqrt(self.config.axis_weight) * (robot_axis - target_axis))

    def _append_arm_collinearity_residuals(self, residuals: list[np.ndarray]) -> None:
        for shoulder_joint, elbow_joint, wrist_joint, _name in ARM_COLLINEARITY_TARGETS:
            upper_axis = _normalize(
                self._joint_position(elbow_joint) - self._joint_position(shoulder_joint),
                np.array([1.0, 0.0, 0.0]),
            )
            forearm_axis = _normalize(
                self._joint_position(wrist_joint) - self._joint_position(elbow_joint),
                upper_axis,
            )
            residuals.append(np.sqrt(self.config.arm_collinearity_weight) * (forearm_axis - upper_axis))

    def _append_arm_horizontal_residuals(self, residuals: list[np.ndarray]) -> None:
        for shoulder_joint, _elbow_joint, wrist_joint, name in ARM_COLLINEARITY_TARGETS:
            arm_axis = _normalize(
                self._joint_position(wrist_joint) - self._joint_position(shoulder_joint),
                self.target_axes[1],
            )
            target_axis = self.target_axes[1] if name == "left_arm" else -self.target_axes[1]
            residuals.append(np.sqrt(self.config.arm_horizontal_weight) * (arm_axis - target_axis))

    def _append_symmetry_residuals(self, residuals: list[np.ndarray]) -> None:
        for left_link, right_link, _name in SYMMETRY_LINK_PAIRS:
            residuals.append(
                np.sqrt(self.config.symmetry_weight)
                * symmetry_residual(self._frame_position(left_link), self._frame_position(right_link))
            )

    def _append_joint_priors(self, residuals: list[np.ndarray], qpos: np.ndarray) -> None:
        base_yaw = 2.0 * np.arctan2(qpos[6], qpos[3])
        target_yaw = float(np.arctan2(self.target_axes[0, 1], self.target_axes[0, 0]))
        yaw_error = np.arctan2(np.sin(base_yaw - target_yaw), np.cos(base_yaw - target_yaw))
        residuals.append(np.array([np.sqrt(self.config.base_heading_weight) * yaw_error]))

        straight_joint_names = (
            "left_knee_joint",
            "right_knee_joint",
        )
        residuals.extend(
            np.array([np.sqrt(self.config.straightness_weight) * qpos[self._joint_qpos_index(joint_name)]])
            for joint_name in straight_joint_names
        )

        for joint_name, straight_qpos in self.straight_elbow_qpos.items():
            elbow_qpos = qpos[self._joint_qpos_index(joint_name)]
            residuals.append(np.array([np.sqrt(self.config.straightness_weight) * (elbow_qpos - straight_qpos)]))

        for side in ("left", "right"):
            ankle_roll = qpos[self._joint_qpos_index(f"{side}_ankle_roll_joint")]
            residuals.append(np.array([np.sqrt(self.config.ankle_roll_neutral_weight) * ankle_roll]))

        waist_joint_names = (
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        )
        residuals.extend(
            np.array([np.sqrt(self.config.nominal_weight) * qpos[self._joint_qpos_index(joint_name)]])
            for joint_name in waist_joint_names
        )

        wrist_joint_names = (
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        )
        residuals.extend(
            np.array([np.sqrt(self.config.wrist_neutral_weight) * qpos[self._joint_qpos_index(joint_name)]])
            for joint_name in wrist_joint_names
        )

        for left_joint, right_joint, sign in MIRRORED_JOINT_PAIRS:
            left_value = qpos[self._joint_qpos_index(left_joint)]
            right_value = qpos[self._joint_qpos_index(right_joint)]
            residuals.append(np.array([np.sqrt(self.config.mirror_joint_weight) * (left_value - sign * right_value)]))

    def _append_orientation_priors(self, residuals: list[np.ndarray]) -> None:
        forward = self.target_axes[0]
        up = self.target_axes[2]
        torso_rotation = self._frame_rotation("torso_link")
        torso_forward = torso_rotation[:, 0]
        torso_up = torso_rotation[:, 2]
        residuals.append(np.sqrt(self.config.torso_heading_weight) * (torso_forward - forward))
        residuals.append(np.sqrt(self.config.torso_upright_weight) * (torso_up - up))

        for side in FOOT_CONTACT_GEOMS:
            foot_forward, foot_up = self._foot_sole_axes(side)
            residuals.append(np.sqrt(self.config.foot_flat_weight) * (foot_up - up))
            residuals.append(np.sqrt(self.config.foot_heading_weight) * (foot_forward - self.target_axes[0]))

        for hand_link, local_axis, target_axis_index, target_sign, _name in HAND_ORIENTATION_TARGETS:
            hand_axis = self._frame_rotation(hand_link) @ local_axis
            target_axis = target_sign * self.target_axes[target_axis_index]
            residuals.append(np.sqrt(self.config.hand_orientation_weight) * (hand_axis - target_axis))


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=float)


def _robot_urdf_for_config(config: XsensTposeCalibrationConfig) -> Path:
    if config.robot_type != "g1":
        raise ValueError("Xsens T-pose calibration currently supports robot_type='g1' only")
    robot_config = RobotConfig(robot_type=config.robot_type, robot_urdf_file=config.robot_urdf_file)
    return _resolve_model_path(robot_config.ROBOT_URDF_FILE)


def _solve_problem(
    problem: _G1CalibrationProblem,
    config: XsensTposeCalibrationConfig,
) -> tuple[np.ndarray, float, bool]:
    x0 = problem.x_from_qpos(problem.nominal_qpos)
    lower, upper = problem.bounds()
    result = least_squares(
        problem.residuals,
        x0,
        bounds=(lower, upper),
        max_nfev=config.max_nfev,
        verbose=config.verbose,
        x_scale="jac",
    )
    qpos = problem.qpos_from_x(result.x)
    problem.set_qpos(qpos)
    lowest_contact_height = min(
        float(problem.data.geom_xpos[problem.geom_name_to_id[name], 2])
        - float(problem.model.geom_size[problem.geom_name_to_id[name], 0])
        for geom_names in FOOT_CONTACT_GEOMS.values()
        for name in geom_names
    )
    qpos[2] -= lowest_contact_height
    final_residuals = problem.residuals(problem.x_from_qpos(qpos))
    final_cost = 0.5 * float(np.dot(final_residuals, final_residuals))
    return qpos, final_cost, bool(result.success)


def _position_errors(
    problem: _G1CalibrationProblem,
    qpos: np.ndarray,
    mapping: dict[str, str],
) -> dict[str, float]:
    problem.set_qpos(qpos)
    errors = {}
    for xsens_name, link_name in mapping.items():
        target = problem.target_positions[_segment_index(xsens_name)]
        current = problem._frame_position(link_name)
        errors[xsens_name] = float(np.linalg.norm(current - target))
    return errors


def _position_offsets(
    problem: _G1CalibrationProblem,
    qpos: np.ndarray,
    mapping: dict[str, str],
) -> dict[str, np.ndarray]:
    """Return calibrated robot-link-origin minus Xsens-segment-origin offsets."""
    problem.set_qpos(qpos)
    return {
        xsens_name: problem._frame_position(link_name) - problem.target_positions[_segment_index(xsens_name)]
        for xsens_name, link_name in mapping.items()
    }


def _axis_errors(problem: _G1CalibrationProblem, qpos: np.ndarray) -> dict[str, float]:
    problem.set_qpos(qpos)
    errors = {}
    for target in LIMB_AXIS_TARGETS:
        target_axis = _normalize(
            problem.target_positions[_segment_index(target.source_end)]
            - problem.target_positions[_segment_index(target.source_start)],
            np.array([1.0, 0.0, 0.0]),
        )
        robot_axis = problem._robot_axis(target, target_axis)
        errors[target.name] = float(np.degrees(np.arccos(np.clip(np.dot(robot_axis, target_axis), -1.0, 1.0))))
    return errors


def _orientation_offsets(
    problem: _G1CalibrationProblem,
    qpos: np.ndarray,
    quaternions_wijk: np.ndarray,
    mapping: dict[str, str],
) -> dict[str, np.ndarray]:
    problem.set_qpos(qpos)
    xsens_rotations = rotation_matrices_from_wxyz(quaternions_wijk)
    offsets = {}
    for xsens_name, link_name in mapping.items():
        xsens_idx = _segment_index(xsens_name)
        robot_rotation = problem._frame_rotation(link_name)
        offset = xsens_rotations[xsens_idx].T @ robot_rotation
        offsets[xsens_name] = rotation_matrices_as_wxyz(offset, canonical=False)
    return offsets


def _offset_angle_deg(offset_wijk: np.ndarray) -> float:
    matrix = rotation_matrices_from_wxyz(np.asarray(offset_wijk).reshape(1, 4))[0]
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(matrix).as_rotvec())))


def evaluate_head_candidate(
    *,
    head_axis_error_deg: float,
    head_orientation_offset_wijk: np.ndarray,
    config: XsensTposeCalibrationConfig,
) -> tuple[bool, str]:
    """Return whether `Head -> head_link` passes calibration diagnostics."""
    offset_angle_deg = _offset_angle_deg(head_orientation_offset_wijk)
    accepted = (
        head_axis_error_deg <= config.head_axis_error_threshold_deg
        and offset_angle_deg <= config.head_orientation_offset_threshold_deg
    )
    status = "accepted" if accepted else "rejected"
    return (
        accepted,
        (f"{status}: axis_error_deg={head_axis_error_deg:.2f}, offset_angle_deg={offset_angle_deg:.1f}"),
    )


def _active_orientation_mapping(
    *,
    position_errors: dict[str, float],
    axis_errors: dict[str, float],
    offsets: dict[str, np.ndarray],
    config: XsensTposeCalibrationConfig,
) -> tuple[dict[str, str], str]:
    active = {}
    head_status = "not evaluated"
    for xsens_name, link_name in config.candidate_orientation_mapping.items():
        if xsens_name == "Head":
            accepted, head_status = evaluate_head_candidate(
                head_axis_error_deg=axis_errors["head"],
                head_orientation_offset_wijk=offsets[xsens_name],
                config=config,
            )
            if accepted:
                active[xsens_name] = link_name
            continue

        if position_errors[xsens_name] <= config.active_position_error_threshold_m:
            active[xsens_name] = link_name
    return active, head_status


def solve_xsens_tpose_calibration(
    hdf5_path: str | Path,
    config: XsensTposeCalibrationConfig | None = None,
) -> XsensTposeCalibrationResult:
    """Solve the dedicated G1 calibration IK for an Xsens HDF5 T-pose."""
    config = config or XsensTposeCalibrationConfig()
    tpose = load_xsens_hdf5_tpose(hdf5_path, variant=config.variant)
    return solve_xsens_tpose_calibration_from_data(tpose, config=config)


def solve_xsens_tpose_calibration_from_data(
    tpose: XsensHdf5Tpose,
    config: XsensTposeCalibrationConfig | None = None,
    *,
    position_scale_factor: float | None = None,
) -> XsensTposeCalibrationResult:
    """Solve calibration IK from an Xsens T-pose-like positional reference.

    When ``position_scale_factor`` is omitted, positions follow the legacy
    direct-human path and are scaled by the configured human/robot height
    ratio. G1-proportioned callers pass ``1.0`` because their positions already
    encode the robot morphology. Segment orientations are never rescaled.
    """

    config = config or XsensTposeCalibrationConfig()
    scale_factor = position_scale_factor
    if scale_factor is None:
        scale_factor = RobotConfig(robot_type=config.robot_type).ROBOT_HEIGHT / config.default_human_height
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("position_scale_factor must be finite and positive")
    scale_factor = float(scale_factor)
    target_positions, target_axes = align_and_scale_tpose_targets(tpose.positions_m, scale_factor=scale_factor)
    model = mujoco.MjModel.from_xml_path(str(_robot_urdf_for_config(config)).replace(".urdf", ".xml"))
    problem = _G1CalibrationProblem(
        model=model,
        target_positions=target_positions,
        target_axes=target_axes,
        config=config,
    )
    qpos, solver_cost, solver_success = _solve_problem(problem, config)

    candidate_mapping = dict(config.candidate_orientation_mapping)
    candidate_position_errors = _position_errors(problem, qpos, candidate_mapping)
    candidate_offsets = _orientation_offsets(problem, qpos, tpose.quaternions_wijk, candidate_mapping)
    axis_errors = _axis_errors(problem, qpos)
    active_mapping, head_status = _active_orientation_mapping(
        position_errors=candidate_position_errors,
        axis_errors=axis_errors,
        offsets=candidate_offsets,
        config=config,
    )
    active_offsets = [candidate_offsets[name] for name in active_mapping]
    all_position_errors = _position_errors(problem, qpos, CALIBRATION_POSITION_MAPPING)
    all_position_offsets = _position_offsets(problem, qpos, CALIBRATION_POSITION_MAPPING)
    axis_metadata = build_xsens_axis_calibration_metadata(
        tpose_positions_m=target_positions,
        tpose_quaternions_wijk=tpose.quaternions_wijk,
    )

    return XsensTposeCalibrationResult(
        qpos=qpos.reshape(1, -1),
        xsens_tpose_positions_m=target_positions,
        xsens_tpose_quaternions_wijk=tpose.quaternions_wijk,
        candidate_orientation_mapping_names=list(candidate_mapping.keys()),
        candidate_robot_link_names=list(candidate_mapping.values()),
        active_orientation_mapping_names=list(active_mapping.keys()),
        robot_link_names=list(active_mapping.values()),
        orientation_offsets_wijk=np.asarray(active_offsets, dtype=float).reshape(-1, 4),
        axis_names=[str(value) for value in axis_metadata["axis_names"]],
        axis_xsens_segment_names=[str(value) for value in axis_metadata["axis_xsens_segment_names"]],
        axis_local_tpose_xyz=np.asarray(axis_metadata["axis_local_tpose_xyz"], dtype=float),
        axis_robot_start_link_names=[str(value) for value in axis_metadata["axis_robot_start_link_names"]],
        axis_robot_end_link_names=[str(value) for value in axis_metadata["axis_robot_end_link_names"]],
        axis_robot_local_vectors=np.asarray(axis_metadata["axis_robot_local_vectors"], dtype=float),
        axis_weights=np.asarray(axis_metadata["axis_weights"], dtype=float),
        position_error_names=list(all_position_errors.keys()),
        position_errors_m=np.asarray(list(all_position_errors.values()), dtype=float),
        position_offsets_robot_minus_xsens_m=np.asarray(list(all_position_offsets.values()), dtype=float),
        axis_error_names=list(axis_errors.keys()),
        axis_errors_deg=np.asarray(list(axis_errors.values()), dtype=float),
        head_candidate_status=head_status,
        solver_cost=solver_cost,
        solver_success=solver_success,
        variant=tpose.variant,
        scale_factor=scale_factor,
    )


def save_xsens_tpose_calibration(result: XsensTposeCalibrationResult, path: str | Path, fps: int = 30) -> None:
    """Save calibration output as a Viser-compatible NPZ plus diagnostics."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        qpos=result.qpos,
        fps=np.array(fps, dtype=int),
        xsens_tpose_positions_m=result.xsens_tpose_positions_m,
        xsens_tpose_quaternions_wijk=result.xsens_tpose_quaternions_wijk,
        candidate_orientation_mapping_names=np.asarray(result.candidate_orientation_mapping_names, dtype=str),
        candidate_robot_link_names=np.asarray(result.candidate_robot_link_names, dtype=str),
        active_orientation_mapping_names=np.asarray(result.active_orientation_mapping_names, dtype=str),
        robot_link_names=np.asarray(result.robot_link_names, dtype=str),
        orientation_offsets_wijk=result.orientation_offsets_wijk,
        axis_names=np.asarray(result.axis_names, dtype=str),
        axis_xsens_segment_names=np.asarray(result.axis_xsens_segment_names, dtype=str),
        axis_local_tpose_xyz=result.axis_local_tpose_xyz,
        axis_robot_start_link_names=np.asarray(result.axis_robot_start_link_names, dtype=str),
        axis_robot_end_link_names=np.asarray(result.axis_robot_end_link_names, dtype=str),
        axis_robot_local_vectors=result.axis_robot_local_vectors,
        axis_weights=result.axis_weights,
        position_error_names=np.asarray(result.position_error_names, dtype=str),
        position_errors_m=result.position_errors_m,
        position_offsets_robot_minus_xsens_m=result.position_offsets_robot_minus_xsens_m,
        axis_error_names=np.asarray(result.axis_error_names, dtype=str),
        axis_errors_deg=result.axis_errors_deg,
        head_candidate_status=np.asarray(result.head_candidate_status),
        solver_cost=np.asarray(result.solver_cost, dtype=float),
        solver_success=np.asarray(result.solver_success, dtype=bool),
        variant=np.asarray(result.variant),
        scale_factor=np.asarray(result.scale_factor, dtype=float),
    )


def resolve_xsens_tennis_hdf5_path(data_path: str | Path, task_name: str) -> Path:
    """Resolve an Xsens tennis task name to its HDF5 file."""
    return resolve_xsens_hdf5_path(data_path, task_name)
