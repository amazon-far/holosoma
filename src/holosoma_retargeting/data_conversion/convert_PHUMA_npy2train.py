"""Convert PHUMA npy files to holosoma npz format.

This script:
1. Loads npy files (PHUMA format with root_trans, root_ori, dof_pos)
2. Converts to qpos format (like create_npy2npz.py)
3. Runs MuJoCo forward kinematics (headless, no viewer)
4. Saves final npz with body positions, velocities, etc.

Usage:
    # 기본 (prepend/append 둘 다 활성화)
    python src/holosoma_retargeting/data_conversion/convert_PHUMA_npy2train.py \
        --input-folder /path/to/npy \
        --output-folder /path/to/output

    # Transition 비활성화
    python ... --no-enable-default-pose-prepend --no-enable-default-pose-append

    # Transition 시간 조절
    python ... --default-pose-prepend-duration-s 1.0 --default-pose-append-duration-s 1.0
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple, cast

import mujoco  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402

# Base path for models directory
MODELS_BASE_PATH = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class PHUMAConversionConfig:
    """Configuration for PHUMA npy to npz conversion."""

    input_file: str | None = None
    """Path to input npy file (single file mode)."""

    input_folder: str | None = None
    """Path to folder containing npy files (batch mode)."""

    output_name: str | None = None
    """Output npz file path (single file mode)."""

    output_folder: str | None = None
    """Output folder for npz files (batch mode)."""

    robot: str = "g1"
    """Robot model to use."""

    input_fps: int = 30
    """FPS of the input motion."""

    output_fps: int = 50
    """FPS of the output motion."""

    data_format: str = "lafan"
    """Motion data format."""

    object_name: str = "ground"
    """Object name for the scene."""

    # Default pose transition settings
    enable_default_pose_prepend: bool = True
    """Whether to prepend interpolated frames from default pose to motion start."""

    enable_default_pose_append: bool = True
    """Whether to append interpolated frames from motion end to default pose."""

    default_pose_prepend_duration_s: float = 2
    """Duration in seconds for prepend transition."""

    default_pose_append_duration_s: float = 2
    """Duration in seconds for append transition."""


args_cli = tyro.cli(PHUMAConversionConfig)


# =============================================================================
# Quaternion utilities
# =============================================================================


def quat_conjugate(q):  # (...,4) [w,x,y,z]
    qc = q.clone()
    qc[..., 1:] = -qc[..., 1:]
    return qc


def quat_mul(a, b):  # Hamilton product, (...,4)
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_to_rotvec(q, eps=1e-8):  # axis-angle vector (rotvec), (...,3)
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    axis = torch.where(s[..., None] > eps, q[..., 1:] / s[..., None], torch.zeros_like(q[..., 1:]))
    return axis * angle[..., None]


def slerp_np(q0: np.ndarray, q1: np.ndarray, t: float, eps: float = 1e-8) -> np.ndarray:
    """Spherical linear interpolation for quaternions (numpy, single t).
    
    Args:
        q0: Start quaternion (..., 4) in xyzw format
        q1: End quaternion (..., 4) in xyzw format  
        t: Interpolation factor [0, 1]
    Returns:
        Interpolated quaternion (..., 4) in xyzw format
    """
    # Normalize
    q0 = q0 / (np.linalg.norm(q0, axis=-1, keepdims=True) + eps)
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + eps)
    
    # Dot product
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    
    # If dot < 0, negate q1 for shortest path
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    # Avoid division by zero
    close_mask = np.abs(sin_theta) < eps
    
    s0 = np.sin((1.0 - t) * theta) / (sin_theta + eps)
    s1 = np.sin(t * theta) / (sin_theta + eps)
    
    result = s0 * q0 + s1 * q1
    
    # For close quaternions, use lerp
    lerp_result = (1.0 - t) * q0 + t * q1
    result = np.where(close_mask, lerp_result, result)
    
    # Normalize result
    result = result / (np.linalg.norm(result, axis=-1, keepdims=True) + eps)
    return result


def lerp_np(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation."""
    return a * (1.0 - t) + b * t


def build_transition_frames(
    start_state: dict[str, np.ndarray],
    end_state: dict[str, np.ndarray],
    num_frames: int,
    drop_first: bool = False,
    drop_last: bool = False,
) -> dict[str, np.ndarray]:
    """Build interpolated transition frames between two states.
    
    Args:
        start_state: Dict with keys: joint_pos, joint_vel, body_pos_w, body_quat_w, 
                     body_lin_vel_w, body_ang_vel_w
        end_state: Same keys as start_state
        num_frames: Number of interpolation frames (including start and end)
        drop_first: If True, exclude the first frame (t=0)
        drop_last: If True, exclude the last frame (t=1)
    
    Returns:
        Dict with interpolated arrays, shape (num_frames, ...)
    """
    # Generate interpolation factors
    alphas = np.linspace(0.0, 1.0, num_frames + 1)
    if drop_first:
        alphas = alphas[1:]
    if drop_last:
        alphas = alphas[:-1]
    
    if len(alphas) == 0:
        return {k: np.zeros((0,) + v.shape, dtype=v.dtype) for k, v in start_state.items()}
    
    frames = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    
    for t in alphas:
        # Linear interpolation for positions and velocities
        frames["joint_pos"].append(lerp_np(start_state["joint_pos"], end_state["joint_pos"], t))
        frames["joint_vel"].append(lerp_np(start_state["joint_vel"], end_state["joint_vel"], t))
        frames["body_pos_w"].append(lerp_np(start_state["body_pos_w"], end_state["body_pos_w"], t))
        frames["body_lin_vel_w"].append(lerp_np(start_state["body_lin_vel_w"], end_state["body_lin_vel_w"], t))
        frames["body_ang_vel_w"].append(lerp_np(start_state["body_ang_vel_w"], end_state["body_ang_vel_w"], t))
        
        # Spherical interpolation for quaternions
        frames["body_quat_w"].append(slerp_np(start_state["body_quat_w"], end_state["body_quat_w"], t))
    
    return {k: np.stack(v, axis=0) for k, v in frames.items()}


def get_default_pose_state(
    robot_model,
    robot_data,
    default_joint_pos: np.ndarray,
    motion_first_frame: dict[str, np.ndarray],
    use_motion_xy: bool = True,
) -> dict[str, np.ndarray]:
    """Get the default standing pose state for the robot.
    
    Args:
        robot_model: MuJoCo model
        robot_data: MuJoCo data
        default_joint_pos: Default joint positions (29 DOF)
        motion_first_frame: First frame of motion data to align position/yaw
        use_motion_xy: If True, use motion's x,y position and yaw
    
    Returns:
        Dict with default pose state
    """
    # Default root position (standing height)
    default_root_pos = np.array([0.0, 0.0, 0.793])  # G1 standing height
    default_root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, identity
    
    if use_motion_xy and motion_first_frame is not None:
        # Use motion's x, y position
        motion_body_pos = motion_first_frame["body_pos_w"]
        default_root_pos[0] = motion_body_pos[0, 0]  # x from pelvis
        default_root_pos[1] = motion_body_pos[0, 1]  # y from pelvis
        
        # Extract yaw from motion's first frame root quaternion (xyzw format in motion)
        motion_root_quat_xyzw = motion_first_frame["body_quat_w"][0]  # pelvis quat
        # Convert to wxyz for easier yaw extraction
        motion_root_quat_wxyz = motion_root_quat_xyzw[[3, 0, 1, 2]]
        # Extract yaw (simplified - assumes mostly upright)
        yaw = np.arctan2(2.0 * (motion_root_quat_wxyz[0] * motion_root_quat_wxyz[3] + 
                                 motion_root_quat_wxyz[1] * motion_root_quat_wxyz[2]),
                         1.0 - 2.0 * (motion_root_quat_wxyz[2]**2 + motion_root_quat_wxyz[3]**2))
        # Create quaternion with only yaw (wxyz format)
        default_root_quat = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])
    
    # Set qpos and run forward kinematics
    robot_data.qpos[:3] = default_root_pos
    robot_data.qpos[3:7] = default_root_quat  # wxyz
    robot_data.qpos[7:] = default_joint_pos
    robot_data.qvel[:] = 0.0
    
    mujoco.mj_forward(robot_model, robot_data)
    
    # Extract state (convert quat from wxyz to xyzw)
    body_quat_wxyz = robot_data.xquat.copy()
    body_quat_xyzw = body_quat_wxyz[:, [1, 2, 3, 0]]
    
    return {
        "joint_pos": default_joint_pos.copy(),
        "joint_vel": np.zeros_like(default_joint_pos),
        "body_pos_w": robot_data.xpos.copy(),
        "body_quat_w": body_quat_xyzw,
        "body_lin_vel_w": np.zeros((robot_model.nbody, 3)),
        "body_ang_vel_w": np.zeros((robot_model.nbody, 3)),
    }


# =============================================================================
# Task constants creation
# =============================================================================


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    *,
    object_name: str | None = None,
) -> SimpleNamespace:
    """Create a mutable namespace with robot and motion data attributes."""
    namespace = SimpleNamespace()

    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(namespace, attr, getattr(robot_config, attr))

    for attr, value in motion_data_config.legacy_constants().items():
        setattr(namespace, attr, value)

    if object_name is not None:
        namespace.OBJECT_NAME = object_name

    # Convert relative paths to absolute paths using MODELS_BASE_PATH
    # ROBOT_URDF_FILE is like "models/g1/g1_29dof.urdf", we need to extract the relative part
    robot_urdf_relative = namespace.ROBOT_URDF_FILE.replace("models/", "")
    namespace.ROBOT_URDF_FILE = str(MODELS_BASE_PATH / robot_urdf_relative)

    if namespace.OBJECT_NAME != "ground":
        namespace.OBJECT_URDF_FILE = str(MODELS_BASE_PATH / namespace.OBJECT_NAME / f"{namespace.OBJECT_NAME}.urdf")
        namespace.OBJECT_MESH_FILE = str(MODELS_BASE_PATH / namespace.OBJECT_NAME / f"{namespace.OBJECT_NAME}.obj")
        namespace.OBJECT_URDF_TEMPLATE = str(MODELS_BASE_PATH / "templates" / f"{namespace.OBJECT_NAME}.urdf.jinja")
        namespace.SCENE_XML_FILE = str(
            MODELS_BASE_PATH / robot_config.robot_type /
            f"{robot_config.robot_type}_{namespace.ROBOT_DOF}dof_w_{namespace.OBJECT_NAME}.xml"
        )
    else:
        namespace.OBJECT_URDF_FILE = namespace.ROBOT_URDF_FILE
        namespace.OBJECT_MESH_FILE = ""
        namespace.SCENE_XML_FILE = namespace.ROBOT_URDF_FILE.replace(".urdf", ".xml")

    return namespace


# =============================================================================
# Motion loader with interpolation
# =============================================================================

DynamicState = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool,
]
StaticState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]


class MotionLoader:
    """Load and interpolate motion data from npz format."""

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the npz file."""
        data = np.load(self.motion_file)
        if "fps" in data:
            self.input_fps = int(data["fps"])
            self.input_dt = 1.0 / self.input_fps
        motion = torch.from_numpy(data["qpos"]).to(torch.float32).to(self.device)

        # qpos format: [root_trans(3), root_ori(4), dof_pos(29)]
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_dof_poss_input = motion[:, 7:36]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration:.2f} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated: {self.input_frames} frames @ {self.input_fps} fps -> "
            f"{self.output_frames} frames @ {self.output_fps} fps"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-8):
        q0 = F.normalize(q0, dim=-1)
        q1 = F.normalize(q1, dim=-1)

        if t.ndim == q0.ndim - 1:
            t = t.unsqueeze(-1)

        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        close = sin_theta.abs() < eps

        s0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
        s1 = torch.sin(t * theta) / (sin_theta + eps)
        out = s0 * q0 + s1 * q1
        out = torch.where(close, (1.0 - t) * q0 + t * q1, out)
        return F.normalize(out, dim=-1)

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        rotvec = quat_to_rotvec(q_rel)
        omega = rotvec / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self) -> StaticState:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag:
            self.current_idx = 0
        return (*state, reset_flag)


# =============================================================================
# MuJoCo utilities
# =============================================================================


def world_body_velocities(model, data):
    """Per-body COM velocities in the world frame."""
    v = np.zeros((model.nbody, 6))
    for b in range(model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, v[b], 0)
    return v[:, 3:6], v[:, 0:3]  # lin_w, ang_w


# =============================================================================
# PHUMA npy to intermediate npz conversion
# =============================================================================


def convert_phuma_npy_to_qpos_npz(npy_path: str, input_fps: int = 30) -> str:
    """Convert PHUMA npy file to intermediate qpos npz format.

    Args:
        npy_path: Path to the input npy file.
        input_fps: FPS of the input motion.

    Returns:
        Path to the temporary npz file.
    """
    print(f"Loading PHUMA npy: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()

    root_trans = data.get("root_trans", data.get("root_position"))
    root_ori = data.get("root_ori", data.get("root_orientation"))
    dof_pos = data.get("dof_pos", data.get("joint_pos"))
    fps = data.get("fps", input_fps)

    if root_trans is None or root_ori is None or dof_pos is None:
        raise ValueError(f"Missing required keys in {npy_path}. Need root_trans, root_ori, dof_pos")

    # Convert quaternion from x,y,z,w to w,x,y,z format
    root_ori = root_ori[:, [3, 0, 1, 2]]

    # Concatenate to create qpos [root_trans(3), root_ori(4), dof_pos(29)]
    qpos = np.concatenate([root_trans, root_ori, dof_pos], axis=1)

    print(f"  root_trans: {root_trans.shape}, root_ori: {root_ori.shape}, dof_pos: {dof_pos.shape}")
    print(f"  qpos shape: {qpos.shape}, fps: {fps}")

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(temp_file.name, fps=fps, qpos=qpos)
    print(f"  Saved intermediate npz to: {temp_file.name}")

    return temp_file.name


# =============================================================================
# Headless MuJoCo conversion (no viewer)
# =============================================================================


def run_headless_conversion(
    input_npz: str,
    output_npz: str,
    robot: str,
    data_format: str,
    object_name: str,
    input_fps: int,
    output_fps: int,
    joint_names: list[str],
) -> None:
    """Run MuJoCo forward kinematics headlessly to convert motion data.

    Args:
        input_npz: Path to input npz file (qpos format).
        output_npz: Path to output npz file.
        robot: Robot type (e.g., "g1").
        data_format: Motion data format (e.g., "lafan").
        object_name: Object name (e.g., "ground").
        input_fps: Input FPS.
        output_fps: Output FPS.
        joint_names: List of joint names in the expected order.
    """
    device = torch.device("cpu")

    # Load motion
    motion = MotionLoader(
        motion_file=input_npz,
        input_fps=input_fps,
        output_fps=output_fps,
        device=device,
    )

    # Create task constants
    robot_config = RobotConfig(robot_type=robot)
    motion_config = MotionDataConfig(data_format=data_format, robot_type=robot)
    constants = create_task_constants(robot_config, motion_config, object_name=object_name)

    # Load MuJoCo model
    robot_model_path = constants.ROBOT_URDF_FILE
    if object_name == "ground":
        robot_xml_path = robot_model_path.replace(".urdf", ".xml")
    elif object_name == "multi_boxes":
        robot_xml_path = constants.SCENE_XML_FILE
    else:
        robot_xml_path = robot_model_path.replace(".urdf", "_w_" + object_name + ".xml")

    robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
    robot_data = mujoco.MjData(robot_model)
    print(f"Loading robot model from: {robot_xml_path}")

    # Prepare DOF index mapping
    dof_name_list = []
    for i in range(robot_model.njnt):
        if robot_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_name_list.append(dof_name)
    print(f"Robot DOFs: {len(dof_name_list)}")
    dof_index_list = [joint_names.index(dof_name) for dof_name in dof_name_list]

    # Initialize log
    log: dict[str, Any] = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    # Process all frames (headless - no viewer, no sleep)
    print(f"Processing {motion.output_frames} frames...")
    for frame_idx in range(motion.output_frames):
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
            reset_flag,
        ) = cast("StaticState", motion.get_next_state())

        # Set robot state
        robot_data.qpos[:] = torch.cat(
            [motion_base_pos, motion_base_rot, motion_dof_pos[:, dof_index_list]], dim=1
        )
        robot_data.qvel[:] = torch.cat(
            [motion_base_lin_vel, motion_base_ang_vel, motion_dof_vel[:, dof_index_list]], dim=1
        )

        # Forward kinematics
        mujoco.mj_forward(robot_model, robot_data)

        # Log data
        lin_vel_w, ang_vel_w = world_body_velocities(robot_model, robot_data)
        log["joint_pos"].append(robot_data.qpos[:].copy())
        log["joint_vel"].append(robot_data.qvel[:].copy())
        log["body_pos_w"].append(robot_data.xpos[:].copy())
        log["body_quat_w"].append(robot_data.xquat[:].copy())
        log["body_lin_vel_w"].append(lin_vel_w[:].copy())
        log["body_ang_vel_w"].append(ang_vel_w[:].copy())

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{motion.output_frames} frames")

    # Stack arrays
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    # ==========================================================================
    # Apply wbt.py-style transformations for final training format
    # ==========================================================================
    # 1. Remove pelvis info from joint_pos and joint_vel
    #    - joint_pos: first 7 values are [xyz, wxyz] of pelvis -> remove
    #    - joint_vel: first 6 values are [vel_xyz, vel_wxyz] of pelvis -> remove
    log["joint_pos"] = log["joint_pos"][:, 7:]  # (T, 36) -> (T, 29)
    log["joint_vel"] = log["joint_vel"][:, 6:]  # (T, 35) -> (T, 29)

    # 2. Convert body_quat_w from wxyz to xyzw format
    #    MuJoCo uses wxyz, but wbt.py internally uses xyzw
    body_quat_wxyz = log["body_quat_w"]  # (T, num_bodies, 4) in wxyz
    log["body_quat_w"] = body_quat_wxyz[:, :, [1, 2, 3, 0]]  # Convert to xyzw

    print(f"  Applied wbt.py-style transformations:")
    print(f"    joint_pos: {log['joint_pos'].shape} (pelvis removed)")
    print(f"    joint_vel: {log['joint_vel'].shape} (pelvis removed)")
    print(f"    body_quat_w: {log['body_quat_w'].shape} (converted to xyzw)")

    # ==========================================================================
    # Add default pose transitions (prepend and/or append)
    # ==========================================================================
    dt = 1.0 / output_fps
    
    # Default joint positions for G1 (all zeros = default standing pose)
    default_joint_pos = np.zeros(log["joint_pos"].shape[1])
    
    # Get first and last frame states
    first_frame = {k: log[k][0] for k in log if k not in ["fps", "joint_names", "body_names"]}
    last_frame = {k: log[k][-1] for k in log if k not in ["fps", "joint_names", "body_names"]}
    
    # Get default pose state (aligned with motion start for prepend, motion end for append)
    default_pose_start = get_default_pose_state(
        robot_model, robot_data, default_joint_pos, first_frame, use_motion_xy=True
    )
    default_pose_end = get_default_pose_state(
        robot_model, robot_data, default_joint_pos, last_frame, use_motion_xy=True
    )
    
    # Prepend transition: default pose -> first motion frame
    if args_cli.enable_default_pose_prepend and args_cli.default_pose_prepend_duration_s > 0:
        num_prepend_frames = int(args_cli.default_pose_prepend_duration_s * output_fps)
        if num_prepend_frames > 1:
            prepend_frames = build_transition_frames(
                start_state=default_pose_start,
                end_state=first_frame,
                num_frames=num_prepend_frames,
                drop_first=False,  # Include default pose frame
                drop_last=True,    # Exclude first motion frame (already in log)
            )
            # Prepend to log
            for k in prepend_frames:
                log[k] = np.concatenate([prepend_frames[k], log[k]], axis=0)
            print(f"  Prepended {len(prepend_frames['joint_pos'])} transition frames ({args_cli.default_pose_prepend_duration_s}s)")
    
    # Append transition: last motion frame -> default pose
    if args_cli.enable_default_pose_append and args_cli.default_pose_append_duration_s > 0:
        num_append_frames = int(args_cli.default_pose_append_duration_s * output_fps)
        if num_append_frames > 1:
            append_frames = build_transition_frames(
                start_state=last_frame,
                end_state=default_pose_end,
                num_frames=num_append_frames,
                drop_first=True,   # Exclude last motion frame (already in log)
                drop_last=False,   # Include default pose frame
            )
            # Append to log
            for k in append_frames:
                log[k] = np.concatenate([log[k], append_frames[k]], axis=0)
            print(f"  Appended {len(append_frames['joint_pos'])} transition frames ({args_cli.default_pose_append_duration_s}s)")
    
    print(f"  Final motion length: {log['joint_pos'].shape[0]} frames")

    # Add joint and body names
    joint_names_mj = [mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(robot_model.njnt)]
    body_names = [mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(robot_model.nbody)]
    log["joint_names"] = joint_names_mj[1:]  # Remove root free joint name
    log["body_names"] = body_names

    # Save output
    output_folder = Path(output_npz).parent
    os.makedirs(output_folder, exist_ok=True)
    np.savez(output_npz, **log)
    print(f"Saved output to: {output_npz}")


# =============================================================================
# Main conversion pipeline
# =============================================================================

G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def convert_single_file(
    input_npy: str,
    output_npz: str,
    robot: str,
    data_format: str,
    object_name: str,
    input_fps: int,
    output_fps: int,
) -> None:
    """Convert a single PHUMA npy file to holosoma npz format."""
    print(f"\n{'='*60}")
    print(f"Converting: {input_npy}")
    print(f"Output: {output_npz}")
    print(f"{'='*60}")

    # Step 1: Convert PHUMA npy to intermediate qpos npz
    temp_npz = convert_phuma_npy_to_qpos_npz(input_npy, input_fps)

    try:
        # Step 2: Run headless MuJoCo conversion
        run_headless_conversion(
            input_npz=temp_npz,
            output_npz=output_npz,
            robot=robot,
            data_format=data_format,
            object_name=object_name,
            input_fps=input_fps,
            output_fps=output_fps,
            joint_names=G1_JOINT_NAMES,
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_npz):
            os.remove(temp_npz)
            print(f"Cleaned up temp file: {temp_npz}")

    print(f"[SUCCESS] Conversion completed: {output_npz}")


def main():
    """Main function."""
    # Single file mode
    if args_cli.input_file is not None:
        if args_cli.output_name is None:
            # Generate output name from input
            input_path = Path(args_cli.input_file)
            output_name = str(input_path.with_suffix(".npz"))
        else:
            output_name = args_cli.output_name

        convert_single_file(
            input_npy=args_cli.input_file,
            output_npz=output_name,
            robot=args_cli.robot,
            data_format=args_cli.data_format,
            object_name=args_cli.object_name,
            input_fps=args_cli.input_fps,
            output_fps=args_cli.output_fps,
        )

    # Batch mode
    elif args_cli.input_folder is not None:
        if args_cli.output_folder is None:
            output_folder = args_cli.input_folder + "_converted"
        else:
            output_folder = args_cli.output_folder

        os.makedirs(output_folder, exist_ok=True)

        input_folder = Path(args_cli.input_folder)
        npy_files = sorted(input_folder.glob("*.npy"))
        print(f"Found {len(npy_files)} npy files in {input_folder}")

        for npy_file in npy_files:
            output_name = os.path.join(output_folder, npy_file.stem + ".npz")
            try:
                convert_single_file(
                    input_npy=str(npy_file),
                    output_npz=output_name,
                    robot=args_cli.robot,
                    data_format=args_cli.data_format,
                    object_name=args_cli.object_name,
                    input_fps=args_cli.input_fps,
                    output_fps=args_cli.output_fps,
                )
            except Exception as e:
                print(f"[ERROR] Failed to convert {npy_file}: {e}")
                continue

        print(f"\n[DONE] Converted {len(npy_files)} files to {output_folder}")

    else:
        print("Error: Please specify either --input-file or --input-folder")
        sys.exit(1)


if __name__ == "__main__":
    main()
