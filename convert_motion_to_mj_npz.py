#!/usr/bin/env python3
"""
Convert motion file (pkl or other formats) to full npz format using MuJoCo FK.

Tries multiple load strategies to handle:
- Standard Python pickle (with numpy compatibility)
- Joblib-serialized files
- PyTorch .pth / torch.save format
- NPZ misnamed as .pkl

Usage:
    python convert_motion_to_mj_npz.py input.pkl output.npz --input-fps 30 --output-fps 50
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Optional: MuJoCo and joblib
try:
    import mujoco
except ImportError:
    mujoco = None
try:
    import joblib
except ImportError:
    joblib = None


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core":
            module = "numpy.core"
        elif module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)


def _load_pickle(filepath):
    with open(filepath, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def _load_joblib(filepath):
    if joblib is None:
        raise ImportError("joblib is required for this format. pip install joblib")
    return joblib.load(filepath)


def _load_torch(filepath):
    return torch.load(filepath, map_location="cpu", weights_only=False)


def _load_npz(filepath):
    loaded = np.load(filepath, allow_pickle=True)
    # Handle .npy files that contain a dict (0-d array with dict)
    if isinstance(loaded, np.ndarray) and loaded.shape == () and isinstance(loaded.item(), dict):
        data = loaded.item()
    elif isinstance(loaded, np.lib.npyio.NpzFile):
        data = dict(loaded)
    else:
        data = dict(loaded) if hasattr(loaded, "keys") else {"data": loaded}
    # Convert 0-dim arrays to scalars for fps etc.
    for k in list(data.keys()):
        if hasattr(data[k], "shape") and data[k].shape == ():
            data[k] = np.asarray(data[k]).item()
        elif isinstance(data[k], np.ndarray) and data[k].dtype.kind == "U":
            data[k] = data[k].tolist()
    return data


def load_motion_file(filepath: str):
    """
    Load motion data from file. Tries pickle, joblib, torch, npz in order.
    Returns a dict with at least: root_pos, root_rot (xyzw), dof_pos, dof_names, fps (optional).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        header = f.read(16)
    # Log format hint for debugging (e.g. pickle \x80\x03, joblib magic, etc.)
    if len(header) >= 2:
        print(f"  File header (hex): {header[:8].hex()}")

    errors = []

    # 1. Try standard pickle
    try:
        data = _load_pickle(filepath)
        if _is_motion_dict(data):
            return _normalize_motion_dict(data)
        errors.append("pickle: loaded but not a motion dict")
    except Exception as e:
        errors.append(f"pickle: {e}")

    # 2. Try joblib (header 80049502... is joblib format)
    if joblib is not None:
        try:
            data = _load_joblib(filepath)
            if _is_motion_dict(data):
                return _normalize_motion_dict(data)
            # Unwrap single-key dict (e.g. {"walk_45cms_grounded": <motion>})
            if isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))
            if _is_motion_dict(data):
                return _normalize_motion_dict(data)
            out = _normalize_joblib_motion(data)
            if out is not None:
                return out
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            errors.append(f"joblib: loaded but not a motion dict (keys: {keys})")
        except Exception as e:
            errors.append(f"joblib: {e}")

    # 3. Try torch.load (e.g. .pth saved as .pkl)
    try:
        data = _load_torch(filepath)
        if isinstance(data, dict) and _is_motion_dict(data):
            return _normalize_motion_dict(data)
        if isinstance(data, dict):
            # Maybe keys are different (e.g. 'position', 'rotation')
            data = _normalize_torch_motion_dict(data)
            if data is not None:
                return data
        errors.append("torch: loaded but not a motion dict")
    except Exception as e:
        errors.append(f"torch: {e}")

    # 4. Try np.load (file named .pkl but actually npz or .npy)
    try:
        data = _load_npz(filepath)
        # Check for .npy format with root_trans/root_ori (check this first, before _is_motion_dict)
        if "root_trans" in data or "root_ori" in data:
            normalized = _normalize_npy_motion_dict(data)
            if normalized is not None:
                return normalized
        if _is_motion_dict(data):
            return _normalize_motion_dict(data)
        # Check for common npz motion keys
        if "joint_pos" in data or "qpos" in data:
            normalized = _normalize_npz_motion_dict(data)
            if normalized is not None:
                return normalized
        errors.append("npz: loaded but not a motion dict")
    except Exception as e:
        errors.append(f"npz: {e}")

    raise ValueError(
        f"Could not load motion from {filepath}. Tried:\n  " + "\n  ".join(errors)
    )


def _is_motion_dict(data):
    """True only if dict has exact keys expected by _normalize_motion_dict."""
    if not isinstance(data, dict):
        return False
    has_pos = "root_pos" in data or "position" in data or "root_trans" in data
    has_rot = "root_rot" in data or "rotation" in data or "quat" in data or "root_ori" in data
    has_dof = "dof_pos" in data or "joint_pos" in data or "qpos" in data
    return has_pos and (has_rot or has_dof)


def _normalize_joblib_motion(data) -> Optional[dict]:
    """
    Extract motion from joblib-loaded object (dict with various key names,
    or object with attributes). Returns normalized dict or None.
    """
    # Convert object to dict if it has attributes
    if not isinstance(data, dict):
        if hasattr(data, "__dict__"):
            data = data.__dict__
        elif hasattr(data, "keys"):
            data = dict(data)
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            # List of frames: [{"position": ..., "rotation": ..., "joints": ...}, ...]
            return _normalize_frame_list_motion(data)
        else:
            return None

    def get(key_candidates, default=None):
        for k in key_candidates:
            if k in data:
                return data[k]
        return default

    # Root position: (T, 3)
    root_pos = get(
        [
            "root_pos",
            "root_position",
            "root_trans_offset",
            "root_trans",
            "position",
            "positions",
            "root_positions",
        ]
    )
    if root_pos is None:
        return None
    root_pos = _to_numpy(root_pos)
    if root_pos.ndim == 1:
        root_pos = root_pos.reshape(-1, 3)
    T = root_pos.shape[0]

    # Root rotation: (T, 4) xyzw or wxyz
    root_rot = get(["root_rot", "root_quat", "rotation", "rotations", "quat", "quats"])
    if root_rot is None:
        root_rot = np.tile([0, 0, 0, 1], (T, 1)).astype(np.float64)
    else:
        root_rot = _to_numpy(root_rot)
        if root_rot.ndim == 1:
            root_rot = root_rot.reshape(-1, 4)
        # If wxyz (first elem ~1 for identity), convert to xyzw
        if root_rot.shape[1] == 4 and np.abs(root_rot[:, 0]).mean() > 0.9:
            wxyz = root_rot
            root_rot = np.column_stack([wxyz[:, 1], wxyz[:, 2], wxyz[:, 3], wxyz[:, 0]])

    # DOF/joint positions: (T, N)
    dof_pos = get(
        [
            "dof_pos",
            "dof",
            "joint_pos",
            "qpos",
            "joint_positions",
            "joint_angles",
            "dof_positions",
            "pose_aa",  # axis-angle; use if dof not present (may need conversion later)
        ]
    )
    if dof_pos is None:
        return None
    dof_pos = _to_numpy(dof_pos)
    if dof_pos.ndim == 1:
        dof_pos = dof_pos.reshape(-1, dof_pos.size)

    dof_names = get(["dof_names", "joint_names", "dof_name", "joint_name"])
    if dof_names is not None:
        dof_names = list(dof_names) if not isinstance(dof_names, list) else dof_names
    fps = get(["fps", "frame_rate", "rate"])
    if fps is not None:
        fps = int(np.asarray(fps).item()) if hasattr(fps, "item") else int(fps)

    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_names": dof_names,
        "fps": fps,
    }


def _normalize_frame_list_motion(frames: list) -> Optional[dict]:
    """Convert list of per-frame dicts to single arrays."""
    if not frames:
        return None
    first = frames[0]
    d0 = first if isinstance(first, dict) else (getattr(first, "__dict__", None) or {})
    def get_frame(d, *keys_defaults):
        # keys_defaults: (key1, key2, ..., default)
        default = keys_defaults[-1]
        for k in keys_defaults[:-1]:
            if isinstance(d, dict) and k in d:
                return d[k]
            if hasattr(d, k):
                return getattr(d, k)
        return default
    pos_list = []
    rot_list = []
    dof_list = []
    for f in frames:
        d = f if isinstance(f, dict) else getattr(f, "__dict__", f)
        if d is None:
            return None
        pos = get_frame(d, "position", "root_pos", "root_position", np.zeros(3))
        rot = get_frame(d, "rotation", "root_rot", "quat", np.array([0.0, 0.0, 0.0, 1.0]))
        joints = get_frame(d, "joint_positions", "dof_pos", "joint_pos", None)
        if joints is None:
            return None
        pos_list.append(_to_numpy(pos))
        rot_list.append(_to_numpy(rot))
        dof_list.append(_to_numpy(joints))
    root_pos = np.stack(pos_list)
    root_rot = np.stack(rot_list)
    dof_pos = np.stack(dof_list)
    if root_rot.shape[1] == 4 and np.abs(root_rot[:, 0]).mean() > 0.9:
        wxyz = root_rot
        root_rot = np.column_stack([wxyz[:, 1], wxyz[:, 2], wxyz[:, 3], wxyz[:, 0]])
    fps = get_frame(d0, "fps", "frame_rate", None)
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_names": None,
        "fps": int(fps) if fps is not None else None,
    }


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _normalize_motion_dict(data: dict) -> dict:
    """Expect keys: root_pos, root_rot (xyzw), dof_pos, dof_names, fps (optional)."""
    root_pos = _to_numpy(data["root_pos"])
    root_rot = _to_numpy(data["root_rot"])  # expect xyzw
    dof_pos = _to_numpy(data["dof_pos"])
    dof_names = data.get("dof_names")
    if dof_names is None:
        dof_names = data.get("joint_names")
    if dof_names is not None:
        dof_names = list(dof_names) if not isinstance(dof_names, list) else dof_names
    fps = data.get("fps")
    if fps is not None:
        fps = int(np.asarray(fps).item()) if hasattr(fps, "item") else int(fps)
    # Ensure root_rot is (T, 4) xyzw
    if root_rot.ndim == 1:
        root_rot = root_rot.reshape(-1, 4)
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_names": dof_names,
        "fps": fps,
    }


def _normalize_torch_motion_dict(data: dict) -> Optional[dict]:
    """Handle torch-style or alternate key names."""
    # position / rotation (sometimes root only)
    pos = data.get("root_pos") or data.get("position") or data.get("root_position")
    rot = data.get("root_rot") or data.get("rotation") or data.get("quat") or data.get("root_quat")
    dof = data.get("dof_pos") or data.get("joint_pos") or data.get("qpos")
    if pos is None or dof is None:
        return None
    pos = _to_numpy(pos)
    dof = _to_numpy(dof)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3)
    if dof.ndim == 1:
        dof = dof.reshape(-1, -1)
    if rot is None:
        # Assume identity quaternion xyzw
        T = pos.shape[0]
        rot = np.tile([0, 0, 0, 1], (T, 1)).astype(np.float64)
    else:
        rot = _to_numpy(rot)
        if rot.ndim == 1:
            rot = rot.reshape(-1, 4)
    dof_names = data.get("dof_names") or data.get("joint_names")
    if dof_names is not None:
        dof_names = list(dof_names) if not isinstance(dof_names, list) else dof_names
    fps = data.get("fps")
    if fps is not None:
        fps = int(np.asarray(fps).item()) if hasattr(fps, "item") else int(fps)
    return {
        "root_pos": pos,
        "root_rot": rot,
        "dof_pos": dof,
        "dof_names": dof_names,
        "fps": fps,
    }


def _normalize_npz_motion_dict(data: dict) -> Optional[dict]:
    """From existing npz (e.g. joint_pos 7+29, body_pos_w, etc.)."""
    if "joint_pos" in data:
        jp = _to_numpy(data["joint_pos"])
        if jp.ndim == 1:
            jp = jp.reshape(1, -1)
        root_pos = jp[:, :3]
        root_quat = jp[:, 3:7]  # often wxyz in npz
        dof_pos = jp[:, 7:]
        # Convert wxyz -> xyzw if needed (MuJoCo uses wxyz; we want xyzw for _normalize)
        root_rot = np.zeros_like(root_quat)
        root_rot[:, 0] = root_quat[:, 1]
        root_rot[:, 1] = root_quat[:, 2]
        root_rot[:, 2] = root_quat[:, 3]
        root_rot[:, 3] = root_quat[:, 0]
        dof_names = data.get("joint_names")
        if dof_names is not None:
            dof_names = dof_names.tolist() if hasattr(dof_names, "tolist") else list(dof_names)
        fps = data.get("fps")
        if fps is not None:
            fps = int(np.asarray(fps).item())
        return {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_names": dof_names,
            "fps": fps,
        }
    if "qpos" in data:
        qpos = _to_numpy(data["qpos"])
        if qpos.ndim == 1:
            qpos = qpos.reshape(1, -1)
        root_pos = qpos[:, :3]
        root_quat = qpos[:, 3:7]
        dof_pos = qpos[:, 7:]
        root_rot = np.zeros_like(root_quat)
        root_rot[:, 0] = root_quat[:, 1]
        root_rot[:, 1] = root_quat[:, 2]
        root_rot[:, 2] = root_quat[:, 3]
        root_rot[:, 3] = root_quat[:, 0]
        return {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_names": None,
            "fps": data.get("fps"),
        }
    return None


def _normalize_npy_motion_dict(data: dict) -> Optional[dict]:
    """From .npy format with root_trans/root_ori keys."""
    root_pos = data.get("root_trans")
    if root_pos is None:
        root_pos = data.get("root_pos")
    if root_pos is None:
        root_pos = data.get("position")
    
    root_ori = data.get("root_ori")
    if root_ori is None:
        root_ori = data.get("root_rot")
    if root_ori is None:
        root_ori = data.get("rotation")
    if root_ori is None:
        root_ori = data.get("quat")
    
    dof_pos = data.get("dof_pos")
    if dof_pos is None:
        dof_pos = data.get("joint_pos")
    if dof_pos is None:
        dof_pos = data.get("qpos")
    
    if root_pos is None or dof_pos is None:
        return None
    
    root_pos = _to_numpy(root_pos)
    dof_pos = _to_numpy(dof_pos)
    
    if root_pos.ndim == 1:
        root_pos = root_pos.reshape(-1, 3)
    if dof_pos.ndim == 1:
        dof_pos = dof_pos.reshape(-1, -1)
    
    T = root_pos.shape[0]
    
    if root_ori is None:
        root_rot = np.tile([0, 0, 0, 1], (T, 1)).astype(np.float64)
    else:
        root_rot = _to_numpy(root_ori)
        if root_rot.ndim == 1:
            root_rot = root_rot.reshape(-1, 4)
        # Check if wxyz (first elem ~1 for identity), convert to xyzw
        if root_rot.shape[1] == 4 and np.abs(root_rot[:, 0]).mean() > 0.9:
            wxyz = root_rot
            root_rot = np.column_stack([wxyz[:, 1], wxyz[:, 2], wxyz[:, 3], wxyz[:, 0]])
    
    dof_names = data.get("dof_names") or data.get("joint_names")
    if dof_names is not None:
        dof_names = list(dof_names) if not isinstance(dof_names, list) else dof_names
    
    fps = data.get("fps")
    if fps is not None:
        fps = int(np.asarray(fps).item()) if hasattr(fps, "item") else int(fps)
    
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_names": dof_names,
        "fps": fps,
    }


# Default G1 29dof joint order (actuated joints only, from MuJoCo XML)
G1_29DOF_JOINT_NAMES = [
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
    "torso_joint",
    "head_joint",
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

# H1_2 27dof joint order (handless version, from MuJoCo XML)
H1_2_27DOF_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
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


def interpolate_quaternions_slerp(q0, q1, t, eps=1e-8):
    """SLERP for quaternions (xyzw). q0, q1: (N,4), t: (N,) or (N,1)."""
    q0 = torch.from_numpy(np.asarray(q0, dtype=np.float32))
    q1 = torch.from_numpy(np.asarray(q1, dtype=np.float32))
    t = torch.from_numpy(np.asarray(t, dtype=np.float32)).reshape(-1, 1)
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
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
    return F.normalize(out, dim=-1).numpy()


def interpolate_motion(root_pos, root_rot, dof_pos, input_fps, output_fps):
    """Interpolate motion to output_fps. root_rot in xyzw."""
    device = "cpu"
    root_pos_t = torch.from_numpy(root_pos).float()
    root_rot_t = torch.from_numpy(root_rot).float()
    dof_pos_t = torch.from_numpy(dof_pos).float()
    input_frames = root_pos.shape[0]
    input_dt = 1.0 / input_fps
    duration = (input_frames - 1) * input_dt
    output_dt = 1.0 / output_fps
    times = np.arange(0, duration, output_dt).astype(np.float32)
    output_frames = len(times)
    phase = times / (duration + 1e-9)
    index_0 = (phase * (input_frames - 1)).astype(np.int64).clip(0, input_frames - 2)
    index_1 = index_0 + 1
    blend = phase * (input_frames - 1) - index_0
    root_pos_interp = (1 - blend[:, None]) * root_pos[index_0] + blend[:, None] * root_pos[index_1]
    dof_pos_interp = (1 - blend[:, None]) * dof_pos[index_0] + blend[:, None] * dof_pos[index_1]
    root_rot_interp = interpolate_quaternions_slerp(
        root_rot[index_0], root_rot[index_1], blend
    )
    return root_pos_interp, root_rot_interp, dof_pos_interp


def extract_joint_names_from_xml(robot_xml_path: str):
    """Extract joint names from MuJoCo XML file, excluding free joints."""
    if mujoco is None:
        raise ImportError("mujoco is required. pip install mujoco")
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    dof_name_list = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_name_list.append(dof_name)
    return dof_name_list


def convert_motion_to_mj_npz(
    input_path: str,
    output_path: str,
    robot_xml_path: str,
    input_fps: int = 30,
    output_fps: int = 30,
):
    """Convert motion file to full npz using MuJoCo FK."""
    if mujoco is None:
        raise ImportError("mujoco is required. pip install mujoco")

    print(f"Loading motion file: {input_path}")
    data = load_motion_file(input_path)

    root_pos = data["root_pos"]
    root_rot_xyzw = data["root_rot"]
    dof_pos = data["dof_pos"]
    actual_input_fps = data.get("fps") or input_fps

    T = root_pos.shape[0]
    num_dof = dof_pos.shape[1]

    print(f"Input FPS: {actual_input_fps}")
    print(f"Number of frames: {T}")
    print(f"Duration: {T / actual_input_fps:.2f} seconds")
    print(f"Number of DOFs: {num_dof}")

    # Extract joint names from XML if not provided in data
    dof_names = data.get("dof_names")
    if dof_names is None:
        # Use robot-specific default joint order if available
        if "h1_2" in robot_xml_path.lower() and num_dof == 27:
            print(f"\nUsing H1_2 27DOF default joint order")
            dof_names = H1_2_27DOF_JOINT_NAMES.copy()
            print(f"Using {len(dof_names)} joints from H1_2_27DOF_JOINT_NAMES")
        else:
            print(f"\nExtracting joint names from XML: {robot_xml_path}")
            dof_names = extract_joint_names_from_xml(robot_xml_path)
            print(f"Found {len(dof_names)} joints in XML")

    # Ensure dof_names list length matches
    if len(dof_names) != num_dof:
        print(f"Warning: dof_names length {len(dof_names)} != num_dof {num_dof}")
        if len(dof_names) > num_dof:
            print(f"  Truncating dof_names to match num_dof")
            dof_names = dof_names[:num_dof]
        else:
            print(f"  Padding dof_names with generic names")
            dof_names = list(dof_names) + [f"joint_{i}" for i in range(len(dof_names), num_dof)]

    # xyzw -> wxyz for MuJoCo
    root_rot_wxyz = np.zeros_like(root_rot_xyzw)
    root_rot_wxyz[:, 0] = root_rot_xyzw[:, 3]
    root_rot_wxyz[:, 1] = root_rot_xyzw[:, 0]
    root_rot_wxyz[:, 2] = root_rot_xyzw[:, 1]
    root_rot_wxyz[:, 3] = root_rot_xyzw[:, 2]

    if output_fps != actual_input_fps:
        print(f"\nInterpolating from {actual_input_fps}fps to {output_fps}fps...")
        root_pos, root_rot_xyzw_interp, dof_pos = interpolate_motion(
            root_pos, root_rot_xyzw, dof_pos, actual_input_fps, output_fps
        )
        # interpolate_motion returns root_rot in xyzw; convert to wxyz for MuJoCo
        root_rot_wxyz = np.zeros_like(root_rot_xyzw_interp)
        root_rot_wxyz[:, 0] = root_rot_xyzw_interp[:, 3]
        root_rot_wxyz[:, 1] = root_rot_xyzw_interp[:, 0]
        root_rot_wxyz[:, 2] = root_rot_xyzw_interp[:, 1]
        root_rot_wxyz[:, 3] = root_rot_xyzw_interp[:, 2]
        T = root_pos.shape[0]

    print(f"\nLoading MuJoCo model: {robot_xml_path}")
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    data_mj = mujoco.MjData(model)

    # Extract joint names from XML, using actuator order if available (for h1_2, this excludes finger joints)
    dof_name_list = []
    # Try to get joint names from actuators first (this gives us the order and excludes finger joints for h1_2)
    actuator_joint_names = []
    for i in range(model.nu):
        actuator_id = model.actuator_trnid[i, 0]
        if model.jnt_type[actuator_id] != mujoco.mjtJoint.mjJNT_FREE:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_id)
            if joint_name:
                actuator_joint_names.append(joint_name)
    
    # Use actuator order if we have at least num_dof actuators (for h1_2, first 27 are main joints)
    if len(actuator_joint_names) >= num_dof:
        dof_name_list = actuator_joint_names[:num_dof]
        print(f"Using actuator order (first {num_dof} of {len(actuator_joint_names)} actuators)")
    else:
        # Otherwise, get all joints (excluding free joints)
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_name_list.append(dof_name)
        print(f"Using all joints ({len(dof_name_list)} joints)")

    # Map MuJoCo joint order -> index in our dof_pos
    try:
        dof_index_list = [dof_names.index(n) for n in dof_name_list[:num_dof]]
    except (ValueError, IndexError):
        # If mapping fails, try to match by position or use sequential mapping
        if len(dof_name_list) >= num_dof:
            dof_index_list = list(range(num_dof))
        else:
            dof_index_list = list(range(len(dof_name_list))) + [None] * (num_dof - len(dof_name_list))
    joint_names_out = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]

    joint_pos = np.zeros((T, 7 + num_dof), dtype=np.float32)
    joint_vel = np.zeros((T, 6 + num_dof), dtype=np.float32)
    body_pos_w = np.zeros((T, model.nbody, 3), dtype=np.float64)
    body_quat_w = np.zeros((T, model.nbody, 4), dtype=np.float64)
    body_lin_vel_w = np.zeros((T, model.nbody, 3), dtype=np.float64)
    body_ang_vel_w = np.zeros((T, model.nbody, 3), dtype=np.float64)

    print(f"\nComputing forward kinematics for {T} frames...")
    dt = 1.0 / output_fps

    for t in range(T):
        if t % 100 == 0:
            print(f"  Frame {t}/{T}")
        data_mj.qpos[:3] = root_pos[t]
        data_mj.qpos[3:7] = root_rot_wxyz[t]
        # Map dof_pos to model joints (handle case where we have fewer/more joints)
        qpos_dof = data_mj.qpos[7:]
        for idx, dof_idx in enumerate(dof_index_list[:len(qpos_dof)]):
            if dof_idx is not None and dof_idx < num_dof:
                qpos_dof[idx] = dof_pos[t, dof_idx]
        data_mj.qvel[:] = 0
        mujoco.mj_forward(model, data_mj)
        joint_pos[t, :3] = root_pos[t]
        joint_pos[t, 3:7] = root_rot_wxyz[t]
        joint_pos[t, 7:] = dof_pos[t]
        body_pos_w[t] = data_mj.xpos
        body_quat_w[t] = data_mj.xquat

    print("\nComputing velocities...")
    joint_vel[:, :3] = np.gradient(root_pos, dt, axis=0)
    joint_vel[:, 6:] = np.gradient(dof_pos, dt, axis=0)
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)

    print(f"\nSaving npz file: {output_path}")
    np.savez(
        output_path,
        fps=np.array([output_fps], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_names=np.array(joint_names_out, dtype=str),
        body_names=np.array(body_names, dtype=str),
    )
    print("\nConversion complete.")
    print(f"  joint_pos: {joint_pos.shape}")
    print(f"  body_pos_w: {body_pos_w.shape}")
    print(f"  FPS: {output_fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert motion file (pkl/joblib/torch/npz) to npz using MuJoCo FK"
    )
    parser.add_argument("input_path", type=str, help="Input motion file (.pkl, .pth, .npz, etc.)")
    parser.add_argument("output_path", type=str, help="Output .npz path")
    parser.add_argument("--robot-xml", type=str, default=None, help="MuJoCo robot XML")
    parser.add_argument("--input-fps", type=int, default=30)
    parser.add_argument("--output-fps", type=int, default=30)
    args = parser.parse_args()

    if args.robot_xml is None:
        # Try to infer from input path or use default
        if "h1_2" in args.input_path.lower():
            # Default to handless version (27 DOF) for h1_2
            args.robot_xml = "src/holosoma/holosoma/data/robots/h1_2/h1_2_handless.xml"
            print(f"Detected h1_2 robot from input path (using handless version, 27 DOF)")
        elif "statue" in args.input_path.lower() or "statue_v0_1" in args.input_path.lower():
            args.robot_xml = "src/holosoma/holosoma/data/robots/statue_v0_1/statue_v0_1.xml"
            print(f"Detected statue_v0_1 robot from input path")
        else:
            args.robot_xml = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
            print(f"Using default G1 robot")
        print(f"Robot XML: {args.robot_xml}")

    # Ensure output path is relative to current directory
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    else:
        # If absolute, use as is
        output_path = Path(args.output_path)
    
    convert_motion_to_mj_npz(
        args.input_path,
        str(output_path),
        args.robot_xml,
        args.input_fps,
        args.output_fps,
    )
