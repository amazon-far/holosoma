from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import mujoco  # type: ignore[import-not-found]
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

Pair = Tuple[str, str]
MujocoFrameKind = Literal["body", "geom", "site"]


@dataclass(frozen=True)
class MujocoFrameRef:
    """Resolved reference to a named MuJoCo kinematic frame."""

    name: str
    kind: MujocoFrameKind
    object_id: int
    body_id: int


@dataclass(frozen=True)
class MujocoFramePoseSet:
    """World poses of named MuJoCo bodies, geometries, or sites."""

    names: tuple[str, ...]
    positions_m: np.ndarray
    quaternions_wxyz: np.ndarray


def resolve_mujoco_frame(model: mujoco.MjModel, frame_name: str) -> MujocoFrameRef:
    """Resolve a body, geom, or site name using stable precedence."""

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if body_id >= 0:
        return MujocoFrameRef(frame_name, "body", int(body_id), int(body_id))

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, frame_name)
    if geom_id >= 0:
        return MujocoFrameRef(frame_name, "geom", int(geom_id), int(model.geom_bodyid[geom_id]))

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
    if site_id >= 0:
        return MujocoFrameRef(frame_name, "site", int(site_id), int(model.site_bodyid[site_id]))

    raise KeyError(f"No MuJoCo body, geom, or site named '{frame_name}'")


def resolve_mujoco_frames(
    model: mujoco.MjModel,
    frame_names: Sequence[str],
) -> dict[str, MujocoFrameRef]:
    """Resolve unique frame names for reuse by optimization-time consumers."""

    return {name: resolve_mujoco_frame(model, name) for name in dict.fromkeys(frame_names)}


def mujoco_frame_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: str | MujocoFrameRef,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a resolved MuJoCo frame's world position and rotation matrix."""

    ref = resolve_mujoco_frame(model, frame) if isinstance(frame, str) else frame
    if ref.kind == "body":
        return data.xpos[ref.object_id].copy(), data.xmat[ref.object_id].reshape(3, 3).copy()
    if ref.kind == "geom":
        return data.geom_xpos[ref.object_id].copy(), data.geom_xmat[ref.object_id].reshape(3, 3).copy()
    return data.site_xpos[ref.object_id].copy(), data.site_xmat[ref.object_id].reshape(3, 3).copy()


def mujoco_frame_jacobians(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: str | MujocoFrameRef,
    point_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return translational/rotational qvel Jacobians and pose for a frame point.

    ``point_offset`` is expressed in the resolved frame. Site origins use
    :func:`mujoco.mj_jacSite`; offset points and other frame kinds use the
    owning body's point Jacobian.
    """

    ref = resolve_mujoco_frame(model, frame) if isinstance(frame, str) else frame
    position, rotation = mujoco_frame_pose(model, data, ref)
    offset = None if point_offset is None else np.asarray(point_offset, dtype=float).reshape(3)
    if offset is not None:
        position = position + rotation @ offset

    jacp = np.zeros((3, model.nv), dtype=np.float64, order="C")
    jacr = np.zeros((3, model.nv), dtype=np.float64, order="C")
    if ref.kind == "site" and offset is None:
        mujoco.mj_jacSite(model, data, jacp, jacr, ref.object_id)
    else:
        mujoco.mj_jac(model, data, jacp, jacr, position.reshape(3, 1), ref.body_id)
    return jacp, jacr, np.asarray(position, dtype=float), rotation


def evaluate_mujoco_frame_poses(
    model_path: str | Path,
    qpos: np.ndarray,
    frame_names: Sequence[str],
) -> MujocoFramePoseSet:
    """Evaluate named frame world poses at one configuration of any MuJoCo model."""

    model = mujoco.MjModel.from_xml_path(str(model_path))
    values = np.asarray(qpos, dtype=float).reshape(-1)
    if values.shape != (model.nq,):
        raise ValueError(f"Expected qpos with shape ({model.nq},), got {values.shape}")

    data = mujoco.MjData(model)
    data.qpos[:] = values
    mujoco.mj_forward(model, data)
    positions = []
    quaternions = []
    frame_refs = resolve_mujoco_frames(model, frame_names)
    for frame_name in frame_names:
        position, rotation = mujoco_frame_pose(model, data, frame_refs[frame_name])
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        positions.append(position)
        quaternions.append(quaternion_xyzw[[3, 0, 1, 2]])
    return MujocoFramePoseSet(
        names=tuple(frame_names),
        positions_m=np.asarray(positions, dtype=float).reshape(-1, 3),
        quaternions_wxyz=np.asarray(quaternions, dtype=float).reshape(-1, 4),
    )


def replace_named_joint_qpos(
    model_path: str | Path,
    qpos: np.ndarray,
    joint_values: Mapping[str, float],
) -> np.ndarray:
    """Return a qpos copy with selected scalar joints replaced by name."""

    model = mujoco.MjModel.from_xml_path(str(model_path))
    result = np.asarray(qpos, dtype=float).reshape(-1).copy()
    if result.shape != (model.nq,):
        raise ValueError(f"Expected qpos with shape ({model.nq},), got {result.shape}")
    for joint_name, value in joint_values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"No MuJoCo joint named '{joint_name}'")
        if int(model.jnt_type[joint_id]) not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Joint '{joint_name}' is not scalar")
        result[int(model.jnt_qposadr[joint_id])] = float(value)
    return result


def _mesh_local_vf(model, geom_id):
    """Return local vertices and faces for a MuJoCo mesh geom."""
    mesh_id = int(model.geom_dataid[geom_id])  # Note: sometime geom does not have mesh, mesh_id will be -1

    v0, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    f0, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])

    V = model.mesh_vert[v0 : v0 + nv].astype(np.float64, copy=True)

    F = model.mesh_face[f0 : f0 + nf].astype(np.int32, copy=True)

    return V, F


def _to_world(v_local, data, geom_id):
    """Transform local vertices to world using geom pose."""
    R = data.geom_xmat[geom_id].reshape(3, 3)
    t = data.geom_xpos[geom_id]

    return v_local @ R.T + t


def _world_mesh_from_geom(model, data, geom_id, geom_name):
    V_local, F = _mesh_local_vf(model, geom_id)

    V_world = _to_world(V_local, data, geom_id)

    return V_world, F
