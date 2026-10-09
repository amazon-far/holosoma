"""Pure derivation of the MuJoCo robot/actor index, name and address tables.

Turns the captured spec metadata + the compiled model into the lookup tables the
simulator needs: clean<->prefixed name maps, robot body ids, per-DOF qpos/qvel
addresses, and per-actor freejoint/static addressing. Every function here is a pure
``model (+ metadata) -> plain-Python tables`` computation with no ``torch`` and no
simulator state, so the caller keeps ownership of the ``self.*`` (and tensor)
assignments and this logic stays independently testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import mujoco
from loguru import logger

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager


def build_name_maps(meta: ActorSpecMeta) -> tuple[dict[str, str], dict[str, str]]:
    """Build the clean<->prefixed name maps for the robot's named elements.

    Robot joints (named, actuated), bodies, and actuators come from the recorded spec
    metadata (exact membership captured pre-attach), mapped clean<->prefixed by
    construction with ``meta.prefix``.

    Returns
    -------
    (clean_to_prefixed, prefixed_to_clean)
        e.g. ``{"hip_joint": "robot_hip_joint"}`` and its inverse.
    """
    clean_to_prefixed: dict[str, str] = {}
    prefixed_to_clean: dict[str, str] = {}
    prefix = meta.prefix
    for clean_name in (*meta.dof_joint_names, *meta.body_names, *meta.actuator_names):
        prefixed_name = f"{prefix}{clean_name}"
        clean_to_prefixed[clean_name] = prefixed_name
        prefixed_to_clean[prefixed_name] = clean_name
    logger.info(f"Built name maps: {len(clean_to_prefixed)} clean->prefixed mappings")
    return clean_to_prefixed, prefixed_to_clean


def resolve_robot_body_ids(
    model: mujoco.MjModel, body_names: Sequence[str], clean_to_prefixed: Mapping[str, str]
) -> list[int]:
    """Map each robot body (clean name) to its compiled MuJoCo body id, in ``body_names`` order.

    This is the cross-backend ``body_ids`` mapping (``body_ids[holosoma_idx] -> backend body
    id``) used to gather robot rows out of the full, ``model.nbody``-wide physics tensors.

    Raises
    ------
    ValueError
        If a robot body is not present in the compiled model.
    """
    body_ids: list[int] = []
    for clean_name in body_names:
        prefixed_name = clean_to_prefixed.get(clean_name, clean_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefixed_name)
        if body_id == -1:
            raise ValueError(f"Robot body '{clean_name}' ('{prefixed_name}') not found in compiled model.")
        body_ids.append(body_id)
    return body_ids


def resolve_dof_addresses(
    model: mujoco.MjModel, dof_names: Sequence[str], clean_to_prefixed: Mapping[str, str]
) -> tuple[list[int], list[int]]:
    """Return ``(qpos_addrs, qvel_addrs)`` for the robot's actuated DOF joints, in ``dof_names`` order.

    Root (freejoint) addressing is handled separately in :func:`resolve_actor_addressing`; this
    covers only the robot-specific DOF joints.

    Raises
    ------
    ValueError
        If a DOF joint is not present in the compiled model.
    """
    qpos_addrs: list[int] = []
    qvel_addrs: list[int] = []
    for dof_name in dof_names:
        # dof_names are clean; MuJoCo lookup needs the prefixed name
        joint_name = clean_to_prefixed.get(dof_name, dof_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"DOF joint '{joint_name}' (clean name: '{dof_name}') not found in model")
        qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
        qvel_addrs.append(int(model.jnt_dofadr[joint_id]))
    logger.info(f"Setup {len(qpos_addrs)} DOF joint addresses")
    return qpos_addrs, qvel_addrs


def apply_initial_joint_angles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    default_joint_angles: Mapping[str, float],
    clean_to_prefixed: Mapping[str, str],
) -> None:
    """Write the configured default joint angles into ``data.qpos`` and forward-kinematics.

    Applies each configured angle to its joint's qpos address, then runs ``mj_forward`` so
    body positions reflect the new angles.

    Raises
    ------
    RuntimeError
        If any configured joint failed to resolve/apply.
    """
    logger.info("Setting initial joint angles from robot config")

    joint_angles_set = 0
    joint_angles_failed = 0
    for joint_name, angle in default_joint_angles.items():
        # Add prefix for MuJoCo lookup
        mujoco_joint_name = clean_to_prefixed.get(joint_name, joint_name)
        joint_id = None
        for i in range(model.njnt):
            if model.joint(i).name == mujoco_joint_name:
                joint_id = i
                break

        if joint_id is None:
            logger.warning(f"Joint '{joint_name}' (MuJoCo name: '{mujoco_joint_name}') not found in model")
            joint_angles_failed += 1
            continue

        try:
            # Get the qpos address for this joint
            joint_qposadr = model.jnt_qposadr[joint_id]
            data.qpos[joint_qposadr] = angle
            joint_angles_set += 1
            logger.info(
                f"Set joint '{joint_name}' -> '{mujoco_joint_name}' (ID: {joint_id}, "
                f"qpos_addr: {joint_qposadr}) to angle {angle}"
            )
        except Exception as e:
            logger.warning(f"Failed to set angle for joint '{joint_name}': {e}")
            joint_angles_failed += 1

    if joint_angles_failed > 0:
        raise RuntimeError("Failed to set joint angles")

    logger.info(
        f"Joint angle setting complete: {joint_angles_set} set, {joint_angles_failed} "
        f"failed out of {len(default_joint_angles)} total"
    )

    # Forward kinematics to update body positions based on joint angles
    mujoco.mj_forward(model, data)
    logger.info("Applied forward kinematics with initial joint angles")


def _resolve_freejoint_addrs(model: mujoco.MjModel, name: str, root_body: str) -> dict[str, int]:
    """Return ``{qpos_addr, qvel_addr}`` for the freejoint under ``root_body``.

    Resolves via the body->joint structural link (body_jntadr). A missing root body or
    freejoint is a hard error.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if body_id == -1:
        raise ValueError(f"Root body '{root_body}' for actor '{name}' not found in model.")

    # Find the freejoint among this body's joints (at most one per body).
    jnt_start = model.body_jntadr[body_id]
    jnt_count = model.body_jntnum[body_id]
    fj_id = next(
        (j for j in range(jnt_start, jnt_start + jnt_count) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE),
        -1,
    )
    if fj_id == -1:
        raise ValueError(
            f"No freejoint on root body '{root_body}' for actor '{name}'. "
            "Every actor must be a floating-base free body."
        )
    return {
        "qpos_addr": int(model.jnt_qposadr[fj_id]),
        "qvel_addr": int(model.jnt_dofadr[fj_id]),
    }


def resolve_actor_addressing(
    model: mujoco.MjModel, scene_manager: MujocoSceneManager, static_names: set[str]
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Build the per-actor addressing tables for the unified get/set_actor_states path.

    Free actors (robot + free rigid bodies) get ``object_addrs[name] = {qpos_addr, qvel_addr}``,
    the freejoint's 7-dof pose / 6-dof velocity slices. Static (fixed, jointless) objects have
    no qpos slice, so they get ``static_object_body_ids[name] = body_id`` and their pose is read
    from xpos. Each actor is resolved by its exact root-body name (not prefix-startswith, which
    collides for prefixes like "a_" vs "a_b_").

    Parameters
    ----------
    static_names : set[str]
        Names classified static (standalone ``fixed`` objects + file-marked static scene bodies),
        decided by the caller the same way every backend does.

    Returns
    -------
    (object_addrs, static_object_body_ids)

    Raises
    ------
    ValueError
        If a root body was not set, or a static actor's root body is missing from the model.
    """
    object_addrs: dict[str, dict[str, int]] = {}
    static_object_body_ids: dict[str, int] = {}

    scene_file_actors = {name: root for name, (root, _is_static) in scene_manager.scene_file_bodies.items()}
    actors = {
        "robot": scene_manager.robot_root_body,
        **scene_manager.rigid_object_root_bodies,
        **scene_file_actors,
    }
    for name, root_body in actors.items():
        assert root_body is not None, f"Root body for actor '{name}' was not set."
        if name in static_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
            if body_id == -1:
                raise ValueError(f"Root body '{root_body}' for static actor '{name}' not found in model.")
            static_object_body_ids[name] = body_id
        else:
            object_addrs[name] = _resolve_freejoint_addrs(model, name, root_body)
        logger.info(f"Actor '{name}' addressing resolved (static={name in static_names}).")

    return object_addrs, static_object_body_ids
