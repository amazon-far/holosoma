"""Resolve MuJoCo robot names and state addresses, validate articulations, and apply robot properties.

Robot DOFs must be named hinge or slide joints with a selected stateless, unit-gain, unit-gear torque
motor. All authored actuators are preserved; additional drives can contribute effort to the same joint.
Exact robot names define the public array order; extra asset bodies are allowed.
Validation failures raise ``ValueError``. The ``add_*`` and ``apply_*`` functions mutate their inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import mujoco
from loguru import logger

from holosoma.utils.robot_properties import resolve_dof_properties

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig
    from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager

# Joint types a robot DOF may have: one qpos and one dof address each.
_DOF_JOINT_TYPES = frozenset({mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE})


def _actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    """Compiled actuator name, or a placeholder for an unnamed one."""
    return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or f"<unnamed #{actuator_id}>")


def _joint_type_label(jnt_type: mujoco.mjtJoint) -> str:
    """``mjJNT_BALL`` -> ``"ball"``."""
    return str(jnt_type.name).split("_", 1)[-1].lower()


def validate_robot_joint_types(spec: mujoco.MjSpec, robot_name: str) -> None:
    """Require every non-free joint of the isolated robot spec to be a named hinge or slide.

    Runs on the robot's own ``MjSpec`` before attach. Freejoints are skipped first: t1's root
    freejoint is unnamed and ``_ensure_single_root_body`` adds an unnamed one when absent.

    Raises
    ------
    ValueError
        Listing every offending joint as ``(name, parent body, type)``.
    """
    offenders: list[tuple[str, str, str]] = []
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if not joint.name or joint.type not in _DOF_JOINT_TYPES:
            offenders.append((joint.name or "<unnamed>", str(joint.parent.name), _joint_type_label(joint.type)))
    if offenders:
        raise ValueError(
            f"Robot '{robot_name}' has joints that cannot be robot DOFs; every non-free joint must be a named "
            f"1-DOF hinge or slide joint. Offenders (name, parent body, type): {offenders}"
        )


def add_missing_dof_motors(spec: mujoco.MjSpec, robot_name: str) -> list[str]:
    """Add torque motors for named non-free joints without a same-named actuator; return their joint names.

    Mutates the isolated robot spec before attach. Motor parameters inherit the spec's actuator defaults.
    Authored drives remain active and may exert force at zero selected-motor command.
    """
    present = {actuator.name for actuator in spec.actuators}
    added: list[str] = []
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE or not joint.name or joint.name in present:
            continue
        motor = spec.add_actuator()
        motor.set_to_motor()
        motor.name = joint.name
        motor.target = joint.name
        motor.trntype = mujoco.mjtTrn.mjTRN_JOINT
        added.append(joint.name)
    if added:
        logger.info(f"Robot '{robot_name}': added a torque <motor> for {len(added)} DOF joint(s) without one: {added}")
    return added


def validate_dof_addressing(qpos_addrs: Sequence[int], qvel_addrs: Sequence[int]) -> None:
    """Every canonical DOF must address one distinct scalar position and velocity."""
    if len(qpos_addrs) != len(qvel_addrs):
        raise ValueError("Robot DOF qpos and qvel address counts differ.")
    for label, addrs in (("qpos", qpos_addrs), ("qvel", qvel_addrs)):
        if len(set(addrs)) != len(addrs) or any(addr < 0 for addr in addrs):
            raise ValueError(f"Robot DOF {label} addresses must be unique and nonnegative: {list(addrs)}.")


def dof_address_index(addrs: Sequence[int]) -> slice | list[int]:
    """Retain a zero-copy slice for ascending runs; otherwise gather in config order."""
    if not addrs:
        return slice(0, 0)
    if all(nxt == cur + 1 for cur, nxt in zip(addrs, addrs[1:])):
        return slice(addrs[0], addrs[-1] + 1)
    return list(addrs)


def build_joint_name_map(meta: ActorSpecMeta, robot_config: RobotConfig) -> dict[str, str]:
    """Canonical joint -> prefixed asset joint, independently of the body namespace."""
    from holosoma.utils.robot_names import validate_asset_names

    validate_asset_names(
        robot_config.dof_names,
        meta.dof_joint_names,
        kind="joint",
        require_all=True,
    )
    return {name: f"{meta.prefix}{name}" for name in robot_config.dof_names}


def validate_dof_actuators(
    model: mujoco.MjModel,
    meta: ActorSpecMeta,
    dof_names: Sequence[str],
    clean_to_prefixed: Mapping[str, str],
) -> list[int]:
    """Return the selected torque-motor IDs in ``dof_names`` order.

    Each motor must target its named joint with fixed unit gain, unit gear, no bias and no activation
    dynamics. Missing or incompatible motors raise ``ValueError``. Other authored drives remain active
    outside this selection. Torque commands and force feedback use these IDs.
    """
    hint = "Define one pure torque <motor name=J joint=J/> per DOF."
    robot = f"{meta.prefix}{meta.root_body}"
    actuator_ids: list[int] = []
    for i, dof in enumerate(dof_names):
        prefixed = clean_to_prefixed.get(dof, dof)
        actuator_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefixed))
        if actuator_id == -1:
            present = [_actuator_name(model, k) for k in range(model.nu)]
            raise ValueError(
                f"Robot '{robot}' DOF {i} '{dof}' has no actuator named '{prefixed}'; actuators present: {present}. "
                f"{hint}"
            )
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefixed)
        trntype, trnid = int(model.actuator_trntype[actuator_id]), int(model.actuator_trnid[actuator_id, 0])
        if trntype != mujoco.mjtTrn.mjTRN_JOINT or trnid != joint_id:
            raise ValueError(
                f"Actuator '{prefixed}' (id {actuator_id}) must drive joint '{prefixed}' (id {joint_id}) through a "
                f"joint transmission; got trntype={mujoco.mjtTrn(trntype).name} trnid={trnid}. {hint}"
            )
        biastype = int(model.actuator_biastype[actuator_id])
        if biastype != mujoco.mjtBias.mjBIAS_NONE:
            raise ValueError(
                f"Actuator '{prefixed}' (id {actuator_id}) has biastype {mujoco.mjtBias(biastype).name}; only pure "
                "torque motors are supported, <position>/<velocity> actuators must be removed. Torques are written "
                "to ctrl."
            )
        # Joint effort is written directly to ctrl, so neither gain nor activation dynamics may alter it.
        gaintype = int(model.actuator_gaintype[actuator_id])
        gain = float(model.actuator_gainprm[actuator_id, 0])
        if gaintype != mujoco.mjtGain.mjGAIN_FIXED or gain != 1.0:
            raise ValueError(
                f"Actuator '{prefixed}' (id {actuator_id}) must have fixed unit gain; "
                f"got gaintype={mujoco.mjtGain(gaintype).name} gainprm[0]={gain}. {hint}"
            )
        dyntype = int(model.actuator_dyntype[actuator_id])
        if dyntype != mujoco.mjtDyn.mjDYN_NONE:
            raise ValueError(
                f"Actuator '{prefixed}' (id {actuator_id}) must have no activation dynamics; "
                f"got dyntype={mujoco.mjtDyn(dyntype).name}. {hint}"
            )
        gear = float(model.actuator_gear[actuator_id, 0])
        if gear != 1.0:
            raise ValueError(
                f"Actuator '{prefixed}' (id {actuator_id}) has gear[0]={gear}; expected 1.0 because holosoma writes "
                "joint effort directly to ctrl. Fold the gear into the asset's inertial/transmission parameters "
                f"and define a unit-gear torque motor. {hint}"
            )
        for limited, value, label in (
            (model.actuator_ctrllimited[actuator_id], model.actuator_ctrlrange[actuator_id], "ctrlrange"),
            (model.actuator_forcelimited[actuator_id], model.actuator_forcerange[actuator_id], "forcerange"),
            (model.jnt_actfrclimited[joint_id], model.jnt_actfrcrange[joint_id], "joint actuatorfrcrange"),
        ):
            low, high = (float(v) for v in value)
            if limited and (low != 0.0 or high != 0.0) and not low < 0.0 < high:
                raise ValueError(
                    f"Actuator '{prefixed}' (id {actuator_id}) has {label}=[{low}, {high}], which does not provide "
                    "effort in both directions. Holosoma's scalar effort limit and symmetric torque clipping "
                    f"require a negative and positive endpoint (or [0, 0]). Author a bidirectional range or set "
                    f"dof_effort_limits. {hint}"
                )
        actuator_ids.append(actuator_id)
    return actuator_ids


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


def _dof_joint_ids(model: mujoco.MjModel, dof_names: Sequence[str], clean_to_prefixed: Mapping[str, str]) -> list[int]:
    ids: list[int] = []
    for dof_name in dof_names:
        joint_name = clean_to_prefixed.get(dof_name, dof_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"DOF joint '{joint_name}' (clean name: '{dof_name}') not found in model")
        ids.append(int(joint_id))
    return ids


def _dof_actuator_ids(
    model: mujoco.MjModel, dof_names: Sequence[str], clean_to_prefixed: Mapping[str, str]
) -> list[int]:
    """Compiled id of the actuator named like each DOF joint, ``-1`` when absent (``validate_dof_actuators`` raises)."""
    return [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, clean_to_prefixed.get(dof, dof)))
        for dof in dof_names
    ]


def _effective_effort(model: mujoco.MjModel, actuator_id: int, joint_id: int) -> float:
    """Symmetric effort limit from the selected motor and joint clamps; other drives' limits are excluded."""
    limits = []
    gear = abs(float(model.actuator_gear[actuator_id, 0]))
    if model.actuator_ctrllimited[actuator_id]:
        low, high = (float(v) for v in model.actuator_ctrlrange[actuator_id])
        limits.append(min(abs(low), abs(high)) * gear)
    if model.actuator_forcelimited[actuator_id]:
        low, high = (float(v) for v in model.actuator_forcerange[actuator_id])
        actuator_force_max = min(abs(low), abs(high))
        limits.append(actuator_force_max * gear)
    if model.jnt_actfrclimited[joint_id]:
        low, high = (float(v) for v in model.jnt_actfrcrange[joint_id])
        limits.append(min(abs(low), abs(high)))  # Already in joint units.
    return min(limits) if limits else float("inf")


def apply_dof_properties_from_config(model: mujoco.MjModel, meta: ActorSpecMeta, robot_config: RobotConfig) -> None:
    """Apply configured position/effort limits, armature and friction to the compiled robot.

    ``None`` preserves asset values. Effort limits clamp each selected motor and combined actuator
    effort at its joint; other drives keep their parameters. Velocity limits are reported only.
    Call before creating backend model copies. Changed authored values are logged.
    """
    overrides = resolve_dof_properties(robot_config)
    dof_names = robot_config.dof_names
    if not dof_names:
        return
    clean_to_prefixed = build_joint_name_map(meta, robot_config)
    joint_ids = _dof_joint_ids(model, dof_names, clean_to_prefixed)
    dof_addrs = [int(model.jnt_dofadr[j]) for j in joint_ids]
    actuator_ids = _dof_actuator_ids(model, dof_names, clean_to_prefixed)
    inf = float("inf")

    def _set_range(column: int, sign: float) -> Callable[[int, float], float]:
        def write(i: int, value: float) -> float:
            j = joint_ids[i]
            authored = float(model.jnt_range[j, column]) if model.jnt_limited[j] else sign * inf
            model.jnt_range[j, column] = value
            model.jnt_limited[j] = 1
            return authored

        return write

    def _set_effort(i: int, value: float) -> float:
        a = actuator_ids[i]
        if a == -1:
            return value
        j = joint_ids[i]
        authored = _effective_effort(model, a, j)
        # The joint clamp acts on the sum of all drives, while the actuator clamps affect only
        # the selected motor. Override all three; other actuators keep their authored parameters.
        model.actuator_ctrlrange[a] = (-value, value)
        model.actuator_ctrllimited[a] = 1
        model.actuator_forcerange[a] = (-value, value)
        model.actuator_forcelimited[a] = 1
        model.jnt_actfrcrange[j] = (-value, value)
        model.jnt_actfrclimited[j] = 1
        return authored

    def _set_dof_array(array: Any) -> Callable[[int, float], float]:
        def write(i: int, value: float) -> float:
            authored = float(array[dof_addrs[i]])
            array[dof_addrs[i]] = value
            return authored

        return write

    writes = (
        ("jnt_range lower", {n: v[0] for n, v in overrides.position.items()}, _set_range(0, -1.0)),
        ("jnt_range upper", {n: v[1] for n, v in overrides.position.items()}, _set_range(1, 1.0)),
        ("actuator_ctrlrange (effort)", overrides.effort, _set_effort),
        ("dof_armature", overrides.armature, _set_dof_array(model.dof_armature)),
        ("dof_frictionloss", overrides.friction, _set_dof_array(model.dof_frictionloss)),
    )
    for attr, values, write in writes:
        changed = []
        for dof, value in values.items():
            i = dof_names.index(dof)
            authored = write(i, float(value))
            if abs(authored - float(value)) > 1e-9:
                changed.append((dof, authored, float(value)))
        if changed:
            preview = ", ".join(f"{dof}: {authored:g}->{cfg:g}" for dof, authored, cfg in changed[:5])
            more = f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""
            logger.info(
                f"MuJoCo {attr}: config overrides the MJCF value on {len(changed)}/{len(values)} DOFs: {preview}{more}"
            )


def read_dof_limits(
    model: mujoco.MjModel, meta: ActorSpecMeta, robot_config: RobotConfig
) -> tuple[list[list[float]], list[float], list[float]]:
    """Per-DOF ``([lower, upper] ...)``, velocity and effort limits as the compiled model enforces them.

    Position from ``jnt_range`` (``-inf``/``inf`` for an unlimited joint); effort is the tightest active clamp
    among the motor's control/force ranges after transmission and the joint's actuator-force range
    (``inf`` when none is limited); velocity is the config value or ``inf`` (MuJoCo has no joint velocity limit).
    Other drives' individual limits are excluded. They can add effort beyond a motor-only limit;
    a joint actuator-force clamp, when enabled, bounds their combined contribution.
    Run after :func:`apply_dof_properties_from_config`.
    """
    overrides = resolve_dof_properties(robot_config)
    dof_names = robot_config.dof_names
    if not dof_names:
        return [], [], []
    clean_to_prefixed = build_joint_name_map(meta, robot_config)
    inf = float("inf")
    joint_ids = _dof_joint_ids(model, dof_names, clean_to_prefixed)
    position = [
        [float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])] if model.jnt_limited[j] else [-inf, inf]
        for j in joint_ids
    ]
    effort = [
        _effective_effort(model, a, j) if a != -1 else inf
        for a, j in zip(_dof_actuator_ids(model, dof_names, clean_to_prefixed), joint_ids)
    ]
    velocity = [overrides.velocity.get(name, inf) for name in dof_names]
    return position, velocity, effort


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
    """Return ``(free_actor_addresses, static_actor_body_ids)`` keyed by registered actor name.

    Free entries contain ``qpos_addr`` and ``qvel_addr`` for their root freejoint. Static entries
    contain native body IDs. ``static_names`` must include every welded actor, including a fixed robot.
    Missing root metadata raises ``AssertionError``; missing bodies or incompatible joints raise ``ValueError``.
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
            jnt_count = int(model.body_jntnum[body_id])
            if jnt_count != 0:
                raise ValueError(
                    f"Static actor '{name}' has {jnt_count} joint(s) on its root body '{root_body}'; a static actor "
                    "must be jointless (welded to the world), its pose is written to the body, not to qpos."
                )
            static_object_body_ids[name] = body_id
        else:
            object_addrs[name] = _resolve_freejoint_addrs(model, name, root_body)
        logger.info(f"Actor '{name}' addressing resolved (static={name in static_names}).")

    return object_addrs, static_object_body_ids
