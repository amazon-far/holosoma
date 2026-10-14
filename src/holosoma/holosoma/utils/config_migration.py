"""Convert saved ``ExperimentConfig`` dictionaries to the current robot schema.

Use ``migrate_legacy_experiment_config`` before constructing a config from checkpoint metadata or
``holosoma_config.yaml``. It preserves legacy gain matching, logs conversions and dropped fields, and
returns a copy. For source presets and overrides, follow ``docs/robotconfig-migration.md``.
"""

from __future__ import annotations

import copy
import dataclasses
import typing
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from loguru import logger

from holosoma.config_types.command import NoiseToInitialPoseConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.utils.robot_properties import dof_limits_from_sequences, dof_values_from_sequence

# Robot fields that no longer exist. Zero-consumer fields are simply dropped; the rest are mapped
# below before being dropped.
_DELETED_ROBOT_KEYS = frozenset(
    {
        "num_bodies",
        "dof_obs_size",
        "algo_obs_dim_dict",
        "actions_dim",
        "policy_obs_dim",
        "critic_obs_dim",
        "contact_pairs_multiplier",
        "key_bodies",
        "num_feet",
        "foot_body_name",
        "foot_height_name",
        "knee_name",
        "torso_name",
        "upper_dof_names",
        "upper_left_arm_dof_names",
        "upper_right_arm_dof_names",
        "lower_dof_names",
        "has_torso",
        "has_upper_body_dof",
        "left_ankle_dof_names",
        "right_ankle_dof_names",
        "knee_dof_names",
        "hips_dof_names",
        "terminate_after_contacts_on",
        "penalize_contacts_on",
        "randomize_link_body_names",
        "waist_dof_names",
        "waist_yaw_dof_name",
        "waist_roll_dof_name",
        "waist_pitch_dof_name",
        "arm_dof_names",
        "left_arm_dof_names",
        "right_arm_dof_names",
        "symmetry_joint_names",
        "flip_sign_joint_names",
        "apply_dof_armature_in_isaacgym",
        "knee_joint_min_threshold",
        "lidar_height_offset",
        "soft_dof_pos_limit",
        "termination_close_to_dof_pos_limit",
        # Removed 2026-07-09 (8254b123) and silently ignored since.
        "object",
    }
)
_DEFAULT_SOFT_DOF_POS_LIMIT = NoiseToInitialPoseConfig().soft_dof_pos_limit
_MOTION_COMMAND_FUNCS = {
    "holosoma.managers.command.terms.wbt:MotionCommand",
    "holosoma.managers.command.terms.wbt.MotionCommand",
}


def _migrate_dof_property_lists(robot: dict[str, Any]) -> None:
    """Convert positional legacy properties using the saved canonical order, never a preset."""
    names = _names(robot, "dof_names")

    def take(field: str) -> list[Any] | None:
        values = robot.pop(field, None)
        if values is not None and (not isinstance(values, list) or len(values) != len(names)):
            raise ValueError(f"robot.{field} must have {len(names)} entries in saved dof_names order.")
        return values

    def put(field: str, values: Any) -> None:
        if field in robot and robot[field] != values:
            raise ValueError(f"robot.{field} conflicts with its legacy positional property; keep one representation.")
        robot[field] = values

    if "dof_pos_lower_limit_list" in robot or "dof_pos_upper_limit_list" in robot:
        lower, upper = take("dof_pos_lower_limit_list"), take("dof_pos_upper_limit_list")
        if (lower is None) != (upper is None):
            raise ValueError("Legacy position lower and upper limits must be supplied together.")
        positions = None
        if lower is not None:
            assert upper is not None
            positions = dof_limits_from_sequences(names, lower, upper, what="legacy robot position limits")
        put("dof_pos_limits", positions)
    for old, new in (
        ("dof_vel_limit_list", "dof_vel_limits"),
        ("dof_effort_limit_list", "dof_effort_limits"),
        ("dof_armature_list", "dof_armature"),
        ("dof_joint_friction_list", "dof_joint_friction"),
    ):
        if old in robot:
            values = take(old)
            put(
                new,
                None
                if values is None
                else dof_values_from_sequence(names, values, what=f"legacy robot {new.replace('_', ' ')}"),
            )


def _names(robot: Mapping[str, Any], key: str) -> list[str]:
    value = robot.get(key)
    return [str(n) for n in value] if isinstance(value, list) else []


def _expand_substring(patterns: Iterable[str], body_names: Sequence[str]) -> list[str]:
    """The pre-refactor ``LeggedRobotLocomotionManager._setup_robot_body_indices`` rule: every body
    whose name contains a pattern, in ``body_names`` order per pattern, deduped."""
    out: list[str] = []
    for pattern in patterns:
        for name in body_names:
            if pattern in name and name not in out:
                out.append(name)
    return out


def _iter_terms(manager: Any, groups: Iterable[str]) -> Iterable[dict[str, Any]]:
    if not isinstance(manager, dict):
        return
    for group in groups:
        terms = manager.get(group)
        if isinstance(terms, dict):
            for term in terms.values():
                if isinstance(term, dict):
                    yield term


def _get_path(root: Any, dotted: str) -> dict[str, Any] | None:
    """The dict at ``dotted`` (``"a.b.c"``), or None when any level is missing or not a dict."""
    node: Any = root
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, dict) else None


def _nested_dataclass(type_hint: Any) -> type | None:
    """The dataclass inside a field annotation (``Cfg`` or ``Cfg | None``), else None."""
    candidates = [type_hint, *typing.get_args(type_hint)]
    return next((c for c in candidates if isinstance(c, type) and dataclasses.is_dataclass(c)), None)


def _field_annotations(cls: type) -> dict[str, Any]:
    """Field name -> annotation, resolved. Pydantic dataclasses carry resolved annotations on every supported
    Python (``typing.get_type_hints`` cannot evaluate ``list[str]`` on 3.8); plain dataclasses give raw types."""
    fields = getattr(cls, "__pydantic_fields__", None)
    if fields:
        return {name: info.annotation for name, info in fields.items()}
    return {f.name: f.type for f in dataclasses.fields(cls)}


def _unknown_keys(node: Mapping[str, Any], cls: type, prefix: str = "") -> list[str]:
    """Dotted keys under ``node`` that the dataclass ``cls`` does not declare, descending into nested dataclasses."""
    hints = _field_annotations(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = [f"{prefix}{key}" for key in node if key not in known]
    for key, value in node.items():
        nested = _nested_dataclass(hints.get(key)) if key in known else None
        if nested is not None and isinstance(value, dict):
            unknown += _unknown_keys(value, nested, f"{prefix}{key}.")
    return sorted(unknown)


def _unknown_robot_keys(robot: Mapping[str, Any]) -> list[str]:
    """Dotted robot keys the current schema does not declare (legacy, fork or subclass fields), at any depth."""
    return _unknown_keys(robot, RobotConfig)


def _pop_path(root: dict[str, Any], dotted: str) -> None:
    *parents, leaf = dotted.split(".")
    node = _get_path(root, ".".join(parents)) if parents else root
    if node is not None:
        node.pop(leaf, None)


def _substring_gain_keys(control: Mapping[str, Any], dof_names: Sequence[str]) -> list[str]:
    """Stiffness/damping keys that are neither exact DOF names nor ``#tag`` references (the pre-refactor
    substring scheme)."""
    known = set(dof_names)
    keys = set(control.get("stiffness") or {}) | set(control.get("damping") or {})
    return sorted(str(k) for k in keys if k not in known and not str(k).startswith("#"))


def _expand_gains_per_dof(control: dict[str, Any], dof_names: Sequence[str]) -> None:
    """Rewrite ``stiffness``/``damping`` to one exact key per DOF with the pre-refactor
    ``JointPositionActionTerm._configure_pd_gains`` rule: every key that is a substring of the DOF name
    matches, iterated in dict order, and the last match wins. An unmatched DOF raised for ``P``/``V``
    control and got zero gains for ``T``; both are reproduced."""
    stiffness = control.get("stiffness") or {}
    damping = control.get("damping") or {}
    tag_keys = sorted(key for key in set(stiffness) | set(damping) if str(key).startswith("#"))
    if tag_keys:
        raise ValueError(
            f"Cannot mix legacy substring PD gain keys {_substring_gain_keys(control, dof_names)} "
            f"with '#tag' keys {tag_keys}. "
            "Use exact DOF names / '#tag' keys only, or keep legacy substring keys without tags."
        )
    new_stiffness: dict[str, float] = {}
    new_damping: dict[str, float] = {}
    for dof in dof_names:
        matches = [key for key in stiffness if key in dof]
        if not matches:
            if control.get("control_type") in ("P", "V"):
                raise ValueError(
                    f"PD gains for joint '{dof}' were not defined: no robot.control.stiffness key is a substring "
                    f"of it (keys: {list(stiffness)}). Add a stiffness and damping entry for '{dof}'."
                )
            new_stiffness[dof] = 0.0
            new_damping[dof] = 0.0
            continue
        key = matches[-1]
        if key not in damping:
            raise ValueError(f"robot.control.damping has no entry for stiffness key {key!r} (DOF '{dof}'); add one.")
        new_stiffness[dof] = float(stiffness[key])
        new_damping[dof] = float(damping[key])
    control["stiffness"] = new_stiffness
    control["damping"] = new_damping


def _migrate_robot_control(
    out: dict[str, Any],
    robot: dict[str, Any],
    control: dict[str, Any],
    changes: list[str],
) -> None:
    """Move the pre-refactor robot controller into its joint action term."""
    if "control" not in robot:
        return
    candidates = [
        term
        for term in _iter_terms(out.get("action"), ("terms",))
        if str(term.get("func", "")).endswith(":JointPositionActionTerm")
    ]
    if len(candidates) != 1:
        if not _names(robot, "dof_names") and control.get("control_type") in (None, "none", "T"):
            robot.pop("control", None)
            changes.append("unused zero-DOF robot.control dropped")
            return
        raise ValueError(
            "Cannot migrate robot.control: expected exactly one JointPositionActionTerm in action.terms, "
            f"found {len(candidates)}."
        )
    term = candidates[0]
    if term.get("controller") not in (None, {}):
        raise ValueError("Cannot migrate robot.control because the joint action term already defines controller.")
    term["controller"] = {
        "control_type": control.get("control_type", "P"),
        "stiffness": control.get("stiffness") or {},
        "damping": control.get("damping") or {},
        "clip_torques": bool(control.get("clip_torques", True)),
        "action_scales_by_effort_limit_over_p_gain": bool(
            control.get("action_scales_by_effort_limit_over_p_gain", False)
        ),
    }
    term["scale"] = float(control.get("action_scale", 1.0))
    clip_value = float(control.get("action_clip_value", 100.0))
    term["clip"] = [-clip_value, clip_value] if control.get("clip_actions", False) else None
    robot.pop("control", None)
    changes.append("robot.control -> JointPositionActionTerm controller/scale/clip")


def _migrate_motion_root(out: dict[str, Any], robot: Mapping[str, Any], *, legacy_body_schema: bool) -> None:
    """Preserve the old WBT column-zero choice only when its body identity is known."""
    for term in _iter_terms(out.get("command"), ("setup_terms", "reset_terms", "step_terms")):
        if term.get("func") not in _MOTION_COMMAND_FUNCS:
            continue
        motion = _get_path(term, "params.motion_config")
        if motion is None or "root_body" in motion:
            continue
        simulator = _get_path(out, "simulator") or {}
        if legacy_body_schema and simulator.get("_target_") not in {
            "holosoma.simulator.isaacgym.isaacgym.IsaacGym",
            "holosoma.simulator.isaacsim.isaacsim.IsaacSim",
        }:
            raise ValueError(
                "Cannot infer MotionConfig.root_body for this pre-refactor simulator: its runtime body "
                "order was not guaranteed to match robot.body_names. Set root_body to the exact body "
                "that supplied motion column zero in the old runtime (or an explicit single-body tag)."
            )
        body_names = _names(robot, "body_names")
        if not body_names:
            raise ValueError("Cannot migrate MotionConfig.root_body without a nonempty saved robot.body_names list.")
        motion["root_body"] = body_names[0]
        logger.warning(
            "Migrated MotionConfig.root_body to {!r}, preserving the saved config's former first-body "
            "selection. This does not infer the physical root or use body_name_ref.",
            body_names[0],
        )


def migrate_legacy_experiment_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``raw`` rewritten to the current schema.

    Current dumps pass through unchanged. Unknown robot keys are dropped with a warning after known
    fields are migrated. Calling this function again on its result makes no further changes.

    Raises
    ------
    ValueError
        ``apply_dof_armature_in_isaacgym: false`` (IsaacGym now always applies ``dof_armature``;
        there is no equivalent); a ``contact_forces_exceeded`` term with a non-default
        ``contact_indices_attr`` (it pointed at an env attribute that no longer exists); or a ``P``/``V``
        control dump whose substring gain keys leave a DOF without gains (the old action term raised too).
        WBT dumps without ``root_body`` also require an explicit selection when the old runtime body
        order cannot be recovered from the config (pre-refactor MuJoCo or unknown simulator).
    """
    out = copy.deepcopy(dict(raw))
    robot = out.get("robot")
    if not isinstance(robot, dict):
        return out
    _migrate_dof_property_lists(robot)

    init_raw = robot.get("init_state")
    init_state: dict[str, Any] = init_raw if isinstance(init_raw, dict) else {}
    control_raw = robot.get("control")
    control: dict[str, Any] = control_raw if isinstance(control_raw, dict) else {}
    dof_names = _names(robot, "dof_names")
    if not dof_names and control.get("control_type") == "none":
        control["control_type"] = "T"
    substring_gain_keys = _substring_gain_keys(control, dof_names) if dof_names else []
    legacy_robot = set(robot) & _DELETED_ROBOT_KEYS
    gain_keys = set(control.get("stiffness") or {}) | set(control.get("damping") or {})
    # Complete DOF names also overlapped under the old substring rule. The old spawn/morphology
    # fields identify that schema; default_joint_angles alone also occurs in tag-aware dumps.
    # 'object' was already obsolete before this refactor and is not evidence of old gain semantics.
    legacy_gain_schema = bool(legacy_robot - {"object"}) or "pos" in init_state or "rot" in init_state
    legacy_gains = legacy_gain_schema and not any(str(key).startswith("#") for key in gain_keys)
    _migrate_motion_root(out, robot, legacy_body_schema=legacy_gain_schema)
    if not _unknown_robot_keys(robot) and not substring_gain_keys:
        return out

    changes: list[str] = []
    body_names = _names(robot, "body_names")

    # init_state: pos/rot (xyzw) -> position/orientation (wxyz)
    if "pos" in init_state:
        init_state["position"] = init_state.pop("pos")
        changes.append("init_state.pos -> init_state.position")
    if "rot" in init_state:
        x, y, z, w = init_state.pop("rot")
        init_state["orientation"] = [w, x, y, z]
        changes.append("init_state.rot [x,y,z,w] -> init_state.orientation [w,x,y,z]")
    if "default_joint_angles" in init_state:
        defaults = init_state.pop("default_joint_angles")
        if "default_joint_positions" in init_state and init_state["default_joint_positions"] != defaults:
            raise ValueError(
                "robot.init_state has conflicting default_joint_angles and default_joint_positions; "
                "keep only the intended default_joint_positions map."
            )
        # The old exact-name map was indexed only by dof_names, so surplus entries were unused.
        # Require old schema evidence and no tag-aware signals; intermediate dumps also used the
        # default_joint_angles spelling. Explicit current maps retain strict validation.
        if (
            legacy_gains
            and dof_names
            and "joint_tags" not in robot
            and "default_joint_positions" not in init_state
            and isinstance(defaults, dict)
            and all(isinstance(name, str) and not name.startswith("#") for name in defaults)
        ):
            default_dofs = set(dof_names)
            unused_defaults = [name for name in defaults if name not in default_dofs]
            if unused_defaults:
                defaults = {name: value for name, value in defaults.items() if name in default_dofs}
                changes.append(f"init_state.default_joint_angles entries {unused_defaults} dropped: not in dof_names")
        init_state["default_joint_positions"] = defaults
        changes.append("init_state.default_joint_angles -> init_state.default_joint_positions")

    # Morphology facts -> body tags (existing tags win). The base tag comes from torso_name alone: the
    # old DR terms read it whether or not has_torso was set.
    body_tags: dict[str, list[str]] = dict(robot.get("body_tags") or {})
    for legacy_key, tag in (("foot_body_name", "#feet"), ("foot_height_name", "#feet_height")):
        pattern = robot.get(legacy_key)
        if not pattern or tag in body_tags:
            continue
        bodies = _expand_substring([pattern], body_names)
        if bodies:
            body_tags[tag] = bodies
            changes.append(f"{legacy_key} -> body_tags['{tag}'] = {bodies}")
        else:
            logger.warning(
                f"Legacy {legacy_key}={pattern!r} matches no body_names entry; no body_tags['{tag}'] derived."
            )
    torso = robot.get("torso_name")
    if torso and torso in body_names and "#base" not in body_tags:
        body_tags["#base"] = [torso]
        changes.append(f"torso_name -> body_tags['#base'] = ['{torso}']")
    if body_tags:
        robot["body_tags"] = body_tags

    # Symmetry data -> joint tags read by the algo's default JointSymmetryConfig. Names outside dof_names
    # are dropped (the old SymmetryUtils skipped them); a pair goes when either side is stale.
    joint_tags: dict[str, list[str]] = dict(robot.get("joint_tags") or {})
    known_dofs = set(dof_names) if dof_names else None
    mirror = robot.get("symmetry_joint_names")
    if isinstance(mirror, dict) and mirror:
        left: list[str] = []
        right: list[str] = []
        seen: set[str] = set()
        stale_pairs: list[tuple[str, str]] = []
        one_way: list[tuple[str, str]] = []
        for k, v in mirror.items():
            if k == v or k in seen:
                continue  # self-map (central joint) or the inverse of a pair already handled
            seen.update((k, v))
            if known_dofs is not None and not {k, v} <= known_dofs:
                stale_pairs.append((k, v))
            elif mirror.get(v) != k:
                one_way.append((k, v))
            else:
                left.append(k)
                right.append(v)
        if stale_pairs:
            logger.warning(f"Legacy symmetry_joint_names pairs {stale_pairs} dropped: not in dof_names.")
        if one_way:
            algo_config = _get_path(out, "algo.config")
            if algo_config is not None and algo_config.get("use_symmetry") is False:
                logger.warning(
                    f"Legacy symmetry_joint_names pairs {one_way} dropped: one-way, and use_symmetry is off."
                )
            else:
                raise ValueError(
                    f"Legacy symmetry_joint_names is not reciprocal: {one_way} have no reverse entry. The old code "
                    "applied such a map one-way, which is not a mirror; add the reverse entries, or set "
                    "algo.config.use_symmetry=False in the dump so the map is dropped."
                )
        if left and "#mirror_left" not in joint_tags and "#mirror_right" not in joint_tags:
            joint_tags["#mirror_left"] = left
            joint_tags["#mirror_right"] = right
            changes.append(f"symmetry_joint_names -> joint_tags['#mirror_left'/'#mirror_right'] ({len(left)} pairs)")
    flip = robot.get("flip_sign_joint_names")
    if flip and "#mirror_flip_sign" not in joint_tags:
        kept = [str(n) for n in flip if known_dofs is None or n in known_dofs]
        stale = [str(n) for n in flip if str(n) not in kept]
        if stale:
            logger.warning(f"Legacy flip_sign_joint_names {stale} dropped: not in dof_names.")
        if kept:
            joint_tags["#mirror_flip_sign"] = kept
            changes.append("flip_sign_joint_names -> joint_tags['#mirror_flip_sign']")
    if joint_tags:
        robot["joint_tags"] = joint_tags

    # Task decisions -> term params (setdefault: an already-migrated dump wins). Only for a pre-refactor
    # robot dump; the old fields were required, so they are present whenever any legacy robot key is.
    if legacy_robot:
        termination_bodies = _expand_substring(robot.get("terminate_after_contacts_on", []), body_names)
        for term in _iter_terms(out.get("termination"), ("terms",)):
            if str(term.get("func", "")).endswith(":contact_forces_exceeded"):
                params = term.setdefault("params", {})
                attr = params.pop("contact_indices_attr", None)
                if attr not in (None, "termination_contact_indices"):
                    raise ValueError(
                        f"contact_forces_exceeded used contact_indices_attr={attr!r}; only the default "
                        "'termination_contact_indices' can be migrated to an explicit 'bodies' list."
                    )
                if "bodies" not in params:
                    params["bodies"] = termination_bodies
                    changes.append(
                        f"terminate_after_contacts_on -> termination params['bodies'] = {termination_bodies}"
                    )
        has_base = "#base" in body_tags
        unset_base_params: list[str] = []
        for term in _iter_terms(out.get("randomization"), ("setup_terms", "reset_terms", "step_terms")):
            func = str(term.get("func", ""))
            if func.endswith(":randomize_mass_startup"):
                params = term.setdefault("params", {})
                if "link_bodies" not in params:
                    # The old IsaacGym path skipped unknown link names with a warning; the resolver now raises.
                    link_names = _names(robot, "randomize_link_body_names")
                    stale_links = [n for n in link_names if body_names and n not in body_names]
                    if stale_links:
                        logger.warning(f"Legacy randomize_link_body_names {stale_links} dropped: not in body_names.")
                    params["link_bodies"] = [n for n in link_names if n not in stale_links]
                    changes.append("randomize_link_body_names -> randomize_mass_startup params['link_bodies']")
                    if "randomize_link_body_names" in robot and not params["link_bodies"]:
                        # An empty legacy selection skipped link scaling while base mass could still run.
                        # Only repair derived selections; explicit/current params retain strict validation.
                        params["enable_link_mass"] = False
                        logger.warning(
                            "Legacy randomize_link_body_names resolves to no bodies; setting "
                            "randomize_mass_startup enable_link_mass=False to preserve its no-op behavior. "
                            "Base-mass settings are unchanged."
                        )
                if has_base:
                    params.setdefault("base_body", "#base")
                elif "base_body" not in params:
                    unset_base_params.append("randomize_mass_startup params['base_body']")
            elif func.endswith(":randomize_base_com_startup"):
                params = term.setdefault("params", {})
                if has_base:
                    params.setdefault("body", "#base")
                elif "body" not in params:
                    unset_base_params.append("randomize_base_com_startup params['body']")
        if unset_base_params:
            logger.warning(
                f"Legacy torso_name={torso!r} is not in body_names, so no body_tags['#base'] was derived; "
                f"{unset_base_params} left unset and their '#base' default will not resolve. Set them to a body "
                "name or add the tag."
            )

    # Relocated knobs.
    if "contact_pairs_multiplier" in robot:
        physx = _get_path(out, "simulator.config.sim.physx")
        if physx is None:
            changes.append("contact_pairs_multiplier dropped: no simulator.config.sim.physx dict to relocate it into")
        elif "max_gpu_contact_pairs" not in physx:
            physx["max_gpu_contact_pairs"] = int(robot["contact_pairs_multiplier"]) * 1024 * 1024
            changes.append("contact_pairs_multiplier -> simulator.config.sim.physx.max_gpu_contact_pairs")
    if robot.get("apply_dof_armature_in_isaacgym") is False:
        raise ValueError(
            "apply_dof_armature_in_isaacgym=false cannot be migrated: every backend now applies "
            "dof_armature. Set dof_armature=None to retain asset values instead."
        )
    soft = robot.get("soft_dof_pos_limit")
    if soft is not None and abs(float(soft) - _DEFAULT_SOFT_DOF_POS_LIMIT) > 1e-9:
        # The WBT initial-pose clamp is the consumer that moved; carry the value into every MotionCommand term
        # (an explicit term value wins). The limits_dof_pos reward terms keep their own param over hard limits
        # on every backend, so IsaacGym no longer contracts dof_pos_limits twice.
        carried = 0
        for term in _iter_terms(out.get("command"), ("setup_terms", "reset_terms", "step_terms")):
            # Preserve legacy extension matching; the builtin dotted spelling is also recognized.
            func = str(term.get("func", ""))
            if not func.endswith(":MotionCommand") and func not in _MOTION_COMMAND_FUNCS:
                continue
            motion = _get_path(term, "params.motion_config")
            if motion is None:
                continue
            noise = motion.setdefault("noise_to_initial_pose", {})
            if isinstance(noise, dict) and "soft_dof_pos_limit" not in noise:
                noise["soft_dof_pos_limit"] = float(soft)
                carried += 1
        if carried:
            changes.append(f"soft_dof_pos_limit={soft} -> noise_to_initial_pose of {carried} MotionCommand term(s)")
        else:
            logger.warning(
                f"Legacy robot.soft_dof_pos_limit={soft} dropped: no MotionCommand term carries it; the "
                f"limits_dof_pos reward terms use their own param (default {_DEFAULT_SOFT_DOF_POS_LIMIT}) over "
                "hard limits on every backend."
            )

    # Explicit substring keys retain their old interpretation even in an otherwise migrated dump.
    if dof_names and (substring_gain_keys or legacy_gains):
        _expand_gains_per_dof(control, dof_names)
        changes.append("control.stiffness/damping legacy substring matching -> exact per-DOF keys")

    _migrate_robot_control(out, robot, control, changes)

    # Drop everything the current schema does not declare: the mapped legacy keys above and any fork or
    # subclass field the dump carries.
    dropped = _unknown_robot_keys(robot)
    for path in dropped:
        _pop_path(robot, path)
    if dropped:
        changes.append(f"dropped {dropped}")

    logger.warning(f"Migrated a pre-refactor experiment config: {'; '.join(changes)}")
    return out
