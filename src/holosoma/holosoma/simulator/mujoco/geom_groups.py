# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""MuJoCo geometry-group policy for LiDAR queries and rendering.

Robot assets conventionally use groups 0-2 for visuals and 3-5 for collision/debug geometry.
After compilation, Holosoma moves robot geometry into two reserved groups so developers can exclude
the whole robot from LiDAR queries without hiding unrelated scene visuals. Groups 4 and 5 are
therefore unavailable to nonrobot geoms, and MuJoCo viewer group 4 controls all robot visuals after
compilation rather than their authored 0-2 groups.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco

# Fixed mask width used by MuJoCo and MuJoCo Warp ray APIs.
MUJOCO_GEOM_GROUP_COUNT = 6

# Authored groups treated as visible robot surfaces.
AUTHORED_ROBOT_VISUAL_GEOM_GROUPS = frozenset((0, 1, 2))
# Authored groups treated as robot collision or debug geometry.
AUTHORED_ROBOT_COLLISION_GEOM_GROUPS = frozenset((3, 4, 5))

# Compiled group containing all robot visual geometry.
ROBOT_VISUAL_GEOM_GROUP = 4
# Compiled group containing all robot collision/debug geometry.
ROBOT_COLLISION_GEOM_GROUP = 5
# Groups reserved globally after robot tagging.
ROBOT_RESERVED_GEOM_GROUPS = frozenset((ROBOT_VISUAL_GEOM_GROUP, ROBOT_COLLISION_GEOM_GROUP))

# Groups rendered by default after robot visuals move from their authored 0-2 group into group 4.
# Robot collision/debug geometry stays out of the default render mask in group 5.
DEFAULT_RENDER_GEOM_GROUPS = (0, 1, 2, ROBOT_VISUAL_GEOM_GROUP)


def map_authored_geom_group_mask(authored_mask: Sequence[bool]) -> tuple[bool, ...]:
    """Map an authored six-group mask onto the compiled robot's two reserved layers.

    Nonrobot geometry keeps its authored group. Robot groups 0-2 are compacted into the reserved
    visual layer and groups 3-5 into the reserved collision/debug layer, so each reserved bit is
    enabled when any authored group in its category is enabled. An all-false input remains all
    false.
    """

    if len(authored_mask) != MUJOCO_GEOM_GROUP_COUNT:
        raise ValueError(f"MuJoCo geom-group masks need {MUJOCO_GEOM_GROUP_COUNT} entries, got {len(authored_mask)}.")

    source_mask = tuple(bool(enabled) for enabled in authored_mask)
    mapped_mask = list(source_mask)
    mapped_mask[ROBOT_VISUAL_GEOM_GROUP] = any(source_mask[group] for group in AUTHORED_ROBOT_VISUAL_GEOM_GROUPS)
    mapped_mask[ROBOT_COLLISION_GEOM_GROUP] = any(source_mask[group] for group in AUTHORED_ROBOT_COLLISION_GEOM_GROUPS)
    return tuple(mapped_mask)


def tag_compiled_robot_geom_groups(
    model: mujoco.MjModel,
    robot_root_body_id: int | None,
) -> None:
    """Retag robot-subtree geoms in an already compiled MuJoCo model.

    Compiling first preserves ``compiler.inertiagrouprange`` behavior exactly. Groups 4 and
    5 are globally reserved after this call, so any nonrobot geom already using either group
    is rejected before the model is mutated.
    """

    def is_robot_body(body_id: int) -> bool:
        if robot_root_body_id is None:
            return False
        while body_id != 0:
            if body_id == robot_root_body_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    robot_geom_ids: list[int] = []
    nonrobot_reserved_geoms: list[str] = []

    for geom_id in range(model.ngeom):
        authored_group = int(model.geom_group[geom_id])
        if not 0 <= authored_group < MUJOCO_GEOM_GROUP_COUNT:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"#{geom_id}"
            raise ValueError(
                f"MuJoCo geom '{geom_name}' uses unsupported group {authored_group}; expected a group from 0 to 5."
            )

        if not is_robot_body(int(model.geom_bodyid[geom_id])):
            if authored_group in ROBOT_RESERVED_GEOM_GROUPS:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"#{geom_id}"
                body_id = int(model.geom_bodyid[geom_id])
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"#{body_id}"
                nonrobot_reserved_geoms.append(f"{body_name}/{geom_name} (group {authored_group})")
            continue

        robot_geom_ids.append(geom_id)

    if nonrobot_reserved_geoms:
        joined = ", ".join(nonrobot_reserved_geoms)
        raise ValueError(
            "MuJoCo geom groups 4 and 5 are reserved for compiled robot visual and collision layers; "
            f"nonrobot geometry already uses them: {joined}."
        )

    for geom_id in robot_geom_ids:
        authored_group = int(model.geom_group[geom_id])
        model.geom_group[geom_id] = (
            ROBOT_VISUAL_GEOM_GROUP
            if authored_group in AUTHORED_ROBOT_VISUAL_GEOM_GROUPS
            else ROBOT_COLLISION_GEOM_GROUP
        )
