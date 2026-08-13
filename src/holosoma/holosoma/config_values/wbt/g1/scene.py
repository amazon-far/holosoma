"""Whole Body Tracking scene preset for the G1 robot."""

from __future__ import annotations

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig

# WBT object-tracking spawns a single free box, addressed by the actor name "object"
# (see managers/command/terms/wbt.py).
g1_29dof_wbt_object_scene = SceneConfig(
    env_spacing=0.0,
    rigid_objects={
        "object": RigidObjectConfig(
            urdf_file="holosoma/data/scene_objects/boxes/large_box.urdf",
        )
    },
)
