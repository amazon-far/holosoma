"""Named scene presets, selectable via the ``scene:`` subcommand.

Each preset hard-defines which assets are spawned; can override fields of a preset's assets
at the CLI (e.g. ``scene:object-managers-demo --scene.rigid-objects.free-box.position 0.9 0 0.7``),
but cannot add/remove assets (the preset fixes the set of keys).

This module holds ONLY production scenes — the ones a shipped experiment, the run-sim CLI, or the
object-managers example actually use. The much larger set of scenes that exist only to drive the
cross-backend test harnesses lives in ``tests/simulators/_scene_presets.py`` and is registered into
:data:`SCENE_REGISTRY` at test time (so the production ``scene:`` menu stays small and core never
imports from ``tests/``). ``SCENE_REGISTRY`` is a dict, so tests append to it via that module's
``register()``.
"""

from holosoma.config_types.scene import (
    RigidObjectConfig,
    SceneConfig,
    SceneFileConfig,
)
from holosoma.config_values.wbt.g1.scene import g1_29dof_wbt_object_scene
from holosoma.utils.config_registry import ConfigRegistry, deprecated_defaults_alias

SCENE_REGISTRY = ConfigRegistry(SceneConfig, group="holosoma.config.scene")

# Box asset paths.
_SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
_SMALL_BOX_USD = "holosoma/data/scene_objects/boxes/small_box.usda"

# No scene assets (the default).
empty = SceneConfig()

# Showcase scene for the object-managers example (holosoma/examples/object_managers_demo.py):
# combines BOTH asset kinds in one scene so the demo exercises the full surface —
#   - standalone rigid bodies: a free box, a second free box with a configured initial
#     velocity (so reset/jitter velocity-restore is observable), and a static pillar;
#   - a 1->N scene FILE that expands to 'scene_free_box' (free) + 'scene_static_post' (static).
# Tri-format on every asset (urdf for IsaacGym, usd for IsaacSim, urdf/xml for MuJoCo) so the
# same preset runs under all four backends. Names are the ObjectRegistry keys the managers
# address: free bodies = {free_box, moving_box, scene_free_box}, static = {pillar, scene_static_post}.
object_managers_demo = SceneConfig(
    rigid_objects={
        "free_box": RigidObjectConfig(urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=[0.4, 0.0, 0.6]),
        "moving_box": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.4, 0.4, 0.6],
            linear_velocity=[0.5, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 1.0],
        ),
        "pillar": RigidObjectConfig(
            urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=[0.8, 0.0, 0.3], fixed=True
        ),
    },
    scene_files={
        "scene": SceneFileConfig(
            xml_path="holosoma/data/scene_objects/multibody/multibody.xml",
            urdf_path="holosoma/data/scene_objects/multibody/multibody.urdf",
            usd_path="holosoma/data/scene_objects/multibody/multibody.usda",
            position=[0.0, -0.6, 0.6],
        )
    },
)

# Production scene presets only. Test-only scenes are added at test time by
# tests/simulators/_scene_presets.register() (see this module's docstring).
SCENE_REGISTRY.add("empty", empty)
SCENE_REGISTRY.add("object-managers-demo", object_managers_demo)
SCENE_REGISTRY.add("g1_29dof_wbt_object", g1_29dof_wbt_object_scene)

__getattr__ = deprecated_defaults_alias(__name__, SCENE_REGISTRY)
