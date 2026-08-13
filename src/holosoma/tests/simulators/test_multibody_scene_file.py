"""Unit tests for multi-body scene-file (1->N) config (no simulator).

Covers the backend-agnostic config-level guarantees of multi-body scene-file loading:
- collision-safe naming: duplicate names across rigid_objects/scene_files fail loud;
- the shared format selector works for SceneFileConfig's ``{fmt}_path`` fields too;
- the per-object static/dynamic override (SceneFileConfig.resolve_fixed).

The live per-backend spawn + override are covered by the per-backend test_scene_spawn_*.py
suites.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.scene import (
    ObjectPatternConfig,
    PhysicsConfig,
    PhysXPhysicsConfig,
    RigidObjectConfig,
    SceneConfig,
    SceneFileConfig,
)
from holosoma.simulator.shared.asset_format import select_asset_format

# ----- collision-safe naming (SceneConfig validator) -----


def test_scene_file_name_colliding_with_rigid_object_fails_loud():
    with pytest.raises(ValueError, match="(?i)duplicate"):
        SceneConfig(
            rigid_objects={"box": RigidObjectConfig(urdf_file="b.urdf")},
            scene_files={"box": SceneFileConfig(xml_path="a.xml")},
        )


def test_distinct_names_ok():
    cfg = SceneConfig(
        rigid_objects={"box": RigidObjectConfig(urdf_file="b.urdf")},
        scene_files={"scene": SceneFileConfig(xml_path="a.xml")},
    )
    assert list(cfg.scene_files) == ["scene"]


# ----- shared format selector works for SceneFileConfig ({fmt}_path) -----


def test_scene_file_format_selection_per_backend():
    sf = SceneFileConfig(xml_path="a.xml", urdf_path="a.urdf", usd_path="a.usda")
    # MuJoCo prefers xml; IsaacGym only urdf; IsaacSim prefers usd.
    assert select_asset_format(sf, ["xml", "urdf"]) == ("xml", "a.xml")
    assert select_asset_format(sf, ["urdf"]) == ("urdf", "a.urdf")
    assert select_asset_format(sf, ["usd", "urdf"]) == ("usd", "a.usda")


# ----- per-object static/dynamic override via object_configs (SceneFileConfig.resolve_fixed) -----


def test_resolve_fixed_defaults_to_structure_when_no_override():
    sf = SceneFileConfig(xml_path="a.xml")
    assert sf.resolve_fixed("free_box", structural_default=False) is False
    assert sf.resolve_fixed("static_post", structural_default=True) is True


def test_resolve_fixed_override_forces_static_and_free():
    sf = SceneFileConfig(
        name="s",
        xml_path="a.xml",
        object_configs={
            "free_box": ObjectPatternConfig(fixed=True),  # force a structurally-free body static
            "static_post": ObjectPatternConfig(fixed=False),  # force a structurally-static body free
        },
    )
    assert sf.resolve_fixed("free_box", structural_default=False) is True
    assert sf.resolve_fixed("static_post", structural_default=True) is False


def test_resolve_fixed_none_override_keeps_structure():
    # An object_configs entry that sets physics but leaves fixed=None must NOT override.
    sf = SceneFileConfig(xml_path="a.xml", object_configs={"*": ObjectPatternConfig()})
    assert sf.resolve_fixed("anything", structural_default=False) is False


def test_resolve_fixed_matches_glob_patterns():
    sf = SceneFileConfig(xml_path="a.xml", object_configs={"box_*": ObjectPatternConfig(fixed=True)})
    assert sf.resolve_fixed("box_3", structural_default=False) is True
    assert sf.resolve_fixed("post", structural_default=False) is False


# ----- per-object physics override via object_configs (SceneFileConfig.resolve_physics) -----
# This is the SHARED rule every backend uses (IsaacGym + IsaacSim both call resolve_physics),
# so a scene file's per-body physics override is applied identically across backends.


def test_resolve_physics_none_when_no_object_configs():
    sf = SceneFileConfig(xml_path="a.xml")
    assert sf.resolve_physics("any_body") is None


def test_resolve_physics_returns_matching_override():
    phys = PhysicsConfig(mass=4.0, physx=PhysXPhysicsConfig(linear_damping=0.5))
    sf = SceneFileConfig(xml_path="a.xml", object_configs={"box_*": ObjectPatternConfig(physics=phys)})
    assert sf.resolve_physics("box_1") is phys
    assert sf.resolve_physics("post") is None


def test_resolve_physics_ignores_fixed_only_entries():
    # An entry that overrides `fixed` but carries no physics must NOT yield a physics config
    # (it would otherwise return a None physics and mask the "no override" case).
    sf = SceneFileConfig(xml_path="a.xml", object_configs={"*": ObjectPatternConfig(fixed=True)})
    assert sf.resolve_physics("anything") is None


def test_resolve_physics_strips_leading_glob_like_resolve_fixed():
    # Mirrors resolve_fixed's "*/" stripping so the two share one matching rule.
    phys = PhysicsConfig(mass=2.0)
    sf = SceneFileConfig(xml_path="a.xml", object_configs={"*/free_box": ObjectPatternConfig(physics=phys)})
    assert sf.resolve_physics("free_box") is phys


def test_scene_file_has_no_scale_field():
    # `scale` was removed as dead config (no backend honored it); guard against re-introducing
    # it silently. Add it back WITH backend wiring + a test if/when it's actually needed.
    sf = SceneFileConfig(xml_path="a.xml")
    assert not hasattr(sf, "scale")


# ----- include/exclude SUBSET filter (SceneFileConfig.should_include) -----
# This is the SHARED rule every backend uses (MuJoCo / IsaacGym / IsaacSim), so a scene file's
# body set is identical regardless of which backend loads it. Distinct from object_configs, which
# only OVERRIDES per-body settings and never filters (see test_object_configs_does_not_filter).


def test_should_include_defaults_to_all():
    # Default include_patterns=["*"], no excludes => every body loads.
    sf = SceneFileConfig(xml_path="a.xml")
    assert sf.should_include("free_box") is True
    assert sf.should_include("static_post") is True


def test_should_include_respects_include_patterns():
    sf = SceneFileConfig(xml_path="a.xml", include_patterns=["box_*"])
    assert sf.should_include("box_1") is True
    assert sf.should_include("post") is False


def test_should_include_exclude_wins_over_include():
    # A body matching both include and exclude is dropped (exclude takes precedence).
    sf = SceneFileConfig(xml_path="a.xml", include_patterns=["*"], exclude_patterns=["*_collision"])
    assert sf.should_include("leg_1") is True
    assert sf.should_include("leg_1_collision") is False


def test_should_include_empty_include_means_no_filter():
    # An explicit empty include list is treated as "no include filter" (load everything not
    # excluded), matching the pre-refactor loader / IsaacGym semantics.
    sf = SceneFileConfig(xml_path="a.xml", include_patterns=[], exclude_patterns=["drop_*"])
    assert sf.should_include("keep") is True
    assert sf.should_include("drop_this") is False


def test_should_include_strips_leading_glob_like_resolve_fixed():
    # Mirrors resolve_fixed/resolve_physics "*/" stripping so all three share one matching rule.
    sf = SceneFileConfig(xml_path="a.xml", include_patterns=["*/free_box"])
    assert sf.should_include("free_box") is True
    assert sf.should_include("static_post") is False


def test_object_configs_does_not_filter():
    # object_configs is OVERRIDE-ONLY: a body matching no object_configs pattern still loads
    # (should_include is the only filter). This pins the Fix-2 documented contract.
    sf = SceneFileConfig(xml_path="a.xml", object_configs={"only_this": ObjectPatternConfig(fixed=True)})
    assert sf.should_include("unmatched_body") is True


# ----- velocity requires a free body (RigidObjectConfig.validate_velocity_requires_free) -----
# A static (fixed=True) object is welded to the world and cannot move, so an initial velocity on
# it can never take effect — the model_validator rejects it at construction rather than letting it
# silently no-op at spawn.


def test_fixed_object_with_linear_velocity_rejected():
    with pytest.raises(ValueError, match="(?i)fixed=True.*velocity|velocity.*fixed"):
        RigidObjectConfig(urdf_file="b.urdf", fixed=True, linear_velocity=[1.0, 0.0, 0.0])


def test_fixed_object_with_angular_velocity_rejected():
    with pytest.raises(ValueError, match="(?i)fixed=True.*velocity|velocity.*fixed"):
        RigidObjectConfig(urdf_file="b.urdf", fixed=True, angular_velocity=[0.0, 0.0, 1.0])


def test_fixed_object_with_zero_velocity_ok():
    # The validator only rejects NON-zero velocity on a fixed body; a fixed body with the default
    # zero velocities (or explicit zeros) is fine.
    cfg = RigidObjectConfig(urdf_file="b.urdf", fixed=True, linear_velocity=[0.0, 0.0, 0.0])
    assert cfg.fixed is True


def test_free_object_with_velocity_ok():
    # A free (fixed=False) body may carry an initial velocity — that is exactly what it's for.
    cfg = RigidObjectConfig(urdf_file="b.urdf", linear_velocity=[1.0, 0.0, 0.0])
    assert cfg.linear_velocity == [1.0, 0.0, 0.0]
