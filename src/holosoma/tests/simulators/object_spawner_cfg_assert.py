"""Headless IsaacSim spawn-cfg assertion harness (boots the app, no full sim build).

``object_spawner`` imports ``isaaclab.sim`` at module load, which needs ``carb`` — only present
once a ``SimulationApp`` is running. So, like the other IsaacSim harnesses, this boots a minimal
headless app FIRST, then imports the spawner and asserts the cfg-layer invariants it is
responsible for. It does NOT build a SimulationContext / full env, so it's far cheaper than the
scene-spawn matrix (just app boot + dataclass construction).

Invariants pinned here:
  - URDF and USD honor the SAME PhysicsConfig (rigid/mass/collision props + contact sensors), so
    the tri-format selector picking one format can't change the body's mass or collision behavior.
  - `fixed` folds into kinematic_enabled on BOTH formats (never a welded URDF base).
  - Friction/restitution (`physics.isaacsim`) is carried identically on BOTH formats as a
    ``physics_material`` on the spawn cfg (URDF reaches USD via the same converter).

Writes ``OK`` to ``--result-file`` after all checks pass (IsaacSim teardown can mask the exit
code, so the result-file is the authoritative signal — same convention as dr_matrix_assert.py).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# tests/simulators/ has an ``isaacsim/`` subpackage that would shadow the real IsaacSim
# ``isaacsim`` package when run as a script; drop sys.path[0] (mirrors the other harnesses).
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

_SMALL_BOX_USD = "holosoma/data/scene_objects/boxes/small_box.usda"
_MULTIBODY_USD = "holosoma/data/scene_objects/multibody/multibody.usda"
_UNAUTHORED_USD = "holosoma/data/scene_objects/multibody/unauthored_table.usda"


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _check_format_symmetry(select_spawn_cfg, sim_utils) -> None:
    from holosoma.config_types.scene import PhysicsConfig, PhysXPhysicsConfig

    phys = PhysicsConfig(
        mass=3.5,
        physx=PhysXPhysicsConfig(linear_damping=0.25, angular_damping=0.3, max_linear_velocity=7.0),
    )
    for fmt in ("urdf", "usd"):
        spawn = select_spawn_cfg(fmt, "/tmp/whatever.asset", fixed=False, physics=phys, source_prim_path=None)

        # Mass override lands on BOTH formats (the URDF mass-drop fix).
        assert spawn.mass_props is not None, f"{fmt}: mass override dropped"
        assert _approx(spawn.mass_props.mass, 3.5), f"{fmt}: mass {spawn.mass_props.mass} != 3.5"

        # Rigid-body props carry the same damping/limits regardless of format.
        assert spawn.rigid_props is not None, f"{fmt}: rigid_props missing"
        assert _approx(spawn.rigid_props.linear_damping, 0.25), f"{fmt}: linear_damping wrong"
        assert _approx(spawn.rigid_props.angular_damping, 0.3), f"{fmt}: angular_damping wrong"
        assert _approx(spawn.rigid_props.max_linear_velocity, 7.0), f"{fmt}: max_linear_velocity wrong"
        assert spawn.rigid_props.kinematic_enabled is False, f"{fmt}: free body must not be kinematic"

        # Collision props + contact sensors present on both (symmetric contact behavior).
        assert spawn.collision_props is not None, f"{fmt}: collision_props missing"
        assert spawn.activate_contact_sensors is True, f"{fmt}: contact sensors not activated"

    # No mass/density override => mass_props None on both (asset's authored mass kept, identically).
    for fmt in ("urdf", "usd"):
        s = select_spawn_cfg(fmt, "/tmp/x.asset", fixed=False, physics=PhysicsConfig(), source_prim_path=None)
        assert s.mass_props is None, f"{fmt}: mass_props should be None without override"

    # `fixed` folds into kinematic on both; URDF static stays kinematic, NOT a welded base.
    for fmt in ("urdf", "usd"):
        s = select_spawn_cfg(fmt, "/tmp/x.asset", fixed=True, physics=None, source_prim_path=None)
        assert s.rigid_props.kinematic_enabled is True, f"{fmt}: fixed must set kinematic_enabled"
    urdf_static = select_spawn_cfg("urdf", "/tmp/x.urdf", fixed=True, physics=None, source_prim_path=None)
    assert urdf_static.fix_base is False, "URDF static MUST stay kinematic, not fix_base"
    assert isinstance(urdf_static, sim_utils.UrdfFileCfg)
    assert urdf_static.replace_cylinders_with_capsules is True, "URDF converter knob dropped"

    # USD 1->N selects a sub-prim via source_path.
    usd_sub = select_spawn_cfg("usd", "/tmp/x.usda", fixed=False, physics=None, source_prim_path="/Scene/Body")
    assert usd_sub.source_path == "/Scene/Body", "USD source_path not threaded"
    print("OK: URDF/USD spawn-cfg symmetry (mass+collision+contact+kinematic) verified")


_TINY_URDF = """<?xml version="1.0"?>
<robot name="probe">
  <link name="base">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
</robot>
"""


def _check_friction_symmetry(select_spawn_cfg) -> None:
    """Friction/restitution (physics.isaacsim) reaches BOTH formats as a spawn physics_material.

    Both USD and URDF carry the configured friction/restitution as a ``physics_material`` on the
    spawn cfg, which the custom USD spawner binds to the body's collider at spawn — URDF reaches
    USD via the same converter, so the two formats are physically identical.
    """
    import tempfile

    from holosoma.config_types.scene import IsaacSimPhysicsConfig, PhysicsConfig

    fr = IsaacSimPhysicsConfig(static_friction=0.31, dynamic_friction=0.27, restitution=0.42)
    phys = PhysicsConfig(isaacsim=fr)

    tmp = Path(tempfile.mkdtemp(prefix="holosoma_urdf_fric_"))
    urdf_file = tmp / "probe.urdf"
    urdf_file.write_text(_TINY_URDF)

    for fmt, path in (("usd", "/tmp/x.usda"), ("urdf", str(urdf_file))):
        spawn = select_spawn_cfg(fmt, path, fixed=False, physics=phys, source_prim_path=None)
        # Both formats end up as a CustomUsdFileCfg carrying the friction as a physics_material.
        assert spawn.__class__.__name__ == "CustomUsdFileCfg", (
            f"{fmt}+friction must carry a physics_material on a CustomUsdFileCfg, got {type(spawn).__name__}"
        )
        mat = spawn.physics_material
        assert mat is not None, f"{fmt}+friction produced no physics_material (friction dropped!)"
        assert _approx(mat.static_friction, 0.31), f"{fmt}: static_friction {mat.static_friction} != 0.31"
        assert _approx(mat.dynamic_friction, 0.27), f"{fmt}: dynamic_friction {mat.dynamic_friction} != 0.27"
        assert _approx(mat.restitution, 0.42), f"{fmt}: restitution {mat.restitution} != 0.42"

    # No isaacsim override => no physics_material attached (asset keeps its authored material).
    plain = select_spawn_cfg("usd", "/tmp/x.usda", fixed=False, physics=PhysicsConfig(), source_prim_path=None)
    assert plain.physics_material is None, "physics_material should be None without an isaacsim override"
    print("OK: URDF/USD friction symmetry — physics_material carried identically on both formats")


def _check_robot_link_physics_cfg() -> None:
    """Robot link_physics to spawn-cfg mapping and its gating, via the same converters objects use.

    The robot is an articulation (not a RigidObject), so it builds its spawn cfg from the converter
    functions directly (converters.physics_to_rigid_body_props / _collision_props / _mass_props) plus
    object_spawner.physics_material_cfg, used by isaacsim._setup_scene / _bind_robot_link_material.
    Checks at the cfg layer (no full sim):
      - physx to RigidBodyPropertiesCfg (damping/velocity caps), and the robot's fixed=False contract;
      - isaacsim collision offsets to CollisionPropertiesCfg (contact/rest offset, torsional patch);
      - gating: collision_props is built only when the isaacsim sub-config is present (else None, so no
        empty PhysxCollisionAPI stamp on every link); mass_props is built only when density (or mass) is set;
      - friction/restitution and both combine modes to physics_material_cfg (the bound-material channel).
    """
    from holosoma.config_types.scene import IsaacSimPhysicsConfig, PhysicsConfig, PhysXPhysicsConfig
    from holosoma.simulator.isaacsim.converters import (
        physics_to_collision_props,
        physics_to_mass_props,
        physics_to_rigid_body_props,
    )
    from holosoma.simulator.isaacsim.object_spawner import physics_material_cfg

    # A link_physics exercising every robot channel: physx, collision offsets, material with combines.
    lp = PhysicsConfig(
        density=250.0,
        physx=PhysXPhysicsConfig(linear_damping=0.4, angular_damping=0.5, max_linear_velocity=12.0),
        isaacsim=IsaacSimPhysicsConfig(
            static_friction=0.6,
            dynamic_friction=0.55,
            restitution=0.2,
            friction_combine_mode="max",
            restitution_combine_mode="min",
            contact_offset=0.005,
            rest_offset=0.0025,
            torsional_patch_radius=0.05,
        ),
    )

    # physx to rigid props (robot is a free-base articulation: fixed=False, not kinematic).
    rigid = physics_to_rigid_body_props(lp, fixed=False)
    assert _approx(rigid.linear_damping, 0.4) and _approx(rigid.angular_damping, 0.5), "robot physx damping dropped"
    assert _approx(rigid.max_linear_velocity, 12.0), "robot physx max_linear_velocity dropped"
    assert rigid.kinematic_enabled is False, "robot rigid props must be free-base (kinematic_enabled False)"

    # isaacsim offsets to collision props (contact/rest offset, torsional patch).
    coll = physics_to_collision_props(lp)
    assert _approx(coll.contact_offset, 0.005), f"robot contact_offset {coll.contact_offset} != 0.005"
    assert _approx(coll.rest_offset, 0.0025), f"robot rest_offset {coll.rest_offset} != 0.0025"
    assert _approx(coll.torsional_patch_radius, 0.05), "robot torsional_patch_radius dropped"

    # density to mass props (mass stays None; the validator forbids mass on link_physics).
    mass = physics_to_mass_props(lp)
    assert mass is not None and _approx(mass.density, 250.0), "robot link_physics density dropped"
    assert mass.mass is None, "robot mass_props must not carry mass (validator-forbidden on link_physics)"

    # friction/restitution and both combine modes to material (the channel a per-shape value write can't carry).
    mat = physics_material_cfg(lp)
    assert mat is not None and _approx(mat.static_friction, 0.6) and _approx(mat.dynamic_friction, 0.55)
    assert _approx(mat.restitution, 0.2), "robot material restitution dropped"
    assert mat.friction_combine_mode == "max", "robot friction_combine_mode dropped"
    assert mat.restitution_combine_mode == "min", "robot restitution_combine_mode dropped"

    # Gating. isaacsim.py builds collision_props only when the isaacsim sub-config is present;
    # mass_props only when density/mass set; material only when isaacsim set. The predicates are
    # mirrored here so a flip is caught at the cfg layer.
    physx_only = PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=0.1))
    assert physics_to_mass_props(physx_only) is None, "physx-only must yield None mass_props (no density)"
    assert physics_material_cfg(physx_only) is None, "physx-only must yield None material (no isaacsim sub-config)"
    # The robot collision_props gate is `link_physics.isaacsim is not None` (isaacsim.py): physx-only
    # means isaacsim is None, yielding None, so no empty PhysxCollisionAPI is stamped on every link.
    assert physx_only.isaacsim is None, "physx-only sentinel: isaacsim sub-config must be None (gate input)"
    none_mass = physics_to_mass_props(PhysicsConfig())
    assert none_mass is None, "empty PhysicsConfig must yield None mass_props (byte-for-byte prior spawn)"
    print("OK: robot link_physics -> rigid/collision/mass/material cfg mapping + gating verified")


def _abs_asset(rel: str) -> str:
    import holosoma

    return str(Path(holosoma.__file__).resolve().parents[1] / rel)


def _check_scene_body_discovery() -> None:
    """The 1->N body-discovery rules (Fix 3) on real USD assets, via the live USD runtime.

    Pins all three decomposition branches of ``_discover_scene_body_prim_paths`` plus the
    distinguishing-name step, with a booted app so ``pxr`` is real (these can't run in the CPU
    env). The shipped ``multibody.usda`` covers the AUTHORED branch; ``unauthored_table.usda``
    (no RigidBodyAPI anywhere) covers both unauthored branches.
    """
    from pxr import Usd

    from holosoma.config_types.scene import SceneFileConfig
    from holosoma.simulator.isaacsim.object_spawner import (
        _discover_scene_body_prim_paths,
        resolve_asset_root,
    )
    from holosoma.simulator.isaacsim.prim_naming import distinguishing_names

    # AUTHORED: the file's enabled-RigidBodyAPI prims ARE the bodies (shipped multibody.usda).
    authored_path = _abs_asset(_MULTIBODY_USD)
    sf = SceneFileConfig(usd_path=authored_path)
    stage = Usd.Stage.Open(authored_path)
    authored = _discover_scene_body_prim_paths(stage, "scene", sf, authored_path)
    authored_names = sorted(distinguishing_names(authored).values())
    assert authored_names == ["free_box", "static_post"], f"authored discovery: {authored_names}"

    # UNAUTHORED + DEFAULT patterns: collapse to ONE body on the defaultPrim ('table'). The legs
    # are NOT split out — geometry is never inspected to guess a multi-body decomposition.
    un_path = _abs_asset(_UNAUTHORED_USD)
    sf_default = SceneFileConfig(usd_path=un_path)
    un_stage = Usd.Stage.Open(un_path)
    collapsed = _discover_scene_body_prim_paths(un_stage, "t", sf_default, un_path)
    collapsed_names = sorted(distinguishing_names(collapsed).values())
    assert collapsed_names == ["table"], f"unauthored default-collapse: {collapsed_names}"

    # UNAUTHORED + EXPLICIT patterns: opt into the multi-body split — the geometry-owning leg
    # Xforms become candidate bodies ('table' has only Xform children, so it's NOT a candidate).
    sf_explicit = SceneFileConfig(usd_path=un_path, include_patterns=["*/leg_*"])
    promoted = _discover_scene_body_prim_paths(un_stage, "t", sf_explicit, un_path)
    promoted_names = distinguishing_names(promoted)
    selected = sorted(n for p, n in promoted_names.items() if sf_explicit.should_include(n))
    assert selected == ["leg_a", "leg_b"], f"unauthored explicit-promote: {selected}"

    # Standalone (1->1) asset-root resolution references the enclosure (defaultPrim), not an
    # interior physics prim, so geometry and materials always compose. It does not inspect
    # RigidBodyAPI, so a multi-body USD resolves to its root (the spawn tail collapses to one
    # body); the single-body contract is enforced downstream, not by target selection.
    one = resolve_asset_root(_abs_asset(_SMALL_BOX_USD), "pillar")
    assert one == "/small_box", f"asset-root (defaultPrim): {one}"
    un_root = resolve_asset_root(un_path, "table_obj")
    assert un_root == "/table", f"unauthored asset-root (defaultPrim 'table'): {un_root}"
    multi_root = resolve_asset_root(authored_path, "ok")
    assert multi_root == "/multibody", f"multi-body asset-root (defaultPrim, not a raise): {multi_root}"
    print("OK: scene-body discovery (authored / collapse / promote) + asset-root resolution verified")


def main() -> int:
    parser = argparse.ArgumentParser(description="IsaacSim spawn-cfg assertion harness.")
    parser.add_argument("--result-file", default=None, help="write 'OK' here after all checks pass")
    args = parser.parse_args()

    # Boot a minimal headless IsaacSim app so isaaclab.sim (carb) is importable.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    try:
        import isaaclab.sim as sim_utils

        from holosoma.simulator.isaacsim.object_spawner import select_spawn_cfg

        _check_format_symmetry(select_spawn_cfg, sim_utils)
        _check_friction_symmetry(select_spawn_cfg)
        _check_scene_body_discovery()
        _check_robot_link_physics_cfg()
    except BaseException:
        print("SPAWNER CFG ASSERT FAILED:\n" + traceback.format_exc(), flush=True)
        simulation_app.close()
        return 1

    print("SPAWNER CFG OK: all spawn-cfg invariants verified", flush=True)
    if args.result_file:
        with open(args.result_file, "w") as f:
            f.write("OK\n")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
