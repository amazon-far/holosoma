"""Headless cross-backend assertion harness for scene-asset spawning.

Builds a sim for a chosen backend + scene preset WITHOUT the infinite run_sim loop,
steps it a few times, and asserts the configured objects spawned and behave correctly
(free bodies fall under gravity; get_actor_states returns the right shape). Exits 0 on
success, non-zero with a message on failure — so it works as a live integration test
under each backend's launcher (MuJoCo venv, isaacgym setup, isaacsim DISPLAY/EULA).

Usage:
  python scripts/scene_spawn_assert.py --simulator mujoco   --scene g1-largebox
  python scripts/scene_spawn_assert.py --simulator isaacgym --scene g1-largebox --num-envs 4
  python scripts/scene_spawn_assert.py --simulator isaacsim --scene g1-largebox

Asserts, in EVERY env (--num-envs spreads each env to a distinct origin), for the preset:
  - get_actor_states([name]) returns shape [num_envs, 13]
  - each free object spawns at its own env_origin (per-env placement, not all stacked)
  - after stepping, every env's free object z decreased (it fell); static objects held
  - configured initial velocity is live in every env and integrates into motion
  - for 1->N scene files, the authored body-to-body offset holds in every env
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

# This file lives in tests/simulators/, which contains an ``isaacsim/`` subpackage.
# Run as a script, that dir lands on sys.path[0] and shadows the real IsaacSim
# ``isaacsim`` package (breaking ``from isaacsim import SimulationApp``). Drop it; the
# package is imported via its installed location / PYTHONPATH=src/holosoma instead.
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators import _scene_presets
from tests.simulators._sim_harness import build_run_sim_config


def _build_run_sim_config(simulator: str, scene: str, robot: str, terrain: str, record_dir: str | None) -> RunSimConfig:
    """Bridge/gantry-disabled RunSimConfig with optional (default-camera) video — see _sim_harness."""
    return build_run_sim_config(simulator, scene, robot, terrain, record_dir=record_dir)


def _isaacgym_object_mass(sim, name, env_id):
    """Total rigid-body mass of object ``name`` in env ``env_id`` (IsaacGym)."""
    env_ptr = sim.envs[env_id]
    actor = sim.object_handles[name][env_id]
    return sum(p.mass for p in sim.gym.get_actor_rigid_body_properties(env_ptr, actor))


def _object_total_mass(sim, name, env_id):
    """Total mass of object ``name`` in env ``env_id``, read from the live backend.

    Reads each backend's native property so a configured ``physics.mass`` override can be
    verified end-to-end at spawn. Returns ``None`` on a backend without a mass read here.
    """
    from holosoma.utils.simulator_config import SimulatorType

    sim_type = sim.get_simulator_type()
    if sim_type == SimulatorType.ISAACGYM:
        return _isaacgym_object_mass(sim, name, env_id)
    if sim_type == SimulatorType.ISAACSIM:
        # IsaacLab RigidObject: per-body masses from the PhysX view, shape [num_envs, num_bodies].
        masses = sim.scene.rigid_objects[name].root_physx_view.get_masses()
        return float(masses[env_id].sum())
    if sim_type == SimulatorType.MUJOCO:
        import mujoco

        model = sim.backend.model
        root = sim.scene_manager.rigid_object_root_bodies.get(name)
        if root is None:
            return None
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
        # Sum the root body and its descendant subtree (single-body objects -> just the root).
        total = 0.0
        for bid in range(model.nbody):
            anc = bid
            while anc not in (0, root_id):
                anc = int(model.body_parentid[anc])
            if anc == root_id:
                total += float(model.body_mass[bid])
        return total
    return None


def _object_sliding_friction(sim, name, env_id):
    """Sliding/static friction of object ``name`` in env ``env_id``, read from the live backend.

    Each backend stores friction in its own form, all reduced to the single sliding/static
    coefficient: IsaacGym shape ``friction``, IsaacSim material ``static_friction`` (column 0),
    MuJoCo geom ``friction[0]``. Returns ``None`` on a backend without a read here. Reads the
    first shape/geom of the object (the preset's box is a single uniform-friction geom).
    """
    from holosoma.utils.simulator_config import SimulatorType

    sim_type = sim.get_simulator_type()
    if sim_type == SimulatorType.ISAACGYM:
        env_ptr = sim.envs[env_id]
        actor = sim.object_handles[name][env_id]
        props = sim.gym.get_actor_rigid_shape_properties(env_ptr, actor)
        return float(props[0].friction) if props else None
    if sim_type == SimulatorType.ISAACSIM:
        # Friction is a physics material bound to the object's collider prim at spawn. Read the
        # bound material's static friction from the live composed stage. The behavioral
        # friction-slide preset is the independent confirmation that PhysX integrates it.
        import omni.usd
        from pxr import UsdPhysics, UsdShade

        stage = omni.usd.get_context().get_stage()
        prefix = f"/World/envs/env_{env_id}/{name}"
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(prefix) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel(materialPurpose="physics")
            for mp in [str(t) for t in rel.GetTargets()] if rel else []:
                mat = stage.GetPrimAtPath(mp)
                if mat and mat.HasAPI(UsdPhysics.MaterialAPI):
                    return float(UsdPhysics.MaterialAPI(mat).GetStaticFrictionAttr().Get())
        return None
    if sim_type == SimulatorType.MUJOCO:
        import mujoco

        model = sim.backend.model
        root = sim.scene_manager.rigid_object_root_bodies.get(name)
        if root is None:
            return None
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) == root_id:
                return float(model.geom_friction[g][0])  # axis 0 = sliding
        return None
    return None


def _run_object_dr_checks(sim, free_names, env_ids) -> bool:
    """Run object mass+material DR on the live backend and verify it took effect.

    Asserts (a) mass increased by the configured additive band on every (object, env), (b) the
    object_names subset narrows targeting (only the named body changes). Currently verifies
    read-back on IsaacGym (native rigid-body-property API); other backends just exercise the
    dispatch without crashing (their DR effect is verified in the per-backend DR unit tests).
    """
    import types

    from holosoma.managers.randomization.terms.objects import (
        randomize_object_rigid_body_mass_startup,
        randomize_object_rigid_body_material_startup,
    )
    from holosoma.utils.simulator_config import SimulatorType
    from tests.simulators._dr_matrix import _sampler

    env_shell = types.SimpleNamespace(simulator=sim, num_envs=sim.num_envs, device=sim.sim_device)
    sampler = _sampler(env_shell)  # shell has no dr_base_seed -> _sampler falls back to fixed seed 0
    is_isaacgym = sim.get_simulator_type() == SimulatorType.ISAACGYM

    mass_lo, mass_hi = 5.0, 6.0
    fric_lo, fric_hi = 0.2, 0.3
    if is_isaacgym:
        before = {(nm, e): _isaacgym_object_mass(sim, nm, e) for nm in free_names for e in range(sim.num_envs)}
        # Snapshot sliding friction per (object, env) BEFORE the material DR so we can assert it
        # actually moved (a no-op write must fail), not merely that it lands in band.
        fric_before = {(nm, e): _object_sliding_friction(sim, nm, e) for nm in free_names for e in range(sim.num_envs)}

    randomize_object_rigid_body_mass_startup(
        env_shell, env_ids, sampler=sampler, mass_distribution_params=[mass_lo, mass_hi]
    )
    randomize_object_rigid_body_material_startup(
        env_shell,
        env_ids,
        sampler=sampler,
        material={
            "isaacgym": {"friction": [fric_lo, fric_hi]},
            "isaacsim": {"static_friction": [fric_lo, fric_hi], "dynamic_friction": [fric_lo, fric_hi]},
            "mujoco": {"sliding_friction": [fric_lo, fric_hi]},
        },
    )

    if is_isaacgym:
        nbody = {
            nm: len(sim.gym.get_actor_rigid_body_properties(sim.envs[0], sim.object_handles[nm][0]))
            for nm in free_names
        }
        for nm in free_names:
            for e in range(sim.num_envs):
                delta = _isaacgym_object_mass(sim, nm, e) - before[(nm, e)]
                # Additive offset applied to EACH body of the object => band scales by body count.
                lo, hi = mass_lo * nbody[nm], mass_hi * nbody[nm]
                if not (lo - 1e-3 <= delta <= hi + 1e-3):
                    print(f"FAIL: object-DR mass delta {delta:.3f} for {nm} env{e} outside [{lo},{hi}]")
                    return False
        print(f"OBJECT-DR mass band [{mass_lo},{mass_hi}]/body applied to {free_names} across {sim.num_envs} envs OK")

        # Friction read-back: the material DR set sliding friction in [fric_lo, fric_hi] on every
        # named shape. Mirror the mass check — read the LIVE shape friction back per (object, env)
        # and assert it (a) CHANGED from the pre-DR baseline and (b) lands in band. The DR samples
        # one friction value per env (from a bucket) applied to all named objects in that env, so
        # this also confirms it reached each object. Reads the same shape-property field the DR
        # wrote (and that the per-object physics check reads), so it can never be a config echo.
        for nm in free_names:
            for e in range(sim.num_envs):
                got = _object_sliding_friction(sim, nm, e)
                if got is None:
                    print(f"FAIL: object-DR friction read-back returned None for {nm} env{e} on IsaacGym")
                    return False
                base = fric_before[(nm, e)]
                if base is not None and abs(got - base) <= 1e-6:
                    print(f"FAIL: object-DR friction for {nm} env{e} did not change from baseline {base:.4f}")
                    return False
                if not (fric_lo - 1e-3 <= got <= fric_hi + 1e-3):
                    print(f"FAIL: object-DR friction {got:.4f} for {nm} env{e} outside [{fric_lo},{fric_hi}]")
                    return False
        print(f"OBJECT-DR friction band [{fric_lo},{fric_hi}] applied to {free_names} across {sim.num_envs} envs OK")

        # object_names subset: randomize only the first free body further; others must not change.
        if len(free_names) >= 2:
            target, other = free_names[0], free_names[1]
            mass_other = {e: _isaacgym_object_mass(sim, other, e) for e in range(sim.num_envs)}
            randomize_object_rigid_body_mass_startup(
                env_shell, env_ids, sampler=sampler, mass_distribution_params=[mass_lo, mass_hi], object_names=[target]
            )
            for e in range(sim.num_envs):
                if abs(_isaacgym_object_mass(sim, other, e) - mass_other[e]) > 1e-4:
                    print(f"FAIL: object-DR subset leaked onto '{other}' env{e}")
                    return False
            print(f"OBJECT-DR object_names subset isolation OK ('{target}' only, '{other}' untouched)")
        return True

    # Non-IsaacGym backends have no live read-back here yet. The DR terms dispatched above, but we
    # have NOT verified the effect landed — returning True now would be a vacuous green that lets a
    # future `--object-dr` wiring of IsaacSim/Warp pass without ever checking the result. Fail loud
    # instead so read-back must be implemented before this branch can report success. (Do not
    # implement IsaacSim/Warp read-back here — just prevent the silent pass.)
    raise NotImplementedError(
        f"--object-dr read-back is not implemented for {sim.get_simulator_type()}; "
        f"mass+material DR dispatched on {free_names} but the effect was not verified. "
        "Implement a live read-back for this backend before wiring it through --object-dr."
    )


def main() -> int:
    # Register the test-only scene presets into scene.DEFAULTS so `scene:<testkey>` resolves through
    # the normal tyro path inside this (sub)process. Core never imports these; tests register them.
    _scene_presets.register()

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--scene", default="g1-largebox")
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--steps", type=int, default=40)
    # >1 spreads each env to a distinct origin and checks per-env placement (the offset/fall/
    # hold/velocity asserts iterate over every env, not just env 0). MuJoCo ClassicBackend
    # rejects >1 (use mjwarp); IsaacGym/IsaacSim/Warp support it.
    parser.add_argument("--num-envs", type=int, default=1)
    # The real run_sim loop calls sim.render() every step regardless of headless, so the
    # harness does too — this is the only path that exercises render() + setup_viewer().
    # --headless false opens a real window (needs a display) and drives the headful render;
    # --headless true drives the headless render (viewer is None — must not crash).
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    # Run the object physics-DR terms (mass + friction) after prepare_sim and assert they took
    # effect on the live backend (read mass/friction back per object/env). Works on any
    # DR-supporting backend (IsaacGym / IsaacSim / Warp).
    parser.add_argument("--object-dr", action="store_true")
    # Save a movie of the run to this directory (records every episode, headless or headful).
    # Uses the backend's own video recorder, so it works on every backend.
    parser.add_argument("--record", default=None, metavar="DIR", help="save a video of the run to DIR")
    args = parser.parse_args()
    headless = args.headless == "true"

    config = _build_run_sim_config(args.simulator, args.scene, args.robot, args.terrain, args.record)
    device = "cuda:0" if args.simulator != "mujoco" else "cpu"
    # num_envs must be set on the training config too: backends size their per-env state
    # tensors (Warp qpos/qvel, IsaacGym/IsaacSim buffers) from training_config.num_envs at
    # setup()/load_assets(), before create_envs(n,...) runs. Passing n only to create_envs
    # leaves the tensors sized for 1 env and indexing env 1..n-1 is out of bounds.
    config = dataclasses.replace(
        config,
        device=device,
        training=dataclasses.replace(config.training, num_envs=args.num_envs),
    )

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(headless)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    # Distinct per-env origins on a line in x (spacing 5m), so per-env placement is
    # observable: each env's objects must sit at their own origin, not all stacked at one
    # world point. num_envs=1 keeps the origin at zero (the original single-env behavior).
    n = args.num_envs
    env_origins = torch.zeros(n, 3, device=device)
    if n > 1:
        env_origins[:, 0] = torch.arange(n, device=device, dtype=torch.float32) * 5.0
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=device,
        dtype=torch.float32,
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()

    # Mirror DirectSimulation: open the interactive viewer only when headful. When headless
    # this is skipped, so render() below must tolerate a None viewer.
    if not headless:
        sim.setup_viewer()

    # Start recording if --record was given. Each backend captures a frame inside its own
    # simulate_at_each_physics_step (gated on is_recording), so the existing step loop fills the
    # movie; on_episode_end (after the loop) encodes and writes the file.
    if sim.video_recorder is not None:
        sim.video_recorder.setup_recording()
        sim.video_recorder.on_episode_start(env_id=sim.video_recorder.config.record_env_id)

    free_names = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    static_names = sim.object_registry.get_names_by_type(ObjectType.SCENE)
    env_ids = torch.arange(sim.num_envs, device=sim.sim_device)

    mode = "headless" if headless else "headful"
    print(f"[{args.simulator}/{mode}] free(INDIVIDUAL)={free_names} static(SCENE)={static_names}")
    if not free_names and not static_names:
        print("FAIL: no scene assets registered")
        return 1

    # Object physics DR (opt-in): run mass + material DR on the free bodies and verify the
    # effect landed on the live backend.
    if args.object_dr:
        if not _run_object_dr_checks(sim, free_names, env_ids):
            return 1

    # All snapshots keep the full env dimension; per-env checks below index every row.
    z0 = {nm: sim.get_actor_states([nm], env_ids)[:, 2].clone() for nm in free_names + static_names}
    p0 = {nm: sim.get_actor_states([nm], env_ids)[:, :3].clone() for nm in free_names}
    q0 = {nm: sim.get_actor_states([nm], env_ids)[:, 3:7].clone() for nm in free_names}
    for nm in free_names + static_names:
        st = sim.get_actor_states([nm], env_ids)
        assert st.shape == (sim.num_envs, 13), f"{nm}: get_actor_states shape {st.shape} != [{sim.num_envs},13]"

    # Per-env origin placement: each free object must spawn at its own env_origin (object
    # world pos = env_origin + the object's configured local position), so the objects are
    # NOT all stacked at one world point. Checked on x/y (the spread axes); z is left to the
    # fall check. At num_envs=1 origins are zero and this reduces to the original placement.
    free_cfg = {name: o for name, o in config.scene.rigid_objects.items() if not o.fixed}
    if n > 1:
        for nm in free_names:
            obj = free_cfg.get(nm)
            if obj is None:  # scene-file body (no standalone RigidObjectConfig) — skip here
                continue
            pos = sim.get_actor_states([nm], env_ids)[:, :3]
            for axis in (0, 1):
                expected = env_origins[:, axis] + obj.position[axis]
                ok_axis = torch.allclose(pos[:, axis], expected, atol=1e-2)
                spread = [round(float(v), 3) for v in pos[:, axis].tolist()]
                exp_round = [round(float(v), 3) for v in expected.tolist()]
                print(f"  origin {nm}: axis{axis} {spread} (expected {exp_round})  ok={ok_axis}")
                if not ok_axis:
                    print(f"FAIL: {nm} not placed at per-env origin on axis{axis}: {spread} != {expected.tolist()}")
                    return 1

    # Configured initial velocities (world frame) per standalone free object. The preset
    # uses an identity initial orientation, so world == body frame and the read-back here
    # matches the config directly on every backend (MuJoCo returns body-local velocity).
    vel_cfg = {
        name: (list(o.linear_velocity), list(o.angular_velocity))
        for name, o in config.scene.rigid_objects.items()
        if not o.fixed
    }
    moving = {nm: v for nm, v in vel_cfg.items() if (any(v[0]) or any(v[1])) and nm in free_names}
    for nm, (lin, ang) in moving.items():
        st = sim.get_actor_states([nm], env_ids)  # [num_envs, 13] — check every env
        lin_t = torch.tensor(lin, device=sim.sim_device).expand(sim.num_envs, 3)
        ang_t = torch.tensor(ang, device=sim.sim_device).expand(sim.num_envs, 3)
        lin_ok = torch.allclose(st[:, 7:10], lin_t, atol=5e-2)
        ang_ok = torch.allclose(st[:, 10:13], ang_t, atol=5e-2)
        print(
            f"  vel   {nm}: lin {st[0, 7:10].tolist()} (cfg {lin})  "
            f"ang {st[0, 10:13].tolist()} (cfg {ang})  ok={lin_ok and ang_ok}"
        )
        if not (lin_ok and ang_ok):
            print(
                f"FAIL: {nm} initial velocity read-back "
                f"lin={st[:, 7:10].tolist()}/{lin} ang={st[:, 10:13].tolist()}/{ang}"
            )
            return 1

    # Multi-body 1->N: the file's two bodies must keep their authored relative offset,
    # identically across backends AND in every env. The multibody asset authors static_post
    # at +0.5m x from free_box; with the file at an axis-aligned world pose the body-to-body
    # translation must be [0.5, 0, 0] in each env (cross-backend + per-env consistency). The
    # relative offset must survive per-env origin translation: a bug that offsets one body by
    # env_origin but not the other would break this in envs 1..N while passing in env 0.
    if "scene_free_box" in (free_names + static_names) and "scene_static_post" in (free_names + static_names):
        p_free = sim.get_actor_states(["scene_free_box"], env_ids)[:, :3]  # [num_envs, 3]
        p_static = sim.get_actor_states(["scene_static_post"], env_ids)[:, :3]
        offset = p_static - p_free  # [num_envs, 3]
        expected = torch.tensor([0.5, 0.0, 0.0], device=sim.sim_device).expand(sim.num_envs, 3)
        offset_ok = torch.allclose(offset, expected, atol=1e-2)
        offset_env0 = [round(float(v), 4) for v in offset[0].tolist()]
        print(f"OFFSET scene_static_post-scene_free_box env0={offset_env0} all_envs_ok={offset_ok}")
        if not offset_ok:
            print(f"FAIL: body-to-body offset per env {offset.tolist()} != expected [0.5,0,0]")
            return 1

    # Per-object physics override: a configured ``physics`` must reach the spawned body on every
    # backend. Read the live mass and sliding friction back and compare to the config (verifies
    # the physics override path end-to-end, free bodies only).
    from holosoma.utils.simulator_config import SimulatorType

    sim_type = sim.get_simulator_type()

    def _configured_sliding_friction(physics):
        """The sliding/static friction this backend should apply, from its own sub-config."""
        if sim_type == SimulatorType.ISAACGYM:
            return physics.isaacgym.friction if physics.isaacgym is not None else None
        if sim_type == SimulatorType.ISAACSIM:
            return physics.isaacsim.static_friction if physics.isaacsim is not None else None
        if sim_type == SimulatorType.MUJOCO:
            return physics.mujoco.friction[0] if (physics.mujoco is not None and physics.mujoco.friction) else None
        return None

    for nm in free_names:
        obj = free_cfg.get(nm)
        if obj is None or obj.physics is None:
            continue

        if obj.physics.mass is not None:
            for e in range(sim.num_envs):
                got = _object_total_mass(sim, nm, e)
                if got is None:
                    print(f"  mass  {nm}: read-back unsupported on {args.simulator}, skipping")
                    break
                if abs(got - obj.physics.mass) > 1e-3:
                    print(f"FAIL: {nm} env{e} mass {got:.4f} != configured {obj.physics.mass}")
                    return 1
            else:
                print(f"  mass  {nm}: {obj.physics.mass} applied across {sim.num_envs} env(s) OK")

        want_friction = _configured_sliding_friction(obj.physics)
        if want_friction is not None:
            for e in range(sim.num_envs):
                got = _object_sliding_friction(sim, nm, e)
                if got is None:
                    print(f"  fric  {nm}: read-back unsupported on {args.simulator}, skipping")
                    break
                if abs(got - want_friction) > 1e-3:
                    print(f"FAIL: {nm} env{e} sliding friction {got:.4f} != configured {want_friction}")
                    return 1
            else:
                print(f"  fric  {nm}: {want_friction} applied across {sim.num_envs} env(s) OK")

    ok = True
    # Track the MINIMUM z each free body reaches over the window. A free body under gravity
    # always dips below its spawn height; asserting the per-step minimum (rather than only the
    # final z) makes the fall check robust to a transient upward nudge from the un-actuated
    # robot collapsing onto a nearby object near the end of the window — a test-rig artifact
    # (no policy drives the robot here), not a spawn defect. The per-env origin spread above
    # is what this multi-env scenario actually validates.
    z_min = {nm: z0[nm].clone() for nm in free_names}
    for _ in range(args.steps):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
        # Drive the render path every step, exactly as the run_sim loop does. Headless: this
        # must be a no-op that does not touch a None viewer. Headful: it draws to the window.
        sim.render()
        for nm in free_names:
            z_min[nm] = torch.minimum(z_min[nm], sim.get_actor_states([nm], env_ids)[:, 2])

    for nm in free_names:
        z1 = sim.get_actor_states([nm], env_ids)[:, 2]  # [num_envs]
        fell = bool(torch.all(z_min[nm] < z0[nm] - 1e-3))  # dipped below spawn height in every env
        print(f"  free  {nm}: z {z0[nm][0]:.4f} -> {z1[0]:.4f} (min {z_min[nm].min():.4f})  fell_all_envs={fell}")
        ok = ok and fell

    # Integrated-motion check: a read-back can pass even if the sim ignored the value, so also
    # confirm the configured velocity actually moved the body, in every env. We assert the net
    # HORIZONTAL displacement MAGNITUDE (gravity excludes z), not the signed projection onto the
    # configured velocity direction: the freejoint velocity is body-local, so a strong angular
    # velocity (here 3 rad/s about z) rotates the linear-velocity direction over the window and
    # the body traces a near-circular arc whose net displacement no longer points along the
    # initial lin_h. A signed projection onto lin_h therefore shrinks toward (or past) zero even
    # though the body clearly moved. The magnitude is direction-agnostic: it can't be cancelled by
    # rotation, still fails loud if the velocity were dropped (a stationary body stays put), and
    # z is excluded so gravity/fall never contributes. (The velocity_box preset also spawns the
    # box clear of the robot so the falling un-actuated robot can't perturb this measurement.)
    for nm, (lin, ang) in moving.items():
        st1 = sim.get_actor_states([nm], env_ids)  # [num_envs, 13]
        lin_h = torch.tensor(lin[:2], device=sim.sim_device)  # configured horizontal velocity
        if torch.linalg.norm(lin_h) > 1e-6:
            disp_h = st1[:, :2] - p0[nm][:, :2]  # [num_envs, 2] net horizontal displacement
            dist_h = torch.linalg.norm(disp_h, dim=1)  # [num_envs] direction-agnostic distance
            moved = bool(torch.all(dist_h > 1e-2))
            print(
                f"  move  {nm}: horiz-dist {[round(float(d), 4) for d in dist_h.tolist()]} "
                f"(cfg lin_h {lin[:2]})  ok_all_envs={moved}"
            )
            ok = ok and moved
        if any(ang):
            st1 = sim.get_actor_states([nm], env_ids)  # [num_envs, 13]
            dq = (st1[:, 3:7] - q0[nm]).abs().amax(dim=1)  # [num_envs]
            spun = bool(torch.all(dq > 1e-2))
            print(f"  spin  {nm}: max|dquat|_env0 {float(dq[0]):.4f} (ang_vel {ang})  spun_all_envs={spun}")
            ok = ok and spun
    for nm in static_names:
        z1 = sim.get_actor_states([nm], env_ids)[:, 2]  # [num_envs]
        fixed = bool(torch.all((z1 - z0[nm]).abs() < 1e-3))
        print(f"  static {nm}: z {z0[nm][0]:.4f} -> {z1[0]:.4f}  fixed_all_envs={fixed}")
        ok = ok and fixed

    # Behavioral friction probe: two boxes pushed identically, differing ONLY in friction, must
    # slide different distances — the low-friction box farther. This proves the configured
    # friction governs what PhysX/MuJoCo integrate at contact, independent of which buffer a
    # read-back inspects. Net horizontal travel from spawn; checked in every env.
    if "box_lowfric" in free_names and "box_highfric" in free_names:
        lo = sim.get_actor_states(["box_lowfric"], env_ids)[:, :2] - p0["box_lowfric"][:, :2]
        hi = sim.get_actor_states(["box_highfric"], env_ids)[:, :2] - p0["box_highfric"][:, :2]
        lo_d = torch.linalg.norm(lo, dim=1)  # [num_envs] slide distance, low friction
        hi_d = torch.linalg.norm(hi, dim=1)  # [num_envs] slide distance, high friction
        # Require a clear separation (not just >) so numerical noise can't pass it.
        slid_farther = bool(torch.all(lo_d > hi_d + 0.05))
        print(
            f"  fric-slide: low={[round(float(v), 3) for v in lo_d.tolist()]} "
            f"high={[round(float(v), 3) for v in hi_d.tolist()]}  low>high_all_envs={slid_farther}"
        )
        if not slid_farther:
            print("FAIL: low-friction box did not slide measurably farther than high-friction box")
            ok = False

    print("PASS" if ok else "FAIL")
    code = 0 if ok else 1

    # Finalize the movie (encode + write) BEFORE any hard-exit below, which would skip it.
    if sim.video_recorder is not None:
        sim.video_recorder.on_episode_end(env_id=sim.video_recorder.config.record_env_id)
        print(f"VIDEO saved under {args.record}")

    # MuJoCo headful teardown race: the passive viewer runs a GLFW window on a background
    # thread, and at normal interpreter exit the main thread's glfw.terminate() atexit handler
    # races that still-live thread, segfaulting (exit 139) AFTER a correct PASS — masking the
    # real result. We have everything we need by now (result printed, code computed), so for
    # the MuJoCo backends with a viewer open, flush and hard-exit via os._exit to bypass the
    # GLFW atexit teardown. Other backends (and all headless runs) fall through to a normal
    # return so graceful shutdown (e.g. IsaacSim SimulationApp.close()) still happens.
    viewer = getattr(sim, "viewer", None)
    if viewer is not None and args.simulator in ("mujoco", "mjwarp"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    return code


if __name__ == "__main__":
    sys.exit(main())
