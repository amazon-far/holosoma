"""End-to-end example: the object-aware managers on a scene of free + static + scene-file bodies.

This is a runnable, self-documenting tour of the scene-asset + object-manager features. It
builds a REAL, fully-managed locomotion environment (the same production path training uses —
``training_context`` + ``get_class(env_class)(...)``, NO shims) on a scene that mixes every
asset kind, then drives each manager-layer feature in turn and narrates what happened. Reading
the printed output (or this source) shows how the pieces compose in a real setup. It is
backend-switchable: sections 1-4 are identical code on all four backends (only ``--backend``
changes); section 5 (physics DR) dispatches per backend and reads the result back on MuJoCo.

What it demonstrates, in order:

  1. SCENE SPAWN — a declarative ``SceneConfig`` (sibling of robot/terrain) spawns standalone
     rigid bodies (a free box, a free box with a configured initial velocity, a static pillar)
     AND a 1->N scene FILE that expands into a free + a static body. Every body is addressed by
     its ObjectRegistry name through the unified actor API ``get_actor_states(names, env_ids)``.

  2. OBJECT OBSERVATIONS — the task-independent base-frame observation terms
     (``object_pos_b`` / ``object_quat_b`` / ``object_lin_vel_b`` / ``object_ang_vel_b``) are
     wired into the env's REAL ``ObservationManager`` as an ``object_obs`` group and computed via
     ``env.observation_manager.compute()``. The group's dimension (k * N for N free bodies) is
     MEASURED by the manager, not declared — so the same term config works for any object count.

  3. EPISODE RESET — ``env.reset_envs_idx(env_ids)`` runs the real per-env reset chain;
     ``BaseTask._reset_objects_callback`` restores every free body to its configured initial pose
     AND initial velocity (so a body set to start moving re-starts moving). Static bodies untouched.

  4. POSE JITTER ON RESET — ``jitter_object_pose_on_reset`` is wired into the env's REAL
     ``RandomizationManager`` as a *reset* term, so step 3's reset already ran it: each free body's
     XY/yaw is perturbed per env (around the baseline reset pose) while its velocity is preserved.
     Omitting the term gives a clean deterministic reset; eval disables the manager, so eval is clean.

  5. PHYSICS DR — the object mass / material(friction) / inertia randomizers are wired as the
     RandomizationManager's *setup* terms, so the manager APPLIES them once at env build (the demo
     never calls a term directly — it sets config and reads the result back). Each term dispatches
     per backend: on MuJoCo (Warp GPU + Classic CPU) every property routes through the shared
     ``randomize_field`` chokepoint, while IsaacGym uses native ``gym.set_actor_rigid_*_properties``
     and IsaacSim uses isaaclab event helpers. Mass, friction, and inertia are ALL cross-backend.
     The demo reads mass + friction back from the live model on MuJoCo (verifying they landed in
     the configured bands); on IsaacGym/IsaacSim it reports which setup terms the manager ran (the
     per-backend physics effect is asserted in the dedicated DR tests).

Run it (MuJoCo CPU needs no GPU; the others need their SDK + a CUDA device):

    # MuJoCo, single env, CPU classic backend:
    python -m holosoma.examples.object_managers_demo --backend mujoco

    # MuJoCo Warp, 4 envs, GPU (per-env randomization is visible with >1 env):
    python -m holosoma.examples.object_managers_demo --backend mjwarp --num-envs 4

    # Isaac backends (need the respective SDK):
    python -m holosoma.examples.object_managers_demo --backend isaacgym --num-envs 4
    python -m holosoma.examples.object_managers_demo --backend isaacsim

The scene is the ``object-managers-demo`` preset (``config_values/scene.py``); it is also
selectable from the normal CLI, e.g. ``run_sim.py simulator:mujoco scene:object-managers-demo``.

Exits 0 on success, non-zero on failure, so it doubles as a smoke test (see
``tests/simulators/mujoco/test_object_managers_demo.py``).
"""

from __future__ import annotations

import argparse
import dataclasses

# name -> config_values.simulator preset attribute. MuJoCo classic is CPU + single-env only;
# mjwarp/isaacgym/isaacsim are GPU and support multiple envs.
_BACKENDS = {
    "mujoco": ("mujoco", "cpu"),
    "mjwarp": ("mjwarp", "cuda:0"),
    "isaacgym": ("isaacgym", "cuda:0"),
    "isaacsim": ("isaacsim", "cuda:0"),
}

# The four base-frame object observation terms, as a reusable group config. Each defaults to
# "all registered free bodies", so the group is object-count- and scene-agnostic; the manager
# measures the resulting k*N dimension at init.
_OBJ_OBS_TERMS = "holosoma.managers.observation.terms.objects"
# The pose-jitter randomization reset term.
_OBJ_RAND = "holosoma.managers.randomization.terms.objects"


def _experiment_config(backend: str, num_envs: int, headless: bool):
    """Build the ExperimentConfig for the demo: the g1_29dof locomotion preset, retargeted to the
    chosen backend + the object-managers demo scene, with the object managers wired in PURELY via
    config — the env then runs them through its real manager stack, exactly as training does:

    - object observation terms -> a new ``object_obs`` group on the ObservationManager (its OWN
      group, not actor/critic, so it doesn't perturb the policy's fixed obs dimension);
    - object physics-DR terms (mass / material / inertia) -> the RandomizationManager's
      ``setup_terms`` (the manager applies them once at env build, like any startup randomizer);
    - the pose-jitter term -> the RandomizationManager's ``reset_terms`` (runs on every reset).

    The demo never calls a randomization term directly — it sets config knobs and lets the
    managers drive, then reads the results back.
    """
    from holosoma.config_types.observation import ObsGroupCfg, ObsTermCfg
    from holosoma.config_types.randomization import RandomizationTermCfg
    from holosoma.config_values import experiment as experiment_values
    from holosoma.config_values import scene as scene_values
    from holosoma.config_values import simulator as simulator_values

    base = experiment_values.g1_29dof
    sim_cfg = getattr(simulator_values, _BACKENDS[backend][0])
    assert base.observation is not None, "g1_29dof preset must define an ObservationManager"
    assert base.randomization is not None, "g1_29dof preset must define a RandomizationManager"

    # Add an object_obs group alongside the preset's existing actor/critic groups.
    obs_groups = dict(base.observation.groups)
    obs_groups["object_obs"] = ObsGroupCfg(
        concatenate=True,
        enable_noise=False,
        history_length=1,
        terms={
            "object_pos_b": ObsTermCfg(func=f"{_OBJ_OBS_TERMS}:object_pos_b"),
            "object_quat_b": ObsTermCfg(func=f"{_OBJ_OBS_TERMS}:object_quat_b"),
            "object_lin_vel_b": ObsTermCfg(func=f"{_OBJ_OBS_TERMS}:object_lin_vel_b"),
            "object_ang_vel_b": ObsTermCfg(func=f"{_OBJ_OBS_TERMS}:object_ang_vel_b"),
        },
    )
    observation = dataclasses.replace(base.observation, groups=obs_groups)

    # Object physics DR as startup randomizers: the RandomizationManager applies these once at
    # env build (after expanding the required model fields). Exaggerated, far-from-default bands
    # so the effect is unambiguous when we read it back (default box mass is 0.1 kg << [5,6]).
    setup_terms = dict(base.randomization.setup_terms)
    setup_terms["randomize_object_mass"] = RandomizationTermCfg(
        func=f"{_OBJ_RAND}:randomize_object_rigid_body_mass_startup",
        params={"mass_distribution_params": [5.0, 6.0]},
    )
    setup_terms["randomize_object_material"] = RandomizationTermCfg(
        func=f"{_OBJ_RAND}:randomize_object_rigid_body_material_startup",
        params={
            "material": {
                "isaacgym": {"friction": [0.2, 0.3], "restitution": [0.0, 0.0]},
                "isaacsim": {
                    "static_friction": [0.2, 0.3],
                    "dynamic_friction": [0.2, 0.3],
                    "restitution": [0.0, 0.0],
                },
                "mujoco": {"sliding_friction": [0.2, 0.3]},
            }
        },
    )
    setup_terms["randomize_object_inertia"] = RandomizationTermCfg(
        func=f"{_OBJ_RAND}:randomize_object_rigid_body_inertia_startup",
        params={
            "inertia_distribution_params_dict": {
                "Ixx": [2.0, 3.0],
                "Iyy": [1.0, 1.0],
                "Izz": [1.0, 1.0],
                "Ixy": [1.0, 1.0],
                "Iyz": [1.0, 1.0],
                "Ixz": [1.0, 1.0],
            }
        },
    )

    # Pose-jitter as a reset randomizer (runs on every reset).
    reset_terms = dict(base.randomization.reset_terms)
    reset_terms["jitter_object_pose"] = RandomizationTermCfg(
        func=f"{_OBJ_RAND}:jitter_object_pose_on_reset",
        params={"xy_range": 0.1, "yaw_range": 0.3, "enabled": True},
    )
    # ignore_unsupported=True: a backend that can't do a given randomizer is skipped (and listed
    # in the manager's _failed_randomizers) rather than crashing the build. The object DR terms
    # are cross-backend, so nothing is skipped here today — this just keeps the demo robust.
    randomization = dataclasses.replace(
        base.randomization, setup_terms=setup_terms, reset_terms=reset_terms, ignore_unsupported=True
    )

    return dataclasses.replace(
        base,
        simulator=sim_cfg,
        scene=scene_values.object_managers_demo,
        observation=observation,
        randomization=randomization,
        training=dataclasses.replace(
            base.training, num_envs=num_envs, headless=headless, seed=0, torch_deterministic=False
        ),
    )


def _section(title: str) -> None:
    # flush=True so output survives even if a backend hard-terminates the process at teardown
    # (IsaacSim's app close can kill the interpreter before block-buffered stdout is written).
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def run_demo(backend: str, num_envs: int, headless: bool = True) -> int:
    """Run the full object-managers tour on ``backend`` via a real env. Returns an exit code.

    Builds the env inside ``training_context`` (which owns the IsaacSim app lifecycle — closing it
    only at block exit), so the whole demo runs while the simulator is alive.
    """
    # Imported lazily so the per-backend launcher (isaacgym/isaacsim) initializes before torch.
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.simulator.shared.object_registry import ObjectType
    from holosoma.train_agent import training_context
    from holosoma.utils.common import seeding
    from holosoma.utils.helpers import get_class
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.simulator_config import SimulatorType

    device = _BACKENDS[backend][1]
    _section(f"BUILD — backend={backend}  num_envs={num_envs}  device={device}")
    cfg = _experiment_config(backend, num_envs, headless)
    seeding(cfg.training.seed)

    with training_context(cfg):
        env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device=device)
        sim = env.simulator
        env_ids = torch.arange(env.num_envs, device=env.device)

        # ---- 1. SCENE SPAWN --------------------------------------------------------------
        _section("1. SCENE SPAWN — free + static standalone bodies AND a 1->N scene file")
        free_names = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
        static_names = sim.object_registry.get_names_by_type(ObjectType.SCENE)
        print(f"free  (INDIVIDUAL) bodies: {free_names}")
        print(f"static (SCENE)     bodies: {static_names}")
        if not free_names or not static_names:
            print("FAIL: expected both free and static bodies registered")
            return 1
        for nm in free_names + static_names:
            st = sim.get_actor_states([nm], env_ids)  # [num_envs, 13] = [pos3, quat_xyzw4, linvel3, angvel3]
            print(f"  {nm:18s} env0 pos={_fmt(st[0, :3])} quat={_fmt(st[0, 3:7])}")
            if st.shape != (env.num_envs, 13):
                print(f"FAIL: {nm} state shape {tuple(st.shape)} != ({env.num_envs}, 13)")
                return 1

        # ---- 2. OBJECT OBSERVATIONS ------------------------------------------------------
        _section("2. OBJECT OBSERVATIONS — base-frame state via the env's real ObservationManager")
        n_free = len(free_names)
        dims = env.observation_manager.get_obs_dims()  # measured by calling each term once
        print(f"observation groups: {sorted(dims)}")
        print(f"measured 'object_obs' dim for {n_free} free bodies: {dims['object_obs']}")
        print(f"  expected k*N = (3 pos + 4 quat + 3 linvel + 3 angvel) * {n_free} = {13 * n_free}")
        if dims["object_obs"] != 13 * n_free:
            print(f"FAIL: object obs dim {dims['object_obs']} != {13 * n_free}")
            return 1
        obs = env.observation_manager.compute()["object_obs"]
        print(f"  computed 'object_obs' tensor shape: {tuple(obs.shape)}  (finite={bool(torch.isfinite(obs).all())})")
        if obs.shape != (env.num_envs, 13 * n_free) or not torch.isfinite(obs).all():
            print("FAIL: object observation tensor malformed")
            return 1

        # ---- 3. EPISODE RESET (pose + velocity restore) + 4. POSE JITTER -----------------
        # reset_envs_idx runs the real per-env reset chain: baseline _reset_objects_callback
        # (restore pose + configured velocity) THEN the randomization manager's reset terms
        # (our jitter term perturbs the pose). We snapshot before/after one reset and check both.
        _section("3. EPISODE RESET — reset_envs_idx restores pose + configured velocity (real chain)")
        moving = "moving_box"
        want_vel = sim.get_actor_initial_velocities([moving], env_ids)[0]  # [6], configured
        before_reset = {nm: sim.get_actor_states([nm], env_ids).clone() for nm in free_names}
        static_before = sim.get_actor_states([static_names[0]], env_ids)[:, :2].clone()

        env.reset_envs_idx(env_ids)

        after = sim.get_actor_states([moving], env_ids)[0]
        print(f"'{moving}' env0 after reset: pos={_fmt(after[:3])} vel={_fmt(after[7:10])}/{_fmt(after[10:13])}")
        print(f"  configured initial velocity (restored, not zeroed): {_fmt(want_vel[:3])} / {_fmt(want_vel[3:6])}")
        if not torch.allclose(after[7:13], want_vel, atol=1e-2):
            print("FAIL: reset did not restore the configured initial velocity")
            return 1

        _section("4. POSE JITTER ON RESET — jitter term (a real reset randomizer) perturbed XY/yaw")
        max_dxy = 0.0
        for nm in free_names:
            st = sim.get_actor_states([nm], env_ids)
            dxy = (st[:, :2] - before_reset[nm][:, :2]).abs().max().item()
            max_dxy = max(max_dxy, dxy)
            # The jitter is around the BASELINE reset pose, so dxy is bounded by xy_range (0.1),
            # not by the drift before the reset. A nonzero dxy proves the reset-time jitter ran.
            z_kept = _close(st[:, 2], before_reset[nm][:, 2])
            print(f"  {nm:18s} XY moved {dxy:.4f} m by the reset+jitter (z kept={z_kept})")
        if max_dxy <= 1e-6:
            print("FAIL: reset-time pose jitter moved no free body (jitter term inert)")
            return 1
        if not _close(sim.get_actor_states([static_names[0]], env_ids)[:, :2], static_before):
            print(f"FAIL: static body '{static_names[0]}' was moved on reset")
            return 1
        print(f"  static '{static_names[0]}' untouched on reset: OK")

        # ---- 5. PHYSICS DR ---------------------------------------------------------------
        # The object mass/material/inertia randomizers are SETUP terms on the env's
        # RandomizationManager (see _experiment_config), so the manager ALREADY applied them at
        # build via randomization_manager.setup() — we don't call any term here, we read the
        # result the manager produced. Default box mass is 0.1 kg, so a per-body mass in the
        # configured [5,6] band is unambiguous proof the manager's startup DR ran.
        _section("5. PHYSICS DR — object mass/material/inertia applied by the RandomizationManager at build")
        rm = env.randomization_manager
        ran = [n for n in rm._setup_names if n not in rm._failed_randomizers and n.startswith("randomize_object_")]
        skipped = [n for n in rm._failed_randomizers if n.startswith("randomize_object_")]
        print(f"manager setup DR terms applied: {ran}")
        if skipped:
            print(f"  (skipped as unsupported on this backend: {skipped})")
        if not _report_object_dr(env, free_names, is_mujoco=sim.get_simulator_type() == SimulatorType.MUJOCO):
            return 1

        _section("DEMO COMPLETE — all object-manager features ran successfully")
        return 0


def _report_object_dr(env, free_names, *, is_mujoco: bool) -> bool:
    """Read back the physics state the RandomizationManager's startup DR already produced.

    Reads (does NOT apply) each free body's mass + sliding friction. On MuJoCo, reads from the
    live model (per-world GPU bridge on WarpBackend, else the CPU model) and verifies the values
    landed in the configured bands (default 0.1 kg mass << [5,6] proves the manager ran the DR).
    On IsaacGym/IsaacSim the model fields aren't introspected here — the per-backend physics
    effect is asserted in the dedicated DR tests — so just confirm the bodies are addressable.
    """
    sim = env.simulator

    if not is_mujoco:
        print(f"  startup DR applied on {sim.get_simulator_type()} for {free_names}")
        print("  (per-backend physics effect is asserted in the dedicated DR tests)")
        return True

    from holosoma.managers.randomization.terms.objects import _mujoco_object_bodies, _mujoco_object_geom_ids

    is_warp = hasattr(sim.backend, "warp_model_bridge")

    def _mass(name):
        body_ids = [bid for bid, _ in _mujoco_object_bodies(sim, name)]
        if is_warp:
            bm = sim.backend.warp_model_bridge.body_mass  # [num_envs, nbody]
            return float(sum(bm[0, bid] for bid in body_ids))
        return float(sum(sim.backend.model.body_mass[bid] for bid in body_ids))

    def _friction0(name):
        geoms = _mujoco_object_geom_ids(sim, name)
        if not geoms:
            return None
        if is_warp:
            return float(sim.backend.warp_model_bridge.geom_friction[0, geoms[0], 0])
        return float(sim.backend.model.geom_friction[geoms[0], 0])

    for nm in free_names:
        m = _mass(nm)
        print(f"  mass '{nm:16s}' = {m:.3f} kg  (default 0.1; configured startup-DR band [5,6]/body)")
        if m < 5.0 - 1e-3:  # at least the per-body additive floor over the 0.1 default
            print(f"FAIL: object '{nm}' mass {m:.3f} below configured band — startup DR did not apply")
            return False
    for nm in free_names:
        fr = _friction0(nm)
        if fr is not None:
            print(f"  friction '{nm:16s}' geom[0] sliding = {fr:.3f}  (configured band [0.2,0.3])")
            if not (0.2 - 1e-3 <= fr <= 0.3 + 1e-3):
                print(f"FAIL: object '{nm}' friction {fr:.3f} outside configured band [0.2,0.3]")
                return False
    return True


def _fmt(t):
    return "[" + ", ".join(f"{float(v):+.3f}" for v in t) + "]"


def _close(a, b, atol=1e-4):
    from holosoma.utils.safe_torch_import import torch

    return bool(torch.allclose(a, b, atol=atol))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=list(_BACKENDS), default="mujoco", help="Simulator backend.")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel envs (>1 needs mjwarp/isaacgym/isaacsim).")
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    args = parser.parse_args()

    if args.backend == "mujoco" and args.num_envs > 1:
        parser.error("the MuJoCo classic (CPU) backend is single-env; use --backend mjwarp for --num-envs > 1.")

    return run_demo(args.backend, args.num_envs, headless=args.headless == "true")


if __name__ == "__main__":
    raise SystemExit(main())
