"""Cross-backend BEHAVIORAL assertion harness for object spawning + interaction.

Unlike scene_spawn_assert.py (which reads physical values back and checks them against config),
this harness asserts physical OUTCOMES after stepping real physics: a body stops at a wall,
rests on a post, bounces, spins by the commanded angle, slides farther when low-friction, etc.
A behavioral pass proves the configured value is doing physically meaningful work — you cannot
get the right outcome by querying the wrong field.

Every scenario asserts in EVERY env (``--num-envs`` spreads each env to its own origin on a line
in x, spacing 5 m). Some scenarios require multi-env or a specific backend. When run via the
pytest wrappers those unsupported cells are expressed as ``pytest.mark.skip`` and never reach the
harness. If the harness is invoked directly on an unsupported combo (or a scenario detects a skip
condition only at runtime), it prints ``SKIP: <reason>`` and exits with ``SKIP_EXIT_CODE`` (77),
which ``run_harness`` translates into a real ``pytest.skip`` — never a silent pass.

Usage:
  python tests/simulators/behavior_assert.py --scenario galileo-freefall --simulator mujoco
  python tests/simulators/behavior_assert.py --scenario collide-into-fixed --simulator isaacgym --num-envs 4
  python tests/simulators/behavior_assert.py --scenario restitution-bounce --simulator isaacsim --record /tmp/vids

Constants: g = 9.81 m/s^2, sim dt = 1/200 = 0.005 s, small_box = 0.1 m cube (half-extent 0.05).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys

# This file lives in tests/simulators/, which contains an ``isaacsim/`` subpackage. Run as a
# script, that dir lands on sys.path[0] and shadows the real IsaacSim ``isaacsim`` package.
# Drop it; the package is imported via its installed location / PYTHONPATH=src/holosoma.
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.sampler import STAGE_RESET, STAGE_SETUP, TermSampler
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators import _scene_presets
from tests.simulators._sim_harness import build_run_sim_config, step, steps_for_seconds

GRAVITY = 9.81


def _behavior_sampler(stage: int, term: str = "behavior_assert") -> TermSampler:
    """A bound TermSampler for calling DR/jitter terms directly in these behavioral checks.

    DR term functions take a REQUIRED ``sampler=`` kwarg, and TermSampler REQUIRES a seed (no
    global-RNG fallback). These scenarios assert physical effects/per-env independence (not seed
    reproducibility), so bind a fixed seed — keyed draws are per-env independent regardless.
    """
    return TermSampler.bind(0, term, stage, None)


# Autotools-convention "skip" exit code. The harness exits with this (and prints a ``SKIP:``
# line) when a scenario cannot run on the chosen (backend, num_envs) combo. The pytest wrappers
# express the STATICALLY-knowable skips as marks (so those cells never spawn a subprocess); this
# code is the runtime path for skips a wrapper cannot predict, and a defensive fallback for direct
# harness invocations with an unsupported combo. run_harness (tests/simulators/_run_harness.py)
# translates it into ``pytest.skip(reason)``. Kept in sync with ``_run_harness.SKIP_EXIT_CODE``;
# chosen so it cannot collide with 0 (pass), 1 (FAIL), or a signal crash (128+N >= 129).
SKIP_EXIT_CODE = 77


# --------------------------------------------------------------------------------------------
# Boot machinery: the generic pieces live in _sim_harness; this harness records video, so it always
# renders each step and supplies an object-framing camera.
# --------------------------------------------------------------------------------------------
def _build_run_sim_config(simulator, scene, robot, terrain, record_dir):
    """Bridge/gantry-disabled RunSimConfig; when recording, frame the camera on the scene OBJECTS."""
    cam = None
    if record_dir is not None:
        # Object-framing camera: the default cartesian offset/smoothing is tuned for a ~1.7 m robot;
        # the behavioral bodies are ~0.1 m boxes moving over ~1-2 m. Pull in closer, look at object
        # height, and drop smoothing so the camera follows fast motion (collide/bounce/spin) instead
        # of lagging it. The TARGET is overridden to the object centroid at runtime by
        # _focus_camera_on_objects; these offsets frame around that point.
        from holosoma.config_types.video import CartesianCameraConfig

        cam = CartesianCameraConfig(offset=[1.1, 1.1, 0.7], target_offset=[0.0, 0.0, 0.0], smoothing=0.6)
    return build_run_sim_config(
        simulator,
        scene,
        robot,
        terrain,
        record_dir=record_dir,
        video_camera=cam,
        show_command_overlay=False,  # robot command text is irrelevant to object scenarios
    )


def _step(sim, n):
    """Advance ``n`` physics steps WITH render each step (this harness records video)."""
    step(sim, n, render=True)


def _focus_camera_on_objects(sim, scene_config):
    """Point the recording camera at the SCENE OBJECTS instead of the robot.

    The shared camera controller orbits whatever world point ``_get_camera_parameters`` hands it,
    which by default is ``robot_root_states[record_env_id]`` — so in these object-centric scenarios
    the robot fills frame while the interesting bodies (boxes colliding, bouncing, sliding) sit off
    to the side. This is a HARNESS-only override: it monkey-patches the live recorder instance's
    ``_get_camera_parameters`` to feed the centroid of the rigid objects at ``record_env_id`` each
    frame (re-read every call, so the camera follows moving bodies), leaving all shipped camera /
    video code untouched. Falls back to the robot when the scene has no rigid objects.
    """
    recorder = sim.video_recorder
    if recorder is None:
        return
    env_id = recorder.config.record_env_id
    names = list(scene_config.rigid_objects)
    if not names:
        return  # no objects -> keep the default robot-tracking behavior

    import torch

    def _object_centroid():
        # Live world positions of all rigid objects in the recorded env, averaged. get_actor_states
        # is WORLD-frame on every backend, [num_envs, 13]; column 0:3 is position.
        env_ids = torch.tensor([env_id], device=sim.sim_device)
        pts = [sim.get_actor_states([nm], env_ids)[0, :3] for nm in names]
        c = torch.stack(pts, dim=0).mean(dim=0)
        return (float(c[0]), float(c[1]), float(c[2]))

    def _patched_get_camera_parameters(robot_pos=None):
        # Ignore the passed robot_pos; track the object centroid instead.
        return recorder.camera_controller.update(robot_pos=_object_centroid())

    recorder._get_camera_parameters = _patched_get_camera_parameters


# --------------------------------------------------------------------------------------------
# Per-scenario assertions. Each returns True on PASS. ``ctx`` bundles the live sim + run info.
# --------------------------------------------------------------------------------------------
class _Ctx:
    def __init__(self, sim, config, args, env_origins, env_ids, torch):
        self.sim = sim
        self.config = config
        self.args = args
        self.env_origins = env_origins
        self.env_ids = env_ids
        self.torch = torch
        self.n = sim.num_envs

    def states(self, name):
        return self.sim.get_actor_states([name], self.env_ids)  # [num_envs, 13]

    def free_cfg(self):
        return {name: o for name, o in self.config.scene.rigid_objects.items() if not o.fixed}


def _all(t):
    """Python bool of torch.all (works on a [num_envs] bool tensor)."""
    return bool(t.all())


def assert_collide_into_fixed(ctx):
    wall0 = ctx.states("wall")[:, :3].clone()
    hammer0_x = ctx.states("hammer")[:, 0].clone()  # spawn x = 1.4
    wall_x = wall0[:, 0]
    face_x = wall_x - 0.10  # hammer center flush against the near face (two 0.05 half-extents)
    # Step the full window manually so we can track the hammer's CLOSEST APPROACH to the wall (peak
    # +x) separately from where it ENDS UP. The two together tell the collision story without pinning
    # a specific stopping position: backends differ on the post-contact settle (a flush rest vs. a
    # small rebound), so we assert the hammer reached the wall and came back, not where it parked.
    n = steps_for_seconds(ctx.sim, 1.5)
    peak_x = hammer0_x.clone()
    for _ in range(n):
        ctx.sim.refresh_sim_tensors()
        ctx.sim.simulate_at_each_physics_step()
        ctx.sim.render()
        peak_x = ctx.torch.maximum(peak_x, ctx.states("hammer")[:, 0])
    hammer = ctx.states("hammer")
    wall = ctx.states("wall")
    hammer_x = hammer[:, 0]
    rebound = peak_x - hammer_x  # how far it retreated from its closest approach
    # Assert the collision STORY, not a stopping position. The hammer must (a) advance from spawn
    # and REACH the wall — its closest approach (peak +x) lands within a hair of the near face
    # face_x = wall_x - 0.10 (the two 0.05 half-extents); and (b) end up AT-OR-BEHIND that face
    # rather than tunneling through it. Backends differ on the post-contact settle: IsaacSim and
    # IsaacGym typically come to rest flush (rebound ~0), MuJoCo (and IsaacGym in some runs) rebounds
    # a few cm back off the face. Both are correct "hit a wall and stopped/bounced" outcomes, so the
    # final-position band is loose (face_x - 0.10 .. face_x + 0.02): it still rejects a dead velocity
    # write (never advances, peak stays at spawn) and a tunnel-through (ends well past the face).
    reached_wall = _all(peak_x > hammer0_x + 0.15) and _all(peak_x > face_x - 3e-2)
    bounced_or_rests = _all((hammer_x > face_x - 0.10) & (hammer_x < face_x + 2e-2))
    # Wall (static) must not move on any axis.
    wall_fixed = _all((wall[:, :3] - wall0).abs().amax(dim=1) < 1e-3)
    print(
        f"  collide: hammer_x env0 {float(hammer_x[0]):.3f} peak {float(peak_x[0]):.3f} "
        f"face {float(face_x[0]):.3f} rebound {float(rebound[0]):+.3f} "
        f"reached_wall={reached_wall} bounced_or_rests={bounced_or_rests} wall_fixed={wall_fixed}"
    )
    return reached_wall and bounced_or_rests and wall_fixed


def assert_momentum_transfer(ctx):
    g0 = ctx.states("target")[:, :3].clone()
    sv0 = ctx.states("striker")[:, 7].clone()  # striker initial +x velocity (~2.0)
    # Track the target's PEAK +x velocity across the window: the striker (x=1.4, v=2) reaches the
    # resting target (x=1.55) in ~0.025 s, the target launches forward, then BOTH boxes' friction
    # (0.05) bleeds it back toward 0 — so a single end-of-window velocity sample misses the launch.
    # The peak is reached right after impact. A resting target can ONLY acquire +x velocity from the
    # collision (friction never accelerates a body from rest), so a clear peak proves momentum
    # transfer, well above the ~mu*g*dt friction noise floor.
    n = steps_for_seconds(ctx.sim, 0.4)
    target_vx_peak = ctx.states("target")[:, 7].clone()
    for _ in range(n):
        ctx.sim.refresh_sim_tensors()
        ctx.sim.simulate_at_each_physics_step()
        ctx.sim.render()
        target_vx_peak = ctx.torch.maximum(target_vx_peak, ctx.states("target")[:, 7])
    g1 = ctx.states("target")[:, :3]
    sv1 = ctx.states("striker")[:, 7]
    # (a) The initially-resting target was LAUNCHED forward (peak +x velocity well above friction
    # noise) AND ended up displaced +x — only the collision can do this.
    target_launched = _all(target_vx_peak > 0.5) and _all((g1[:, 0] - g0[:, 0]) > 0.02)
    # (b) The striker gave up most of its forward momentum into the target (lost >> the ~0.2 m/s
    # friction-only loss over this window).
    striker_lost = _all((sv0 - sv1) > 1.0)
    ok = target_launched and striker_lost
    print(
        f"  momentum: target_vx_peak env0 {float(target_vx_peak[0]):.2f} disp {float(g1[0, 0] - g0[0, 0]):+.3f} "
        f"striker_vx {float(sv0[0]):.2f}->{float(sv1[0]):.2f}  launched={target_launched} "
        f"striker_lost={striker_lost}  ok={ok}"
    )
    return ok


def assert_static_support(ctx):
    post0 = ctx.states("post")[:, :3].clone()
    _step(ctx.sim, steps_for_seconds(ctx.sim, 1.5))
    rest = ctx.states("restbox")
    post = ctx.states("post")
    rest_z = rest[:, 2]
    # Clean rest on the post top (0.35) puts the 0.1-cube center at exactly 0.40. Tight band (8mm)
    # so a box that came to rest PENETRATING the post (soft/mis-scaled contact, center < ~0.39)
    # fails — a 3cm band would accept a box sunk most of the way into the post.
    rests_on = _all((rest_z - 0.40).abs() < 8e-3)
    # Still over the post in x,y (didn't slide off the edge / tip) — an independent guard, not the
    # dead "above floor" clause the tight z-band already subsumes. Post + box both at x=1.5,y=0.
    over_post = _all((rest[:, :2] - post0[:, :2]).abs().amax(dim=1) < 3e-2)
    post_fixed = _all((post[:, :3] - post0).abs().amax(dim=1) < 1e-3)
    print(
        f"  support: restbox_z env0 {float(rest_z[0]):.4f} (want ~0.40)  rests_on_all={rests_on}  "
        f"over_post={over_post} post_fixed_all={post_fixed}"
    )
    return rests_on and over_post and post_fixed


def _slide_distance(ctx, name, p0):
    st = ctx.states(name)[:, :2]
    return ctx.torch.linalg.norm(st - p0, dim=1)  # [num_envs]


def assert_friction_slide(ctx):
    lo0 = ctx.states("box_lowfric")[:, :2].clone()
    hi0 = ctx.states("box_highfric")[:, :2].clone()
    _step(ctx.sim, steps_for_seconds(ctx.sim, 1.0))
    lo_d = _slide_distance(ctx, "box_lowfric", lo0)
    hi_d = _slide_distance(ctx, "box_highfric", hi0)
    ok = _all(lo_d > hi_d + 0.05)
    print(
        f"  fric-slide: low {[round(float(v), 3) for v in lo_d.tolist()]} "
        f"high {[round(float(v), 3) for v in hi_d.tolist()]}  low>high_all={ok}"
    )
    return ok


def _pre_prepare_dr_friction(sim, env_ids, torch):
    """Apply friction DR to dr_lo (low) / dr_hi (high) BEFORE prepare_sim.

    Every backend only honors a friction write in a specific window: IsaacGym applies
    set_actor_rigid_shape_properties between create_envs and prepare_sim; Warp must have the
    per-world geom_friction expanded before its step graph is captured (done in __init__);
    MuJoCo Classic writes the model field directly. Running the DR here (pre-prepare) is the one
    point that takes effect on ALL backends, so the configured friction governs the slide.
    """
    import types

    from holosoma.managers.randomization.terms.objects import randomize_object_rigid_body_material_startup

    shell = types.SimpleNamespace(simulator=sim, num_envs=len(env_ids), device=sim.sim_device)
    if hasattr(sim, "prepare_randomization_fields"):
        sim.prepare_randomization_fields(["geom_friction"])  # MuJoCo Warp: expand per-world field
    # Degenerate [x,x] band => deterministic friction per env. Friction-only (this scenario tests the
    # low-vs-high sliding-friction slide); per-backend material config, each names its friction channel.
    sampler = _behavior_sampler(STAGE_SETUP)

    def _friction_material(f):
        return {
            "isaacgym": {"friction": [f, f]},
            "isaacsim": {"static_friction": [f, f], "dynamic_friction": [f, f]},
            "mujoco": {"sliding_friction": [f, f]},
        }

    randomize_object_rigid_body_material_startup(
        shell,
        env_ids,
        sampler=sampler,
        material=_friction_material(0.02),
        object_names=["dr_lo"],
    )
    randomize_object_rigid_body_material_startup(
        shell,
        env_ids,
        sampler=sampler,
        material=_friction_material(5.0),
        object_names=["dr_hi"],
    )


def assert_dr_friction_governs(ctx):
    # Friction was set by the pre-prepare hook (the only window all backends honor); here we just
    # run the slide and assert the DR-set low-friction box travels farther than the high-friction one.
    lo0 = ctx.states("dr_lo")[:, :2].clone()
    hi0 = ctx.states("dr_hi")[:, :2].clone()
    _step(ctx.sim, steps_for_seconds(ctx.sim, 1.0))
    lo_d = _slide_distance(ctx, "dr_lo", lo0)
    hi_d = _slide_distance(ctx, "dr_hi", hi0)
    ok = _all(lo_d > hi_d + 0.05)
    print(
        f"  dr-fric: low {[round(float(v), 3) for v in lo_d.tolist()]} "
        f"high {[round(float(v), 3) for v in hi_d.tolist()]}  low>high_all={ok}"
    )
    return ok


def _pre_prepare_dr_damping(sim, env_ids, torch):
    """Apply linear-damping DR to `damped` (leave `control` at zero) BEFORE prepare_sim.

    MuJoCo Warp must have dof_damping per-world-expanded before its step graph is captured (done
    in __init__) and the write must land before capture; MuJoCo Classic writes the model field
    directly; IsaacSim writes the live USD PhysxRigidBodyAPI damping. Running here (pre-prepare)
    is the window that takes effect on all damping-capable backends. mujoco_warp NaNs above ~0.05,
    so use a Warp-safe value there and a sharper value on the stable backends.
    """
    import types

    from holosoma.managers.randomization.terms.objects import randomize_object_linear_damping_startup
    from holosoma.utils.simulator_config import SimulatorType

    shell = types.SimpleNamespace(simulator=sim, num_envs=len(env_ids), device=sim.sim_device)
    if hasattr(sim, "prepare_randomization_fields"):
        sim.prepare_randomization_fields(["dof_damping"])  # MuJoCo Warp: expand per-world field
    is_warp = sim.get_simulator_type() == SimulatorType.MUJOCO and hasattr(sim.backend, "warp_model_bridge")
    d = 0.05 if is_warp else 3.0  # Warp stable band vs a sharp signal on classic/IsaacSim
    # Degenerate [d, d] band => deterministic damping per env on `damped`; `control` stays undamped.
    randomize_object_linear_damping_startup(
        shell, env_ids, sampler=_behavior_sampler(STAGE_SETUP), damping_range=[d, d], object_names=["damped"]
    )


def assert_dr_damping_governs(ctx):
    # Damping was set by the pre-prepare hook on `damped` only. Both boxes fly +x high in the air
    # (no floor contact), so the ONLY difference is the DR-applied damping: `damped` must lose more
    # horizontal speed than the undamped `control`. Proves runtime damping DR governs the dynamics.
    vd0 = ctx.states("damped")[:, 7].clone()
    vc0 = ctx.states("control")[:, 7].clone()
    _step(ctx.sim, steps_for_seconds(ctx.sim, 0.5))
    vd1 = ctx.states("damped")[:, 7]
    vc1 = ctx.states("control")[:, 7]
    damped_loss = vd0 - vd1
    control_loss = vc0 - vc1
    # The damped box decelerates clearly more than the control, in every env.
    ok = _all(damped_loss > control_loss + 0.10)
    print(
        f"  dr-damp: damped vx {float(vd0[0]):.3f}->{float(vd1[0]):.3f} (loss {float(damped_loss[0]):.3f}) "
        f"control loss {float(control_loss[0]):.3f}  damped>control_all={ok}"
    )
    return ok


def assert_galileo_freefall(ctx):
    zl0 = ctx.states("light")[:, 2].clone()
    zh0 = ctx.states("heavy")[:, 2].clone()
    n = steps_for_seconds(ctx.sim, 0.3)
    _step(ctx.sim, n)
    elapsed = n * ctx.sim.sim_dt
    expected_drop = 0.5 * GRAVITY * elapsed * elapsed
    zl1 = ctx.states("light")[:, 2]
    zh1 = ctx.states("heavy")[:, 2]
    drop_l = zl0 - zl1
    drop_h = zh0 - zh1
    # Tight 5% band: analytic semi-implicit-Euler error over 0.3 s is <2%, so 5% covers cross-backend
    # integrator spread but is tight enough that a stray default damping (Isaac free-body default 0.1,
    # which _no_damping must zero) would push the drop out of band rather than hide inside a loose 12%.
    kin_l = _all(((drop_l - expected_drop).abs() / expected_drop) < 0.05)
    kin_h = _all(((drop_h - expected_drop).abs() / expected_drop) < 0.05)
    # Mass-independence: light and heavy fall identically.
    same = _all((zl1 - zh1).abs() < 0.02)
    print(
        f"  galileo: drop_light env0 {float(drop_l[0]):.4f} drop_heavy {float(drop_h[0]):.4f} "
        f"(expected {expected_drop:.4f})  kin_l={kin_l} kin_h={kin_h} mass_indep={same}"
    )
    return kin_l and kin_h and same


def assert_damping_decay(ctx):
    # Damped box vs undamped CONTROL, both launched +x at z=5.5 and kept AIRBORNE for the window
    # (z stays >> floor, so floor friction cannot contribute — only drag slows them). The damped
    # box must lose meaningfully MORE +x speed than the control, in every env. This is sensitive to
    # the configured damping itself (a no-op damping write => damped behaves like control => FAIL),
    # unlike an absolute "% decay" threshold that floor friction could satisfy at damping≈0.
    vd0 = ctx.states("dbox")[:, 7].clone()
    vc0 = ctx.states("dctrl")[:, 7].clone()
    _step(ctx.sim, steps_for_seconds(ctx.sim, 0.5))
    vd1 = ctx.states("dbox")[:, 7]
    vc1 = ctx.states("dctrl")[:, 7]
    z_d1 = ctx.states("dbox")[:, 2]
    damped_loss = vd0 - vd1
    control_loss = vc0 - vc1
    airborne = _all(z_d1 > 0.2)  # confirm it never hit the floor (no friction contamination)
    decayed_more = _all(damped_loss > control_loss + 0.10)
    print(
        f"  damping: damped vx {float(vd0[0]):.3f}->{float(vd1[0]):.3f} (loss {float(damped_loss[0]):.3f}) "
        f"control loss {float(control_loss[0]):.3f}  airborne={airborne} damped>control={decayed_more}"
    )
    return airborne and decayed_more


def assert_restitution_bounce(ctx):
    t = ctx.torch
    n = steps_for_seconds(ctx.sim, 1.0)
    # Record the full z trajectory, then detect the FIRST floor contact (first local minimum) and
    # the rebound apex AFTER it. A global argmin is wrong: the box settles back near the floor at
    # the END, so "max after the global min" misses the early rebound entirely.
    z_hist = []
    for _ in range(n):
        ctx.sim.refresh_sim_tensors()
        ctx.sim.simulate_at_each_physics_step()
        ctx.sim.render()
        z_hist.append(ctx.states("ball")[:, 2].clone())
    z = t.stack(z_hist, dim=0)  # [steps, num_envs]
    REST_Z = 0.052  # box half-extent on the floor
    rebound = t.zeros(ctx.n, device=z.device)
    contact_z = t.zeros(ctx.n, device=z.device)
    for e in range(ctx.n):
        col = z[:, e]
        # First contact = first step the box is back near the floor (within 1cm of resting height).
        near_floor = (col <= REST_Z + 0.01).nonzero().flatten()
        first_contact = int(near_floor[0]) if near_floor.numel() else int(col.argmin())
        contact_z[e] = col[first_contact]
        rebound[e] = col[first_contact:].max()  # apex after the first contact
    # Threshold tied to the CONFIGURED restitution e=0.9: from a fall height h ≈ 0.448 m the apex
    # above the floor should be e^2*h + REST_Z = 0.81*0.448 + 0.052 ≈ 0.41 m. Require apex > 0.30
    # (effective e > ~0.77) so a box that bounces at only HALF the configured restitution FAILS.
    # Holds on EVERY backend, IsaacGym included: with the ground restitution set to 0.9 (see the
    # terrain_term override in the runner), IsaacGym honors e≈0.88 (apex ~0.40) via its geometric-mean
    # PhysX combine, same as MuJoCo/IsaacSim.
    threshold = 0.30
    bounced = _all(rebound > threshold)
    print(
        f"  bounce: contact_z env0 {float(contact_z[0]):.4f} rebound_apex {float(rebound[0]):.4f}  "
        f"(thresh {threshold}, e_cfg=0.9)  bounced_all={bounced}"
    )
    if os.environ.get("BOUNCE_DEBUG"):
        print(f"  BOUNCE_DEBUG z[::10] env0 = {[round(float(v), 3) for v in z[::10, 0].tolist()]}")
    return bounced


def assert_angular_spin(ctx):
    from holosoma.utils.rotations import quat_angle_axis

    t = ctx.torch
    q0 = ctx.states("spinner")[:, 3:7].clone()
    n = steps_for_seconds(ctx.sim, 0.3)
    _step(ctx.sim, n)
    elapsed = n * ctx.sim.sim_dt
    st1 = ctx.states("spinner")
    w1 = st1[:, 10:13]
    q1 = st1[:, 3:7]
    expected_angle = 4.0 * elapsed  # omega * t (= 1.2 rad < pi, so the [0,pi] clamp is fine)
    # The INTEGRATED rotation angle is the load-bearing, integration-sensitive check: tight band
    # (~5%, vs the ~2% semi-implicit-Euler error over 60 steps) so a constant-fraction rate error
    # (e.g. a units/inertia-frame bug delivering 85% of omega) FAILS rather than slips through a 21%
    # band. Measured via the relative quaternion q0^-1 * q1 (unsigned [0,pi] angle + unit axis).
    q0_conj = q0.clone()
    q0_conj[:, :3] = -q0_conj[:, :3]  # conjugate (xyzw)
    qr = _quat_mul_xyzw(t, q0_conj, q1)  # relative rotation, xyzw
    angle, axis = quat_angle_axis(qr, w_last=True)
    ang_ok = _all((angle - expected_angle).abs() < 0.06)
    axisz_ok = _all(axis[:, 2].abs() > 0.98)  # rotation axis is essentially +/- z
    # Velocity read-back: a tight band so a partial-omega or damping-bleed bug is caught, and no
    # spurious off-axis tumble (a pure-z spin from identity must keep wx,wy ~ 0).
    wz_ok = _all((w1[:, 2] - 4.0).abs() < 0.2)
    off_axis_ok = _all(w1[:, :2].abs().amax(dim=1) < 0.05)
    print(
        f"  spin: wz env0 {float(w1[0, 2]):.3f}  angle env0 {float(angle[0]):.3f} (want {expected_angle:.3f}) "
        f"ang_ok={ang_ok} axisz_ok={axisz_ok} wz_ok={wz_ok} off_axis_ok={off_axis_ok}"
    )
    return ang_ok and axisz_ok and wz_ok and off_axis_ok


def _quat_mul_xyzw(t, a, b):
    """Hamilton product of two xyzw quaternion batches [N,4]."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return t.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=1,
    )


def assert_velocity_restore_on_reset(ctx):
    t = ctx.torch
    sim = ctx.sim
    world0 = ctx.states("rbox")[:, :3].clone()  # spawn world position (origins applied), per env
    _step(sim, steps_for_seconds(sim, 0.5))  # let it move; pose + velocity drift
    # Reset to the configured initial state via the public seam: get_actor_initial_poses and
    # set_actor_states are both WORLD-frame on every backend, so feed the get output STRAIGHT into
    # set, exactly as the production reset/jitter terms do — no manual origin add.
    poses = sim.get_actor_initial_poses(["rbox"], ctx.env_ids)  # [n,7]
    vels = sim.get_actor_initial_velocities(["rbox"], ctx.env_ids)  # [n,6]
    sim.set_actor_states(["rbox"], ctx.env_ids, t.cat([poses, vels], dim=1))
    restored = ctx.states("rbox")
    # Position restored to the original WORLD spawn pose in every env (catches an origin
    # double-count / frame mismatch that the velocity checks alone would miss).
    pos_ok = _all((restored[:, :3] - world0).abs().amax(dim=1) < 5e-2)
    lin_ok = _all((restored[:, 7:10] - t.tensor([1.0, 0.0, 0.0], device=restored.device)).abs().amax(dim=1) < 5e-2)
    ang_ok = _all((restored[:, 10:13] - t.tensor([0.0, 0.0, 2.0], device=restored.device)).abs().amax(dim=1) < 5e-2)
    # Moving again: step and confirm it translates +x and spins.
    p_mid = ctx.states("rbox")[:, :3].clone()
    q_mid = ctx.states("rbox")[:, 3:7].clone()
    _step(sim, steps_for_seconds(sim, 0.3))
    after = ctx.states("rbox")
    moved = _all((after[:, 0] - p_mid[:, 0]) > 1e-2)
    spun = _all((after[:, 3:7] - q_mid).abs().amax(dim=1) > 1e-2)
    print(f"  vel-restore: pos_ok={pos_ok} lin_ok={lin_ok} ang_ok={ang_ok} moving_again={moved and spun}")
    return pos_ok and lin_ok and ang_ok and moved and spun


def _yaw_world_to_base(t, yaw, vec_w):
    """Rotate a world vector into a base frame yawed by ``yaw`` about z — CLOSED FORM (2D rotation
    by -yaw), independent of the source's quat_rotate_inverse so the obs comparison isn't circular.
    yaw: [n], vec_w: [n,3] -> [n,3]."""
    c, s = t.cos(yaw), t.sin(yaw)
    bx = c * vec_w[:, 0] + s * vec_w[:, 1]
    by = -s * vec_w[:, 0] + c * vec_w[:, 1]
    return t.stack([bx, by, vec_w[:, 2]], dim=1)


def assert_object_obs_under_pose(ctx):
    import types

    from holosoma.managers.observation.terms.objects import (
        object_ang_vel_b,
        object_lin_vel_b,
        object_pos_b,
        object_quat_b,
    )

    t = ctx.torch
    sim = ctx.sim
    if ctx.n < 2:
        print("SKIP: object-obs-robot-pose needs num_envs>1 (per-env transform independence)")
        return None
    names = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)  # >=2 objects (obox, obox2)
    # Per-env-distinct robot root pose: env i gets yaw 0.5*i about z. Read the proxy into a plain
    # tensor (IsaacSim's robot_root_states is a RootStatesProxy, no .clone()); mutate, write back, push.
    yaws = t.tensor([0.5 * e for e in range(ctx.n)], device=sim.sim_device)
    root = sim.robot_root_states[:].clone()
    for e in range(ctx.n):
        root[e, 3:7] = t.tensor([0.0, 0.0, math.sin(0.5 * e / 2), math.cos(0.5 * e / 2)], device=root.device)
    sim.robot_root_states[:] = root
    sim.set_actor_root_state_tensor_robots(ctx.env_ids)  # 2nd arg defaults to None on every backend

    shell = types.SimpleNamespace(simulator=sim, num_envs=ctx.n, device=sim.sim_device)
    rp = sim.robot_root_states[:][:, :3]
    # Each obs term returns [num_envs, k*N] flattened object-major; reshape to [num_envs, N, k].
    N = len(names)
    pos_b = object_pos_b(shell, names).reshape(ctx.n, N, 3)
    lin_b = object_lin_vel_b(shell, names).reshape(ctx.n, N, 3)
    ang_b = object_ang_vel_b(shell, names).reshape(ctx.n, N, 3)
    quat_b = object_quat_b(shell, names).reshape(ctx.n, N, 4)
    ok = True
    for j, nm in enumerate(names):
        st = ctx.states(nm)  # world-frame ground truth [n,13]
        # Independent closed-form expectation (NOT the source rotation helper).
        exp_pos = _yaw_world_to_base(t, yaws, st[:, :3] - rp)
        exp_lin = _yaw_world_to_base(t, yaws, st[:, 7:10])
        exp_ang = _yaw_world_to_base(t, yaws, st[:, 10:13])
        pos_ok = _all((pos_b[:, j] - exp_pos).abs().amax(dim=1) < 2e-3)
        lin_ok = _all((lin_b[:, j] - exp_lin).abs().amax(dim=1) < 2e-3)
        ang_ok = _all((ang_b[:, j] - exp_ang).abs().amax(dim=1) < 2e-3)
        # Orientation obs: the object spawns at identity, so its base-frame yaw must be -robot_yaw.
        # quat_b is xyzw; for a pure-z relative rotation the z-component is sin(theta/2), w cos(theta/2).
        obj_world_yaw = 2.0 * t.atan2(st[:, 5], st[:, 6])  # object world yaw from its xyzw quat (z,w)
        exp_rel_yaw = obj_world_yaw - yaws
        got_rel_yaw = 2.0 * t.atan2(quat_b[:, j, 2], quat_b[:, j, 3])
        dyaw = (got_rel_yaw - exp_rel_yaw + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi,pi]
        quat_ok = _all(dyaw.abs() < 5e-3)
        ok = ok and pos_ok and lin_ok and ang_ok and quat_ok
        print(f"  obs-pose [{nm}]: pos_ok={pos_ok} lin_ok={lin_ok} ang_ok={ang_ok} quat_ok={quat_ok}")
    # This scenario asserts a single-frame obs read-back (no stepping), so the video frame buffer
    # would be empty. Drive a short settle with the render path so a clip is produced — purely for
    # video; the assertion above already decided PASS/FAIL.
    _step(sim, steps_for_seconds(sim, 0.5))
    return ok


def assert_pose_jitter_settle(ctx):
    import types

    from holosoma.managers.randomization.terms.objects import jitter_object_pose_on_reset

    t = ctx.torch
    if ctx.n < 2:
        print("SKIP: pose-jitter-settle needs num_envs>1 (per-env independence)")
        return None
    XY_RANGE, YAW_RANGE = 0.5, 0.785
    base_xy = ctx.states("jbox")[:, :2].clone()  # baseline (pre-jitter) XY, per env
    shell = types.SimpleNamespace(simulator=ctx.sim, num_envs=ctx.n, device=ctx.sim.sim_device)
    jitter_object_pose_on_reset(
        shell,
        ctx.env_ids,
        sampler=_behavior_sampler(STAGE_RESET),
        xy_range=XY_RANGE,
        yaw_range=YAW_RANGE,
        object_names=["jbox"],
    )
    jpose = ctx.states("jbox")
    target_xy = jpose[:, :2].clone()  # jittered XY per env
    target_yaw = 2.0 * t.atan2(jpose[:, 5], jpose[:, 6])  # jittered yaw from xyzw quat (z,w)

    # (1) Per-env independence — ALL envs mutually distinct, not just "some env differs from env0".
    # Require the rounded per-env XY to be unique across every env, so a partial-broadcast bug that
    # collapses envs 1..N-1 to one sample is caught.
    xy_keys = {tuple(round(float(v), 3) for v in target_xy[e].tolist()) for e in range(ctx.n)}
    independent = len(xy_keys) == ctx.n
    # (2) Sampled offsets lie within the configured ranges (not just non-equal).
    xy_in_range = _all((target_xy - base_xy).abs().amax(dim=1) <= XY_RANGE + 1e-3)
    yaw_in_range = _all(target_yaw.abs() <= YAW_RANGE + 1e-3)
    # (3) Yaw jitter actually happened and is per-env distinct (the yaw path is separate code from
    # the XY offset — a no-op / wrong-axis / wrong-compose yaw would otherwise pass on XY alone).
    yaw_keys = {round(float(target_yaw[e]), 3) for e in range(ctx.n)}
    yaw_jittered = (len(yaw_keys) == ctx.n) and bool(target_yaw.abs().max() > 0.05)

    _step(ctx.sim, steps_for_seconds(ctx.sim, 0.5))
    settled = ctx.states("jbox")
    near_xy = _all((settled[:, :2] - target_xy).abs().amax(dim=1) < 0.1)
    settled_yaw = 2.0 * t.atan2(settled[:, 5], settled[:, 6])
    near_yaw = _all(((settled_yaw - target_yaw + math.pi) % (2 * math.pi) - math.pi).abs() < 0.1)
    ok = independent and xy_in_range and yaw_in_range and yaw_jittered and near_xy and near_yaw
    print(
        f"  jitter: independent={independent} xy_in_range={xy_in_range} yaw_in_range={yaw_in_range} "
        f"yaw_jittered={yaw_jittered} near_xy={near_xy} near_yaw={near_yaw}"
    )
    return ok


def assert_loader_invariance(ctx):
    # EVERY free body (whether loaded standalone as `lbox` or as 1->N scene-file bodies) must
    # free-fall along the SAME analytic curve — that is the loader-invariance claim. Assert all of
    # them (not just free[0], which would let a regression in one file-expanded body slip through),
    # at the same tight 5% band as galileo so a loader-path damping/mass drift can't hide.
    free = ctx.sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    if not free:
        print("FAIL: loader-invariance preset has no free body")
        return False
    z0 = {nm: ctx.states(nm)[:, 2].clone() for nm in free}
    n = steps_for_seconds(ctx.sim, 0.3)
    _step(ctx.sim, n)
    elapsed = n * ctx.sim.sim_dt
    expected = 0.5 * GRAVITY * elapsed * elapsed
    ok = True
    drops = {}
    for nm in free:
        drop = z0[nm] - ctx.states(nm)[:, 2]
        drops[nm] = round(float(drop[0]), 4)
        ok = ok and _all(((drop - expected).abs() / expected) < 0.05)
    print(f"  loader-inv {drops} (expected {expected:.4f})  kin_ok_all={ok}")
    return ok


def assert_multibody_independence(ctx):
    t = ctx.torch
    free = "scene_free_box"
    static = "scene_static_post"
    fp0 = ctx.states(free)[:, :3].clone()
    sp0 = ctx.states(static)[:, :3].clone()
    # 1->N relative placement: the file authors static_post at +0.5 m x from free_box. Assert the
    # body-to-body offset at SPAWN (before stepping moves the free body's z), every env — the core
    # claim the multibody preset exists to verify (get_actor_states is world-frame on every backend,
    # so static-free cancels env_origins and the x/y offset is the cross-backend invariant).
    offset0 = sp0 - fp0  # [num_envs, 3]
    want = t.tensor([0.5, 0.0, 0.0], device=offset0.device)
    offset_ok = _all((offset0[:, :2] - want[:2]).abs().amax(dim=1) < 1e-2)  # x,y (z differs as free falls)
    _step(ctx.sim, steps_for_seconds(ctx.sim, 0.5))
    fz1 = ctx.states(free)[:, 2]
    sp1 = ctx.states(static)[:, :3]
    fell = _all(fz1 < 0.10)  # the free body actually fell to the floor (not just a >1mm twitch)
    held = _all((sp1 - sp0).abs().amax(dim=1) < 1e-3)  # static sibling held its pose (all axes)
    print(
        f"  multibody: offset0_xy env0 {[round(float(v), 3) for v in offset0[0, :2].tolist()]} "
        f"offset_ok={offset_ok} free_fell_all={fell} static_held_all={held}"
    )
    return offset_ok and fell and held


def assert_per_env_relocation(ctx):
    t = ctx.torch
    sim = ctx.sim
    if ctx.n < 2:
        print("SKIP: per-env-relocation needs num_envs>=2 (relocate in SOME envs)")
        return None
    # The pillar spawns UNDER the box (x=2.0) so every box starts resting on it. Settle, then
    # relocate the pillar AWAY to a PER-ENV-DISTINCT x (6.0 + env*0.5) in one write over the full
    # contiguous env range. Two behavioral claims, both cross-backend:
    #   (a) per-env relocation INDEXING: the pillar lands at each env's OWN distinct target (a
    #       transposition / subset-scramble bug would mis-assign these — see the subset-write bug),
    #   (b) removing the support makes the box FALL in every env.
    # Relocate AWAY (not "some rest, some fall"): a moved support genuinely disappears, so the box
    # must fall in every env, which reads identically on all four backends.
    _step(sim, steps_for_seconds(sim, 0.8))
    rested = ctx.states("freebox")[:, 2]
    rested_ok = _all((rested - 0.40).abs() < 3e-2)  # all boxes rest on the spawn-placed pillar first
    target_x = 6.0 + ctx.env_ids.to(t.float32) * 0.5  # per-env-distinct relocation target (env-local)
    # set_static_body_pose / set_actor_states take WORLD coordinates on EVERY backend (the unified
    # frame contract), so build the world pose directly — env origin + per-env target — and pass it
    # as-is. No `if simulator == ...` branch: a uniform interface means the caller never special-cases.
    poses = t.zeros(ctx.n, 7, device=sim.sim_device, dtype=t.float32)
    poses[:, :3] = ctx.env_origins  # world = origin + local target (below)
    poses[:, 0] += target_x  # per-env-distinct x
    poses[:, 2] += 0.30  # lift off the floor
    poses[:, 6] = 1.0  # identity quat (xyzw)
    sim.set_static_body_pose(["pillar"], ctx.env_ids, poses)
    pillar_x = ctx.states("pillar")[:, 0]
    # get_actor_states returns WORLD frame on every backend, so the expected x is the world target.
    want_x = target_x + ctx.env_origins[:, 0]
    indexed_ok = _all((pillar_x - want_x).abs() < 1e-2)  # each env's pillar at its OWN target
    _step(sim, steps_for_seconds(sim, 1.5))
    box_z = ctx.states("freebox")[:, 2]
    fell_ok = _all(box_z < 0.10)  # support removed -> every box fell to the floor
    ok = rested_ok and indexed_ok and fell_ok
    print(
        f"  per-env-reloc: pillar_x {[round(float(v), 3) for v in pillar_x.tolist()]} "
        f"box_z {[round(float(v), 3) for v in box_z.tolist()]}  rested={rested_ok} "
        f"per_env_indexed={indexed_ok} all_fell={fell_ok}"
    )
    return ok


def assert_actor_set_states_robot_object(ctx):
    """set_actor_states writes ROBOT and OBJECTS together, uniformly, in WORLD frame.

    This is the actor-INTERSECTION counterpart to the object-only scenarios: it drives the single
    unified seam — ``set_actor_states(["robot", "obox", "obox2"], ...)`` — that treats the robot as
    just another actor (the path IsaacGym's old "Cannot set 'robot' state" guard rejected). In ONE
    name-major call over the full env range it writes per-env-distinct WORLD poses + a known
    WORLD velocity for all three actors, then asserts (every env, every backend):
      (a) each actor lands at its OWN per-env-distinct world position (no robot<->object or
          cross-env transposition in the unified index path),
      (b) the ROBOT specifically moved to its commanded world pose via set_actor_states (not only
          objects — the deleted-guard concern), with NO env_origins double-count,
      (c) the written WORLD velocity reads straight back for robot AND object (the angular check
          guards MuJoCo's freejoint body-local<->world conversion for the robot too).
    Velocity is read back BEFORE stepping so dynamics/gravity can't mask a frame error.
    """
    t = ctx.torch
    sim = ctx.sim
    if ctx.n < 2:
        print("SKIP: actor-set-states-robot-object needs num_envs>=2 (per-env-distinct targets)")
        return None
    names = ["robot", "obox", "obox2"]
    _step(sim, steps_for_seconds(sim, 0.3))  # let the scene settle to a deterministic pre-write state

    # Per-actor, per-env-distinct WORLD target. name-major / env-minor to match get/set_actor_states.
    # Each actor gets a distinct base x and each env a distinct offset, so a transposition (actor<->
    # actor or env<->env) lands at the wrong place and (a) fails.
    base_x = {"robot": 0.0, "obox": 3.0, "obox2": 6.0}
    world_lin = t.tensor([0.5, -0.3, 0.0], device=sim.sim_device)  # z=0 -> gravity is the only z term
    world_ang = t.tensor([0.4, 0.5, -0.6], device=sim.sim_device)
    rows = []
    for nm in names:
        s = t.zeros(ctx.n, 13, device=sim.sim_device, dtype=t.float32)
        s[:, :3] = ctx.env_origins  # WORLD = env origin + local target
        s[:, 0] += base_x[nm] + ctx.env_ids.to(t.float32) * 0.5  # per-actor + per-env-distinct x
        s[:, 2] += 1.5  # lift clear of the ground
        s[:, 6] = 1.0  # identity quat (xyzw)
        s[:, 7:10] = world_lin
        s[:, 10:13] = world_ang
        rows.append(s)
    states = t.cat(rows, dim=0)  # [n_names * num_envs, 13], name-major

    sim.set_actor_states(names, ctx.env_ids, states)  # <-- the unified seam, robot + objects together
    sim.write_state_updates()
    is_isaacsim = sim.get_simulator_type().value == "isaacsim"

    # (c) WORLD-frame velocity round-trip FIRST, read BEFORE stepping so the written value is exact
    # (no dynamics drift). MuJoCo (classic+warp) and IsaacGym reflect the write immediately; IsaacSim
    # refreshes its read cache (.data.root_state_w) only on a sim step, so a pre-step read is stale
    # there — skip the strict velocity read for IsaacSim (the all_root_states harness covers IsaacSim
    # velocity with its dedicated stepped, gravity-aware, frame-checked test).
    if is_isaacsim:
        lin_ok = ang_ok = True
        vel_note = "skip(isaacsim cache; covered by all_root_states harness)"
    else:
        vb = sim.get_actor_states(names, ctx.env_ids)
        lin_ok = _all((vb[:, 7:10] - world_lin).abs().amax(dim=1) < 1e-3)
        ang_ok = _all((vb[:, 10:13] - world_ang).abs().amax(dim=1) < 1e-3)
        vel_note = f"lin_ok={lin_ok} ang_ok={ang_ok}"

    # (a)+(b) Per-actor world position landed at its own per-env target. IsaacSim's pose read cache
    # is also stale until a sim step, so refresh with a couple steps there (the actors were lifted
    # 1.5 m and given zero vertical velocity, so they barely move over 2 steps). We assert on the X
    # column, which is gravity-invariant AND the per-env/per-actor-distinct discriminator (a
    # transposition lands at the wrong x), so the refresh step doesn't perturb the check.
    if is_isaacsim:
        _step(sim, 2)  # refresh IsaacSim's PhysX read cache (negligible horizontal drift)
    pos_ok = True
    robot_moved_ok = True
    for nm in names:
        want_x = base_x[nm] + ctx.env_ids.to(t.float32) * 0.5 + ctx.env_origins[:, 0]
        got = ctx.states(nm)
        landed = _all((got[:, 0] - want_x).abs() < 2e-2)
        pos_ok = pos_ok and landed
        if nm == "robot":
            # Robot rose to the lifted z via set_actor_states (a small gravity drop over the IsaacSim
            # refresh steps is well within tolerance; the point is it MOVED there, not stayed at spawn).
            robot_moved_ok = landed and _all((got[:, 2] - (ctx.env_origins[:, 2] + 1.5)).abs() < 1e-1)

    ok = pos_ok and robot_moved_ok and lin_ok and ang_ok
    print(
        f"  actor-set-states: per_actor_per_env_pos_ok={pos_ok} robot_moved_via_set_actor_states={robot_moved_ok} "
        f"vel[{vel_note}]"
    )
    return ok


# --------------------------------------------------------------------------------------------
# link_physics parity twins: a one-link robot (spawned with a link_physics) vs a free object "twin"
# carrying the same PhysicsConfig. The robot link reaches the same physical outcome as the object,
# proving the shared config drives a robot link the same way it drives an object.
# --------------------------------------------------------------------------------------------


def _warmup(ctx, n=2):
    """Step a couple of frames before sampling baselines.

    IsaacSim's robot ``root_state_w`` is cached and reads stale immediately after ``prepare_sim``;
    one warm-up step refreshes it. On the other backends a 2-step drop is sub-mm.
    """
    _step(ctx.sim, n)


def assert_galileo_twin(ctx):
    # Robot link and object twin, both dropped with matched zero damping, fall identically. The
    # warm-up step (so IsaacSim's robot root_state is live) gives the bodies a small initial
    # downward velocity, so the kinematic prediction is the full v0*t + 1/2 g t^2 rather than
    # 1/2 g t^2 from rest. Measure baseline z and vz after warm-up so the prediction stays exact.
    _warmup(ctx)
    zr0 = ctx.states("robot")[:, 2].clone()
    zt0 = ctx.states("twin")[:, 2].clone()
    vzr0 = ctx.states("robot")[:, 9].clone()  # world-frame vz (state layout: [pos3, quat4, lin3, ang3])
    n = steps_for_seconds(ctx.sim, 0.3)
    _step(ctx.sim, n)
    elapsed = n * ctx.sim.sim_dt
    drop_r = zr0 - ctx.states("robot")[:, 2]
    drop_t = zt0 - ctx.states("twin")[:, 2]
    # Velocity-aware free-fall prediction from the post-warmup baseline (downward positive).
    pred_r = -vzr0 * elapsed + 0.5 * GRAVITY * elapsed * elapsed
    # kin_ok is a sanity check (the link obeys gravity), not a link_physics-discrimination test: the
    # config here is zero damping, gravity is config-independent, and a stray 0.1 default damping bleeds
    # only ~1% over 0.3 s, inside this 5% band. The physx damping/velocity path is discriminated by
    # the `damped-fall-twin` scenario (strong damping that visibly slows the fall). This scenario's
    # claim is parity: the robot link falls like the object twin (within 2 cm), on every backend.
    kin_ok = _all(((drop_r - pred_r).abs() / pred_r) < 0.05)
    parity_ok = _all((drop_r - drop_t).abs() < 0.02)
    print(
        f"  galileo-twin: robot_drop env0 {float(drop_r[0]):.4f} twin_drop {float(drop_t[0]):.4f} "
        f"(pred {float(pred_r[0]):.4f}) kin_ok={kin_ok} parity_ok={parity_ok}"
    )
    return kin_ok and parity_ok


def assert_damped_fall_twin(ctx):
    # Robot link and object twin, both with a strong physx linear_damping, dropped from a tall height.
    # This discriminates the physx damping path (unlike galileo-twin's gravity-only fall): a damped
    # body falls measurably less than free-fall, so if link_physics.physx were a no-op the robot would
    # free-fall and `damped < free-fall` fails. Isaac-only (MuJoCo ignores the physx sub-config).
    _warmup(ctx)
    zr0 = ctx.states("robot")[:, 2].clone()
    zt0 = ctx.states("twin")[:, 2].clone()
    n = steps_for_seconds(ctx.sim, 0.3)
    _step(ctx.sim, n)
    elapsed = n * ctx.sim.sim_dt
    drop_r = zr0 - ctx.states("robot")[:, 2]
    drop_t = zt0 - ctx.states("twin")[:, 2]
    free_fall = 0.5 * GRAVITY * elapsed * elapsed  # undamped drop from rest over the window
    # (a) damping took effect: the robot link fell less than an undamped body would (the strong
    # damping more-than-halves the drop on Isaac). A no-op physx path means free-fall, which fails.
    damped_ok = _all(drop_r < 0.85 * free_fall)
    # (b) parity: the robot link's damped drop matches the object twin's within 2 cm.
    parity_ok = _all((drop_r - drop_t).abs() < 0.02)
    print(
        f"  damped-fall-twin: robot_drop env0 {float(drop_r[0]):.4f} twin_drop {float(drop_t[0]):.4f} "
        f"(free-fall {float(free_fall):.4f}) damped_ok={damped_ok} parity_ok={parity_ok}"
    )
    return damped_ok and parity_ok


def assert_restitution_twin(ctx):
    # Robot link and object twin, both dropped with restitution 0.9, rebound to a matched apex.
    # The robot's friction/restitution went on via the post-play material write; the twin carries the
    # same isaacsim material with a matching multiply combine mode, so their apexes agree.
    _warmup(ctx)
    n = steps_for_seconds(ctx.sim, 1.0)
    # Step both together, recording each one's z per step. Two separate passes over the same window
    # would desync the sim.
    t = ctx.torch
    zr_hist, zt_hist = [], []
    for _ in range(n):
        ctx.sim.refresh_sim_tensors()
        ctx.sim.simulate_at_each_physics_step()
        ctx.sim.render()
        zr_hist.append(ctx.states("robot")[:, 2].clone())
        zt_hist.append(ctx.states("twin")[:, 2].clone())
    rest_z = 0.052

    def _apex(hist):
        z = t.stack(hist, dim=0)
        out = t.zeros(ctx.n, device=z.device)
        for e in range(ctx.n):
            col = z[:, e]
            near = (col <= rest_z + 0.01).nonzero().flatten()
            first = int(near[0]) if near.numel() else int(col.argmin())
            out[e] = col[first:].max()
        return out

    apex_r = _apex(zr_hist)
    apex_t = _apex(zt_hist)
    # The robot link bounced (apex well above resting height)...
    bounced_ok = _all(apex_r > 0.30)
    # ...and rebounds to within 6 cm of the object twin (the parity check).
    parity_ok = _all((apex_r - apex_t).abs() < 0.06)
    print(
        f"  restitution-twin: robot_apex env0 {float(apex_r[0]):.3f} twin_apex {float(apex_t[0]):.3f} "
        f"bounced_ok={bounced_ok} parity_ok={parity_ok}"
    )
    return bounced_ok and parity_ok


def assert_combine_mode_collision(ctx):
    """Friction-combine-mode semantics test (IsaacSim): the combine mode changes the outcome.

    Catches a config value that reads back correctly via ``get_material_properties()`` but has no
    effect on physics (PhysX queries the property differently than expected, or IsaacSim changed how
    it ingests it). Only a physical outcome catches that.

    Two boxes with identical low per-shape friction (mu=0.1) but different IsaacSim friction combine
    modes are launched along +x onto a moderate-friction floor. PhysX combines the contact-pair
    friction by the pair's combine mode:
      - ``slide_min`` ("min" combine): effective mu = min(0.1, mu_ground) = 0.1 -> slides far.
      - ``slide_max`` ("max" combine): effective mu = max(0.1, mu_ground) = mu_ground (>> 0.1) ->
        slides short.
    So the ``min`` box slides farther than the ``max`` box. If the combine mode were a no-op (the
    regression this guards), the two would slide the same distance and the test fails: the
    unit-green-but-physics-dead case a value read-back cannot detect.

    This varies the combine mode on objects, the entity that carries it at spawn. The robot's
    post-play ``set_material_properties`` path writes only the friction/restitution values, not the
    combine mode, so a robot leg here would not isolate the combine mode. The object path is the
    vehicle for this semantics check.
    """
    zmin0 = ctx.states("slide_min")[:, 0].clone()
    zmax0 = ctx.states("slide_max")[:, 0].clone()
    # Step long enough for both to decelerate toward rest (1.2 s; the high-friction box stops early).
    _step(ctx.sim, steps_for_seconds(ctx.sim, 1.2))
    slide_min = ctx.states("slide_min")[:, 0] - zmin0
    slide_max = ctx.states("slide_max")[:, 0] - zmax0
    # Both moved forward (launch took effect, not stuck)...
    moved_ok = _all(slide_min > 0.02) and _all(slide_max > 0.02)
    # ...and the min-combine box slides farther than the max-combine box. The margin (>0.05 m) is
    # above integrator noise: if the combine mode were ignored both would carry the same effective
    # friction and slide equal distances (margin ~0).
    combine_governs = _all(slide_min > slide_max + 0.05)
    print(
        f"  combine-mode: slide_min env0 {float(slide_min[0]):.3f} slide_max {float(slide_max[0]):.3f} "
        f"moved_ok={moved_ok} min>max={combine_governs}"
    )
    return moved_ok and combine_governs


def assert_robot_combine_mode(ctx):
    """The robot link honors ``friction_combine_mode`` end-to-end (IsaacSim).

    The robot's ``link_physics.isaacsim`` is bound as a material prim at spawn
    (``_bind_robot_link_material``), so the combine mode reaches PhysX, unlike a per-shape value
    write, which carries friction/restitution values but not the mode.

    The robot link carries a "min"-combine material and the object ``twin`` a "max"-combine material,
    same per-shape friction, both launched +x on the same floor. PhysX pairs each with the floor:
      - robot (min): effective mu = min(0.1, mu_ground) = 0.1 -> slides far.
      - twin  (max): effective mu = max(0.1, mu_ground) = mu_ground (>> 0.1) -> slides short.
    So the robot slides farther than the twin. If the combine mode failed to reach the robot (the
    regression this guards, the property silently defaulting), the robot would carry the floor's
    effective friction and slide like the twin (margin ~0).

    Velocity is seeded via set_actor_states (robot init-velocity is dropped on Isaac), then both
    decelerate. The robot's material and combine came from the spawn bind; the twin's from its scene
    cfg.
    """
    sim = ctx.sim
    names = ["robot", "twin"]
    seed = sim.get_actor_states(names, ctx.env_ids).clone()  # [n_actors*n_env, 13], name-major
    n_env = ctx.n
    for i in range(len(names)):
        block = slice(i * n_env, (i + 1) * n_env)
        seed[block, 2] = 0.06  # pos_z: resting height on the floor
        seed[block, 7] = 2.0  # lin_vel_x: +x launch (state layout [pos3, quat4, lin3, ang3])
        seed[block, 8:13] = 0.0  # zero lin_y/lin_z + all angular
    # set_actor_states signature is (names, env_ids, states), not (names, states, env_ids).
    sim.set_actor_states(names, ctx.env_ids, seed)
    _step(sim, 1)
    x0_r = sim.get_actor_states(["robot"], ctx.env_ids)[:, 0].clone()
    x0_t = sim.get_actor_states(["twin"], ctx.env_ids)[:, 0].clone()
    _step(sim, steps_for_seconds(sim, 1.2))
    slide_r = sim.get_actor_states(["robot"], ctx.env_ids)[:, 0] - x0_r
    slide_t = sim.get_actor_states(["twin"], ctx.env_ids)[:, 0] - x0_t
    moved_ok = _all(slide_r > 0.02) and _all(slide_t > 0.02)
    # robot (min combine) must slide clearly farther than twin (max combine).
    combine_reaches_robot = _all(slide_r > slide_t + 0.05)
    print(
        f"  robot-combine-mode: robot_slide(min) env0 {float(slide_r[0]):.3f} twin_slide(max) {float(slide_t[0]):.3f} "
        f"moved_ok={moved_ok} robot>twin={combine_reaches_robot}"
    )
    return moved_ok and combine_reaches_robot


# =================================================================================================
# SCENARIO CATALOG — what each behavioral scenario is meant to SHOW (the physics it proves, and so
# what to look for in its recorded video). Each asserts a physical OUTCOME after stepping, in every
# env; the assertion fn holds the exact thresholds.
#
#   collide-into-fixed     A driven box hits a FIXED wall and comes to rest against its face — proves
#                          a static/fixed body is solid and stops a moving body (no pass-through, no
#                          sink-in). Video: box slides in, stops flush against the wall.
#   momentum-transfer      A moving "striker" box hits a resting "target" box; the target launches
#                          forward and the striker gives up its momentum — proves dynamic-vs-dynamic
#                          collision and momentum transfer. Video: one box hits another, both move.
#   static-support         A box dropped onto a fixed POST comes to rest ON TOP (center at post_top +
#                          half-extent), not penetrating and not sliding off — proves a static body
#                          supports a resting load. Video: box falls, lands on the post, sits.
#   friction-slide         Two boxes given equal initial speed; the LOW-friction box slides farther
#                          than the HIGH-friction one — proves configured friction governs sliding.
#                          Video: two boxes slide, one travels noticeably farther.
#   dr-friction-governs    Same as friction-slide, but friction is set via the runtime DOMAIN
#                          RANDOMIZATION term — proves DR-applied friction takes effect on every
#                          backend. Video: the DR-low box outruns the DR-high box.
#   dr-damping-governs     A DR-damped box loses speed faster than an undamped control — proves
#                          DR-applied linear DAMPING governs decay. (No IsaacGym: it has no runtime
#                          free-body damping setter.) Video: damped box slows visibly more.
#   galileo-freefall       A light and a heavy box free-fall identically along z = ½gt² — proves
#                          gravity integration + mass-independence (and that stray default damping is
#                          zeroed). Video: two boxes fall together at the same rate.
#   damping-decay          A damped box vs an undamped control, both launched and kept AIRBORNE — the
#                          damped one loses more forward speed (drag, not floor friction). Video: two
#                          airborne boxes, the damped one slows faster.
#   restitution-bounce     A box dropped onto a bouncy ground rebounds to ~e²·height (e≈0.9) — proves
#                          configured restitution drives the bounce. Video: box drops, bounces high.
#   angular-velocity-spin  A box given a commanded angular velocity about z rotates by the expected
#                          angle with no off-axis tumble — proves angular-velocity init + integration.
#                          Video: box spins in place about the vertical axis.
#   velocity-restore-reset A box is stepped, then RESET via get_actor_initial_poses->set_actor_states;
#                          its world spawn pose AND configured initial velocity are restored, and it
#                          moves again — proves the reset seam round-trips pose+velocity. Video: box
#                          drifts, snaps back to spawn, then moves off again.
#   object-obs-robot-pose  With the robot at a per-env-distinct yaw, the object observation terms
#                          (pos/lin_vel/ang_vel/quat in the robot BASE frame) match an independent
#                          closed-form transform — proves object obs are computed in the right frame
#                          per env. (Mostly a numeric check; the video is a short static settle.)
#   pose-jitter-settle     jitter_object_pose_on_reset places the single box at a per-env-distinct
#                          XY+yaw within the configured ranges — proves reset jitter is per-env
#                          independent (not broadcast) on both XY and yaw. The payoff is the spread
#                          ACROSS envs, which the single-env (record_env_id=0) video can't show:
#                          the clip is just one box jittered off its spawn point, then settling.
#   loader-invariance      A free box loaded STANDALONE free-falls along the analytic curve — proves
#                          the standalone-asset load path yields correct dynamics. Video: box falls.
#   loader-invariance-file Same analytic free-fall, but the body comes from a 1->N SCENE FILE — proves
#                          the file-expanded load path matches the standalone one. Video: box falls.
#   multibody-independence A 1->N scene file spawns a FREE box + a STATIC post at the authored offset;
#                          the free body falls while the static sibling holds — proves multi-body files
#                          spawn at the right relative pose with independent dynamics. Video: one body
#                          falls, the other stays put.
#   per-env-relocation     A box rests on a pillar; the pillar is kinematically RELOCATED to a
#                          per-env-distinct target, removing support, and every box falls — proves
#                          per-env-indexed runtime relocation (set_static_body_pose) + support removal.
#                          Video: pillar jumps away, box drops to the floor.
#   actor-set-states-robot-object  ROBOT + two free objects written TOGETHER in one
#                          set_actor_states call to per-env-distinct WORLD poses + a known WORLD
#                          velocity — proves the unified actor seam treats the robot as just another
#                          actor (the path IsaacGym's old "Cannot set 'robot' state" guard rejected):
#                          each lands at its own per-env target (no transposition), the robot moved
#                          via set_actor_states (no origin double-count), and world velocity (incl.
#                          the freejoint body-local<->world angular conversion) round-trips.
#                          Video: robot + both boxes jump to distinct lifted poses.
#
# scenario key -> (scene preset key, assertion fn, min_envs, mujoco_supported, unsupported_backends)
# unsupported_backends: backends that statically cannot run the scenario (skip with a reason).
# =================================================================================================
SCENARIOS = {
    "collide-into-fixed": ("collide-into-fixed", assert_collide_into_fixed, 1, True, ()),
    "momentum-transfer": ("momentum-transfer", assert_momentum_transfer, 1, True, ()),
    "static-support": ("static-support", assert_static_support, 1, True, ()),
    "friction-slide": ("friction-slide", assert_friction_slide, 1, True, ()),
    "dr-friction-governs": ("dr-friction-pair", assert_dr_friction_governs, 1, True, ()),
    # IsaacGym has no runtime free-body damping setter (damping is import-time AssetOptions only);
    # MuJoCo (dof_damping) and IsaacSim (live USD PhysxRigidBodyAPI damping) support it.
    "dr-damping-governs": ("dr-damping-pair", assert_dr_damping_governs, 1, True, ("isaacgym",)),
    "galileo-freefall": ("galileo-freefall", assert_galileo_freefall, 1, True, ()),
    # Isaac-only: physx config-time linear_damping drives the drag. There is no static MuJoCo
    # freejoint-damping config; runtime damping DR is covered by dr-damping-governs.
    "damping-decay": ("damping-decay", assert_damping_decay, 1, False, ()),
    "restitution-bounce": ("restitution-bounce", assert_restitution_bounce, 1, True, ()),
    "angular-velocity-spin": ("angular-spin", assert_angular_spin, 1, True, ()),
    "velocity-restore-reset": ("velocity-restore", assert_velocity_restore_on_reset, 1, True, ()),
    "object-obs-robot-pose": ("obs-under-pose", assert_object_obs_under_pose, 2, True, ()),
    "pose-jitter-settle": ("jitter-settle", assert_pose_jitter_settle, 2, True, ()),
    "loader-invariance": ("loader-invariance-freefall", assert_loader_invariance, 1, True, ()),
    "loader-invariance-file": ("loader-invariance-multibody", assert_loader_invariance, 1, True, ()),
    "multibody-independence": ("multibody", assert_multibody_independence, 1, True, ()),
    "per-env-relocation": ("per-env-relocation", assert_per_env_relocation, 2, True, ()),
    "actor-set-states-robot-object": ("obs-under-pose", assert_actor_set_states_robot_object, 2, True, ()),
    # link_physics robot/object parity twins (one-link robot vs free object with the same config).
    # Drop-from-rest, so no robot init velocity is needed (which Isaac drops). Not on IsaacGym: the
    # one-link robot is a 0-DOF floating-base actor, which IsaacGym cannot represent as an
    # articulation ("Cannot get mass matrix for actors without DOFs", a null DOF tensor crash in
    # prepare_sim). Runs on MuJoCo (classic + warp) and IsaacSim.
    "galileo-twin": ("galileo-twin", assert_galileo_twin, 1, True, ("isaacgym",)),
    "restitution-twin": ("restitution-twin", assert_restitution_twin, 1, True, ("isaacgym",)),
    # damped-fall-twin: strong physx damping visibly slows the fall (discriminates the physx path,
    # which galileo-twin's zero-damping fall cannot). Isaac-only: MuJoCo ignores the physx sub-config;
    # IsaacGym can't run the one-link robot.
    "damped-fall-twin": ("damped-fall-twin", assert_damped_fall_twin, 1, False, ("isaacgym",)),
    # Friction-combine-mode semantics (IsaacSim): two objects, identical friction, different combine
    # modes give different slide distance. Guards the unit-green-but-physics-noop case. Pure objects
    # (the entity that carries a combine mode at spawn). Isaac-only: MuJoCo has no physx combine mode,
    # and IsaacGym's RigidShapeProperties has no per-pair friction combine-mode field either.
    "combine-mode-collision": ("combine-mode", assert_combine_mode_collision, 1, False, ("isaacgym",)),
    # robot-link combine mode: a robot link honors friction_combine_mode (its link_physics is bound
    # as a material prim at spawn). Twin scenario (robot min vs object max). Isaac-only and
    # IsaacGym-unsupported (same reasons as combine-mode-collision).
    "robot-combine-mode": ("robot-combine-mode", assert_robot_combine_mode, 1, False, ("isaacgym",)),
}


# Scenarios that spawn the one-link `onelink-box` robot (with a per-scenario link_physics) instead of
# the default g1, paired with an object `twin` carrying the same PhysicsConfig. Maps scenario to the
# link_physics applied to the robot, which matches the twin scene's object physics. Built lazily in
# main() after the config import so PhysicsConfig is available without a module-level sim dependency.
def _twin_robot_link_physics():
    from holosoma.config_types.scene import (
        IsaacGymPhysicsConfig,
        IsaacSimPhysicsConfig,
        MujocoPhysicsConfig,
        PhysicsConfig,
        PhysXPhysicsConfig,
    )

    # Mirror the twin scene-object PhysicsConfig so robot and twin are physically identical.
    galileo = PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0))
    # Strong physx damping; matches damped_fall_twin's twin object (_DAMPED_FALL_DAMPING=3.0).
    damped_fall = PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=3.0, angular_damping=0.0))
    restitution = PhysicsConfig(
        isaacgym=IsaacGymPhysicsConfig(friction=0.5, restitution=0.9),
        # Matches the restitution-twin scene object, including the "max" combine mode (which the
        # robot carries via the spawn material bind), so robot and twin rebound identically.
        isaacsim=IsaacSimPhysicsConfig(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.9, restitution_combine_mode="max"
        ),
        mujoco=MujocoPhysicsConfig(friction=[0.5, 0.005, 0.001], solref=[-3000.0, -3.0]),
    )
    # robot-combine-mode: the robot link carries a "min"-combine material (same low friction as its
    # "max"-combine twin). The bound material prim makes the combine mode reach PhysX.
    robot_combine = PhysicsConfig(
        isaacsim=IsaacSimPhysicsConfig(static_friction=0.1, dynamic_friction=0.1, friction_combine_mode="min"),
    )
    # combine-mode-collision is not here: it is a pure-object scenario (two objects with different
    # combine modes), so it uses the default robot and needs no twin.
    return {
        "galileo-twin": galileo,
        "damped-fall-twin": damped_fall,
        "restitution-twin": restitution,
        "robot-combine-mode": robot_combine,
    }


# Scenario -> hook run AFTER create_envs but BEFORE prepare_sim (the window IsaacGym honors a
# friction write and before Warp captures its step graph). Signature: (sim, env_ids, torch).
_PRE_PREPARE = {
    "dr-friction-governs": _pre_prepare_dr_friction,
    "dr-damping-governs": _pre_prepare_dr_damping,
}


def main() -> int:
    # Register the test-only scene and robot presets into the production DEFAULTS so the scenario's
    # scene/robot keys resolve through the normal tyro path inside this (sub)process. Core never
    # imports these; tests register them.
    _scene_presets.register()
    from tests.simulators import _robot_presets

    _robot_presets.register()

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="override the scenario's computed step count")
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    parser.add_argument("--record", default=None, metavar="DIR", help="save a video of the run to DIR")
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    args = parser.parse_args()
    headless = args.headless == "true"

    scene_key, assertion_fn, min_envs, mujoco_ok, unsupported_backends = SCENARIOS[args.scenario]

    # Statically-knowable unsupported combos. The pytest wrappers already mark these as skips (so
    # they never spawn a subprocess); these guards are the DEFENSIVE fallback for a direct harness
    # invocation with an unsupported combo — they exit with SKIP_EXIT_CODE (translated to a real
    # pytest.skip by run_harness) rather than a silent pass, and bail before any sim build.
    is_mujoco = args.simulator in ("mujoco", "mjwarp")
    if not mujoco_ok and is_mujoco:
        print(f"SKIP: scenario '{args.scenario}' is Isaac-only (no equivalent on MuJoCo)")
        return SKIP_EXIT_CODE
    if args.simulator in unsupported_backends:
        print(f"SKIP: scenario '{args.scenario}' is unsupported on {args.simulator} (no runtime API for this feature)")
        return SKIP_EXIT_CODE
    if args.simulator == "mujoco" and (min_envs > 1 or args.num_envs > 1):
        print(f"SKIP: scenario '{args.scenario}' needs multi-env; MuJoCo ClassicBackend is single-env (use mjwarp)")
        return SKIP_EXIT_CODE
    if min_envs > 1 and args.num_envs < min_envs:
        print(f"SKIP: scenario '{args.scenario}' needs --num-envs>={min_envs} (got {args.num_envs})")
        return SKIP_EXIT_CODE

    # Twin scenarios spawn the one-link `onelink-box` robot (not g1) carrying a per-scenario
    # link_physics that mirrors the twin object's PhysicsConfig. A non-twin scenario keeps --robot.
    twin_link_physics = _twin_robot_link_physics()
    robot_key = "onelink-box" if args.scenario in twin_link_physics else args.robot

    config = _build_run_sim_config(args.simulator, scene_key, robot_key, args.terrain, args.record)
    device = "cuda:0" if args.simulator != "mujoco" else "cpu"
    config = dataclasses.replace(
        config,
        device=device,
        training=dataclasses.replace(config.training, num_envs=args.num_envs),
    )

    # Apply the scenario's link_physics onto the one-link robot's asset, so the robot link carries the
    # same PhysicsConfig the twin object does (replace the nested frozen asset).
    if args.scenario in twin_link_physics:
        lp = twin_link_physics[args.scenario]
        config = dataclasses.replace(
            config,
            robot=dataclasses.replace(config.robot, asset=dataclasses.replace(config.robot.asset, link_physics=lp)),
        )

    # The restitution bounce needs a BOUNCY GROUND too: PhysX combines contact-pair restitution
    # (geometric-mean-like), and the terrain restitution defaults to 0 -> combine(0.9, 0) = 0, no
    # bounce on the Isaac backends (IsaacGym PlaneParams.restitution / IsaacSim ground material).
    # Give the ground high restitution for this scenario so the box's restitution registers. NOTE:
    # restitution lives at terrain.terrain_term.restitution — TerrainManagerCfg itself has no such
    # field, so a flat dataclasses.replace(config.terrain, restitution=0.9) is a SILENT no-op (the
    # ground stays at 0.0). Must replace the nested term. (MuJoCo bounces via the box's negative-form
    # solref, independent of ground restitution, so this is harmless there.) restitution-twin needs
    # the same bouncy ground (the robot link bounces against it just like the object); combine-mode-
    # collision reuses the same restitution-bearing material and scene.
    if args.scenario in ("restitution-bounce", "restitution-twin", "combine-mode-collision"):
        config = dataclasses.replace(
            config,
            terrain=dataclasses.replace(
                config.terrain,
                terrain_term=dataclasses.replace(config.terrain.terrain_term, restitution=0.9),
            ),
        )

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(headless)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

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

    # Pre-prepare hook: some scenarios must mutate physics AFTER actors exist (create_envs) but
    # BEFORE finalization (prepare_sim) — IsaacGym only applies set_actor_rigid_shape_properties
    # (friction) in that window, and Warp captures its step graph in put_model. Runtime DR after
    # prepare_sim is a no-op on those backends. Hooks run on every backend (harmless where the
    # write also works post-prepare).
    pre_hook = _PRE_PREPARE.get(args.scenario)
    if pre_hook is not None:
        pre_env_ids = torch.arange(n, device=sim.sim_device)
        pre_hook(sim, pre_env_ids, torch)

    sim.prepare_sim()

    if not headless:
        sim.setup_viewer()
    if sim.video_recorder is not None:
        sim.video_recorder.setup_recording()
        # These scenarios are object-centric; aim the camera at the scene objects, not the robot.
        _focus_camera_on_objects(sim, config.scene)
        sim.video_recorder.on_episode_start(env_id=sim.video_recorder.config.record_env_id)

    env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
    free_names = sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    static_names = sim.object_registry.get_names_by_type(ObjectType.SCENE)
    mode = "headless" if headless else "headful"
    print(f"[{args.simulator}/{mode}] scenario={args.scenario} free={free_names} static={static_names} num_envs={n}")

    # Boot-time guard for the twin scenarios: the one-link robot and its object twin must both exist.
    # Fail here rather than let a missing actor (e.g. a silent fallback to g1, or an unregistered
    # robot key) surface as an assertion failure deep in the scenario.
    if args.scenario in twin_link_physics:
        assert config.robot.asset.robot_type == "onelink_box", (
            f"twin scenario '{args.scenario}' must run the onelink-box robot, got "
            f"'{config.robot.asset.robot_type}' (did _robot_presets.register() run?)"
        )
        assert "twin" in free_names, (
            f"twin scenario '{args.scenario}' is missing the 'twin' object (free={free_names}); "
            f"the twin scene preset did not load."
        )

    ctx = _Ctx(sim, config, args, env_origins, env_ids, torch)
    result = assertion_fn(ctx)
    if result is None:
        # Scenario self-skipped at runtime (it printed its own SKIP line). Signal a real skip via
        # the autotools exit code so run_harness reports it as pytest.skip, never a pass.
        code = SKIP_EXIT_CODE
        passed = None
    else:
        passed = bool(result)
        code = 0 if passed else 1
        print("PASS" if passed else "FAIL")

    if sim.video_recorder is not None:
        sim.video_recorder.on_episode_end(env_id=sim.video_recorder.config.record_env_id)
        print(f"VIDEO saved under {args.record}")

    if passed and args.result_file:
        with open(args.result_file, "w") as f:
            f.write("OK")

    viewer = getattr(sim, "viewer", None)
    if viewer is not None and args.simulator in ("mujoco", "mjwarp"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
