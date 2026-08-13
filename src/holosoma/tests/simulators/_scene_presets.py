"""Test-only scene presets — NOT shipped on the production CLI.

These presets exist purely to drive the cross-backend test harnesses (``scene_spawn_assert``,
``behavior_assert``, ``_dr_matrix``). They are deliberately kept OUT of
``holosoma.config_values.scene`` so the production ``scene:`` subcommand menu (built from
``scene.DEFAULTS``) advertises only real, shipped scenes (``empty``, ``object-managers-demo``,
``g1_29dof_wbt_object``).

Two ways the harnesses consume them:

- BY OBJECT: a harness that builds its config in-process imports the preset as a plain attribute
  (e.g. ``_dr_matrix`` uses :data:`free_and_static`) and injects it via ``dataclasses.replace``.
- BY STRING: a harness that resolves ``--scene <key>`` / ``--scenario`` through tyro calls
  :func:`register` first, which merges :data:`TEST_PRESETS` into ``scene.DEFAULTS``. Because the
  config types use deferred annotations (``from __future__ import annotations``), tyro reads
  ``scene.DEFAULTS`` LAZILY at ``tyro.cli()`` time, so a key registered here resolves through the
  EXACT SAME production tyro path — no separate resolution branch. The dependency stays strictly
  tests -> core: core never imports this module.
"""

from __future__ import annotations

from holosoma.config_types.scene import (
    IsaacGymPhysicsConfig,
    IsaacSimPhysicsConfig,
    MujocoPhysicsConfig,
    ObjectPatternConfig,
    PhysicsConfig,
    PhysXPhysicsConfig,
    RigidObjectConfig,
    SceneConfig,
    SceneFileConfig,
)

# Box asset paths. Duplicated from config_values/scene.py (which keeps its own copies for the
# production presets) rather than imported, so this test module does not couple to a core private.
_LARGE_BOX = "holosoma/data/scene_objects/boxes/large_box.urdf"
_SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
_SMALL_BOX_USD = "holosoma/data/scene_objects/boxes/small_box.usda"

# A single free-body box (the shipped WBT large-box asset). y=1.5 keeps it clear of the
# un-actuated robot at the origin: at multi-env the robot collapses under gravity, and a box
# directly beneath it is nudged UP by the falling limbs so its per-env min-z never dips below
# spawn (the multi-env fall check then fails in that one env). Offsetting isolates the free-fall.
g1_largebox = SceneConfig(
    rigid_objects={
        "box": RigidObjectConfig(
            urdf_file=_LARGE_BOX,
            position=[0.0, 1.5, 0.5],
        )
    }
)

# One static box — holds its pose, does not fall. (urdf form)
static_box = SceneConfig(
    rigid_objects={"pillar": RigidObjectConfig(urdf_file=_SMALL_BOX, position=[0.6, 0.0, 0.3], fixed=True)}
)

# One static box from a USD asset — exercises the static (kinematic) USD path on IsaacSim.
static_box_usd = SceneConfig(
    rigid_objects={"pillar": RigidObjectConfig(usd_file=_SMALL_BOX_USD, position=[0.6, 0.0, 0.3], fixed=True)}
)

# An arbitrary mix of free + static bodies, to exercise both paths together. Free bodies sit at
# y=1.5/1.8 to clear the un-actuated robot at the origin (see g1_largebox): a free box beneath the
# collapsing robot is nudged up, failing the multi-env fall check. The static pillar holds its pose.
free_and_static = SceneConfig(
    rigid_objects={
        "free0": RigidObjectConfig(urdf_file=_SMALL_BOX, position=[0.0, 1.5, 0.6]),
        "free1": RigidObjectConfig(urdf_file=_SMALL_BOX, position=[0.0, 1.8, 0.6]),
        "pillar": RigidObjectConfig(urdf_file=_SMALL_BOX, position=[0.7, 0.0, 0.3], fixed=True),
    }
)

# A single free-body box from a USD asset — exercises the USD format path (IsaacSim).
usd_box = SceneConfig(rigid_objects={"box": RigidObjectConfig(usd_file=_SMALL_BOX_USD, position=[0.4, 0.0, 0.6])})

# A single free box given a known non-zero initial linear + angular velocity. Drives the
# velocity wiring on every backend: the box should read its configured velocity back
# immediately, translate along +x/+y (gravity-independent axes) and visibly spin about z.
# Tri-format (urdf for IsaacGym, usd for IsaacSim, urdf->xml selected for MuJoCo) so the
# same preset runs under all backends. Initial orientation is identity so the world-frame
# config velocity and each backend's body-local read-back frame coincide.
#
# y=1.5 keeps the box clear of the robot, which spawns at [0,0,0.8]. The harness's robot is
# un-actuated (no policy), so it collapses under gravity; with the box directly beneath it the
# falling limbs strike the box and contaminate its velocity-driven motion with a large, per-env-
# and per-run-chaotic displacement (observed 0.02-0.6 m spread). Offsetting the box isolates the
# free-body motion this scene is meant to verify, making the displacement deterministic.
velocity_box = SceneConfig(
    rigid_objects={
        "vbox": RigidObjectConfig(
            urdf_file=_LARGE_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.0, 1.5, 0.6],
            linear_velocity=[1.0, 0.5, 0.0],
            angular_velocity=[0.0, 0.0, 3.0],
        )
    }
)

# A free box carrying an explicit per-object physics override: a known mass (cross-backend
# core) plus each backend's own friction sub-config.
physics_box = SceneConfig(
    rigid_objects={
        "pbox": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.0, 1.5, 0.6],  # y=1.5 clears the origin robot's collapse footprint (see g1_largebox)
            physics=PhysicsConfig(
                mass=3.0,
                isaacgym=IsaacGymPhysicsConfig(friction=0.4),
                isaacsim=IsaacSimPhysicsConfig(static_friction=0.4, dynamic_friction=0.4),
                mujoco=MujocoPhysicsConfig(friction=[0.4, 0.01, 0.001]),
            ),
        )
    }
)


def _friction_box(y: float, friction: float) -> RigidObjectConfig:
    """A free box at ``[0, y, 0.07]`` with a +x push and a per-backend friction override.

    The friction value is set on every backend's sub-config so the same box behaves the same
    regardless of which backend loads it. The two ``friction_slide`` boxes differ ONLY in this
    value, so their relative slide distance isolates the contact-friction effect.
    """
    return RigidObjectConfig(
        urdf_file=_SMALL_BOX,
        usd_file=_SMALL_BOX_USD,
        position=[0.0, y, 0.07],
        linear_velocity=[2.0, 0.0, 0.0],
        physics=PhysicsConfig(
            isaacgym=IsaacGymPhysicsConfig(friction=friction),
            isaacsim=IsaacSimPhysicsConfig(static_friction=friction, dynamic_friction=friction),
            mujoco=MujocoPhysicsConfig(friction=[friction, 0.005, 0.001]),
        ),
    )


# Two identical boxes pushed along +x, differing ONLY in friction (low vs high). A behavioral
# probe: a low-friction box must slide measurably farther than a high-friction one, proving the
# configured friction governs contact dynamics (not merely that it is authored on the asset).
# The values straddle a typical ground friction so the low<high slide ordering holds under any
# friction-combine mode. Boxes sit off to either side in y, clear of the robot at the origin.
friction_slide = SceneConfig(
    rigid_objects={
        # Wide spread (near-frictionless vs very grippy) so the slide-distance gap is large and
        # unambiguous within the harness window — the low box keeps gliding while the high box
        # arrests almost immediately. The behavioral probe needs a clear separation, not just >.
        "box_lowfric": _friction_box(y=2.0, friction=0.02),
        "box_highfric": _friction_box(y=-2.0, friction=5.0),
    }
)

# A multi-body scene FILE (1->N): one file expands to a free box + a static post at a
# known +0.5m-x relative offset. Tri-format so each backend loads its own (MuJoCo xml,
# IsaacGym urdf, IsaacSim usd); registered bodies are 'scene_free_box' (free) and
# 'scene_static_post' (static). Placed at a non-trivial world pose to exercise composition.
# y=1.5 clears the un-actuated harness robot at the origin (which collapses under gravity and
# would strike the free body, contaminating the fall check — see g1_largebox); the +0.5m-x
# body-to-body offset the tests assert is translation-invariant, so the move is inert to it.
multibody = SceneConfig(
    scene_files={
        "scene": SceneFileConfig(
            xml_path="holosoma/data/scene_objects/multibody/multibody.xml",
            urdf_path="holosoma/data/scene_objects/multibody/multibody.urdf",
            usd_path="holosoma/data/scene_objects/multibody/multibody.usda",
            position=[0.4, 1.5, 0.6],
        )
    }
)

# Same multi-body file, but object_configs FLIPS each body's type relative to the file:
# the structurally-free 'free_box' is forced static, the structurally-static 'static_post'
# is forced free. Exercises the per-object override path on every backend.
multibody_override = SceneConfig(
    scene_files={
        "scene": SceneFileConfig(
            xml_path="holosoma/data/scene_objects/multibody/multibody.xml",
            urdf_path="holosoma/data/scene_objects/multibody/multibody.urdf",
            usd_path="holosoma/data/scene_objects/multibody/multibody.usda",
            position=[0.4, 1.5, 0.6],  # y=1.5 clears the origin robot's collapse footprint (see multibody)
            object_configs={
                "free_box": ObjectPatternConfig(fixed=True),
                "static_post": ObjectPatternConfig(fixed=False),
            },
        )
    }
)


_RUBBER_DUCK = "holosoma/data/scene_objects/rubber_duck/rubber_duck.urdf"
_RUBBER_DUCK_XML = "holosoma/data/scene_objects/rubber_duck/rubber_duck.xml"
_LARGE_RUBBER_DUCK = "holosoma/data/scene_objects/rubber_duck/large_rubber_duck.urdf"
_LARGE_RUBBER_DUCK_XML = "holosoma/data/scene_objects/rubber_duck/large_rubber_duck.xml"
_HUGE_RUBBER_DUCK = "holosoma/data/scene_objects/rubber_duck/huge_rubber_duck.urdf"
_HUGE_RUBBER_DUCK_XML = "holosoma/data/scene_objects/rubber_duck/huge_rubber_duck.xml"


def _falling_duck(
    urdf: str, xml: str, spawn_height: float, impact_time: float, impact_height: float, spin: float = 0.3
) -> RigidObjectConfig:
    """A duck tossed straight up from (0, 0, spawn_height) that falls back to impact_height at
    the origin after impact_time seconds. The upward throw buys the wait without an absurd spawn
    height (a 5 s delay needs vz ~25 m/s, not a 30 m drop). vz from z0 + vz*t - g*t^2/2 = z_hit.
    """
    _GRAVITY = 9.81  # m/s^2; matches the simulators' default downward acceleration.

    vz = (impact_height - spawn_height) / impact_time + 0.5 * _GRAVITY * impact_time
    return RigidObjectConfig(
        urdf_file=urdf,
        xml_file=xml,
        position=[0.0, 0.0, spawn_height],
        linear_velocity=[0.0, 0.0, vz],
        angular_velocity=[0.0, 0.0, spin],  # gentle tumble; high spin drifts the duck off-center
    )


# DUCKNADO: rubber ducks rain straight down onto the robot at the origin. An unreferenced showcase
# preset (no automated test consumes it) kept reachable via ``--scene ducknado``.
#   1. A large static duck stands guard nearby.
#   2. A cluster of regular ducks drops, staggered in time and impact height (first hit past 5 s).
#   3. Two large ducks follow, each on its own beat.
#   4. A dramatic pause.
#   5. A huge duck plummets dead-center for the finale.
# All ducks spawn at (0, 0) but at DISTINCT heights in a vertical column — overlapping spawns make
# the solver eject bodies sideways, so they'd miss. vx=vy=0 means each falls straight back down.
def _ducknado_scene() -> SceneConfig:
    robot_height = 1.5
    column_z = 3.0  # running spawn height, bumped per duck so nothing overlaps at t=0
    ducks: dict[str, RigidObjectConfig] = {}

    # Cluster: 8 regular ducks, impacts 5.2 -> 7.0 s, heights swept feet -> head.
    num_ducks, ring_t0, ring_t1 = 8, 5.2, 7.0
    for i in range(num_ducks):
        impact_time = ring_t0 + (ring_t1 - ring_t0) * (i / (num_ducks - 1))
        impact_height = robot_height * (i + 0.5) / num_ducks
        ducks[f"duck_{i}"] = _falling_duck(_RUBBER_DUCK, _RUBBER_DUCK_XML, column_z, impact_time, impact_height)
        column_z += 0.4

    # Two large ducks, solo beats at 8.0 and 9.0 s.
    column_z += 0.6
    for j in range(2):
        impact_height = robot_height * (0.85 if j == 0 else 0.4)
        ducks[f"large_duck_{j}"] = _falling_duck(
            _LARGE_RUBBER_DUCK,
            _LARGE_RUBBER_DUCK_XML,
            column_z,
            ring_t1 + 1.0 + j,
            impact_height,
        )
        column_z += 0.8

    # Pause, then the huge finale dead-center at 12 s.
    column_z += 2.0
    ducks["finale_huge_duck"] = _falling_duck(
        _HUGE_RUBBER_DUCK, _HUGE_RUBBER_DUCK_XML, column_z, ring_t1 + 2.0 + 3.0, robot_height
    )

    # Static set-piece duck standing guard (no velocity allowed on fixed bodies).
    large_static_duck = RigidObjectConfig(
        urdf_file=_LARGE_RUBBER_DUCK,
        xml_file=_LARGE_RUBBER_DUCK_XML,
        position=[1.0, 1.0, 0.0],
        fixed=True,
    )

    return SceneConfig(rigid_objects={"large_static_duck": large_static_duck, **ducks})


ducknado = _ducknado_scene()


# =====================================================================================
# Behavioral-test presets (consumed by tests/simulators/behavior_assert.py).
#
# Each isolates ONE physical invariant so the harness can assert a behavioral OUTCOME
# (a body stops / rests / bounces / spins) rather than a value read-back. Objects sit
# >=1.4 m from the robot at the origin so the un-actuated robot collapsing during the
# window never perturbs them. Tri-format (urdf+usd; MuJoCo derives xml from urdf) so the
# same preset runs on every backend. HALF=0.05 (0.1 m cube), g=9.81, dt=1/200=0.005.
# =====================================================================================

# #1 collide-into-fixed: a low-friction box slides +x into a static wall; it must advance, come
# to rest (stopped, not tunneled through), and the wall must not move.
collide_into_fixed = SceneConfig(
    rigid_objects={
        "hammer": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.4, 0.0, 0.05],
            linear_velocity=[2.0, 0.0, 0.0],
            physics=PhysicsConfig(
                isaacgym=IsaacGymPhysicsConfig(friction=0.05),
                isaacsim=IsaacSimPhysicsConfig(static_friction=0.05, dynamic_friction=0.05),
                mujoco=MujocoPhysicsConfig(friction=[0.05, 0.005, 0.001]),
            ),
        ),
        "wall": RigidObjectConfig(urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=[1.7, 0.0, 0.05], fixed=True),
    }
)

# #2 momentum-transfer: a moving striker hits a resting target (equal mass); the target must be
# set in motion +x and the striker must not pass through it. Qualitative transfer (cubes, not
# spheres — no exact velocity swap asserted).
momentum_transfer = SceneConfig(
    rigid_objects={
        "striker": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.4, 0.0, 0.05],
            linear_velocity=[2.0, 0.0, 0.0],
            physics=PhysicsConfig(
                isaacgym=IsaacGymPhysicsConfig(friction=0.05),
                isaacsim=IsaacSimPhysicsConfig(static_friction=0.05, dynamic_friction=0.05),
                mujoco=MujocoPhysicsConfig(friction=[0.05, 0.005, 0.001]),
            ),
        ),
        "target": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.55, 0.0, 0.05],
            physics=PhysicsConfig(
                isaacgym=IsaacGymPhysicsConfig(friction=0.05),
                isaacsim=IsaacSimPhysicsConfig(static_friction=0.05, dynamic_friction=0.05),
                mujoco=MujocoPhysicsConfig(friction=[0.05, 0.005, 0.001]),
            ),
        ),
    }
)

# #3 static-support: a box dropped onto a static post must come to rest ON the post (z ~= 0.40,
# post top 0.35), clearly above the floor, and the post must not move.
static_support = SceneConfig(
    rigid_objects={
        "restbox": RigidObjectConfig(urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=[1.5, 0.0, 0.45]),
        "post": RigidObjectConfig(urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=[1.5, 0.0, 0.30], fixed=True),
    }
)

# #5 dr-friction-governs: two identical boxes pushed +x with NO preset friction — the DR term
# sets low/high friction at runtime, and the low-friction box must slide farther. Proves the DR
# chokepoint governs contact dynamics, not just the model field.
dr_friction_pair = SceneConfig(
    rigid_objects={
        "dr_lo": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.0, 2.0, 0.07],
            linear_velocity=[2.0, 0.0, 0.0],
        ),
        "dr_hi": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.0, -2.0, 0.07],
            linear_velocity=[2.0, 0.0, 0.0],
        ),
    }
)

# dr-damping-governs (MuJoCo only): two boxes launched +x high in the air (no floor contact, so
# only damping acts). The DR term sets freejoint linear damping on `damped` at runtime; `control`
# keeps zero. The damped box must lose more horizontal speed than the control — proving runtime
# dof_damping DR governs the dynamics. Isaac SDKs have no runtime body-damping setter, so this
# scenario is MuJoCo-only. Spawn high so they free-fly through the window without hitting the floor.
dr_damping_pair = SceneConfig(
    rigid_objects={
        "damped": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.0, 1.0, 5.0],
            linear_velocity=[3.0, 0.0, 0.0],
        ),
        "control": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.0, -1.0, 5.0],
            linear_velocity=[3.0, 0.0, 0.0],
        ),
    }
)


def _no_damping():
    """A physics override zeroing PhysX damping (Isaac defaults free bodies to 0.1) so freefall /
    spin tests are pure. MuJoCo ignores physx (no body damping there) — harmless."""
    return PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0))


# #6 galileo-freefall: two boxes of different mass dropped from the same height must fall
# identically (z independent of mass) and track z0 - 0.5*g*t^2. Spawn high (z=5) so neither
# nears the floor in the 0.3 s window. Damping zeroed so the fall is pure.
galileo_freefall = SceneConfig(
    rigid_objects={
        "light": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.5, 0.5, 5.0],
            physics=PhysicsConfig(mass=0.1, physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0)),
        ),
        "heavy": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.5, -0.5, 5.0],
            physics=PhysicsConfig(mass=10.0, physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0)),
        ),
    }
)


# #7 damping-decay: a box launched +x with linear damping must lose horizontal speed to drag.
# PhysX linear_damping on Isaac takes a strong value (5.0) that more than halves the speed. This
# scenario is Isaac-only: there is no static MuJoCo freejoint-damping config (runtime damping DR is
# exercised by the dr-damping-governs scenario), so the MuJoCo backends have no per-object
# config-time drag to assert here.
def _damping_box(physx_damp):
    """A box launched +x at z=5.5, staying airborne the whole window so only drag (not floor
    friction) can slow it. PhysX (Isaac) takes ``physx_damp``. The harness asserts the damped box
    loses more +x speed than the undamped control, which holds for any positive damping."""
    return RigidObjectConfig(
        urdf_file=_SMALL_BOX,
        usd_file=_SMALL_BOX_USD,
        position=[1.5, 0.0, 5.5],
        linear_velocity=[3.0, 0.0, 0.0],
        physics=PhysicsConfig(
            physx=PhysXPhysicsConfig(linear_damping=physx_damp, angular_damping=0.0),
        ),
    )


# A damped box vs an undamped CONTROL, both launched +x and kept airborne. The damped box must
# lose clearly more horizontal speed than the control — isolating DRAG from floor friction (both
# stay airborne through the window, so only damping can slow them).
damping_decay = SceneConfig(
    rigid_objects={
        "dbox": _damping_box(physx_damp=5.0),
        "dctrl": _damping_box(physx_damp=0.0),  # undamped control
    }
)

# #8 restitution-bounce: a box dropped from z=0.5 with high restitution must rebound off the
# floor (rise back well above box height after the first contact). Isaac uses native restitution
# (+ max combine so the floor's 0 restitution doesn't null it); MuJoCo uses a negative-form
# solref for a springy contact (cluster-confirmed values).
restitution_bounce = SceneConfig(
    rigid_objects={
        "ball": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.5, 0.0, 0.5],
            physics=PhysicsConfig(
                isaacgym=IsaacGymPhysicsConfig(friction=0.5, restitution=0.9),
                isaacsim=IsaacSimPhysicsConfig(
                    static_friction=0.5, dynamic_friction=0.5, restitution=0.9, restitution_combine_mode="max"
                ),
                mujoco=MujocoPhysicsConfig(friction=[0.5, 0.005, 0.001], solref=[-3000.0, -3.0]),
            ),
        ),
    }
)

# #9 angular-velocity-spin: a box high in the air given a pure +z spin (no linear vel) must
# rotate about z by ~omega*t with the spin axis preserved. Spawn high so it free-spins without
# floor contact; damping zeroed so the spin doesn't decay (Isaac default would bleed it).
angular_spin = SceneConfig(
    rigid_objects={
        "spinner": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.5, 0.0, 3.0],
            angular_velocity=[0.0, 0.0, 4.0],
            physics=PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0)),
        ),
    }
)

# #10 velocity-restore-reset: a box with a configured initial velocity that, after running and
# being reset to its initial state, must be MOVING again (translating + spinning) — the restored
# velocity is live, not merely stored.
velocity_restore = SceneConfig(
    rigid_objects={
        "rbox": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.5, 0.0, 0.6],
            linear_velocity=[1.0, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 2.0],
        ),
    }
)

# #11 object-obs-robot-pose (multi-env): one free box with known world velocity + spin; the test
# sets a per-env-distinct robot root pose and asserts the base-frame object obs equal the
# hand-computed world->base transform in EVERY env (catches per-env indexing bugs).
obs_under_pose = SceneConfig(
    rigid_objects={
        "obox": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.5, 0.0, 0.6],
            linear_velocity=[0.5, 0.3, 0.0],
            angular_velocity=[0.0, 0.0, 1.0],
        ),
        # A SECOND free body at a distinct pose/velocity so the [N, num_envs, 13] object-major
        # reshape in the obs terms is actually exercised (with N=1 it's the identity for any
        # ordering, hiding an object/env transposition).
        "obox2": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.5, 0.6, 0.6],
            linear_velocity=[-0.2, 0.4, 0.0],
            angular_velocity=[0.0, 0.0, -1.5],
        ),
    }
)

# #12 pose-jitter-settle (multi-env): a box resting on the floor; the jitter reset term perturbs
# its XY/yaw per-env, and each env must settle near ITS OWN jittered pose (per-env independence).
jitter_settle = SceneConfig(
    rigid_objects={
        "jbox": RigidObjectConfig(urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=[1.5, 0.0, 0.06]),
    }
)

# #13 loader-invariance: same freefall, one as a standalone rigid body (tri-format) and one as a
# 1->N scene-file body — both must track 0.5*g*t^2, proving the loader path doesn't perturb
# physics. Spawn high so the fall is clean.
loader_invariance_freefall = SceneConfig(
    rigid_objects={
        "lbox": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.5, 0.0, 5.0],
            physics=PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0)),
        ),
    }
)

# Scene-file counterpart of #13: the multibody file lifted high with the static post excluded,
# so only the free body falls (compared against the standalone freefall above).
loader_invariance_multibody = SceneConfig(
    scene_files={
        "scene": SceneFileConfig(
            xml_path="holosoma/data/scene_objects/multibody/multibody.xml",
            urdf_path="holosoma/data/scene_objects/multibody/multibody.urdf",
            usd_path="holosoma/data/scene_objects/multibody/multibody.usda",
            position=[1.5, 0.0, 5.0],
            # Re-type BOTH bodies to free and zero their damping, so each file-expanded body
            # free-falls under the SAME pure-gravity dynamics as the standalone `lbox` — otherwise
            # a scene-file free body inherits the default 0.1 linear damping (vs the standalone's
            # 0.0), and the loader-invariance comparison would be against differently-damped bodies.
            object_configs={
                "free_box": ObjectPatternConfig(fixed=False, physics=_no_damping()),
                "static_post": ObjectPatternConfig(fixed=False, physics=_no_damping()),
            },
        )
    }
)

# #15 per-env-relocation (multi-env): the box rests on a spawn-placed pillar, then the test
# relocates the pillar to a PER-ENV-DISTINCT target x (6.0 + env*0.5) over the full env range and
# asserts (a) each env's pillar landed at its OWN target — the per-env set_static_body_pose
# indexing — and (b) every box then falls once its support moved away. (See assert_per_env_relocation
# for why it relocates AWAY for all envs rather than keeping some resting: IsaacSim drops a resting
# contact through any mid-sim kinematic pose-write, so "keep resting through a relocation" is not
# cross-backend; relocating OUT is the direction static_move_assert validates on every backend.)
per_env_relocation = SceneConfig(
    rigid_objects={
        # Pillar spawns UNDER the box (x=2.0, top at 0.35) so the box rests on it from the start.
        "freebox": RigidObjectConfig(urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=[2.0, 0.0, 0.45]),
        "pillar": RigidObjectConfig(
            urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=[2.0, 0.0, 0.30], fixed=True
        ),
    }
)


# --------------------------------------------------------------------------------------------
# Robot/object parity twins for the link_physics feature. Each pairs the one-link `onelink-box`
# robot (spawned by behavior_assert with a matching link_physics) against a free `twin` object
# carrying the same PhysicsConfig the robot's link_physics carries. The robot and the twin reach the
# same physical outcome, confirming the robot link is driven by the shared config the same way the
# object is, not merely that a field echoes back.
#
# The twin sits at y >= 2.0 to clear IsaacGym's +-1 m robot spawn jitter (the robot spawns near the
# origin); gravity outcomes are x/y-invariant, so the lateral offset is harmless. The robot's own
# spawn pose comes from the onelink-box RobotConfig.init_state (z=0.6), set in _robot_presets.

# galileo-twin: drop from rest, matched zero damping. Robot link and object fall identically
# (z0 - 1/2 g t^2). No init velocity needed (robot init-velocity is dropped on Isaac), so this is
# the all-backend case.
galileo_twin = SceneConfig(
    rigid_objects={
        "twin": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.0, 2.0, 0.6],  # same z as the robot's init_state; clear in y
            physics=PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0)),
        ),
    }
)

# damped-fall-twin (Isaac-only): the robot link and the object twin carry a strong physx
# linear_damping, dropped from a tall height so they never hit the floor in the window. Unlike
# galileo-twin (zero damping, gravity-only, config-independent), this discriminates the physx
# damping path: a strongly-damped body falls measurably less than free-fall (1/2 g t^2), so if
# link_physics.physx did nothing the robot would free-fall and the test would fail. Isaac-only
# because MuJoCo reads only its `mujoco` sub-config and ignores the physx sub-config. The robot's
# matching link_physics is set in behavior_assert._twin_robot_link_physics.
_DAMPED_FALL_DAMPING = 3.0  # strong linear damping; on Isaac this more than halves the free-fall drop
damped_fall_twin = SceneConfig(
    rigid_objects={
        "twin": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.0, 2.0, 5.0],  # tall: stays airborne the whole window so only drag, no floor
            physics=PhysicsConfig(physx=PhysXPhysicsConfig(linear_damping=_DAMPED_FALL_DAMPING, angular_damping=0.0)),
        ),
    }
)

# restitution-twin: drop from rest with high restitution; robot link and object rebound to a
# matched apex. Both the twin and the robot use restitution_combine_mode="max" (so the box's 0.9
# wins against the ground); the robot carries the same max-combine material via the spawn material
# bind (_bind_robot_link_material authors the combine mode on the bound material prim, which a
# per-shape value write cannot), so robot and twin rebound identically. The harness widens the
# terrain restitution for this scenario so the Isaac ground is bouncy.
restitution_twin = SceneConfig(
    rigid_objects={
        "twin": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            # Same drop height as the robot's init_state z (0.6) so impact speed, and thus rebound
            # apex, matches; a different z would give a different apex and a false fail.
            position=[0.0, 2.0, 0.6],
            physics=PhysicsConfig(
                isaacgym=IsaacGymPhysicsConfig(friction=0.5, restitution=0.9),
                # "max" restitution combine so the box's 0.9 wins against the ground (matches the
                # restitution-bounce scenario; multiplying against a <1 ground kills the bounce).
                # The robot link carries the same max-combine material via the spawn material bind
                # (the bind authors the combine mode, which a per-shape value write cannot), so
                # robot and twin rebound to a matched apex.
                isaacsim=IsaacSimPhysicsConfig(
                    static_friction=0.5, dynamic_friction=0.5, restitution=0.9, restitution_combine_mode="max"
                ),
                mujoco=MujocoPhysicsConfig(friction=[0.5, 0.005, 0.001], solref=[-3000.0, -3.0]),
            ),
        ),
    }
)

# combine-mode: two objects with identical per-shape friction but different IsaacSim friction
# combine modes, launched along +x and kept on the floor, against a moderate-friction ground.
# PhysX combines the contact-pair friction by the pair's combine mode, so:
#   - "min" combine -> effective friction = min(mu_obj, mu_ground) = the smaller -> slides far
#   - "max" combine -> effective friction = max(mu_obj, mu_ground) = the larger -> slides short
# With mu_obj=0.1 (low) and a ground mu well above it, min and max diverge sharply. This confirms
# the combine mode reaches PhysX and changes the outcome, not just that it reads back. Both bodies
# launch low and settle onto the floor.
_COMBINE_MU = 0.1  # object per-shape friction; far below the ground's, so min vs max diverge


def _combine_box(y, combine_mode):
    return RigidObjectConfig(
        urdf_file=_SMALL_BOX,
        usd_file=_SMALL_BOX_USD,
        position=[1.0, y, 0.06],  # resting height on the floor
        linear_velocity=[2.0, 0.0, 0.0],  # +x launch; friction decelerates it
        physics=PhysicsConfig(
            isaacsim=IsaacSimPhysicsConfig(
                static_friction=_COMBINE_MU,
                dynamic_friction=_COMBINE_MU,
                friction_combine_mode=combine_mode,
            ),
        ),
    )


combine_mode = SceneConfig(
    rigid_objects={
        "slide_min": _combine_box(y=0.0, combine_mode="min"),  # low effective friction -> far
        "slide_max": _combine_box(y=1.5, combine_mode="max"),  # high effective friction -> short
    }
)

# robot-combine-mode: the robot link carries a "min"-combine material (via link_physics, bound as a
# RigidBodyMaterialCfg so the combine mode reaches PhysX) and the object `twin` carries a
# "max"-combine material; both have the same per-shape friction and are launched +x on the same
# floor. The robot (min: effective mu = the smaller) slides farther than the twin (max: effective mu
# = the larger). This confirms the robot honors friction_combine_mode end-to-end, the property a
# per-shape value write cannot carry. The robot's own material is set on its preset's link_physics
# in behavior_assert (_twin_robot_link_physics); this scene only holds the twin.
robot_combine_mode = SceneConfig(
    rigid_objects={
        "twin": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[1.0, 2.0, 0.06],
            linear_velocity=[2.0, 0.0, 0.0],
            physics=PhysicsConfig(
                isaacsim=IsaacSimPhysicsConfig(
                    static_friction=_COMBINE_MU, dynamic_friction=_COMBINE_MU, friction_combine_mode="max"
                ),
            ),
        ),
    }
)

# Test-scene CLI keys -> SceneConfig. These mirror the keys the harnesses pass as ``--scene``
# (scene_spawn_assert) or derive from ``--scenario`` (behavior_assert). Registered into the
# production ``scene.DEFAULTS`` by :func:`register` so they resolve through the normal tyro path.
TEST_PRESETS: dict[str, SceneConfig] = {
    "galileo-twin": galileo_twin,
    "damped-fall-twin": damped_fall_twin,
    "restitution-twin": restitution_twin,
    "combine-mode": combine_mode,
    "robot-combine-mode": robot_combine_mode,
    "g1-largebox": g1_largebox,
    "static-box": static_box,
    "static-box-usd": static_box_usd,
    "free-and-static": free_and_static,
    "usd-box": usd_box,
    "velocity-box": velocity_box,
    "physics-box": physics_box,
    "friction-slide": friction_slide,
    "ducknado": ducknado,
    "multibody": multibody,
    "multibody-override": multibody_override,
    # Behavioral-test presets (tests/simulators/behavior_assert.py).
    "collide-into-fixed": collide_into_fixed,
    "momentum-transfer": momentum_transfer,
    "static-support": static_support,
    "dr-friction-pair": dr_friction_pair,
    "dr-damping-pair": dr_damping_pair,
    "galileo-freefall": galileo_freefall,
    "damping-decay": damping_decay,
    "restitution-bounce": restitution_bounce,
    "angular-spin": angular_spin,
    "velocity-restore": velocity_restore,
    "obs-under-pose": obs_under_pose,
    "jitter-settle": jitter_settle,
    "loader-invariance-freefall": loader_invariance_freefall,
    "loader-invariance-multibody": loader_invariance_multibody,
    "per-env-relocation": per_env_relocation,
}


def register() -> None:
    """Merge :data:`TEST_PRESETS` into ``holosoma.config_values.scene.DEFAULTS``.

    Call this once before ``tyro.cli()`` in any harness that resolves a test scene by STRING
    (``scene_spawn_assert``, ``behavior_assert``). tyro reads ``scene.DEFAULTS`` lazily (the config
    types use ``from __future__ import annotations``), so a key registered here becomes a valid
    ``scene:<key>`` subcommand on the SAME production resolution path — no separate branch. Idempotent
    (a plain ``dict.update``); the import direction stays tests -> core.
    """
    from holosoma.config_values import scene

    scene.DEFAULTS.update(TEST_PRESETS)
