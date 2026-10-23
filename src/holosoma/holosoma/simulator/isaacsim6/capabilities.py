"""What the Isaac Sim 6.1 backend supports, as measured, plus the startup check that enforces it.

Two things read this module, and that is the point:

* :func:`validate` runs once when the simulator is built and raises on anything unsupported.
* ``scripts/isaac_capability_report.py`` renders these tables into
  ``docs/isaac-sim-backends.md``.

One table means the documentation cannot claim something the code rejects, or omit something the
code quietly allows.

Availability and verification are two different things
------------------------------------------------------
A capability matrix has to answer two questions that are easy to conflate:

1. **Availability** -- is there an implementation of this capability for this backend at all?
2. **Verification** -- has it been *run* and observed to work?

They differ in both directions. A capability can be available but unverified (plausibly fine, no
harness has exercised it), and a run can fail without saying anything about the capability -- if
the simulator never finishes constructing, the capability's own code never executes, so the
result measures the backend, not the capability. Collapsing the two into one yes/no column is how
a table ends up recording "no" for fourteen capabilities when what actually happened was one
shared construction failure, repeated fourteen times.

So :class:`Capability` carries a per-backend :class:`Verification` rather than a boolean, and
:data:`BACKEND_STATUS` carries whether a backend is reachable at all. The generated table renders
both.

Everything here is **measured**, not inferred
---------------------------------------------
The values come from running ``tests/simulators/*_assert.py`` against each preset, plus targeted
probes for the two non-PhysX backends. That matters, because the first version of this table
asserted most capabilities "should" work on every backend, reasoning that nothing engine-specific
was involved -- and measurement disagreed. Reasoning about a capability matrix instead of
measuring it is how a backend ends up advertised as working when it does not.

Why unsupported means *raise*
-----------------------------
The alternative to raising is degrading -- a stock sensor that ignores its configured scan
pattern, a camera parented to a pose that never updates, a physics engine tuned by another
engine's settings. Those produce plausible-looking output that is wrong, which for a perception
sensor or a physics tunable is the worst available outcome: nothing fails, the numbers simply are
not what the configuration says. So anything unsupported fails loudly, names itself, and says
what to do instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from holosoma.config_types.sensor import CameraSensorConfig, LidarSensorConfig
from holosoma.config_types.simulator import IsaacPhysicsBackend

if TYPE_CHECKING:
    from holosoma.config_types.simulator import SimulatorInitConfig


class Verification(str, Enum):
    """How much is actually known about one capability on one physics backend.

    The three "not working" values are deliberately distinct, because they call for completely
    different responses and an undifferentiated ``no`` hides that:

    * :attr:`UNPORTED` -- holosoma has no implementation on the 6.1 lane. Porting work.
    * :attr:`BLOCKED` -- an implementation exists, but the backend cannot construct a simulator,
      so this capability's code never ran. Nothing is known about the capability itself; fix the
      backend and re-measure.
    * :attr:`UNAVAILABLE` -- measured as structurally impossible on this backend specifically,
      independent of whether the backend otherwise works.
    """

    VERIFIED = "verified"
    """A harness ran on this backend and passed. Named in the capability's ``note``."""

    UNVERIFIED = "unverified"
    """Implemented and believed to work, but no harness covers it. Not a claim that it works."""

    BLOCKED = "blocked"
    """The backend never finished constructing, so this capability was never exercised."""

    UNAVAILABLE = "unavailable"
    """Measured as impossible on this backend, for a reason specific to this capability."""

    UNPORTED = "unported"
    """Not implemented on the 6.1 lane at all, so unavailable on every backend."""


_WORKING = frozenset({Verification.VERIFIED, Verification.UNVERIFIED})


@dataclass(frozen=True)
class BackendStatus:
    """Whether a physics engine is reachable at all, and what was measured."""

    backend: IsaacPhysicsBackend

    usable: bool
    """Whether holosoma can run on it. ``False`` means :func:`validate` rejects it outright."""

    reachable: bool
    """Whether a simulator can be *constructed* on it.

    Separate from :attr:`usable` because it is what decides whether per-capability results mean
    anything. When this is ``False``, every capability on this backend is
    :attr:`Verification.BLOCKED` by construction and no per-capability claim is possible.
    """

    summary: str
    """One-line state, for the generated table."""

    detail: str
    """The measured failure and what fixing it would take. Used in the raised message."""


BACKEND_STATUS: tuple[BackendStatus, ...] = (
    BackendStatus(
        IsaacPhysicsBackend.PHYSX,
        usable=True,
        reachable=True,
        # No capability counts here on purpose -- they are derived from CAPABILITIES by the report
        # generator. A tally hand-written next to the data it summarises goes stale silently.
        summary="Constructs and runs. The verified path; every `verified` cell below is PhysX.",
        detail="",
    ),
    BackendStatus(
        IsaacPhysicsBackend.NEWTON,
        usable=True,
        reachable=True,
        summary="Supported. Whole harness suite passes bar one contact scenario (a solver difference).",
        detail=(
            "Two holosoma-side changes were needed, both measured:\n"
            "(1) Model population. Newton does not read its physics model from the USD stage; it "
            "builds it from a ClonePlan. holosoma cloned only with `isaaclab.cloner.usd_replicate`, "
            "which authors USD prims -- right for PhysX, a no-op for Newton's `ModelBuilder` -- so "
            "the first thing to read that model failed with \"Site 'ft_0' ... matched no builder "
            'bodies". `physics_backend.replicate_environments` now also runs '
            "`isaaclab_newton.cloner.newton_physics_replicate` under Newton.\n"
            "(2) Bulk physics properties. The randomization terms read and write mass, inertia, "
            "centre of mass and shape friction through the PhysX tensor view, of whose seven "
            "members Newton's `ArticulationView` provides none. `physics_properties` adapts "
            "Newton's `get_attribute`/`set_attribute` model to the PhysX layout, including three "
            "differences that each produced a distinct failure: the per-articulation axis Newton "
            "adds, its `mat33` inertia dtype (a 5-D float array exceeds warp's 4-dimension "
            "limit), and the fact that PhysX returns these arrays on the host while Newton "
            "returns them on the device.\n"
            "Plus one upstream workaround: `isaaclab_newton`'s `RigidObject` resolves the "
            "`RigidBodyAPI` prim and queries the *articulation* namespace with it, which differ "
            "for holosoma's converted assets -- see :mod:`.newton_rigid_object`.\n"
            "Known difference, not a gap: `behavior_assert --scenario collide-into-fixed` passes "
            "on PhysX and fails on Newton. Newton is a different solver, so it is a separate "
            "physics target -- policies do not transfer from PhysX unchanged."
        ),
    ),
    BackendStatus(
        IsaacPhysicsBackend.OVPHYSX,
        usable=False,
        reachable=False,
        summary="Cannot construct under Kit: OvPhysX is the Kit-free PhysX runtime, and holosoma needs Kit.",
        detail=(
            "OvPhysX is Isaac Lab 3.0's *Kit-free* PhysX runtime, so it is mutually exclusive "
            "with running inside Isaac Sim. Isaac Lab says so in code rather than in prose: "
            "`isaaclab.physics.physics_manager_cfg._resolve_physx_auto_cfg` selects the `ovphysx` "
            "branch only `if not use_isaac_sim`, where `use_isaac_sim = has_kit()`. holosoma "
            "always launches Kit (it needs Kit for USD conversion, `TiledCamera` RTX sensors, "
            "debug draw and the GUI viewport), so a `PhysxAutoCfg` would silently resolve back to "
            "`isaacsim_physx` and holosoma's `_ovphysx_cfg()` -- which passes `OvPhysxCfg` "
            "directly and so does force OvPhysX -- segfaults instead. Measured with the same "
            "scene and config run both ways: with Kit absent it passes; with Kit running it "
            "crashes in `ovphysx.api.attach_ovstage` (via `OvPhysxManager._attach_ovstage`) "
            "during `sim.reset()`. Isaac Lab's own `test_ovphysx_gravity` passes in this "
            "environment, so neither the wheel nor the GPU is at fault. Note the first failure "
            "here was only a missing optional `ovphysx` wheel; installing it moved the failure "
            "from an import error to this crash, so the wheel was never the real blocker. "
            "Supporting OvPhysX would mean a Kit-free holosoma lane, which is a much larger "
            "change than a backend selection."
        ),
    ),
)
"""Per-engine state. ``detail`` records what was measured, not what is expected."""


HEADLESS = "(none)"
"""The empty ``isaac_visualizers`` selection, named so it can appear in a table cell."""

RENDERER_AGNOSTIC: tuple[str, ...] = ()
"""Marks a capability that does not involve the renderer; verified headless.

Distinct from "verified under no renderer". Most physics capabilities never touch the render
path, so enumerating visualizers for them would imply five measurements that were never needed.
An empty tuple says the renderer is not a variable for this capability -- which is a claim, and a
checkable one: if it turns out a capability *does* behave differently under a visualizer, this is
the field that was wrong.
"""


@dataclass(frozen=True)
class Capability:
    """One capability, and what is known about it on each physics and rendering backend."""

    name: str

    physx: Verification
    """State under PhysX, the originally verified backend."""

    note: str
    """The evidence, or why it is unsupported and what to use instead. Rendered verbatim."""

    newton: Verification | None = None
    """State under Newton, or ``None`` to derive it.

    Explicit per backend rather than copied from :attr:`physx`, because the engines genuinely
    differ: Newton passes the contact-support, momentum-transfer and friction-slide scenarios but
    *fails* ``behavior_assert --scenario collide-into-fixed``, where the hammer reaches the wall
    and then neither rebounds nor comes to rest. That is a solver difference, not a porting bug,
    and a matrix that inferred Newton's column from PhysX's would have hidden it.

    ``None`` falls back to the backend's reachability, which yields ``unverified`` -- never
    ``verified``. So a capability nobody measured on Newton says so.
    """

    renderers: tuple[str, ...] = RENDERER_AGNOSTIC
    """Which ``isaac_visualizers`` selections this capability was verified under.

    The second verification dimension. Physics backends and rendering backends are independent
    choices in Isaac Lab 3.0, so "verified" without saying under which renderer is only half an
    answer -- cameras are the obvious case, since they are the capability a visualizer choice can
    actually break.

    :data:`RENDERER_AGNOSTIC` (the default) means the renderer is not a variable here. A non-empty
    tuple lists the selections actually measured, so anything absent from it is unmeasured rather
    than known-good.
    """

    @property
    def ported(self) -> bool:
        """Whether the 6.1 lane implements this at all.

        Drives the generated table's availability column, and is what :func:`validate` rejects
        on. Keyed off PhysX because PhysX is the only constructible backend: a capability that is
        unported has nowhere to be available.
        """
        return self.physx is not Verification.UNPORTED

    def on(self, backend: IsaacPhysicsBackend) -> Verification:
        """This capability's state on ``backend``.

        Unported wins over everything -- there is no implementation to be blocked. Otherwise an
        unreachable backend yields :attr:`Verification.BLOCKED`, which is the honest answer: the
        capability's code never ran, so its own behaviour is simply unknown there.

        A *reachable* non-PhysX backend yields :attr:`Verification.UNVERIFIED` rather than
        ``VERIFIED``: becoming constructible is not evidence that any particular capability works
        on it. Each one has to be measured and recorded here before it may claim otherwise.
        """
        if self.physx is Verification.UNPORTED:
            return Verification.UNPORTED
        if backend is IsaacPhysicsBackend.PHYSX:
            return self.physx
        if backend is IsaacPhysicsBackend.NEWTON and self.newton is not None:
            return self.newton
        return Verification.UNVERIFIED if _backend_status(backend).reachable else Verification.BLOCKED


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "Rigid-body state read/write",
        Verification.VERIFIED,
        "static_move_assert, subset_write_assert. Both engines.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Articulation state read/write",
        Verification.VERIFIED,
        "all_root_states_unified_assert. Both engines.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Scene spawn (USD, multi-body)",
        Verification.VERIFIED,
        "scene_spawn_assert, object_spawner_cfg_assert. Both engines.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Environment cloning",
        Verification.VERIFIED,
        "static_move_assert at 4 envs, both engines. Newton additionally requires "
        "`newton_physics_replicate` -- see physics_backend.replicate_environments.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Lighting rig",
        Verification.VERIFIED,
        "scene_spawn_assert --probe-lights, which checks the authored light prims rather than "
        "pixels, so it holds regardless of visualizer. Worth knowing that a wrong light "
        "*orientation* does not fail this probe and instead renders the scene black -- that is "
        "what the quaternion-convention note below is about.",
        renderers=RENDERER_AGNOSTIC,
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Contact sensors",
        Verification.VERIFIED,
        "behavior_assert: static-support, momentum-transfer and friction-slide all pass on both "
        "engines. One scenario does NOT: `collide-into-fixed` passes on PhysX and fails on "
        "Newton, where the hammer reaches the wall and then neither rebounds nor comes to rest. "
        "That is a contact/restitution solver difference rather than a porting gap, and it is the "
        "concrete reason Newton is a different physics target rather than a drop-in.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Height-scanner ray caster",
        Verification.VERIFIED,
        "Exercised via the locomotion terrain path on PhysX. Not separately measured on Newton, "
        "so its column stays unverified rather than inheriting this.",
    ),
    Capability(
        "Cameras (robot-link / world mount)",
        Verification.VERIFIED,
        "camera_assert, depth_assert, world_camera_assert, camera_obs_assert, camera_multi_assert, "
        "camera_orientation_assert, camera_geometry_assert, camera_projection_assert. Measured as "
        "a matrix against both physics backends and every *installed* visualizer, all passing with "
        "non-blank frames (`rerun` and `viser` are not installed here -- see the rendering table). TiledCamera is an offscreen RTX sensor, so it needs no visualizer -- "
        "Isaac Lab's RTX renderer enables what it requires on demand, which is why headless "
        "passes. Verified under every visualizer selection, on both engines.",
        renderers=(HEADLESS, "kit", "newton"),
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Domain randomization (mass, friction, inertia, CoM)",
        Verification.VERIFIED,
        "dr_matrix_assert, with read-back verification, on both engines. Routed through "
        "`physics_properties`, which adapts Newton's `get_attribute`/`set_attribute` model to the "
        "PhysX tensor-view layout the terms are written against. Note Newton has a single "
        "`shape_material_mu` where PhysX splits static and dynamic friction, so the dynamic "
        "column is what reaches Newton.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "External wrench / push events",
        Verification.VERIFIED,
        "dr_matrix_assert's push stage, both engines. "
        "enable_external_forces_every_iteration pinned False under PhysX for 5.1 parity.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Debug draw / viewer keyboard",
        Verification.UNVERIFIED,
        "No harness covers this -- it is exercised only by running non-headless, so `kit` is "
        "listed as the renderer it needs rather than one it was confirmed under. Isaac Lab 3.0 "
        "does not load `isaacsim.util.debug_draw`, so the lane enables that extension "
        "explicitly; without it every non-headless run fails, since debug_viz defaults to True.",
        renderers=("kit",),
    ),
    Capability(
        "Cameras (actor mount)",
        Verification.UNPORTED,
        "Unported on 6.1, and the **cause is not established**. Measured symptom: a camera "
        "parented to a spawned scene object returns a uniformly blank frame "
        '(`camera_actor_mount_assert`: "actor-mount frame ~uniform; renderer saw nothing"). '
        "The same harness also reports the frame unchanged after the actor moves, but that "
        "follows arithmetically from the frame being blank in both samples, so it is not "
        "independent evidence about transform propagation. Robot-link and world mounts both "
        "render correctly on this lane, so whatever this is, it is specific to having a spawned "
        "rigid object as the parent prim. Use a robot-link or world mount, or the Isaac Sim 5.1 "
        "backend (simulator:isaacsim).",
    ),
    Capability(
        "LiDAR",
        Verification.UNPORTED,
        "Unported on 6.1. holosoma's rotating-pattern LiDAR needs a *mutable* ray caster -- it "
        "rewrites ray directions each control step -- and Isaac Lab exposes no public API for "
        "that, so it subclasses the backend ray caster and reaches into 20 of its members. "
        "Measured: 18 of those 20 survive in 3.0; the two that do not are `_physics_sim_view` "
        "(now a property on SimulationContext) and `mesh_offsets` (restructured into `meshes` + "
        "`_build_mesh_records`/`_update_mesh_transforms`). Both have identified replacements, so "
        "this is a bounded port rather than a rewrite -- it is unported, not infeasible. Note "
        "the subclass must derive from the *concrete* `isaaclab_physx` class, because the "
        "generic one is a FactoryBase that refuses subclasses outside the `isaaclab` package, "
        "which also makes it PhysX-specific. Use simulator:isaacsim meanwhile.",
    ),
    Capability(
        "Video recording",
        Verification.UNPORTED,
        "Unported on 6.1. The recorder creates an `XFormPrim` over its camera prim to drive "
        "robot-follow tracking; Isaac Sim 6.1 removed `isaacsim.core.prims` outright (unlike "
        "`isaacsim.util.debug_draw`, no extension brings it back). Isaac Lab 3.0's replacement "
        "is `isaaclab.sim.views.XformPrimView`, a factory-dispatched FrameView whose "
        "`set_world_poses` takes warp arrays rather than torch tensors and has no `initialize()` "
        "-- so both the construction and the per-frame tracking write need porting. "
        "`omni.replicator.core`, the other dependency, is fine: Isaac Lab's RTX renderer enables "
        "it on demand. Use simulator:isaacsim to record.",
    ),
)
"""Every capability holosoma's config can request, and where it is verified."""

KNOWN_VISUALIZERS = ("kit", "newton", "rerun", "viser")
"""Visualizer backends Isaac Lab 3.0's ``--visualizer`` accepts.

Three of the four names are usable in this environment; two are not installed:

* Empty (headless), ``kit`` and ``newton`` are **verified** -- ``camera_assert`` passes with
  non-blank frames on both physics engines.
* ``rerun`` and ``viser`` are **not installed** here. Isaac Lab resolves a visualizer by name
  (``isaaclab_visualizers.rerun.rerun_visualizer:RerunVisualizer``) and raises when the backing
  distribution is absent, so selecting one now fails at startup with
  ``ModuleNotFoundError: No module named 'rerun'``. They stay in :data:`KNOWN_VISUALIZERS`
  because the *name* is valid -- the gap is the environment, which ``setup_isaacsim6.sh`` could
  close by installing the extra.

A correction worth recording, because it shows how a measurement can be vacuous: an earlier
version of this table claimed all five selections verified across both engines. They had been
run, but ``isaac_visualizers`` was validated here and then **never read by anything** --
``launcher.apply_launcher_args`` ignored it and resolved every run to ``kit`` or ``None``. So ten
runs exercised one configuration ten times, and the two uninstalled backends looked fine
precisely because the selection never reached the launcher. Honouring the field is what surfaced
the real state.

``kit`` remains the one required for a GUI viewport. Cameras do not need any of them: a
``TiledCamera`` is an offscreen RTX sensor and Isaac Lab's RTX renderer enables what it needs on
demand, which is why the headless column passes.
"""


def _capability(name: str) -> Capability:
    return next(c for c in CAPABILITIES if c.name == name)


def _backend_status(backend: IsaacPhysicsBackend) -> BackendStatus:
    return next(b for b in BACKEND_STATUS if b.backend is backend)


def validate(config: SimulatorInitConfig, sensors: Mapping[str, object]) -> None:
    """Reject any configuration this lane cannot honour, before the simulator is built.

    Called once from ``IsaacSim.__init__``. Raises rather than warning: see the module docstring.

    Parameters
    ----------
    config
        The simulator init config -- supplies the physics backend and visualizer selection.
    sensors
        The configured sensors, passed in rather than read off ``config``: sensors live on
        ``FullSimConfig``, not ``SimulatorInitConfig``, so reading them from the latter would
        silently find nothing and let every sensor check pass vacuously.

    Raises
    ------
    NotImplementedError
        For a physics backend measured as not working, or a capability unported on 6.1.
    ValueError
        For a configuration that is internally inconsistent -- engine-specific tuning supplied for
        a different engine, or an unknown visualizer name.
    """
    _validate_backend(config.isaac_physics_backend)
    _validate_unported_sensors(sensors)
    _validate_engine_tuning(config)
    _validate_visualizers(config)


def _validate_backend(backend: IsaacPhysicsBackend) -> None:
    """Reject a physics engine measured as not working.

    Rejected up front rather than letting the run reach the engine's own error, because those
    errors do not identify themselves as "this backend is unsupported": Newton's is a
    body-pattern match failure deep inside its manager, and OvPhysX's is a **segfault** inside
    the ovphysx wheel with no Python exception at all. Neither tells the reader that holosoma has
    never run on that engine, and a segfault does not even leave a message to read.
    """
    status = _backend_status(backend)
    if not status.usable:
        raise NotImplementedError(
            f"The {backend.value!r} physics backend is not supported by holosoma on Isaac Sim "
            f"6.1. {status.detail} Use isaac_physics_backend='physx' (the default), or the "
            f"Isaac Sim 5.1 backend (simulator:isaacsim)."
        )


def _validate_unported_sensors(sensors: Mapping[str, object]) -> None:
    """LiDAR and actor-mounted cameras are not implemented on this lane at all.

    The raised message is the capability's own ``note``, so the error a user sees and the row the
    generated table shows cannot disagree. ``tests/.../test_capabilities.py`` asserts that both
    of these are in fact marked unported, which is what keeps the two in step.
    """
    lidars = sorted(n for n, c in sensors.items() if isinstance(c, LidarSensorConfig))
    if lidars:
        raise NotImplementedError(f"{_capability('LiDAR').note} Configured LiDARs: {lidars}.")

    actor_cams = sorted(
        n
        for n, c in sensors.items()
        if isinstance(c, CameraSensorConfig) and getattr(c.mount, "target_kind", None) == "actor"
    )
    if actor_cams:
        raise NotImplementedError(
            f"{_capability('Cameras (actor mount)').note} Configured actor-mounted cameras: {actor_cams}."
        )


def _validate_engine_tuning(config: SimulatorInitConfig) -> None:
    """Engine-specific tuning must match the selected engine.

    The engines share no tuning surface (see :mod:`.physics_backend`), so applying one engine's
    config to another would drop most of it. Non-default values aimed at the wrong engine are an
    error, not something to partially honour.
    """
    if config.isaac_physics_backend is IsaacPhysicsBackend.NEWTON:
        return
    newton_set = {name: value for name, value in vars(config.isaac_newton).items() if value is not None}
    if newton_set:
        raise ValueError(
            f"isaac_newton tuning {newton_set} was supplied, but isaac_physics_backend is "
            f"{config.isaac_physics_backend.value!r}. Newton's settings have no meaning for "
            f"another engine; either select isaac_physics_backend='newton' or drop these values."
        )


def _validate_visualizers(config: SimulatorInitConfig) -> None:
    """Every requested visualizer must be a name Isaac Lab recognises.

    An unknown name is silently ignored by the launcher, which means a run intended to produce a
    window or camera frames comes up headless with no error -- so check it here.
    """
    unknown = sorted(set(config.isaac_visualizers) - set(KNOWN_VISUALIZERS))
    if unknown:
        raise ValueError(
            f"Unknown isaac_visualizers {unknown}. Isaac Lab 3.0 accepts "
            f"{list(KNOWN_VISUALIZERS)}. An unrecognised name is ignored by the launcher, so the "
            f"run would come up headless without an error."
        )
