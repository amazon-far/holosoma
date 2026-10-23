# Isaac Sim backends

holosoma supports two Isaac Sim generations side by side, as separate backends:

| Preset | Isaac Sim | Isaac Lab | Package | Status |
|---|---|---|---|---|
| `simulator:isaacsim` | 5.1 | 2.3.2 | `holosoma/simulator/isaacsim/` | default, full capability |
| `simulator:isaacsim6` | 6.1 | 3.0 | `holosoma/simulator/isaacsim6/` | opt-in, PhysX + Newton, three gaps (below) |

They are separate implementations rather than one tree with version branches, because Isaac Lab
3.0 is not a superset of 2.3.2 — it inverted the quaternion convention globally, renamed or
removed most of the accessors holosoma calls, and replaced the single physics engine with a
selectable one. Neither generation is a usable base for the other.

Both report `SimulatorType.ISAACSIM`, so code that only cares "is this Isaac Sim?" is unaffected.
Code that needs to know *which* generation asks `holosoma/simulator/isaac_lane.py`.

`simulator:isaacsim` stays the default until a release boundary, so no existing config silently
changes physics engine. At that point the 5.1 package is deleted and the 6.1 one renamed.

## Choosing backends on 6.1

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run these commands from
the repository root. Deactivate any conda environment first. The setup creates a Python 3.12
environment and a pinned Isaac Lab checkout under `~/.holosoma_deps_sim6`, separate from 5.1.

```bash
bash scripts/setup_isaacsim6.sh
source scripts/source_isaacsim6_setup.sh
```

Isaac Lab 3.0 makes both the physics engine and the renderer configuration choices. Isaac Sim 5.1
has one of each, so these fields are **rejected** on the 5.1 backend rather than ignored — a
silently dropped engine choice would be a config that lies about what ran.

```bash
# PhysX (the default)
python -m holosoma.run_sim simulator:isaacsim6

# With a GUI viewport
python -m holosoma.run_sim simulator:isaacsim6 --simulator.config.isaac-visualizers kit
```

- `isaac_physics_backend` — `physx` (default) and `newton` both work; `ovphysx` is rejected at
  startup. The whole harness suite passes on Newton bar one contact scenario (below). `ovphysx`
  is Isaac Lab's **Kit-free** PhysX runtime, so supporting it would mean a Kit-free holosoma —
  a much larger change than a backend selection.
- `isaac_newton` — Newton's tunables. Deliberately separate from `sim.physx`: the engines share
  **no** tuning fields, so one struct cannot serve both. Supplying the wrong engine's tuning is
  an error rather than something partially applied.
- `isaac_visualizers` — rendering backends by name (`kit`, `newton`, `rerun`, `viser`). Empty
  means headless. Headless, `kit`, and `newton` have been verified against both physics
  backends; `rerun` and `viser` require additional packages. `kit` is the one a GUI viewport needs.

**Newton is a different physics target, not a drop-in.** It is a different solver, so contact
response differs — `behavior_assert --scenario collide-into-fixed` passes on PhysX and fails on
Newton. Policies trained under PhysX will not transfer unchanged; switching means retraining and
re-validation. PhysX stays the default partly because it is pinned to stay numerically
comparable with Isaac Sim 5.1, which is what makes the 6.1 migration safe.

For convenience the non-default engines also have presets — `simulator:isaacsim6-newton` and
`simulator:isaacsim6-ovphysx`, following the `mujoco` / `mjwarp` pattern. They exist mainly so the
assert harnesses can exercise each engine, since those take a preset name through plain argparse
and cannot express a nested config override.

Under PhysX the lane pins `use_newton_actuators=False` and
`enable_external_forces_every_iteration=False` to stay numerically comparable with Isaac Sim 5.1;
Isaac Lab 3.0 defaults both the other way.

## Regenerating the tables below

The block between the markers is generated, so it cannot drift from what the code enforces:

```bash
# From the holosoma repository root:
source scripts/source_isaacsim6_setup.sh
python scripts/isaac_capability_report.py --write
```

Re-run it whenever `ISAACLAB3_COMMIT` in `scripts/versions.sh` is re-pinned.

<!-- BEGIN GENERATED: scripts/isaac_capability_report.py -->

### Pinned versions

| Package | Version |
|---|---|
| `isaacsim` | 6.1.0.0 |
| `isaaclab` | 24.0.0 |
| `isaaclab-physx` | 7.2.0 |
| `isaaclab-newton` | 9.0.0 |
| `isaaclab-ov` | 4.1.0 |
| Python | 3.12.14 |

### Physics backends: measured state

Three different things, deliberately separated — a backend can be installed, and still not construct, and construction is still not the same as working:

- **Package** — the config class imports in this environment.
- **Constructs** — a simulator can actually be built on it. This is the column that decides whether per-capability results below mean anything: when it is `no`, no capability on that backend ever got to run.
- **Usable** — holosoma supports it. `no` means `validate()` rejects it at startup.

| `isaac_physics_backend` | Package | Constructs | Usable | Measured state |
|---|---|---|---|---|
| `physx` | yes | yes | yes | Constructs and runs. The verified path; every `verified` cell below is PhysX. |
| `newton` | yes | yes | yes | Supported. Whole harness suite passes bar one contact scenario (a solver difference). |
| `ovphysx` | yes | **no** | **no** | Cannot construct under Kit: OvPhysX is the Kit-free PhysX runtime, and holosoma needs Kit. |

**`ovphysx`** — OvPhysX is Isaac Lab 3.0's *Kit-free* PhysX runtime, so it is mutually exclusive with running inside Isaac Sim. Isaac Lab says so in code rather than in prose: `isaaclab.physics.physics_manager_cfg._resolve_physx_auto_cfg` selects the `ovphysx` branch only `if not use_isaac_sim`, where `use_isaac_sim = has_kit()`. holosoma always launches Kit (it needs Kit for USD conversion, `TiledCamera` RTX sensors, debug draw and the GUI viewport), so a `PhysxAutoCfg` would silently resolve back to `isaacsim_physx` and holosoma's `_ovphysx_cfg()` -- which passes `OvPhysxCfg` directly and so does force OvPhysX -- segfaults instead. Measured with the same scene and config run both ways: with Kit absent it passes; with Kit running it crashes in `ovphysx.api.attach_ovstage` (via `OvPhysxManager._attach_ovstage`) during `sim.reset()`. Isaac Lab's own `test_ovphysx_gravity` passes in this environment, so neither the wheel nor the GPU is at fault. Note the first failure here was only a missing optional `ovphysx` wheel; installing it moved the failure from an import error to this crash, so the wheel was never the real blocker. Supporting OvPhysX would mean a Kit-free holosoma lane, which is a much larger change than a backend selection.

### Rendering backends

`isaac_visualizers` accepts these names, matching Isaac Lab 3.0's `--visualizer`: `kit`, `newton`, `rerun`, `viser`. Empty means headless.

| Visualizer | State | Measured |
|---|---|---|
| *(empty)* | verified | Headless. Cameras and depth still pass — a `TiledCamera` is an offscreen RTX sensor, and Isaac Lab's RTX renderer enables what it needs on demand. |
| `kit` | verified | Required for a GUI viewport. |
| `newton` | verified | `camera_assert` passes with non-blank frames. |
| `rerun` | **not installed** | Raises `ModuleNotFoundError: No module named 'rerun'` at startup. The name is valid; the distribution is absent here. |
| `viser` | **not installed** | Same — `No module named 'viser'`. |

Isaac Lab resolves a visualizer by name (`isaaclab_visualizers.rerun.rerun_visualizer:RerunVisualizer`) and raises when the backing distribution is missing, so an uninstalled backend fails loudly rather than quietly rendering nothing. Installing the extra in `setup_isaacsim6.sh` closes that gap.

A correction worth recording. An earlier version of this table claimed all five selections verified on both engines. They had been run — but `isaac_visualizers` was validated here and then **never read by anything**: `launcher.apply_launcher_args` ignored it and resolved every run to `kit` or `None`, so ten runs exercised one configuration ten times, and the two uninstalled backends looked fine precisely because the selection never reached the launcher. Honouring the field is what surfaced the real state — and `build_run_sim_config(visualizers=...)` is what made the axis reachable from a harness at all.

### Capability matrix

Read from `holosoma/simulator/isaacsim6/capabilities.py`, the same table `validate()` enforces when the simulator is built, so the doc cannot promise what the code rejects.

The columns answer three different questions. `Available on 6.1` is whether the lane implements the capability at all. The physics columns and `Renderers verified` are the two **independent** verification axes Isaac Lab 3.0 introduced — an engine choice and a renderer choice — so a capability verified on `physx` is not thereby verified under every visualizer, and vice versa.

Physics-column values:

- `verified` — a harness ran on that backend and passed. The harness is named under Evidence.
- `unverified` — implemented, no harness covers it. Not a claim that it works, and not a claim that it does not.
- `blocked` — the backend never finished constructing, so this capability's code never executed. **Nothing is known about the capability here**; this records a backend failure, not a capability failure. Fix the backend (see its row above) and re-measure.
- `unported` — holosoma has no implementation on the 6.1 lane, so no backend has it.

Anything not `verified` or `unverified` raises at startup naming itself, rather than degrading quietly.

`Renderers verified` values:

- A list of visualizer names — the `isaac_visualizers` selections actually measured. `(none)` is the empty selection. Anything absent from the list is **unmeasured**, not known-good.
- `n/a (headless)` — the renderer is not a variable for this capability: it involves no render path, so the headless run is the whole story. A claim, not a shrug — if a capability turns out to behave differently under some visualizer, this is the cell that was wrong.
- `—` — unported, so there is nothing to verify under any renderer.

| Capability | Available on 6.1 | `physx` | `newton` | `ovphysx` | Renderers verified | Evidence |
|---|---|---|---|---|---|---|
| Rigid-body state read/write | yes | verified | verified | blocked | n/a (headless) | static_move_assert, subset_write_assert. Both engines. |
| Articulation state read/write | yes | verified | verified | blocked | n/a (headless) | all_root_states_unified_assert. Both engines. |
| Scene spawn (USD, multi-body) | yes | verified | verified | blocked | n/a (headless) | scene_spawn_assert, object_spawner_cfg_assert. Both engines. |
| Environment cloning | yes | verified | verified | blocked | n/a (headless) | static_move_assert at 4 envs, both engines. Newton additionally requires `newton_physics_replicate` -- see physics_backend.replicate_environments. |
| Lighting rig | yes | verified | verified | blocked | n/a (headless) | scene_spawn_assert --probe-lights, which checks the authored light prims rather than pixels, so it holds regardless of visualizer. Worth knowing that a wrong light *orientation* does not fail this probe and instead renders the scene black -- that is what the quaternion-convention note below is about. |
| Contact sensors | yes | verified | verified | blocked | n/a (headless) | behavior_assert: static-support, momentum-transfer and friction-slide all pass on both engines. One scenario does NOT: `collide-into-fixed` passes on PhysX and fails on Newton, where the hammer reaches the wall and then neither rebounds nor comes to rest. That is a contact/restitution solver difference rather than a porting gap, and it is the concrete reason Newton is a different physics target rather than a drop-in. |
| Height-scanner ray caster | yes | verified | *unverified* | blocked | n/a (headless) | Exercised via the locomotion terrain path on PhysX. Not separately measured on Newton, so its column stays unverified rather than inheriting this. |
| Cameras (robot-link / world mount) | yes | verified | verified | blocked | `(none)`, `kit`, `newton` | camera_assert, depth_assert, world_camera_assert, camera_obs_assert, camera_multi_assert, camera_orientation_assert, camera_geometry_assert, camera_projection_assert. Measured as a matrix against both physics backends and every *installed* visualizer, all passing with non-blank frames (`rerun` and `viser` are not installed here -- see the rendering table). TiledCamera is an offscreen RTX sensor, so it needs no visualizer -- Isaac Lab's RTX renderer enables what it requires on demand, which is why headless passes. Verified under every visualizer selection, on both engines. |
| Domain randomization (mass, friction, inertia, CoM) | yes | verified | verified | blocked | n/a (headless) | dr_matrix_assert, with read-back verification, on both engines. Routed through `physics_properties`, which adapts Newton's `get_attribute`/`set_attribute` model to the PhysX tensor-view layout the terms are written against. Note Newton has a single `shape_material_mu` where PhysX splits static and dynamic friction, so the dynamic column is what reaches Newton. |
| External wrench / push events | yes | verified | verified | blocked | n/a (headless) | dr_matrix_assert's push stage, both engines. enable_external_forces_every_iteration pinned False under PhysX for 5.1 parity. |
| Debug draw / viewer keyboard | yes | *unverified* | *unverified* | blocked | `kit` | No harness covers this -- it is exercised only by running non-headless, so `kit` is listed as the renderer it needs rather than one it was confirmed under. Isaac Lab 3.0 does not load `isaacsim.util.debug_draw`, so the lane enables that extension explicitly; without it every non-headless run fails, since debug_viz defaults to True. |
| Cameras (actor mount) | **no** | **unported** | **unported** | **unported** | — | Unported on 6.1, and the **cause is not established**. Measured symptom: a camera parented to a spawned scene object returns a uniformly blank frame (`camera_actor_mount_assert`: "actor-mount frame ~uniform; renderer saw nothing"). The same harness also reports the frame unchanged after the actor moves, but that follows arithmetically from the frame being blank in both samples, so it is not independent evidence about transform propagation. Robot-link and world mounts both render correctly on this lane, so whatever this is, it is specific to having a spawned rigid object as the parent prim. Use a robot-link or world mount, or the Isaac Sim 5.1 backend (simulator:isaacsim). |
| LiDAR | **no** | **unported** | **unported** | **unported** | — | Unported on 6.1. holosoma's rotating-pattern LiDAR needs a *mutable* ray caster -- it rewrites ray directions each control step -- and Isaac Lab exposes no public API for that, so it subclasses the backend ray caster and reaches into 20 of its members. Measured: 18 of those 20 survive in 3.0; the two that do not are `_physics_sim_view` (now a property on SimulationContext) and `mesh_offsets` (restructured into `meshes` + `_build_mesh_records`/`_update_mesh_transforms`). Both have identified replacements, so this is a bounded port rather than a rewrite -- it is unported, not infeasible. Note the subclass must derive from the *concrete* `isaaclab_physx` class, because the generic one is a FactoryBase that refuses subclasses outside the `isaaclab` package, which also makes it PhysX-specific. Use simulator:isaacsim meanwhile. |
| Video recording | **no** | **unported** | **unported** | **unported** | — | Unported on 6.1. The recorder creates an `XFormPrim` over its camera prim to drive robot-follow tracking; Isaac Sim 6.1 removed `isaacsim.core.prims` outright (unlike `isaacsim.util.debug_draw`, no extension brings it back). Isaac Lab 3.0's replacement is `isaaclab.sim.views.XformPrimView`, a factory-dispatched FrameView whose `set_world_poses` takes warp arrays rather than torch tensors and has no `initialize()` -- so both the construction and the per-frame tracking write need porting. `omni.replicator.core`, the other dependency, is fine: Isaac Lab's RTX renderer enables it on demand. Use simulator:isaacsim to record. |

Totals on `physx`, out of 14 capabilities: 10 verified, 1 unverified, 3 unported.

<!-- END GENERATED -->

## Where 5.1 and 6.1 do not align

Hand-written, because each of these needs judgement rather than introspection. The silent ones
matter most: they produce no error, just different numbers.

### Silent differences

**Quaternion order flipped globally.** Isaac Lab 3.0 is `(x, y, z, w)` in config fields, in
runtime `.data` accessors, *and* in `isaaclab.utils.math`; 2.3.2 was `(w, x, y, z)` throughout.
holosoma is internally `(x, y, z, w)`, so 3.0 now *matches* it and the 6.1 lane **deletes** the
conversions the 5.1 lane needs. It adds them only where a w-first config value is handed to a 3.0
config object: `InitialStateCfg.rot`, sensor `OffsetCfg.rot`, `spawn_light(orientation=)` and
`create_prim(orientation=)`. Getting the light one wrong does not raise — the lights are created,
aimed elsewhere, and the scene renders black.

**`isinstance(asset, Articulation)` is always False on 3.0.** The generic `Articulation` is a
`FactoryBase` that dispatches to a per-backend implementation; the object the scene holds derives
from `BaseArticulation`, not from the factory. Code that branches on it silently takes the
single-rigid-body path for an articulation.

**3.0 ignores `ActuatorBaseCfg.armature` and `.friction`.** 2.3.2's `Articulation` applied them
via `write_joint_armature_to_sim`; in 3.0 `.armature` appears nowhere outside Isaac Lab's own
tests. holosoma still configures both from the robot config, so without intervention the robot
runs with zero armature and zero joint friction while the config says otherwise. The 6.1 lane
applies them explicitly once the physics view is live.

**Defaults that changed.** `use_newton_actuators` is new and defaults `True`;
`enable_external_forces_every_iteration` flipped `False` → `True`, which matters because holosoma
applies external pushes. Both are pinned under PhysX for 5.1 parity.

**A bare `InteractiveSceneCfg` produces no replication** in 3.0, while still reporting correct
`env_origins` — so a scene can silently run one environment while reporting N.

### Renamed, moved or removed

| 2.3.2 | 3.0 |
|---|---|
| `SimulationCfg.physx` | `SimulationCfg.physics` (a `PhysicsCfg`) |
| `isaaclab.sim.PhysxCfg` | `isaaclab_physx.physics.PhysxCfg`, wrapped in `PhysxAutoCfg` |
| `asset.root_physx_view` | `asset.root_view` (**removed**, not aliased) |
| `ContactSensorData.net_forces_w` | `net_normal_forces_w` (PhysX cannot report a combined force) |
| `RayCasterCfg(attach_yaw_only=True)` | `ray_alignment="yaw"` |
| `isaacsim.core.utils.*` | `isaaclab.sim.utils.*` |
| `InteractiveScene.clone_environments()` | `isaaclab.cloner.usd_replicate()` |
| `ray_cast_utils.obtain_world_pose_from_view` | `view.get_world_poses()`; PhysX tensor views still use packed `get_transforms` |
| `SimulationContext.render_mode` / `RenderMode` | `has_gui or has_offscreen_render` |
| `SimulationContext.has_rtx_sensors()` | the `/isaaclab/render/rtx_sensors` setting |
| `SimulationContext.current_time` | `get_physics_step_count() * get_physics_dt()` |
| `SimulationContext.has_gui()` | `has_gui` (a property) |
| `--headless`, `--enable_cameras` | `--visualizer` (CSV of backend names) |
| `cfg.class_type` is the class | a lazy `ResolvableString`; callable, but `is` comparisons fail |

### Other behavioural differences

- **Prim path expressions need `env_[^/]+`, not `env_.*`.** `.*` matches across `/`, so 3.0's
  resolver places the environment boundary too shallow and walks an asset's whole subtree —
  `RuntimeError: Expected 1 prims at '/World/envs/env_.*/freebox', found 1267`. `[^/]+` is what
  3.0's own `{ENV_REGEX_NS}` macro expands to.
- **Data accessors return warp-backed values.** `asset.data.*` yields a `ProxyArray` whose
  deprecation bridge forwards operators but **not** tensor methods such as `.clone()`; the physics
  views return bare `warp.array`. Physics-view *setters* want warp arrays with **int32** indices —
  a torch tensor reaches `warp.types.type_ctype` and fails with
  `TypeError: issubclass() arg 1 must be a class`, which reads like a version conflict rather than
  a wrong argument type.
- **Isaac Sim 6.1's URDF importer ignores `dont_collapse`.** Fixed-joint merging moved to a Python
  pre-pass that strips every fixed joint before the native importer sees the attribute. For the G1
  that deletes `left_foot_contact_point` / `right_foot_contact_point`, which are the robot config's
  `key_bodies` and `foot_height_name`, so the 6.1 lane disables merging for any URDF that uses the
  attribute. The cost is that the robot's other fixed-joint frames survive as articulation links.
- **The URDF→USD conversion cache is keyed per lane**, outside the source tree. The two importers
  produce different USD for the same URDF, so a shared cache would serve one generation's output
  to the other.

## Known gaps on 6.1

Two kinds of gap, and they are worth keeping apart because they fail for unrelated reasons:

- **Unported capabilities** (LiDAR, actor-mounted cameras, video recording) — missing on every
  backend. Each raises `NotImplementedError` at startup naming the capability.
- **The unsupported physics backend** (OvPhysX) — nothing to do with any one capability; it
  cannot construct a simulator at all. Newton used to be in this list and now works; the section
  below records what it took, because the same shape of problem is likely for any further engine.

Each entry below separates what was **measured** from what is **inferred**, because several of
my own earlier write-ups of these overstated the evidence.

### Newton — supported, with one measured solver difference

Newton now runs the whole harness suite. Three things had to change, and each was found by
measurement rather than inspection:

**1. The physics model was never populated.** Newton does not read its model from the USD stage;
it builds it from a ClonePlan. holosoma cloned only with `isaaclab.cloner.usd_replicate`, which
authors USD prims — exactly right for PhysX, which parses its model *from* the stage, and a no-op
for Newton's `ModelBuilder`. The first thing to read that empty model failed:

```
ValueError: Site 'ft_0' with body_pattern
  '/World/envs/env_[^/]+/Robot/Geometry/pelvis' matched no builder bodies.
```

That reads like a contact-sensor bug. Instrumenting `_cl_inject_sites` showed the builder held
**zero bodies** — an empty model, not a naming mismatch. `physics_backend.replicate_environments`
now also runs `isaaclab_newton.cloner.newton_physics_replicate` under Newton (12 articulations,
176 bodies, correct per-env paths). The USD clone still runs too: the RTX sensors read the stage
whichever engine owns the physics.

**2. Bulk physics properties had no Newton path.** The randomization terms read and write mass,
inertia, CoM and shape friction through the PhysX tensor view — seven members, of which Newton's
`ArticulationView` provides **none**, exposing `get_attribute`/`set_attribute` over named model
fields instead. `physics_properties` adapts between them. Three layout differences each produced
a distinct failure, and none were guessable:

| Difference | How it failed |
|---|---|
| Newton adds a per-articulation axis | `index out of bounds` in the sampler |
| Inertia is `mat33`, not 9 floats | `Failed to create array with shape (4, 1, 42, 3, 3), the maximum number of dimensions is 4` |
| PhysX returns these on the **host**, Newton on the **device** | `Expected all tensors to be on the same device` |

PhysX's own layout is not uniform either — measured, it keeps a length-1 body axis for
`get_masses` and `get_material_properties` on a rigid object but **drops** it for `get_inertias`.
The adapter reproduces that, so the terms stay engine-agnostic.

Note what was *not* done: Isaac Lab 3.0 has portable asset-level setters
(`set_masses_index`/`set_coms_index`/`set_inertias_index`), but they have no material-properties
equivalent and their reads are authored *defaults* where the terms deliberately compose onto the
**current** value. Moving PhysX onto them would have changed the verified numerics.

**3. An upstream addressing bug.** `isaaclab_newton`'s `RigidObject` resolves the prim holding
`UsdPhysics.RigidBodyAPI` and then queries the *articulation* namespace with it. Both labels exist
and differ for holosoma's converted assets:

```
body_label          /World/envs/env_0/freebox/Geometry/baseLink   <- what it searches
articulation_label  /World/envs/env_0/freebox                     <- what it indexes
```

holosoma passes the asset root, which is what Newton indexes — so this is not a holosoma bug.
Upstream does not hit it because its own test assets put `RigidBodyAPI` on the configured prim
itself. `newton_rigid_object` keeps upstream's body resolution and asks the model which
articulation owns that body, measuring the trim per asset (assets sit at different depths: the
robot's articulation root is four segments deeper than `freebox`). Worth reporting upstream.

**The remaining difference is physics, not porting.** `behavior_assert --scenario
collide-into-fixed` passes on PhysX and fails on Newton: the hammer reaches the wall and then
neither rebounds nor comes to rest. Newton is a different solver, so this is expected and is the
concrete reason Newton is a separate physics target — see the warning above.

### OvPhysX — needs a Kit-free holosoma

**Measured:** OvPhysX is Isaac Lab 3.0's *Kit-free* PhysX runtime, so it is mutually exclusive
with running inside Isaac Sim. Isaac Lab states this in code rather than prose —
`isaaclab.physics.physics_manager_cfg._resolve_physx_auto_cfg` takes the `ovphysx` branch only
`if not use_isaac_sim`, where `use_isaac_sim = has_kit()`:

```python
if not use_isaac_sim and physics_cfg.ovphysx is not None:
    selected = physics_cfg.ovphysx      # OvPhysX only when Kit is absent
else:
    selected = physics_cfg.isaacsim_physx
```

holosoma always launches Kit — it needs Kit for USD conversion, `TiledCamera` RTX sensors, debug
draw and the GUI viewport. So a `PhysxAutoCfg` would silently resolve back to `isaacsim_physx`,
and `physics_backend._ovphysx_cfg()`, which passes `OvPhysxCfg` directly and therefore does force
OvPhysX, **segfaults** instead — inside `ovphysx.api.attach_ovstage`, via
`OvPhysxManager._attach_ovstage`, during `sim.reset()`. There is no Python exception to catch.

Confirmed by running the *same scene and config* both ways: with Kit absent it passes and steps;
with Kit running it crashes. Isaac Lab's own `test_ovphysx_gravity` also passes in this
environment, so neither the wheel nor the GPU is at fault.

**Corrected:** my earlier write-up said this was merely a missing optional `ovphysx` runtime
wheel, fixable with `uv run --extra ovphysx`. The wheel genuinely was absent, but installing it
only moved the failure from an `ImportError` to the segfault above — it was never the real
blocker. Note `ovphysx` caps `packaging<24` while `isaacsim-core` pins `==26.0`; Isaac Lab's own
`pyproject.toml` resolves that with an override widening to `<27`, so install it with
`--no-deps` rather than letting it downgrade `packaging`.

Supporting OvPhysX would mean a Kit-free holosoma lane. That is a much larger change than a
backend selection, and not obviously worth it while PhysX works under Kit.

### LiDAR — a bounded port, not a rewrite

holosoma's LiDAR drives a rotating scan pattern, so it needs a ray caster whose ray directions
can be rewritten every control step. Isaac Lab exposes no public API for that, so
`sensor_setup._mutable_multi_mesh_ray_caster_cls` subclasses the backend ray caster and overrides
five methods (`_update_buffers_impl`, `set_ray_directions`, `mark_outdated`, `ray_starts_w`,
`_obtain_trackable_prim_view`), reaching into 20 members of the base class.

Measured against Isaac Lab 3.0: **18 of those 20 survive.** Only two do not, and both have an
identified replacement:

| Missing in 3.0 | Used for | Replacement |
|---|---|---|
| `self._physics_sim_view` | creating articulation / rigid-body views for mesh targets | `SimulationContext.physics_sim_view` (a public property) |
| `self.mesh_offsets` | per-target position/orientation offsets | restructured into `meshes` plus `_build_mesh_records` / `_update_mesh_transforms` |

So the blocker is the second one — 3.0 reorganised how mesh transforms are tracked, and the
override's offset lookup has to be rewritten against the new mesh-record structure. That is real
work but bounded; it is unported, not infeasible.

One structural constraint worth knowing before attempting it: the subclass must derive from the
**concrete** `isaaclab_physx.sensors.MultiMeshRayCaster`. The generic `isaaclab.sensors` class is
a `FactoryBase` whose `__init_subclass__` raises
`ImportError: Factory class ... must be defined within the 'isaaclab' package`. That makes the
override PhysX-specific by construction, so a Newton LiDAR would need a second subclass.

### Actor-mounted cameras — cause not established

**Measured:** a camera parented to a spawned scene object returns a uniformly blank frame —
`camera_actor_mount_assert` reports `actor-mount frame ~uniform; renderer saw nothing`.
Robot-link and world mounts both render correctly on this lane (`camera_assert`, `depth_assert`
and `world_camera_assert` all pass), so the problem is specific to having a spawned rigid object
as the parent prim.

**Not evidence:** the same harness also reports the frame unchanged after the actor moves. That
follows arithmetically from the frame being blank in both samples (`torch.equal` on two blank
buffers), so it says nothing about whether the transform propagates. An earlier write-up of mine
presented it as a second, independent finding; it is not.

**Unconfirmed.** A Fabric-versus-USD transform-propagation difference is plausible, since 3.0
drives object poses through Fabric and the mount relies on the USD parent/child relationship. USD
instancing was the other candidate and looks unlikely: holosoma already spawns objects with
`disable_instanceable=True`. A live diagnostic to settle it segfaulted inside Kit, so this is
genuinely open.

### Video recording — one removed module, one changed signature

The recorder creates a prim view over its camera prim to drive robot-follow tracking
(`video_recorder.py`, `XFormPrim(self._camera_prim_path, reset_xform_properties=True)` then
`set_world_poses(...)` each frame).

**Measured:** `isaacsim.core.prims` is removed in Isaac Sim 6.1 and — unlike
`isaacsim.util.debug_draw`, which merely needs `enable_extension()` — no extension brings it
back. Isaac Lab 3.0's replacement is `isaaclab.sim.views.XformPrimView`, a factory-dispatched
`FrameView` whose `set_world_poses(positions, orientations, indices)` takes **warp arrays** and
which has no `initialize()`. So both the construction and the per-frame tracking write need
porting.

`omni.replicator.core`, the recorder's other dependency, is **not** a problem: it is absent from
a bare launch but Isaac Lab's RTX renderer calls `enable_extension` on it, so it resolves once a
camera exists. An earlier probe of mine reported it missing because that probe built no camera.

## Verifying a change

Every capability has a cross-backend harness that runs against either generation:

```bash
cd src/holosoma
export OMNI_KIT_ACCEPT_EULA=1 PYTHONPATH=$PWD

# 6.1
~/.holosoma_deps_sim6/hssim6/bin/python -u \
    tests/simulators/static_move_assert.py --simulator isaacsim6 --num-envs 4

# 5.1, same assertions
conda activate hssim
python -u tests/simulators/static_move_assert.py --simulator isaacsim --num-envs 4
```

Two things a passing suite does **not** prove, both worth asserting directly:

- **Cloning.** A bare `InteractiveSceneCfg` reports correct `env_origins` while replicating
  nothing, so run the multi-env case and check per-env prim sets and distinct origins.
- **Quaternion handling.** Identity reads the same under both conventions, so a non-identity mount
  orientation is required to exercise it.

Read verdicts from the harness's `--result-file` sentinel, not the exit code: Isaac Sim's teardown
corrupts the return code, which is why `tests/simulators/_run_harness.py` judges on the sentinel.
Not every harness defines `--result-file`; the rest print `PASS` / `FAIL`.

### Re-measuring a non-PhysX backend

`validate()` rejects Newton and OvPhysX before construction, so the harnesses cannot reach them
as shipped. To measure one, flip its `reachable`/`usable` in `BACKEND_STATUS` to `True`, run the
suite against `--simulator isaacsim6-newton` / `isaacsim6-ovphysx`, then record what you saw and
restore the flags. Three traps, each of which produced a wrong answer once:

- **An argparse rejection is not a backend failure.** Harnesses that keep their own choice list
  (`dr_matrix_assert.py`'s `_READERS`) reject an unknown preset *before launching*, which in a
  capability sweep looks exactly like the backend failing. Add the preset to every such list.
- **Classify where it failed, not just that it failed.** A failure during simulator construction
  says nothing about the capability under test — that is the `blocked` state in the matrix. Only
  a failure after construction is a per-capability result.
- **Never run a probe script from a directory containing an `isaaclab` sibling.** A stray
  `/tmp/isaaclab` lands on `sys.path[0]`, makes `import isaaclab` resolve as a namespace package,
  and yields `ImportError: cannot import name '_deprioritize_prebundle_paths'` — which reads
  convincingly as a broken environment. `/tmp` itself is usually unsafe for this reason.

When a backend does fail, check whether Isaac Lab's own tests pass on it first
(`pytest source/isaaclab_ov/test/physics/`), which separates "holosoma is wrong" from "this
environment is wrong". That is what established the OvPhysX cause above. If Kit crashes rather
than raising, it writes a py-spy Python traceback next to its minidump under
`site-packages/isaacsim/kit/data/Kit/IsaacLab/3.0/*.py.txt` — that is the only way to get a stack
out of a segfault.
