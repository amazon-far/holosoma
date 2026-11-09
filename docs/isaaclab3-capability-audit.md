# Isaac Lab 3 capability audit and implementation proposal

Source audit on November 9, 2026, against Holosoma `53dd7f4`, Isaac Lab
`v3.0.0`, and Newton `v1.6.1`. The APIs below were checked in the pinned source;
the linked upstream documentation was also checked. The inventory below describes
that baseline; the implementation status is tracked here as the work proceeds.

The current change implements native `physics` and `renderer_options` kwargs,
schema fragments with ordered per-prim rules, per-collider materials and saved
configuration migration. It consolidates the former five-field Newton block
into the native physics configuration. Actual syntax and bounds are in
[the backend reference](isaacsim-backends.md). The remaining architecture and
feature proposals below, including the illustrative YAML, are future work.

Holosoma currently exposes a useful rigid-robot subset of Lab 3. Physics and
camera renderer selection, native actuator execution, aggregate contact forces,
property randomization, and standalone startup already work. The main restriction
is that its adapters construct a small set of native configurations internally.
Users cannot supply many other native configurations through a saved experiment.

The recommendation is to retain the portable Holosoma configuration and add a
first-class native Lab 3 configuration path. Native properties should not require
an equivalent setting in every simulator. Most configuration gaps can be closed
with this path; actuators, richer observations, and new asset types also need
changes to Holosoma's runtime contracts.

## Priority and scope

| Area | Current restriction | Native capability to expose | Implementation scope |
| --- | --- | --- | --- |
| Physics schemas and targeting | Legacy property configs; robot `link_physics` applies to every link; no native fragment composition | Per-link mass, per-collider contact tuning, gravity compensation, joint and tendon settings; shared, Newton/MuJoCo and PhysX fragments | **First priority.** Config resolver, ordered targeting, and Lab 3 spawner/material adapter changes |
| Solver and collision pipeline | MJWarp fixed as Newton solver; five Newton options exposed | Integrator, friction cone, convergence, collision source, broad phase, capacities, determinism, SDF/hydroelastic controls | **First priority.** Resolve native physics configs and validate combinations |
| Asset import and conversion | Robot USD or URDF; forced URDF reconversion; selected importer options hardcoded | Explicit MJCF import, native importer settings, conversion caching and preconverted assets | **High priority.** Loader and cache changes, with body/joint identity checks |
| OVRTX, Warp and Isaac RTX cameras | Renderer selection works, but native renderer configs use defaults; RGB/depth outputs only | Renderer options, per-camera settings, ISP, segmentation/normals where supported | Config exposure is modest work; additional outputs and async capture need observation contracts |
| Visualizers | SDK visualizer CLI selection is partly available; Holosoma supplies defaults and Kit viewport controls | Native viewer configs, debug geometry, Viser/Rerun recording, multiple viewers | Config and viewer lifecycle/control integration; keep camera rendering independent |
| Actuators | One zero-gain Ideal PD group, receiving torques computed by Holosoma | DC motor saturation, delayed/remotized PD, actuator networks, implicit drives, different joint groups | New command ownership and target-writing action path |
| Contacts and other sensors | Aggregate normal/total contacts, mounted cameras and visual-mesh LiDAR | Contact partners/points, friction history, shape-level Newton contacts, IMU/frame/raycast sensors | Native sensor configuration plus typed data access and observation integration |
| Other assets and solvers | One controlled robot and rigid scene objects are the main shared contract | Additional articulations, collections, deformables, cables, particles and solver-specific assets | New asset registry/state/reset contracts; config passthrough alone is insufficient |

## 1. Physics schemas: more than adding fields

Lab 3 supports schema **fragments** grouped by property family. For example,
`collision_props` can combine shared USD collision settings with
`NewtonCollisionCfg` and `MujocoCollisionCfg`. Each fragment writes its own
namespace. File spawners also accept ordered regex rules targeting different
parts of an imported asset.

Useful controls currently missing from Holosoma's configuration include:

- `NewtonCollisionCfg.contact_margin` and `contact_gap`.
- `MujocoCollisionCfg.condim`, `priority`, `solref`, `solimp` and `solmix`.
- `MujocoRigidBodyCfg.gravcomp` and `MujocoJointCfg.actuatorgravcomp`.
- `NewtonMaterialCfg` torsional/rolling friction, contact stiffness/damping,
  friction gain and adhesion.
- Mesh hull limits, SDF resolution/voxel settings and hydroelastic properties.
- `NewtonArticulationCfg.self_collision_enabled` and
  `MujocoFixedTendonCfg` stiffness/damping.
- Per-link `MassCfg`, rather than assigning the same mass to every robot link.
  `MassCfg` exposes mass/density; it does not expose a full COM/inertia override.

Holosoma's `PhysicsConfig.mujoco` is for the separate MuJoCo simulator. It is not
currently forwarded to Newton's MJWarp solver.

The changes belong in
[scene physics types](../src/holosoma/holosoma/config_types/scene.py),
[robot asset types](../src/holosoma/holosoma/config_types/robot.py),
[property converters](../src/holosoma/holosoma/simulator/isaacsim/converters.py),
[robot configuration](../src/holosoma/holosoma/simulator/isaacsim/isaacsim_articulation_cfg.py),
and the
[custom file spawner](../src/holosoma/holosoma/simulator/isaacsim/spawners/from_files.py).
The custom spawner currently calls the legacy `modify_*` writers; its
configuration narrows material input to a legacy scalar config, and material
binding calls `cfg.func`. Adding native fields without changing these paths
would leave lists and targeting rules unusable.

For Lab 3, prefer upstream spawners where Holosoma does not need custom behavior.
Where source-prim extraction or rigid-body normalization requires the custom
spawner, use the public `apply_*_properties` family writers and
`spawn_physics_material`. Keep the Lab 2 path separate. Retain normalization for
rigid compound objects, but do not apply it to native articulations or deformables.
Native per-collider materials also need ordered material creation/binding rules;
upstream `physics_material` fragment lists alone do not select individual feet.

Preserve these upstream semantics:

- `None` keeps authored attributes; an empty fragment requests its schema;
  an empty fragment list makes no changes.
- Rules match regex suffixes relative to the spawn prim. These are not the bare
  body-name globs used by Holosoma's scene-file selection.
- Later rules win on overlapping attributes. Only eligible schema carriers
  should receive a property.
- API creation flags, fixed-root topology and drive creation are explicit.
  Narrow the targets when creating missing APIs.
- Instance proxies need an explicit edit strategy. Report requests that reach
  no editable target instead of silently accepting them.

Use ordered rule **lists** in saved Holosoma configs, converting them to the
upstream ordered mappings at the adapter boundary. `ExperimentConfig.save_config`
currently calls `yaml.safe_dump` with its default key sorting. Storing native
rule precedence only in dictionary insertion order would change behavior on
save/reload.

Sources:
[Physics schemas](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/schema_cfgs.html),
[Newton fragments](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/sim/schemas/schemas_cfg.py),
[Newton materials](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/sim/spawners/materials/physics_materials_cfg.py),
[public schema writers](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/sim/schemas/schemas.py).

## 2. Solver configuration and effective physics

[IsaacLab3NewtonConfig](../src/holosoma/holosoma/config_types/simulator.py)
currently exposes CUDA graphs, contact/constraint capacities, solver iterations
and default shape margin. The
[backend](../src/holosoma/holosoma/simulator/isaacsim/backends/isaaclab3.py)
constructs `NewtonCfg(solver_cfg=MJWarpSolverCfg(...))` internally.

Expose native `NewtonCfg`, nested solver config, `NewtonShapeCfg`,
`NewtonSoftContactCfg` and `NewtonCollisionPipelineCfg`. Important additions are
MJWarp integrator/solver selection, line-search settings, friction cone,
`impratio`, tolerance, actuator gears, collision detection source and MJCF
debug export. Scene-wide controls include collision decimation, deterministic
mode, broad phase, capacity and hydroelastic settings. Native config support
should also cover OVPhysX and Kit PhysX settings, including GPU buffers and
cooked-collider caching, instead of fixing them inside constructors.

Configuration must reach the solver that actually runs:

- A custom Newton collision pipeline with MJWarp requires
  `use_mujoco_contacts=False`; upstream rejects the conflicting combination.
  SDF/hydroelastic settings require the relevant collision pipeline.
- MJWarp GPU determinism requires the supported configuration, including
  `disable_sensors=True`. PyTorch determinism alone does not supply this guarantee.
- `ccd_iterations` refers to convex collision detection iterations, not a
  switch for continuous collision detection.
- Newton 1.6.1's MuJoCo solver clears margins in some native CCD/MULTICCD paths
  and warns. An authored USD margin is not proof of an effective solver margin.
- Property authoring happens before model construction. Later USD edits do not
  automatically update the finalized Newton model; randomization needs native
  setters/update notifications, and structural changes may require rebuilding.

Keep the existing five options as compatibility inputs. Reject conflicting
explicit native settings rather than choosing an undocumented winner. Resolve
physics once and use that configuration for launch validation and
`SimulationContext`, with the derived effective configuration recorded for review.

Sources:
[Newton configuration](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py),
[MJWarp configuration](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager_cfg.py),
[collision pipeline](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/physics/newton_collision_cfg.py),
[MJWarp tuning](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/how-to/solver_tuning_mjwarp.html),
[Newton MuJoCo solver](https://github.com/newton-physics/newton/blob/v1.6.1/newton/_src/solvers/mujoco/solver_mujoco.py).

## 3. Asset formats, importer control and loading time

Lab's importer path supports USD, URDF and MJCF. Holosoma's Isaac Lab robot path
uses USD when supplied, otherwise URDF; the robot's `xml_file` belongs to its
separate MuJoCo path. Scene objects can convert URDF, while the Isaac Lab
multi-body scene-file loader currently requires USD.

Add an explicit Lab asset format choice, native URDF/MJCF converter settings,
and conversion policy. Do not silently select an existing MuJoCo XML file when
switching an established task to Newton. Useful importer settings include ROS
package paths, collision generation/approximation, mesh merging, mimic/fixed
joint handling, importer drive settings and multiphysics conversion.

The current robot URDF path sets `force_usd_conversion=True` and creates
rank-specific conversion directories. Replace this with a reusable cache keyed
by source and mesh dependencies, converter options and SDK/importer versions.
Use locking and atomic publication for concurrent processes. Support
preconverted/read-only assets; compose task overrides into the task's stage
after conversion, so cached assets remain reusable.

Conversion caching is a direct opportunity to reduce loading time without
bringing in Kit. Benchmark cold and warm conversion separately from physics
initialization and kernel compilation. Import validation must check body/joint
identity, fixed markers, authored inertia, material bindings and referenced
asset files, not just whether a USD was generated.

Sources:
[asset preparation](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/how-to/prepare_asset_for_newton.html),
[URDF converter](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py),
[MJCF converter](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/sim/converters/mjcf_converter_cfg.py),
[Holosoma object/scene loading](../src/holosoma/holosoma/simulator/isaacsim/object_spawner.py).

## 4. Renderers, cameras and visualizers

OVRTX is supported already, including camera/movie rendering. What is missing is
native configuration: `renderer_config()` constructs default OVRTX, Warp or
Isaac RTX configs, and camera data is restricted to RGB/depth.

Expose global renderer configs, per-camera overrides and `isp_cfg`. OVRTX has
async capture, shadow, segmentation colorization and logging/temp-stage options.
Warp adds textures, shadows, ambient lighting, semantic filtering, clipping,
tiling and kernel options. Add output contracts for supported segmentation,
normals and other requested products: integer IDs, metadata and vector-valued
images must not inherit RGB normalization/interpolation.

Treat async OVRTX as a separate change. Its delayed frames need capture
timestamps, matching camera poses/intrinsics, history and partial-reset policy.
Setting `async_rendering=True` alone would expose stale observations as current.
Warp fisheye also needs an explicit mapping from Holosoma's USD lens calibration
to the supported native distortion model.

Visualizers are a separate configuration surface. SDK CLI selection already
exists in part, but Holosoma should save native viewer configs and support their
camera controls, lifecycle and debug overlays. Lab supports lists of visualizers,
Newton GL/RTX options and Viser/Rerun recording. A camera renderer preference
should not initialize an unused renderer or dictate the viewer.

Use the selected camera, viewer and extension requirements to determine launch
dependencies. An OVRTX or Newton GL workflow should stay standalone. Kit should
load when an active feature requires it.

Sources:
[renderers](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/renderers.html),
[visualization](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/visualization.html),
[OVRTX config](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer_cfg.py),
[Warp config](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer_cfg.py),
[camera config](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py),
[Holosoma sensor setup](../src/holosoma/holosoma/simulator/isaacsim/sensor_setup.py).

## 5. Actuators need command ownership

Native actuator execution is already integrated, but the robot builder selects
one zero-gain `IdealPDActuatorCfg`. Holosoma's
[joint control action](../src/holosoma/holosoma/managers/action/terms/joint_control.py)
computes PD torques itself and sends joint efforts each physics step. This
does not expose the full native actuator model selection.

Add named joint groups and native actuator configs. Support two explicit action
paths: external efforts with Holosoma owning the controller, and native
position/velocity/feedforward targets with the configured actuator owning it.
This prevents two PD controllers from acting on the same joint.

DC motor saturation, delayed/remotized PD, actuator networks and implicit
controllers then become useful rather than merely constructible. Distinguish
actuator limits from solver joint limits. Keep commanded, computed and applied
effort distinct in reward/observation telemetry. Delays/network state need
partial-reset support, and gain randomization must update the active controller.
Do not promise hard joint velocity limits where the selected solver does not
enforce them.

Source:
[actuator concepts and limits](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/actuators.html).

## 6. Native sensors and assets need extension contracts

Contact sensor configuration should expose partners, friction forces, contact
points, pose, air time, history and capacity. Newton additionally supports
shape-level sensing/filtering. Keep the existing aggregate normal/total force
accessors, adding structured contact outputs with counts and explicit frames.

Allow native sensor declarations and observation accessors for IMU,
frame-transform and raycast data. Newton collision-BVH raycasting needs a
distinct contract: it ignores `mesh_prim_paths`, queries collision geometry and
can include the robot. It is not a transparent replacement for Holosoma's
visual-mesh LiDAR.

For broader Lab use, add a typed native asset registry and lifecycle hooks.
Additional articulations and rigid-object collections are a practical first
step. Deformables, cables and MPM assets need different state/action/reset
contracts; they do not fit a rigid actor's 13-value root state. Exposing
XPBD/Featherstone/VBD/MPM/Kamino configs also requires testing the asset,
contacts and commands supported by each solver.

Provide native handles for custom observation/reward terms, with documented
buffer ownership and lifetime. Expose writes through controlled adapters/hooks;
model rebuilds can invalidate views and require sensor/actuator reinitialization.
This lets advanced tasks use Lab without forcing Holosoma to duplicate all its
managers, or pretending every native feature has portable semantics.

Sources:
[contact sensor config](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py),
[Newton raycast config](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/sensors/ray_caster/newton_raycast_sensor_cfg.py),
[Newton backend](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/physics-backends/newton.html),
[asset implementations](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0/source/isaaclab/isaaclab/assets).

## Proposed configuration architecture

Add a data-only `NativeConfigSpec` containing a class path and recursively
serializable constructor arguments. Nested native configs have an explicit
discriminator, so ordinary dictionaries are not mistaken for constructors.
Resolve specs only after selecting the active runtime. Validate the expected
config/fragment family and unknown constructor fields, and include the full
Holosoma config path in errors.

This avoids maintaining a second copy of every SDK config class. Common presets
can still expose typed convenience fields. Saved configs must load for
inspection/evaluation overrides on machines without Isaac Lab, Kit or PhysX.

Illustrative proposed YAML, **not current Holosoma syntax**:

```yaml
robot:
  asset:
    isaaclab3:
      schemas:
        collision_props:
          rules:
            - match: "/.*"
              fragments:
                - class_path: isaaclab_newton.sim.schemas:NewtonCollisionCfg
                  kwargs: {contact_gap: 0.002}
            - match: "/.*foot[^/]*(/.*)?"
              fragments:
                - class_path: isaaclab_newton.sim.schemas:MujocoCollisionCfg
                  kwargs: {condim: 6}
        rigid_props:
          rules:
            - match: "/.*"
              fragments:
                - class_path: isaaclab_newton.sim.schemas:MujocoRigidBodyCfg
                  kwargs: {gravcomp: 0.0}
```

These are USD path rules, so the foot pattern must be checked against the chosen
importer's hierarchy. `condim=6` permits sliding, torsional and rolling contact
dimensions; appropriate material coefficients and a supporting solver are also
needed to produce the intended behavior.

Recommended precedence is authored asset values, current portable/legacy
defaults, native general rules, then native specific rules in declared order.
Apply this to robot assets, standalone objects and scene-file overrides.
Preserve explicit `None` and empty values.

Declare runtime requirements before resolving classes that could import heavy
modules, then pass the resolved prototypes through Lab launch validation.
Advanced extension factories should declare requirements too. Shared USD and
PhysX schema authoring can still be needed by a Newton importer; this does not
require launching the PhysX engine.

Record SDK versions, requested specs and effective native settings alongside
the experiment. Protect shared timing, device and state-ordering ownership
against competing native overrides. A requested property must either be
supported, produce an actionable error, or be explicitly documented as authoring
for an inactive backend.

## Suggested implementation sequence and acceptance evidence

1. **Native config and schemas.** Add serializable specs and ordered rules;
   update Lab 3 robot/object spawners and material binding. Verify save/reload
   precedence, partial overrides, instance handling and unchanged legacy defaults.
2. **Physics settings.** Expose native solver, collision and PhysX configs.
   Validate supported combinations and provide effective-model diagnostics.
3. **Asset loading.** Add explicit formats/importer options and conversion cache.
   Verify cache invalidation, concurrent loading and stable body/joint mapping.
4. **Renderers and visualizers.** Expose configs first; add output types,
   calibration and viewer controls. Handle async observations separately.
5. **Actuator models.** Add groups and native target actions, with effort telemetry,
   reset and gain-randomization coverage.
6. **Native sensors and assets.** Add extension lifecycle/data access, then
   expand contacts and asset types with solver-specific contracts.

For physics, validation must span authored USD, native model values and observed
behavior. Useful cases include different foot/torso material settings, a
single-link mass override, gravity compensation, supported torsional/rolling
contacts, and the selected collision pipeline. Check that non-target links and
environments remain unaffected. Constructor assertions alone are insufficient.

Also verify config round-trips without the SDK, existing training/checkpoint
contracts, partial resets and model/view lifetimes. For loading performance,
compare the same task's cold/warm cache and first/steady startup, and inspect
actual loaded libraries to retain standalone Newton/OVRTX behavior.

Full native configuration access is the foundation. It should not be advertised
as validated support for every solver, sensor or asset merely because its
configuration can be constructed.
