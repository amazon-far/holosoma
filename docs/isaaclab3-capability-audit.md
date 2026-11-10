# Isaac Lab 3 capability audit and implementation status

Source audit on November 9, 2026, against Holosoma `53dd7f4`, Isaac Lab
`v3.0.0`, and Newton `v1.6.1`. The APIs below were checked in the pinned source;
the linked upstream documentation was also checked. The inventory below describes
that baseline; the implementation status is tracked here as the work proceeds.

The current change exposes typed Holosoma settings and backend adapters. Newton
MJWarp reuses `sim.mujoco` and `sim.mujoco.warp`, plus the existing per-asset
`PhysicsConfig.mujoco` contact settings. Newton-specific collision/execution,
contact spring and mesh options have typed fields; renderer options are typed.
Per-link body-name overrides reuse `PhysicsConfig` and the scene configuration's
first-match pattern convention. Existing Newton constructors, CLI and saved
inputs remain supported. See [the backend reference](isaacsim-backends.md) for
current syntax and explicit limitations. The baseline capability inventory below
also identifies future work; it is not a claim that every SDK feature is supported.

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

Controls missing at the audited baseline included:

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

The current adapters forward `PhysicsConfig.mujoco` to Newton's USD importer,
including friction vectors, contact parameters and body gravity compensation.
Shared mass/density remain first-class fields. Contact springs and mesh/SDF
controls use typed `NewtonPhysicsConfig`; no SDK fragment/class names appear in
user configuration. Additional joint/tendon and deformable controls remain future
work. Existing self-collision controls still own robot self-collision policy.

Robot body-name overrides and scene object patterns reuse `PhysicsConfig`.
Adapters build the appropriate SDK fragments after selecting physics. Global
defaults apply first, followed by one matching per-link override. An explicit
mass override targets only its selected rigid body; it does not stamp mass on
all descendant geometry. Per-collider material overrides preserve unmodified
bound material properties. Only ancestors of eligible targets are deinstanced.

`None` leaves authored values unchanged. Missing/unsupported targets produce an
error. Body-name globs are a Holosoma interface; SDK USD path expressions remain
private to the adapter. Experiment YAML now preserves mapping order, retaining
first-match precedence across saving and loading.

The custom spawner retains source-prim extraction and rigid-body normalization.
Lab 3 property adapters use public `apply_*_properties` writers and material
spawning; Lab 2 continues through its existing converters and material binding.
Native articulation/deformable topology should not be normalized into a rigid
compound object when those asset types are added.

Sources:
[Physics schemas](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/schema_cfgs.html),
[Newton fragments](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/sim/schemas/schemas_cfg.py),
[Newton materials](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/sim/spawners/materials/physics_materials_cfg.py),
[public schema writers](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/sim/schemas/schemas.py).

## 2. Solver configuration and effective physics

Shared MuJoCo/MJWarp controls provide solver options and capacities.
[IsaacLab3NewtonConfig](../src/holosoma/holosoma/config_types/simulator.py)
contains Newton-only execution/collision settings and default shape margin. The
[backend](../src/holosoma/holosoma/simulator/isaacsim/backends/isaaclab3.py)
constructs `NewtonCfg(solver_cfg=MJWarpSolverCfg(...))` internally.

Adapt typed Holosoma fields to `NewtonCfg`, its solver config, `NewtonShapeCfg`,
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

Resolve physics once through the shared controls and use that configuration for
launch validation and
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
configuration at the audited baseline: `renderer_config()` constructed default OVRTX, Warp or
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

## Configuration architecture

Keep user configuration in typed Holosoma dataclasses. Reuse existing controls
where semantics agree, and construct SDK configs/fragments inside the selected
backend. Users do not specify SDK class paths, schema family names or nested
constructor dictionaries. Shared config parsing, CLI help and saved experiment
inspection must work without Isaac Lab, Kit or PhysX installed.

For Newton MJWarp, `sim.mujoco` owns solver controls and `sim.mujoco.warp` owns
per-world resources/execution options. `isaaclab3.newton` contains genuinely
Newton-specific execution and collision-pipeline controls. There are no duplicate
solver fields or alias precedence rules. None leaves the direct MuJoCo defaults
unchanged and selects Newton's configured defaults. The internal Newton manager
adapter fills the SDK's missing Jacobian
and line-search tolerance fields before allocation/capture.

Physics properties use the existing mass/density and engine-specific subconfigs.
Newton reads MuJoCo friction/contact settings through schema adapters. The typed
`newton` subconfig adds contact springs and mesh/SDF controls. Robot per-link
physics uses a mapping from body-name globs to `PhysicsConfig`, matching scene
`object_configs`; the first matching override is applied after global defaults.
YAML must preserve mapping order. SDK fragment composition remains an internal
implementation detail. Additional joint/tendon controls should extend the
existing controller or asset configs with typed options as they are implemented.

Resolve runtime requirements before SDK imports. Shared timing, device and
state-ordering ownership remain in existing Holosoma controls. An explicit active
setting must apply or fail with its config path; inactive backend blocks remain
portable saved data. Newton's unsupported total contact-pool allocation,
generic allocation extras and MuJoCo collision masks are documented limitations.
Richer actuator modes, sensor outputs and asset types require runtime contracts
in addition to configuration.

## Suggested implementation sequence and acceptance evidence

1. **Typed physics properties.** Reuse common properties and body-name overrides;
   update Lab 3 robot/object spawners and material binding. Verify save/reload
   precedence, partial overrides, instance handling and unchanged MuJoCo/PhysX defaults.
2. **Physics settings.** Reuse MuJoCo/MJWarp options and add typed Newton collision controls.
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
