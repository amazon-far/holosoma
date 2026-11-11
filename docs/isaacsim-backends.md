# Isaac Sim backends

Both versions use the `IsaacSim` simulator, shared scene configuration,
state accessors, sensors, domain randomization, and hooks. Configuration selects
the version; installed packages validate that selection. Native API differences
live in `holosoma/simulator/isaacsim/backends`.

| Preset / API family | Isaac Lab | Isaac Sim | Setup / activation | Dockerfile |
| --- | --- | --- | --- | --- |
| `simulator:isaacsim` / `isaaclab2` | 2.3.2 | 5.1 | `setup_isaacsim.sh` / `source_isaacsim_setup.sh` | `docker/isaacsim.Dockerfile` |
| `simulator:isaacsim3` / `isaaclab3` | 3.0.0 | 6.2 | `setup_isaacsim3.sh` / `source_isaacsim3_setup.sh` | `docker/isaacsim3.Dockerfile` |

The Lab 3 profile uses Python 3.12, PyTorch 2.11 with CUDA 13.0,
Warp 1.17, and NumPy 2. The old profile retains its dependency constraints.
The new image uses Isaac Sim's bundled Python to create its virtual environment;
the standalone installer uses Lab's pinned Isaac Sim extra.
Lab 3 setup includes the locked `video`, `ov` and `importers` extras for rendering and
recording, including OVRTX 0.5.1.385782, OVPhysX 0.6.4.72118723 and
OVStage 0.2.1.385922.
Kit-less URDF conversion uses the genuine `isaacsim-asset-isolated` 6.2.0.0
wheel. Its `isaacsim.asset` namespace is selected before imports, avoiding
the full Isaac Sim package's Kit bootstrap in the same environment. PhysX USD
schemas needed by URDF conversion are registered from data-only wheel resources;
this does not import or load the PhysX engine for Newton.

## Select a profile

```bash
bash scripts/setup_isaacsim3.sh
source scripts/source_isaacsim3_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac simulator:isaacsim3 logger:disabled
```

Activation only loads dependencies. `simulator:isaacsim` always selects Lab 2;
`simulator:isaacsim3` selects Lab 3 for both training and
`python -m holosoma.run_sim`. Saved configurations lacking `isaacsim_backend`
retain `isaaclab2`. To migrate a saved configuration, override
`--simulator.config.isaacsim-backend=ISAACLAB3`. A selection that conflicts with
the installed runtime fails at launch.

```bash
docker build -f docker/isaacsim3.Dockerfile -t holosoma-isaacsim3 .
docker run --rm --gpus all --ipc=host \
    -e HOLOSOMA_MULTIGPU=False holosoma-isaacsim3 \
    bash tests/ci/isaacsim_ci_tests.sh --backend isaaclab3
```

This CI command checks the runtime and GPU before running the existing Isaac Sim
tests, then hands the same explicit version to the pure Python suite.
For direct pytest invocation, use
`python -m pytest -c pyproject.toml --isaacsim-backend isaaclab3 ...` from the
repository root; simulator wrappers forward that selection to their child harnesses. The pure script also
accepts `--backend isaaclab3` and defaults to `isaaclab2` when omitted.
The workflows build, test, and promote each image separately and include both
presets in nightly training. Multi-GPU tests require at least two visible devices.

## Lab 3 physics and camera rendering

Lab 3 selects physics and camera rendering independently:

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac simulator:isaacsim3 logger:disabled \
    --simulator.config.sim.isaaclab3.physics-backend=newton \
    --simulator.config.sim.isaaclab3.renderer=newton_warp
```

For OVRTX cameras and movies with Newton, use the same command with
`--simulator.config.sim.isaaclab3.renderer=ovrtx`. With
`--simulator.config.sim.isaaclab3.physics-backend=physx`, OVRTX uses OVPhysX.

| `physics-backend` | `renderer` | Configuration |
| --- | --- | --- |
| `physx` | `isaac_rtx` | Default: Isaac Sim PhysX and Isaac RTX cameras. |
| `physx` | `newton_warp` | Kit-less OVPhysX dynamics and Newton Warp cameras and movies. |
| `newton` | `isaac_rtx` | Newton MJWarp dynamics and Isaac RTX cameras. |
| `newton` | `newton_warp` | Kit-less Newton MJWarp dynamics and Newton Warp cameras and movies. |
| `physx` | `ovrtx` | Kit-less OVPhysX dynamics and OVRTX cameras and movies. |
| `newton` | `ovrtx` | Kit-less Newton MJWarp dynamics and OVRTX cameras and movies. |

Kit starts when active Isaac RTX cameras, movies, the default Isaac RTX viewer
or explicit Kit launch settings need it. Headless runs without those requirements leave the renderer
preference inactive, including `isaac_rtx`, and use the standalone physics
runtime. LiDAR alone does not activate a camera renderer.

`newton_warp` and `ovrtx` use the SDK's Kit-less launch, USD asset import and native
camera lifecycle. The `physx` physics choice selects OVPhysX for those runtimes; existing
PhysX scene settings are authored into USD before the engine loads it.
Interactive OVRTX launches select the SDK's `newton_rtx` visualizer unless
`--visualizer` explicitly selects another compatible visualizer; Warp defaults
to `newton_gl`. These interactive visualizers have not been validated here. Kit
experiences, Kit visualizers and Kit livestreaming cannot share an OVRTX
process and are rejected by Isaac Lab's launch validation. Kit rendering-mode
presets and `kit_args` must be unset in standalone runs. The native visualizer owns
its camera controls; Holosoma's Kit keyboard bindings apply to Kit viewers.
Live USD body-damping randomization also requires the Kit PhysX runtime;
set `simulator.config.sim.isaaclab3.require_kit=True` before launch when that
integration is needed. Detached physics does not consume those USD edits on
runtime clones, so that event is rejected. This explicit requirement also
supports other Kit APIs without enabling an unused camera renderer.
The renderer setting controls configured sensors exposed as `CameraRecord`.
Warp cameras trade RTX lighting and material fidelity for a simpler rendering
path; the adapter maps their zero-valued depth misses to infinity before applying
Holosoma's shared depth policy.
Current Warp camera integration supports pinhole projection. Use `isaac_rtx`
or `ovrtx` for Holosoma's fisheye configurations; selecting Warp with those configurations
is rejected rather than rendering a pinhole approximation.

Mounted cameras have independent capture rates through the existing
`update_decimation` field. At a 50 Hz control rate, `1` captures at 50 Hz and
`"10Hz"` captures at 10 Hz; consumers keep its last frame between captures.
Lab 3 uses this same setting to reduce actual camera rendering with Warp,
OVRTX and Isaac RTX. All due camera poses are prepared together.
OVRTX keeps the normal shared batch for single-camera and uniform-rate setups.
Mixed rates and perception/movie coexistence use persistent native renderers
with one product each, updating and capturing only due owners. Stable membership
avoids the pinned OVRTX package's [multiview batch defect](ovrtx-camera-cadence-repro.md)
and repeated product re-entry. This trades additional memory and initialization
for lower measured capture cost; the comparison includes every-tile RGB/depth
checks, rather than capture counts alone.
Multiple native engines add per-product memory and, in the pinned OVRTX build,
disable video textures on scene materials. Camera movie recording still works.
OVRTX retains the SDK's fixed `1/60` native snapshot increment; native sensor
time is separate from physics time. Matched controls found identical slow-camera
frames with and without a fast persistent owner in the standalone scene.
The SDK scene retains some prior material color already present in native output;
this does not establish the exact pinned temporal-filter mechanism.
See the [timing, quality and compatibility investigation](ovrtx-timing-and-quality.md).
Isaac RTX activates only due tiled products and pauses each on its native
completion event, then reads the SDK-owned output buffers. Viewer renders
and logger movie captures do not activate idle perception products.
Lab 3's camera captures are independent of the viewer's `render_interval`.
Lab 2 retains its periodic render behavior and interval divisibility check.
Rates resolve against the control clock; asynchronous sampling and rates above
that clock require an additional observation contract.

Logger movies use Kit/Replicator RTX with `isaac_rtx` and native cameras with
`newton_warp` or `ovrtx`. Native movie cameras are declared before cloning;
Isaac RTX creates one movie camera and render product under the recording
environment. Both paths track the selected `record_env_id` and use the shared
encoding, overlays and recording lifecycle. Camera capture does not integrate
physics. Disable logger video when comparing camera renderer throughput.
Isaac RTX movie products explicitly retain a single-view tile layout before
capture; this prevents active tiled perception products from replacing the
recording camera's view.

The native integration matrix checks all six combinations with two G1
environments, sparse property randomization, dynamics, actuation, contact forces,
and independent RGB/depth geometry. Separate OVRTX cases cover movies, fisheye,
visual-mesh LiDAR and training/checkpoint/evaluation with both physics choices.
The matrix inspects loaded shared libraries: standalone cases must have no Kit,
and standalone Newton must have no PhysX native bindings or libraries.
Physics-only startup cases also assert that no unused renderer is loaded and
record launch/scene-ready timing alongside the library inventory. Native movie
checks cover Warp as well as OVRTX. Timing is diagnostic, not a performance gate.
Newton LiDAR and training also exercise the default renderer preference with no
active cameras or movies, so they use standalone physics. These bounded checks do
not establish equal policy performance or throughput across engines.

## Lab 3 options

These settings are under `--simulator.config.sim.isaaclab3.*`:

| Option | Default | Effect |
| --- | --- | --- |
| `physics-backend` | `physx` | Select PhysX (OVPhysX without Kit) or Newton's MJWarp solver. |
| `renderer` | `isaac_rtx` | Select Isaac RTX, Kit-less Newton Warp or Kit-less OVRTX; unused in physics-only runs. |
| `renderer-options` | Typed defaults | Camera settings for the selected renderer; None retains SDK defaults. |
| `use-fabric` | `True` | Use native Fabric synchronization. |
| `create-stage-in-memory` | `False` | Author assets into the simulation's in-memory stage. |
| `use-newton-actuators` | `True` | Use native actuator evaluation, including the PhysX host adapter. |
| `contact-force-mode` | `normal` | Preserve the prior normal-force reward input; `total` includes friction. |

Newton MJWarp reads the **existing MuJoCo configuration**, so the same experiment
can carry its solver and contact settings across direct MJWarp and Isaac Lab 3:

```bash
--simulator.config.sim.isaaclab3.physics-backend newton
--simulator.config.sim.isaaclab3.renderer newton_warp
--simulator.config.sim.mujoco.iterations 50
--simulator.config.sim.mujoco.cone elliptic
--simulator.config.sim.mujoco.impratio 3.0
--simulator.config.sim.mujoco.warp.nconmax-per-env 256
--simulator.config.sim.isaaclab3.renderer-options.enable-shadows False
```

Renderer options currently expose Warp shadows, textures, ambient lighting,
backface culling and CUDA graph capture. OVRTX RGB/depth always include shadows;
an explicit `enable-shadows` request for OVRTX fails rather than silently doing
nothing. Camera `far` and `depth_clipping_behavior` remain the single controls for
ray range and shared depth clipping. Segmentation outputs are not yet supported.

Supported shared solver fields are `integrator`, `solver`, `cone`, `jacobian`,
`impratio`, `iterations`, `tolerance`, `ls_iterations` and `ls_tolerance`.
`mujoco.warp.nconmax_per_env`, `njmax_per_env`, `graph_conditional` and scalar
execution `extras` reach Newton before CUDA capture. Flags reach model compilation;
no-slip iterations must be zero. Newton supports CG/Newton solvers and Euler,
RK4 or implicitfast integration. Unsupported explicit inputs fail with their
config path. Newton does not expose `warp.naconmax` or generic allocation extras;
use the per-environment capacities. `mujoco.extra` supports writable MuJoCo
options accepted by Newton's solver constructor, rather than arbitrary SDK kwargs.

Unset values keep direct MJWarp's existing 96 contacts per environment.
Newton uses 128 contacts, the SDK's default of 300 constraint rows,
CUDA graphs and shape margin `0.01`. Solver settings have one configuration
path: `sim.mujoco` and `sim.mujoco.warp`, with no duplicate Newton fields.
`sim.substeps` and `sim.physx` own Newton substeps and PhysX solver policy respectively.

Newton-only execution and collision controls are typed:

```python
IsaacLab3NewtonConfig(
    collision_source="newton",
    collision=NewtonCollisionConfig(broad_phase="sap", rigid_contact_max=8192),
    use_cuda_graph=True,
    deterministic_mode="run_to_run",
)
```

MuJoCo collision detection remains the default. `collision` requires the Newton
pipeline selector. Deterministic runs disable MuJoCo's internal solver sensors;
Isaac Lab sensors continue to use Newton state. Camera options have normal typed
CLI fields. Explicit options unsupported by the selected renderer fail at startup.
Asynchronous OVRTX observations still need a timestamp/history contract.

Robot links and scene objects use the existing `PhysicsConfig`:

```python
link_physics = PhysicsConfig(
    mujoco=MujocoPhysicsConfig(
        friction=[0.7, 0.01, 0.002], condim=6, gravity_compensation=0.1,
    ),
)
link_physics_overrides = {
    "pelvis": PhysicsConfig(mass=5.2),
    "*ankle_roll_link": PhysicsConfig(
        mujoco=MujocoPhysicsConfig(friction=[0.45, 0.07, 0.03]),
    ),
}
```

Set these on `robot.asset`; objects use `object.physics` and scene files use
`object_configs`. Body-name globs follow the existing scene pattern: defaults
apply first, then the first matching override supplies its configured fields.
Set mappings in Python presets; YAML saving preserves their order. MuJoCo and
Isaac Lab 3 support robot overrides; Lab 2 reports an error if they are requested.
A whole-robot `link_physics.mass` remains prohibited; targeted mass is allowed.

Newton MJWarp reuses `mujoco.friction`, `solref`, `solimp`, `condim`, `priority`,
`solmix`, `margin`, `gap` and `gravity_compensation`. Sliding/torsional/rolling
friction map to Newton material channels; MuJoCo settings take precedence over
IsaacSim material defaults on Newton. The SDK importer does not expose MuJoCo
collision masks or arbitrary geom extras, so explicit requests fail.
`PhysicsConfig.newton` adds typed contact spring/adhesion, convex-hull and
SDF/hydroelastic properties. These are authored before cloning and compilation;
other physics engines leave the Newton block inactive. Per-link overrides retain
unmodified authored/default properties and cannot reach sibling assets.

`condim=6` enables sliding, torsional and rolling contact dimensions; coefficients
and solver support are still required. Joint gains, armature and friction remain
owned by the existing controller/actuator configuration. Additional joint/tendon
schemas and actuator command modes require their own typed controls and runtime
contracts; this configuration does not imply universal SDK asset support.

Lab 3 distinguishes actuator model limits from joint limits enforced by the
solver. Holosoma maps the robot's DOF effort and velocity limits to
`actuator_effort_limit` and `actuator_velocity_limit`. It leaves authored joint
solver caps intact, so an asset's lower cap can still constrain the applied
effort. Changing `use-newton-actuators` changes evaluation, not this distinction.

PhysX materials have static friction, dynamic friction and restitution.
Newton exposes one Coulomb coefficient (`mu`) plus restitution. Newton material
randomization maps `static_friction` to that coefficient; a paired
`dynamic_friction` distribution is unused and reported, while a dynamic-only
distribution is rejected. This is a physical model difference, not a second
friction channel hidden by the adapter.
Material writes resolve exact shape ownership from Newton's public model topology.
Its articulation material view assumes contiguous shapes and can include an
interleaved scene object; the adapter avoids that assumption.

Holosoma keeps XYZW state quaternions and WXYZ configuration quaternions.
The native adapter exposes explicit Torch views of Lab 3's proxy arrays,
uses indexed state writers, and composes world-frame forces through the native
wrench composer. Environment replication, camera rendering, and video cleanup
use Lab 3's lifecycle APIs.

Lab 3 LiDAR retains the configured visual mesh targets under either physics
engine. Native mesh casters track their motion upstream; the adapter adds
fixed-size ray-pattern updates. Newton's collision-BVH caster is a separate
candidate because it senses different geometry.
World-mounted Newton LiDAR uses a supported bodyless site per environment,
with the fixed mounting pose applied as a ray offset.
Explicit actor state flushing resolves Newton's pending FK before a sensor
capture, so moved targets are visible without first advancing physics.

The native backend restores the robot's Isaac Lab default pose, velocity,
and joint state after physics startup, before domain randomization runs.
Kit PhysX performs a warmup integration while creating its tensor views:
`PhysxManager._warmup_and_create_views()` calls
`update_simulation(get_physics_dt(), 0.0)` directly before and after view creation.
These calls do not increment
Holosoma's experiment clock. Setting the configured defaults after reset
prevents warmup gravity and motion from changing the first observation.

Lab 3 matches prim expressions against the whole path. Generated environment
segments use `env_[^/]+`, which excludes descendants from asset spawn parents.
Caller-provided LiDAR mesh expressions keep their original spelling; use
`/World/envs/env_[^/]+/Actor/...` when the environment wildcard should cover one
path segment.

The Lab 3 launcher sets
`--/exts/omni.kit.telemetry/skipDeferredStartup=true` to run telemetry startup
serially while Kit initializes its native interfaces. Telemetry remains enabled.
An explicit value in `--simulator.config.isaacsim.kit-args` overrides this
startup ordering.

Explicit `performance`, `balanced`, and `quality` rendering modes retain the
Lab 2 preset settings through native RTX Kit arguments. Lab 3 removed the
upstream preset launcher option. Omitting a mode keeps its experience defaults,
and explicit `kit-args` override the preset settings.

For URDFs with fixed joints marked `dont_collapse="true"` or `"1"`, Lab 3
retains fixed joints because Isaac Sim 6.2's merger ignores that preservation
flag. This keeps the configured G1 and T1 foot marker bodies for observations. G1 consequently
has additional native fixed-link bodies; Holosoma still exposes bodies in the
configured order.

## Upstream references

- [Isaac Lab 3.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- [Migration guide at the supported tag](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/isaaclab/migrating.rst)
- [Native simulation tutorial](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0/scripts/tutorials/00_sim)
- [Isaac Sim 6.2 source](https://github.com/isaac-sim/IsaacSim/tree/v6.2.0)
- [Physics backends](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/physics_backends.html)
- [Camera renderers and supported pairings](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/renderers.html)
- [Upgrade feature inventory and follow-up proposals](isaaclab3-upgrade-notes.md)
