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
the full Isaac Sim package's Kit bootstrap in the same environment.

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
| `physx` | `newton_warp` | PhysX dynamics and Newton Warp cameras. |
| `newton` | `isaac_rtx` | Newton MJWarp dynamics and Isaac RTX cameras. |
| `newton` | `newton_warp` | Newton MJWarp dynamics and Newton Warp cameras. |
| `physx` | `ovrtx` | Kit-less OVPhysX dynamics and OVRTX cameras and movies. |
| `newton` | `ovrtx` | Kit-less Newton MJWarp dynamics and OVRTX cameras and movies. |

`isaac_rtx` and `newton_warp` retain Kit for the viewer and recording.
`ovrtx` uses the SDK's Kit-less launch, USD asset import and native camera
lifecycle. The `physx` physics choice selects OVPhysX for that runtime; existing
PhysX scene settings are authored into USD before the engine loads it.
Interactive OVRTX launches select the SDK's `newton_rtx` visualizer unless
`--visualizer` explicitly selects another compatible visualizer. Kit
experiences, Kit visualizers and Kit livestreaming cannot share an OVRTX
process and are rejected by Isaac Lab's launch validation. Kit rendering-mode
presets and `kit_args` must be unset with OVRTX. The native visualizer owns
its camera controls; Holosoma's Kit keyboard bindings apply to Kit viewers.
Live USD body-damping randomization also requires the Kit PhysX runtime;
detached physics does not consume those edits, so that event is rejected.
The renderer setting controls configured sensors exposed as `CameraRecord`.
Warp cameras trade RTX lighting and material fidelity for a simpler rendering
path; the adapter maps their zero-valued depth misses to infinity before applying
Holosoma's shared depth policy.
Current Warp camera integration supports pinhole projection. Use `isaac_rtx`
for Holosoma's fisheye configurations; selecting Warp with those configurations
is rejected rather than rendering a pinhole approximation.

Logger movies use Kit/Replicator RTX with the Kit profiles and native OVRTX
cameras with `ovrtx`. Movie cameras are declared before cloning, track the
selected `record_env_id`, and use the shared encoding, overlays and recording
lifecycle. Camera capture does not integrate physics. Disable logger video
when comparing camera renderer throughput.

The native integration matrix checks all six combinations with two G1
environments, sparse property randomization, dynamics, actuation, contact forces,
and independent RGB/depth geometry. Separate Newton cases cover visual-mesh
LiDAR and the training/checkpoint/evaluation workflow. These bounded checks do
not establish equal policy performance or throughput across engines.

## Lab 3 options

These settings are under `--simulator.config.sim.isaaclab3.*`:

| Option | Default | Effect |
| --- | --- | --- |
| `physics-backend` | `physx` | Select PhysX (OVPhysX with OVRTX) or Newton's MJWarp solver. |
| `renderer` | `isaac_rtx` | Select Isaac RTX, Newton Warp or Kit-less OVRTX. |
| `use-fabric` | `True` | Use native Fabric synchronization. |
| `create-stage-in-memory` | `False` | Author assets into the simulation's in-memory stage. |
| `use-newton-actuators` | `True` | Use native actuator evaluation, including the PhysX host adapter; this does not select Newton physics. |
| `contact-force-mode` | `normal` | Preserve the prior normal-force reward input; `total` includes friction. |

Newton solver options are under
`--simulator.config.sim.isaaclab3.newton.*`:

| Option | Default | Effect |
| --- | --- | --- |
| `use-cuda-graph` | `True` | Capture physics kernels on CUDA. |
| `nconmax` | `128` | Contact capacity per environment; `None` opts into the upstream automatic estimate. |
| `njmax` | `None` | Constraint capacity per environment; retain upstream default when omitted. |
| `solver-iterations` | `None` | MJWarp solver iterations; retain upstream default when omitted. |
| `default-shape-margin` | `0.01` | Default collision margin in meters, including mesh terrain; authored shape settings can override it. |

Configured capacities and iteration counts must be positive; the margin must
be finite and nonnegative. `sim.substeps` controls Newton's solver substeps.
PhysX uses the existing `sim.physx` solver settings.
The Newton contact default avoids the automatic estimate's observed overflow
in G1 scenes. More complex contact configurations can require a higher capacity.

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
