# Isaac Sim backends

Holosoma keeps Isaac Sim 5.1 as the default and provides an opt-in backend for the
released Isaac Lab 3.0 stack.

| Preset | Isaac Sim | Isaac Lab | Python |
|---|---|---|---|
| `simulator:isaacsim` | 5.1 | 2.3.2 | 3.11 |
| `simulator:isaacsim6` | 6.2.0 | 3.0.0 | 3.12 |
| `simulator:isaacsim6-newton` | 6.2.0 | 3.0.0 | 3.12 |

The GA checkout is pinned to `v3.0.0` at
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. The `isaacsim6` name identifies the
major-generation integration; it now uses 6.2, the simulator dependency in the GA lockfile.

## Shared code remains in the 5.1 package

`holosoma/simulator/isaacsim/` owns the shared articulation/contact configuration,
camera and LiDAR setup, object spawning, USD prim utilities, conversion helpers,
randomization events, physics-property access, quaternion/state adaptation, lighting,
viewport controls and recording. Shared entry points default to 5.1 behavior.

`holosoma/simulator/isaacsim6/` owns the Lab 3 launcher and simulation driver plus small
adapters for the native backend, cloning and frame-view APIs. It imports the shared
5.1 helpers; it does not carry copies of those libraries. An explicit `IsaacLabApi`
profile supplies quaternion order, environment namespace and USD API differences.
Neither importing a helper nor choosing a camera launches a viewer.

Both integrations report `SimulatorType.ISAACSIM`. Code that needs a specific
generation uses `holosoma/simulator/isaac_lane.py`.

## Installation and headless operation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and deactivate
Conda before setting up the GA environment:

```bash
bash scripts/setup_isaacsim6.sh
source scripts/source_isaacsim6_setup.sh

python -m holosoma.run_sim simulator:isaacsim6 --training.headless
python -m holosoma.run_sim simulator:isaacsim6-newton --training.headless
```

The setup creates a separate environment, Isaac Lab checkout and conversion/bytecode
caches under `~/.holosoma_deps_sim6`. It uses upstream `uv.lock` with the `isaacsim`
and `video` extras. Holosoma's NumPy and pandas ceilings are overridden only in this
environment because GA requires NumPy 2 and pandas 3. Other locked versions remain
constrained, except protobuf: W&B requires a stable version below the GA lock's 7.x
prerelease. The 5.1 installation and its dependency constraints remain unchanged.

An existing prerelease checkout must be reset with `scripts/reset_isaacsim6.sh`
before reinstalling. `HOLOSOMA_DEPS_DIR` can select another isolated workspace.

`isaac_physics_backend` chooses PhysX or Newton; `isaac_newton` supplies Newton
settings. Under PhysX, `use_newton_actuators=False` and
`enable_external_forces_every_iteration=False` preserve the established actuator and
force paths. Different solvers still require their own policy and dynamics validation.
OvPhysX is rejected because this integration needs Kit for asset import and RTX.

`training.headless=True` with no explicit `isaac_visualizers` opens no viewers.
Cameras and recording request offscreen rendering independently. Explicit viewer
selection remains available through `--simulator.config.isaac-visualizers`; ordinary
non-headless operation defaults to `kit`. GPU regression tests use headless mode.

## Native Lab 3 APIs and compatibility adapters

- **State and quaternions:** Lab 3 uses xyzw and Warp-backed `ProxyArray` buffers.
  Holosoma unwraps them to Torch views and preserves its xyzw state contract.
  Shared builders convert wxyz configuration values only for the Lab 3 profile.
- **Cloning:** the source environment's asset, sensor and fixture configurations
  populate the native `ClonePlan`. Both engines receive all requested environments.
- **Randomization:** shared events use one property adapter for 5.1 tensor views,
  Lab 3 PhysX views and Newton attributes. Newton has a single friction coefficient;
  independent static-only friction randomization raises an explicit error.
- **LiDAR:** the Lab 3 adapter resolves the native `MultiMeshRayCaster` implementation
  and adds mutable ray directions. Upstream owns mesh transforms and tracking.
  Robot, world and actor camera mounts use the same shared camera builder.
- **Recording:** the shared recorder captures and encodes frames. Lab 3 supplies
  `FabricFrameView` pose writes and an on-demand RTX update for headless capture.
- **URDF fixed links:** the GA importer still ignores `dont_collapse` markers while
  merging fixed joints. The shared builder preserves marked frames by disabling
  merging for those URDFs only under the Lab 3 profile. Conversion caches are
  isolated by runtime version.
- **Newton object views:** a small adapter resolves a rigid object's owning
  articulation path for native Newton views. This is separate from the shared
  object-spawning logic.

## Validation

The matrix records runtime checks on the pinned release, not inferred compatibility
from the prerelease implementation. Tests use separate processes and explicit
completion markers or result files, because Kit shutdown can mask process exit codes.
All local simulator runs are headless; desktop keyboard/debug controls retain the
shared implementation but have no GUI validation in this run.

Intermittent native Kit startup crashes were observed on the RTX 5090 validation
host before Holosoma scene construction. Completed checks below establish feature
behavior, not reliable startup across every launch. No cause has been established.
Newton playback remains unverified because its runs hit this startup failure;
the PhysX playback checks complete successfully.

Validation on 2026-10-23 UTC: 1,201 CPU tests pass in the 5.1 environment and 1,222
in the GA environment. Focused helper/cleanup checks and Python 3.8 import checks
also pass. Live headless checks cover both engines' state, scenes, sensors,
randomization and recording, including nonblank encoded video from environment 1.

Regenerate the package versions and capability table from the GA environment:

```bash
python scripts/isaac_capability_report.py --write
python -m pytest -m no_sim src/holosoma
python -m pytest src/holosoma/tests/simulators/isaacsim6
```

For a 5.1 regression, activate `scripts/source_isaacsim_setup.sh` and run the existing
`tests/simulators/isaacsim` suite. The GA and 5.1 suites share their assertion harnesses.

<!-- BEGIN GENERATED: scripts/isaac_capability_report.py -->

### Environment used for validation

| Package | Version |
|---|---|
| `isaacsim` | 6.2.0.0 |
| `isaaclab` | 28.0.0 |
| `isaaclab-physx` | 8.0.0 |
| `isaaclab-newton` | 9.1.4 |
| `isaaclab-ov` | 5.1.5 |
| Python | 3.12.14 |

The Lab release is **v3.0.0**; its Python packages use a separate version scheme.

### Physics backends

| Backend | Supported | Runtime evidence |
|---|---|---|
| `physx` | yes | Four-environment state, cloning and contact-support checks pass on GA. |
| `newton` | yes | Four-environment state, cloning and contact-support checks pass on GA. |
| `ovphysx` | no | Holosoma's current asset and sensor pipeline requires Kit. |

### Existing Holosoma capabilities

`verified` means the named harness completed on this GA environment. `unverified` means runtime coverage is incomplete; `blocked` means the backend is unsupported. All recorded runs were headless. Desktop visualizers have no runtime verification here.

| Capability | PhysX | Newton | OvPhysX | Evidence |
|---|---|---|---|---|
| Rigid-body state read/write | verified | verified | blocked | static_move_assert: four environments, static relocation and dynamic contact support. |
| Articulation state read/write | verified | verified | blocked | all_root_states_unified_assert: robot/object reads, full and partial writes. |
| Kinematic playback | verified | unverified | blocked | playback_assert: PhysX clip tracking, FK and frozen physics clock pass. Newton validation is blocked by native Kit startup crashes before the scene is built. |
| Scene spawn (USD, multi-body) | verified | verified | blocked | scene_spawn_assert: four-environment multibody and multibody-override scenes; object_spawner_cfg_assert. |
| Environment cloning | verified | verified | blocked | static_move_assert: four distinct environments populated through native ClonePlan. |
| Lighting rig | verified | verified | blocked | scene_spawn_assert --probe-lights: UsdLux types, properties and orientations. |
| Contact sensors | verified | verified | blocked | isaac_sensor_state_assert: finite forces/history and contact response in all four environments. |
| Height-scanner ray caster | verified | verified | blocked | isaac_sensor_state_assert: finite ray hits at the known ground height in four environments. |
| Cameras (robot-link / world mount) | verified | verified | blocked | Headless RGB and depth capture; geometric and cadence harnesses cover the shared camera path. |
| Domain randomization (mass, friction, inertia, CoM) | verified | verified | blocked | dr_matrix_assert: read-back of robot and object properties. Newton has one friction coefficient. |
| External wrench / push events | verified | verified | blocked | dr_matrix_assert: robot and object pushes with velocity read-back. |
| Debug draw / viewer keyboard | unverified | unverified | blocked | Shared viewport tracking and Kit keyboard controller; desktop viewers were not launched. |
| Cameras (actor mount) | verified | verified | blocked | camera_actor_mount_assert: nonblank frames change when the owning actor moves. |
| LiDAR | verified | verified | blocked | lidar_assert: two environments, static/moving meshes, ray cycling, filters and update cadence. |
| Video recording | verified | verified | blocked | video_recording_assert: nonblank frames and encoded MP4 from environment 1. |

Accepted visualizer names: `kit`, `newton_gl`, `newton_rtx`, `rerun`, `viser`. These names describe upstream choices, not verified desktop support. Install the corresponding Isaac Lab extras before selecting optional viewers.

<!-- END GENERATED -->
