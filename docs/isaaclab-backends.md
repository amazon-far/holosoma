# Isaac Lab backends

Backend names follow the **Isaac Lab major version**.

| Preset | Runtime | Choose it for |
| --- | --- | --- |
| `simulator:isaaclab2` | Isaac Sim 5.1 / Isaac Lab 2.3.2 (`isaaclab2`) | Existing experiments and checkpoints using the released integration. |
| `simulator:isaaclab3` | Isaac Sim 6.2 / Isaac Lab 3.0.0 (`isaaclab3`) | Selectable physics/renderers and standalone execution. |

Existing presets/configurations stay on `isaaclab2`; activation does not select a
backend. Migrate saved configs with `--simulator.config.isaaclab-backend=ISAACLAB3`.
The installed runtime must match. Changing engines can change policy performance.

`simulator:isaacsim` remains an alias for `simulator:isaaclab2` in training,
evaluation and `run_sim.py`. Released `setup_isaacsim.sh`,
`source_isaacsim_setup.sh`, reset and CI script names forward to the Lab 2
scripts. New configs and scripts use the Lab names; existing shared
`physics.isaacsim`, camera and Kit options retain their field names.

## Isaac Lab 3: choose physics and rendering

Set `--simulator.config.sim.isaaclab3.physics-backend` independently of
`--simulator.config.sim.isaaclab3.renderer`. All six pairings below are supported.

| Physics | Choose it for |
| --- | --- |
| `physx` (default) | Existing PhysX solver/material settings. Uses Isaac Sim PhysX with Kit, OVPhysX standalone. |
| `newton` | Newton MJWarp dynamics using shared MuJoCo/MJWarp settings. Standalone runs avoid Kit and PhysX. |

| Renderer | Choose it for | Main constraint |
| --- | --- | --- |
| `isaac_rtx` (default) | Existing Isaac RTX camera appearance and Kit integrations. | Active cameras, movies or the default viewer require Kit. |
| `newton_warp` | Simple camera rendering when RTX lighting/material fidelity is unnecessary. | Pinhole cameras only; Holosoma fisheye models are rejected. |
| `ovrtx` | RTX lighting/materials and fisheye cameras without Kit. | Cannot share a process with Kit. |

All three support RGB/depth and logger movies. Headless runs without cameras or
movies use standalone physics regardless of renderer preference; visual-mesh
LiDAR does not activate a camera renderer. Leave Kit presets/`kit_args` unset for
standalone runs. Native Warp/OVRTX GUI controls have not been validated here.

For example, Newton with OVRTX:

```bash
bash scripts/setup_isaaclab3.sh
source scripts/source_isaaclab3_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac simulator:isaaclab3 logger:disabled \
    --simulator.config.sim.isaaclab3.physics-backend=newton \
    --simulator.config.sim.isaaclab3.renderer=ovrtx
```

Isaac Lab 2 uses `setup_isaaclab2.sh` / `source_isaaclab2_setup.sh`. Docker builds use
`docker/isaaclab2.Dockerfile` or `docker/isaaclab3.Dockerfile` respectively.

The Isaac Lab 3 image contains both Kit and standalone backends. Shell activation
prepares native library lookup without loading those runtimes; Kit's Python
packages are exposed only when the selected cameras, movie, viewer or explicit
Kit requirement needs them. Standalone importers use ordinary namespace discovery.
Pip setup keeps the full runtime's wheels in a separate directory within the
same environment, sharing the locked dependencies. Source the setup script before
starting Python; select the runtime before importing Isaac Lab.

## Keep shared settings in their existing locations

- **Solver:** Newton MJWarp reads `sim.mujoco` and `sim.mujoco.warp`; PhysX reads
  `sim.physx`. `sim.substeps` remains shared. Newton-only collision/execution
  controls use `sim.isaaclab3.newton`.
- **Objects and robot links:** use the existing `PhysicsConfig` and
  `robot.asset.link_physics`. Newton reuses
  `PhysicsConfig.mujoco` friction/contact parameters; `PhysicsConfig.newton` adds
  contact-spring and mesh/SDF properties. Joint gains/limits remain in the
  controller/asset config.
- **Cameras:** use `update_decimation` for observation updates and held buffers.
  At a 50 Hz control rate, `1` updates at 50 Hz and `"10Hz"` at 10 Hz.
  Isaac RTX and OVRTX render the full SDK batch whenever any camera is due
  (or a movie needs a frame), then publish only due observations. Thus 50/10 Hz
  cameras both render at 50 Hz, while the slow observation holds between 10 Hz updates.
  Warp renders due cameras independently. Capture never advances physics.
  `viewer_update_decimation` controls viewer rendering
  in simulation workflows; `render_interval` is deprecated.
  Rates above the control clock and asynchronous observations are unsupported.

See typed fields/defaults in
[simulator.py](../src/holosoma/holosoma/config_types/simulator.py),
[mujoco.py](../src/holosoma/holosoma/config_types/mujoco.py),
[newton.py](../src/holosoma/holosoma/config_types/newton.py) and
[scene.py](../src/holosoma/holosoma/config_types/scene.py). Unsupported active
options fail at startup.

## Quirks that affect experiments

- Newton maps randomized `static_friction` to its single Coulomb coefficient;
  paired `dynamic_friction` is unused, dynamic-only randomization is rejected.
  Authored joint solver caps can limit effort below actuator limits.
- Live USD body-damping randomization needs Kit PhysX. Request
  `sim.isaaclab3.require_kit=True` with PhysX and a compatible renderer when using
  that event; detached physics rejects it.
- OVRTX uses the SDK's `1/60` native snapshot increment per submission, independently
  of physics time. Native output can retain prior material color; exact temporal
  reconstruction remains undocumented. A faster shared batch changes every
  camera's native capture history, including cameras with slower observation updates.
  All cameras and movies use ordinary shared SDK ownership and batching.
- Moving robot, link and actor camera mounts use the SDK's
  `update_latest_camera_pose` option. The pinned Lab 3 SDK also uses this option
  to publish poses to native renderers; fixed world mounts keep their cached pose.
- With Newton and Isaac RTX, the moving movie camera also writes its pose through
  the public USD view. The pinned Newton camera view updates native sites without
  reliably publishing this pose to Kit; rendering and extraction still use the SDK.
- Renderer quality is not interchangeable. Kit rendering presets apply to
  `isaac_rtx`; the typed `renderer_options` currently control Warp features.
  Camera `far` and `depth_clipping_behavior` remain authoritative for all three.
  Segmentation output is not implemented.

## Behavioral regression coverage

The actual-simulator harnesses in `tests/simulators/isaacsim` exercise physical
outcomes in four environments, independently of camera rendering. Both PhysX
and Newton MJWarp run the shared collision, momentum, support, friction, force,
torque, motion, property-response, restoration and environment-isolation scenarios.
Mass and inertia changes must alter the response to identical impulses, rather
than merely appear in native property readback.

Newton skips the five PhysX-specific contracts: live USD body-damping
randomization, configured body-damping decay, the damped robot/object twin, and
the two material-combine-mode scenarios. These skips do not establish a Newton
equivalent of those semantics. Friction fixtures give object materials higher
MuJoCo priority so the ground coefficient does not mask the tested value.
Newton's object-focused wall-collision and support-relocation controls use the
existing one-link robot fixture. The pinned stack produced intermittent
overflows/NaNs when an unrelated, unactuated G1 collapsed during those controls;
larger arenas and shared GPU streams did not reliably fix them. Their object
outcome assertions remain unchanged, and PhysX retains the G1 fixture.
These controls do not establish stability of unactuated G1 collapse in Newton.

For Newton, `fixed=True` attaches a single compound body to world with a
native `FixedJoint`. It has zero dynamic DOFs: no free joint, locking armature,
large mass or stiff constraint. MuJoCo represents the pose-settable fixed root
as a mocap body. This permits explicit fixture relocation without integrating
fixture motion. Authored kinematic scene bodies retain the SDK's separate
pose-driven free-joint representation unless an explicit `fixed=True` overrides
them. PhysX's existing representation remains unchanged.

Isaac Lab's Newton `RigidObject` requires an articulation view, which cannot
bind a standalone zero-DOF body. Holosoma's fixed actor extends the public
`AssetBase` lifecycle and accesses public Newton model/state arrays instead.
It binds bodies after cloning and rebinds on hard reset; selected pose writes
update the world attachment, both state buffers and the SDK's public dirty-state
notifications. Physics and renderers therefore observe the same placement.
Source assets are unchanged. Disabled rigid bodies, world-attached bodies and
collider-only scene geometry also load as fixed actors; genuine kinematic
authoring remains separate.

The pinned Newton stack still fails to initialize the tested **dynamic** URDF
rigid object: its object view searches for the body path while the model labels
the articulation with the enclosing asset path. That test executes and expects
only the specific SDK initialization error; other failures remain failures.
Fixed URDF objects use the zero-DOF actor path and are tested normally, as are
USD free/fixed objects and URDF robot articulations.

The fixed-fixture regression checks native zero-DOF topology, massless fixtures,
disabled/world-attached scene bodies, genuine kinematic support, robot and prop
contacts, selected relocation, rendered orientation, retained images, SDK scene
state restoration and hard-reset rebinding. The optional UCA1 regression reuses
`/session/resolved-config.json` and its external assets; it skips explicitly when
that dataset is unavailable. It checks the full fixture layout, supported props
and the rendered tabletop before removal and after partial restoration.
Depth must reflect fixture relocation on the first capture. The rotated-panel
test allows two further captures for the default temporal RGB reconstruction
to converge, then checks the projected silhouette. This does not promise a
pixel-perfect RGB silhouette immediately after an abrupt pose change.

Detailed UCA1 prop meshes still exhibit contact jitter with true fixed bodies.
Standalone Newton USD imports reproduce it with both CPU MuJoCo and MJWarp;
ordinary MuJoCo XML controls are calmer. This does not isolate whether imported
collision geometry/properties or solver settings cause the remaining difference.
The apartment regression requires strict settling for the trays/toys and bounded
support for the detailed meshes, recording their residual velocities. It does
not establish that every detailed prop becomes completely stationary.

Active Newton + Isaac RTX and Newton + OVRTX tests reuse the geometry, moving
mount, task frame-end and movie harnesses. The six physics/renderer pairings also
run the cadence and movie-coexistence harness: RGB/depth must match current
geometry in every tile, observations hold between due updates, retained output
buffers survive subsequent captures and teardown, sparse camera resets preserve
other tiles, and captures leave native dynamics and physics time unchanged.
The tested decimations are `1/5`, `2/3`, `1/5/2`, uniform `3/3` and single `3`.
These tests do not cover segmentation, GUI interaction or distributed execution.
Fixture/cadence lifecycle checks use SDK stop/clear before the shared harness's
hard exit; they do not validate natural interpreter shutdown.
