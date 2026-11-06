# Isaac runtimes and native backends

The existing `isaacsim` preset uses Isaac Sim 5.1 / Isaac Lab 2.3.2. The new
presets use Isaac Sim 6.2 / Isaac Lab 3.0 in a separate Python environment.
Both runtimes use `holosoma.simulator.isaacsim.isaacsim.IsaacSim`.

| CLI preset | Runtime | Physics | Camera renderer |
| --- | --- | --- | --- |
| `simulator:isaacsim` | 5.1 | Isaac Sim PhysX | Isaac RTX |
| `simulator:isaacsim6` | 6.2 | Isaac Sim PhysX | Isaac RTX |
| `simulator:isaacsim6-newton` | 6.2 | Newton, MJWarp solver | Newton Warp |
| `simulator:isaacsim6-ovphysx` | 6.2 | Standalone OVPhysX | OVRTX |

## Installation

Install `uv`, then run:

```bash
bash scripts/setup_isaacsim6.sh
source scripts/source_isaacsim6_setup.sh
```

The default installation root is `${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}/isaacsim6`.
Set `ISAACSIM6_ROOT` to use another location. This creates an independent Python
3.12 environment; the 5.1 setup and activation scripts remain available.

`scripts/versions.sh` pins the Isaac Lab release commit. Setup uses that
release's `uv.lock`, importer extras, and USD dependency overrides. Engine and
renderer dependencies remain pinned when installing Holosoma. Pandas and
protobuf are resolved with Holosoma's analysis and telemetry requirements.

The container equivalent is `docker/isaacsim6.Dockerfile`.

## Configuration

Runtime selection lives in `simulator.config.isaacsim`. Existing saved
configurations containing only `rendering_mode` and `kit_args` retain 5.1
defaults.
On Lab 3, `rendering_mode` translates the existing performance, balanced, and
quality presets into native Isaac RTX global settings. Explicit renderer
options override the preset. These Kit settings remain unavailable with
standalone OVPhysX.

Run the existing G1 locomotion task with native Newton:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof \
  simulator:isaacsim6-newton \
  --training.headless=True \
  --logger.video.enabled=False
```

Newton solver choices are `mjwarp`, `kamino_padmm`, `kamino_dvi`, `xpbd`,
`featherstone`, and `vbd`. Each selection constructs the corresponding native
Isaac Lab solver configuration and physics manager. Solver support for a
particular physical model remains solver-specific.

Native settings can be supplied as JSON objects:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof \
  simulator:isaacsim6-newton \
  --training.headless=True \
  --logger.video.enabled=False \
  --simulator.config.isaacsim.physics-options '{"num_substeps": 2}' \
  --simulator.config.isaacsim.solver-options '{"iterations": 20}'
```

`physics_options`, `solver_options`, `renderer_options`, and
`visualizer_options` accept public fields from the selected native
configuration classes. Unknown fields and implementation-class overrides are
rejected. Native constructors run after applying options, preserving their
validation and solver dispatch.

Camera renderers can be selected independently through
`sensor.<name>.isaacsim.renderer.backend` and `.options`. The available backend
names are `isaacsim_rtx`, `newton_renderer`, and `ovrtx`. A camera inherits global
renderer options when it uses the global renderer; selecting another renderer
starts from that renderer's defaults. Global `renderer="none"` disables capture
unless a camera explicitly selects its own renderer.

Newton Warp does not implement Holosoma's F-theta lens calibration. Fisheye
cameras must explicitly select `isaacsim_rtx` or `ovrtx`; incompatible
configurations fail before launch.

Visualizers are selected separately using `visualizers` and configured by name
in `visualizer_options`. Headless execution selects no visualizers and rejects
an explicit nonempty visualizer list. An explicit `visualizers=[]` also disables
viewers in an otherwise interactive configuration. Isaac Lab validates native
engine, renderer, and Kit compatibility before launch.

Available visualizers are `kit`, `newton_gl`, `newton_rtx`, and `viser`.
Viser binds to `127.0.0.1` by default, with browser opening and public sharing
disabled. Use an SSH local port forward for remote access. Rerun is not exposed:
the pinned upstream viewer starts HTTP/gRPC servers whose SDK does not honor
the visualizer's bind address.

The native activation script and image default `OPENBLAS_NUM_THREADS=1`.
This avoids a reproduced SciPy/OpenBLAS at-fork crash during Kit startup.
Set it before starting Python; changing it after importing SciPy is too late.
It controls CPU BLAS threading, not the CUDA physics engine.

## Shared contracts

Robot, scene, sensor, randomization, state, and camera-control code stays under
`simulator/isaacsim/`. The `backends` package contains version and engine
adaptation: launch ownership, native configuration, property writes, ray
updates, and recording. There is no second simulator implementation under an
`isaacsim6` package.

Holosoma state tensors continue to use XYZW quaternions, world-frame poses,
and the configured joint/body ordering. The 5.1 adapter converts its native
WXYZ state; Lab 3 already uses XYZW. Configuration quaternions retain the
existing WXYZ convention and are converted at the native configuration boundary.
Lab 3 simulation timestamps use its context's monotonic physics-step count and
timestep; capturing a sensor frame does not advance that clock. Kinematic
playback retains its separate timestep-driven clock.
Native camera captures refresh the mount's world pose before rendering, and
Newton world-mounted LiDAR uses the native environment-root frame. Standalone launch
registers the data-only PhysX and MuJoCo USD schemas needed by fresh URDF imports.
Both PhysX paths author collision groups so overlapping environments stay
independent while retaining contact with shared terrain.

The Lab 3 URDF importer no longer implements cylinder-to-capsule replacement.
The adapter applies that existing Holosoma option to the generated USD before
physics initialization, preserving cylinder height, radius, local transforms,
visual geometry, and authored mass properties. Selective preprocessing uses the
native fixed-joint merger while protecting named body frames and legacy
`dont_collapse` joints. Other fixed links, including massless sensor frames,
merge into their parents with their transforms and inertial properties.

Native video uses a camera per environment because the pinned camera sensor
and Newton renderer require the full environment batch. Only `record_env_id`
is positioned and encoded. Camera memory and rendering cost scale with
environment count; disable video or reduce its resolution for large batches.
Standalone viewers do not acquire Kit's debug-drawing interface.

Randomization uses the existing shared keyed sampler. Native property writes
use Isaac Lab setters so derived mass and inertia data is updated. Newton has a
single friction coefficient: specify matching static/dynamic distributions or
only one of them. Spawn-time robot, object, and terrain materials must likewise
use equal static and dynamic friction. Independent coefficients are rejected.
Kamino's shared materials do not support this interface's per-environment material
randomization; that operation raises an explicit error.
Newton also rejects PhysX per-body linear/angular damping randomization rather
than writing USD attributes its solver does not consume. The existing WBT
presets request different static and dynamic friction ranges, so they require
an explicit compatible randomization configuration when selecting Newton.

OVPhysX scenes with simulated rigid objects use native full-stage import.
Its native clones do not consume live per-body damping edits, so importing every
world preserves per-environment randomization. This increases USD import work
and memory at startup; robot-only scenes retain native cloning. Damping writes
use OVPhysX's sealed control-stage updates.

## Validation

The same existing Isaac feature suite can target each preset. Run one simulator
process at a time on a GPU:

```bash
DISPLAY= WAYLAND_DISPLAY= HEADLESS=1 python -m pytest \
  --strict-markers --isaac-preset=isaacsim6 \
  -m '(isaacsim or isaacsim6) and not multi_gpu' \
  --ignore=thirdparty --ignore=src/holosoma_inference
```

Use `isaacsim6-newton` or `isaacsim6-ovphysx` for the other native presets.
`tests/ci/isaacsim6_ci_tests.sh` runs those selections sequentially. Software
contract checks, physical outcome tests, and renderer geometry checks serve
different purposes; passing configuration checks alone does not establish
physics or rendering equivalence.

The default Newton/MJWarp preset does not yet pass the complete feature gate.
Remaining failures include friction/contact response, PhysX damping and
restitution/combine-mode expectations, zero-range joints in synthetic robot
fixtures, and the incompatible WBT friction configurations described above.
Other Newton solver selections are exposed through their native configuration
classes but have not been validated with the complete Holosoma feature suite.
