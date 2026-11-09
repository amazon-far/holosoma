# Isaac Lab 3 upgrade notes

This inventory separates changes included in Holosoma's Lab 3 integration from
upstream features proposed for later review. It follows the final
[Lab 3.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
and [2.x migration guide](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/migrating.html),
rather than beta dependency pins or limitations.
The integration tests exercise all six declared physics/camera combinations,
Warp and OVRTX movies, OVRTX fisheye, visual-mesh LiDAR and training/checkpoint/evaluation
with both physics choices, plus physics-only startup with all renderer preferences
and standalone Newton LiDAR and training. They do not
establish equivalent trajectories, policy quality or throughput.

## Included in this integration

| Change | Holosoma behavior | Official source |
| --- | --- | --- |
| Explicit version selection | `simulator:isaacsim` selects Lab 2; `simulator:isaacsim3` selects Lab 3. Activation loads dependencies. CI and child harnesses receive explicit version arguments. | [Migration: backend configuration](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/migrating.html) |
| Physics selection | `physics_backend=physx\|newton` selects PhysX or Newton MJWarp. Standalone runs select the OVPhysX implementation of PhysX. Other Newton solvers are outside this selection. | [Physics backends](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/physics_backends.html) |
| Camera renderer selection | `renderer=isaac_rtx\|newton_warp\|ovrtx` selects configured camera sensors independently of physics. Warp accepts pinhole projection. Warp and OVRTX use the Kit-less launcher and native sensors for both cameras and logger movies. Physics-only runs leave the renderer inactive. | [Renderers](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/renderers.html) |
| Runtime dependencies | Runtime selection precedes SDK launcher imports. Newton standalone cases check actual loaded libraries for accidental Kit/PhysX loading; URDF conversion registers only codeless PhysX schema resources. Physics-only cases check all renderer preferences and record startup timing. | [Launch ownership](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/app/sim_launcher.py) |
| Warp-native data | The adapter reads explicit `.torch` views of `ProxyArray`, preserving shared accessors and cloning only where ownership requires it. | [Proxy arrays](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/proxy_array.rst) |
| Quaternions and partial writes | Holosoma state stays XYZW; configuration WXYZ is converted at the native boundary. Indexed writers preserve partial reset/write semantics. | [Migration](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/migrating.html) |
| Native actuators | `ActuatorCollection` owns commands and telemetry. Model limits remain separate from authored solver limits. The native actuator option does not choose Newton physics. | [Final release: actuator ownership](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0) |
| Contact semantics | `normal` preserves the prior observation contract; `total` includes tangential forces. The selected property feeds existing substep accumulation. | [Final release: sensor contracts](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0) |
| Material and body randomization | Backend adapters use actual material channels and native property setters. Newton has one friction coefficient, a position-only COM update, and exact material shape ownership from public model topology. | [Newton backend](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0/source/isaaclab_newton/isaaclab_newton/assets) |
| Dynamic mesh raycasting | Native mesh casters own target transform tracking. Holosoma retains visual target discovery and fixed-size pattern changes. Dynamic multi-mesh casting already existed in Lab 2.3.2. | [Ray caster concepts](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/concepts/sensors/ray_caster.rst), [2.3.2 configuration](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_cfg.py) |
| Stage and launch ownership | `launch_simulation` owns runtime lifetime; native scene cloning owns the environment subtree. Native cameras use `FrameView`, avoiding the legacy Fabric mount repair. | [PhysX lifecycle](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/concepts/physics-backends/physx.rst) |
| Importers and physics schemas | Conversion and body discovery account for rewritten importer hierarchies. Marked fixed joints remain available for configured robot observations. | [Migration: importers and schemas](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/migrating.html) |
| Dependencies and images | Separate Python 3.12 / Torch 2.11 / Warp 1.17 / NumPy 2 profile, frozen upstream video, OV and standalone importer extras, bundled interpreter, prebundle repointing and Docker CARB shim. Legacy constraints remain separate. | [Pinned project and lock](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/pyproject.toml), [upstream Dockerfile](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docker/Dockerfile.base) |

The configuration reference and command examples are in
[Isaac Sim backends](isaacsim-backends.md).
Physics/rendering selection does not promise equal trajectories, image fidelity,
policy performance or throughput. Upstream's
[PhysX↔Newton policy-transfer guide](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/how-to/transfer_policies_between_physx_and_newton.html)
addresses ordering and tuning differences.

## Upstream opportunities requiring separate review

The [capability audit and implementation proposal](isaaclab3-capability-audit.md)
examines configuration and runtime gaps in more detail, including native physics
fragments, per-prim rules, solver tuning, importer caching, actuators and sensors.
It distinguishes the native configuration/schema additions from the remaining
proposals.

| Proposal | Benefit | Limits and deployment requirements | Official source |
| --- | --- | --- | --- |
| Newton collision-BVH raycasting | Query all live collision shapes, including moving bodies, without maintaining target meshes. | `NewtonRaycastSensorCfg` ignores mesh target paths. Collision geometry, own-body hits and target filtering differ from current visual-mesh observations. Requires Newton physics and contract comparison. | [Configuration](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/sensors/ray_caster/newton_raycast_sensor_cfg.py) |
| Sim physics raycast sensor | Arbitrary ray origins/directions and `ray_time_offsets`; owned snapshots or read-only live buffers. | Uses physics collision queries per ray. Live zero-copy views are host NumPy buffers, with reset/lifetime and driving-thread restrictions. No established GPU batching or speed advantage over current Warp casting. | [Physics sensor overview](https://github.com/isaac-sim/IsaacSim/blob/v6.2.0/source/extensions/isaacsim.sensors.experimental.physics/docs/Overview.md), [implementation](https://github.com/isaac-sim/IsaacSim/blob/v6.2.0/source/extensions/isaacsim.sensors.experimental.physics/plugins/isaacsim.sensors.experimental.physics/RaycastSensorImpl.cpp) |
| RTX LiDAR via `Lidar` / `LidarSensor` | Full scans, vendor profiles, intensity, semantics/object IDs and optical material behavior; GPU output and physics-time scheduling. | Requires Kit/RTX, OmniSensor assets and current experimental sensor APIs. Scan completion, timestamps, buffer ownership and observation layout differ from batched range matrices. RTX IDs/materials already existed in Sim 5.1. | [RTX sensor overview](https://github.com/isaac-sim/IsaacSim/blob/v6.2.0/source/extensions/isaacsim.sensors.experimental.rtx/docs/Overview.md), [Sim 6.0 release](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html) |
| RTX Minimal and scene partitioning | Lower-cost camera shading and environment-local geometry visibility. | Changes image fidelity and lighting. Partition rules differ for sensor cameras, viewport, livestream and XR; partition capacity is bounded. Benchmark identical outputs and delivered frame age. | [Renderers](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/renderers.html) |
| Asynchronous OVRTX captures | Overlap rendering and simulation. | Current OVRTX captures are synchronous. Async capture adds one-capture latency; capture pose/intrinsics must match delayed images and partial resets can re-prime tiled products. Requires observation timestamp and history integration. | [Capture timing](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/renderers.html) |
| Batched image processing, frame history and PPISP | Fused layout/normalization, explicit frame stacks and HDR camera modeling. | Lab manager observation terms are not Holosoma's pipeline. Preserve shapes, channel order and ownership; ISP modeling adds a new fidelity/dependency surface. | [GA camera migration](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0), [PPISP package](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0/source/isaaclab_ppisp) |
| Warp fisheye mapping | Extend cheaper camera rendering to supported distorted projections. | Upstream supports OpenCV fisheye, while Holosoma currently authors USD polynomial fisheye settings. Those require an explicit distortion mapping; the current Warp path rejects them. | [Newton Warp renderer](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py) |
| Additional Newton solvers and deformables | VBD/MPM, particles, cables, coupled solvers, XPBD/Featherstone and experimental Kamino. | Not drop-in rigid humanoid solver modes. Require new asset/contact/material/task contracts and solver-specific capability tests. Stricter determinism is supported only for specific configurations. | [Newton physics configuration](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py) |
| Additional visualizers and recording tools | Newton GL, Viser and Rerun for lighter inspection and tooling. | OVRTX already uses native cameras for logger movies and defaults to Newton RTX for interactive viewing. Other visualizers need their own camera controls, capture and shutdown coverage. | [GA release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0) |
| New robot assets / lighter collision variants | Multiphysics assets and fewer collision primitives can reduce solver work. | Asset dynamics, self-collision, body order and checkpoint behavior can change. Do not silently substitute these for configured G1/T1 assets. | [Assets changelog](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_assets/docs/CHANGELOG.rst) |
| Teleop, system identification, ROS2 Control and deployment tools | MCAP/CloudXR workflows, tuning, external control and LEAPP/ONNX export. | New products and runtime dependencies rather than requirements of this migration. ROS/streaming availability needs its own validation. Holosoma retains its trainers and inference path. | [Sim 6.1](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html), [Sim 6.2](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html), [Lab GA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0) |

Performance evaluation should compare equal environment counts, geometry,
resolution, sensor update rates and outputs. Measure completed GPU work and
image age as well as physics throughput; asynchronous submission time alone
does not establish a faster observation pipeline.

## Release coverage

The review covered Lab 2.3.2, 3.0 beta, beta2, beta2.patch1, EA and GA release
families, plus the core, assets, contrib, experimental, mimic, Newton, OV, PhysX,
PPISP, RL, tasks, experimental tasks, teleop and visualizer package changelogs.
The final GA release supersedes early `wp.array` descriptions with `ProxyArray`
and removes several EA launch/configuration helpers.

Isaac Sim release notes from
[5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html),
[6.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html),
[6.0.1](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html),
[6.1](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html)
and [6.2](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html)
were included. Their relevant progression is:

- **6.0:** new core/sensor API families, Newton integration, physics-time
  multitick rendering and instantaneous RTX LiDAR full scans; removal of old
  compatibility shims and Camera/JSON RTX sensor authoring.
- **6.0.1:** asset toolchain updates and sensor, buffer teardown and asset-loading fixes.
- **6.1:** Newton tensor parity, hydroelastic/solver additions, SPG CUDA sensor
  processing, mixed-rate sensor fixes, renderer GPU selection and ROS2 Control.
- **6.2:** Newton 1.6.1 / MJWarp 3.12 / Warp 1.17, batched native actuators,
  all-substep state-buffer fixes, view invalidation after model rebuild, owned
  sensor snapshots/live views and batched camera authoring.

The final Lab dependency lock also aligns USD 26.08, USD exchange 3.0.1 and
Newton schemas 0.5. Older Isaac Sim or beta exact pins should not replace that
coherent installation. Optional standalone OV libraries need their own extras;
package presence alone does not establish a supported Holosoma integration.
