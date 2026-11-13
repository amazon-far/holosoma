# Source-clock rendering and GPU co-simulation

Holosoma can use one backend for physics and another for camera capture. The
source supplies unified state and simulation time. The receiver applies that
state and publishes cameras without integrating its own physics. A `RenderTask`
drives each frame; each backend supplies its SimState writers and one
render-without-physics primitive.

Isaac keeps normal `ArticulationCfg`, object loading and Isaac Lab `render()` →
`forward()`, including native FK and Fabric synchronization. MuJoCo's default
primitive runs forward kinematics and renders sensors from it.

## Native startup and capture

```python
with receiver_config.create_render_task() as task:
    # inputs maps existing Holosoma field names to read-only GPU tensors.
    task.simulator.apply_state(inputs)
    task.render_frame(sim_time=source_time)
```

`RunSimConfig.create_render_task()` builds a `RenderTask`: normal startup with
`sim.kinematic_playback`, the backend's native placement where it has one
(Isaac's clone grid), camera preparation, and every sensor rendering each frame.
It has no physics step; `task.step()` raises.

`RunSimConfig.create_task()` accepts two optional programmatic inputs for the
source:

- `buffer_allocator`: select SimState cache storage before any cached reads,
  including native initialization reads.
- `env_origins`: float32 `[num_envs, 3]` origins on the task device. A backend
  that cannot honor the requested placement raises an error.

Calls without these options retain normal startup. Runtime
allocator callbacks and tensors stay outside the serializable scene/config
models. Custom task subclasses must accept the new options when they use them;
ordinary construction still passes only the existing configuration argument.

`apply_state()` accepts complete writable SimState fields, such as
`robot_root_states` `[E, 13]`, `dof_pos` and `dof_vel` `[E, D]`, and
`object_root_states` `[E, O, 13]`. Roots use Holosoma world coordinates, xyzw
quaternions and world velocities. Joint order is `dof_names`; object columns follow
`object_state_names`. Inputs are copied into the receiver's SimState tensors and
reach the backend through its registered SimState writers at the next flush, so
any backend with those writers can receive state. Every environment is written
each frame, without reading the simulation first. Inputs are read, not retained.

`render_frame()` sets the kinematic clock to source time, emits `FRAME_BEGIN`,
calls the simulator's `render_applied_state(dt)` with the elapsed source time, and
emits `FRAME_END` for sensor capture and consumers. It returns after GPU work
finishes. Order consumers after `simulator.sensor_render_hook`. It does not emit
physics `PRE_STEP`/`POST_STEP` or integrate physics.

Source time must be finite, nonnegative and strictly increasing, except that
`render_frame(sim_time=..., repeat=True)` may repeat the previous time, for
state the source changed without advancing its clock. `frames_rendered` counts
only frames that are not repeats. Isaac also
checks that its physics clock never moves after preparation; render failure
propagates and leaves its physics stopped. Isaac receivers render cameras only;
source notifications own cadence. A `RenderTask` starts without the terrain height
scanner; other tasks can disable it with `terrain.terrain_term.height_scanner=False`.

## Optional CUDA IPC / ZeroMQ bridge

`holosoma.cosim` is an optional same-host, same-GPU bridge. It shares persistent
SimState cache allocations, sends only metadata through ZeroMQ, and uses CUDA
ready/done events. Scenes, environment placement and process/container deployment
remain caller-owned. The bridge validates joint/object order and environment
origins at startup, and that both processes use the same physical GPU (by UUID,
since `CUDA_VISIBLE_DEVICES` can renumber devices). It contains no Isaac SDK
access or simulator-type dispatch.

Receiver process:

```python
with receiver_config.create_render_task() as task:
    from holosoma.cosim.bridge import Receiver
    Receiver(task, endpoint=endpoint).run()
```

Source process, with the same scene and agreed environment origins:

```python
from holosoma.cosim.cuda import CacheAllocator

fields = ("robot_root_states", "dof_pos", "dof_vel", "object_root_states")
allocator = CacheAllocator(fields)
with source_config.create_task(
    buffer_allocator=allocator, env_origins=matched_origins,
) as task:
    from holosoma.cosim.bridge import Publisher
    publisher = Publisher(task.simulator, allocator, endpoint=endpoint)
    task.run()
```

By default the source publishes its non-empty writable SimState fields, and the
receiver rejects any field it cannot write when the connection opens. An explicit
field list selects a subset. The allocator must own every published field; the
publisher raises before connecting otherwise. Import CUDA helpers after provider
startup when they use Warp; importing `CacheAllocator` itself loads no GPU SDK.

Publication follows the source's camera capture: a frame is published whenever
any source camera is due. MJWarp orders Torch writers and readers around its Warp
physics, and every MuJoCo backend refreshes pending kinematics before rendering
sensors, so published state and source images match the latest step.

The source holds its caches while the receiver applies state, renders, publishes
cameras and completes GPU reads. Physics resumes after completion: serialized
lockstep, with no physics/render overlap or frame pipelining. Startup handles and
layout travel once; subsequent notifications carry sequence, physics step and
simulation time. Publisher close obtains an unmap acknowledgment before storage
can be released. Failures/timeouts propagate; reconnection and scene changes are
not implemented. Startup and the first frame exchange may take up to five
minutes (first-use compilation); later exchanges time out after 30 seconds. On a lost peer, retain exporting allocations until that peer
has exited or confirmed unmapping.

Warp's owning allocations use `ScopedMempool(..., False)` and remain GPU memory.
The caller must retain `CacheAllocator` while SimState or imported tensors refer
to that storage. Ordinary unselected caches continue using Torch allocations.
Normal Holosoma use does not require Warp IPC APIs; the export path requires
Warp's CUDA IPC arrays and events (tested with Warp 1.15 and 1.19).

Sharing caches removes a separate publication copy and CPU payload transfers.
It does not remove native-to-unified SimState refresh, receiver layout conversion
or native SDK writes. No promise of end-to-end zero-copy state application is made.

## Observation loopback

A source camera with `remote=True` takes its images from the receiver, so
`get_camera_data()`, observation terms and camera plugins on the source read the
receiver's frames unchanged. Local and remote cameras can mix on one source;
LiDARs stay local.

```python
from dataclasses import replace

source_cameras = {name: replace(camera, remote=True) for name, camera in cameras.items()}
```

The receiver keeps the same cameras with `remote=False`. Backends create no
native renderer or runtime mount-pose writer for a remote camera, so
`set_mount_pose()` raises for it. Each frame the backend renders its due local
cameras and then hands the due remote cameras to the sensor manager's remote
renderer, all inside sensor rendering, so buffers and `last_due` are final for
every consumer ordered after `simulator.sensor_render_hook`. A remote camera with
no remote renderer raises on its first render.

The `Publisher` is that remote renderer when the source has remote cameras. It
sends their names, data types and sizes when the connection opens, and the
receiver rejects any that differ from its own cameras before mapping state. The
receiver exports one persistent buffer per camera output, rgb `uint8 [E, H, W, 3]`
and depth `float32 [E, H, W, 1]`, and copies its sensor buffers into them before
signalling completion. The source then copies each due camera's image into a new
buffer, so earlier frames held by consumers stay valid. Publication follows the
remote cameras: a frame is published when one of them is due. Without remote
cameras the source publishes after capture, as above.

Sources reset environments after cameras render. When observations re-render the
reset environments, the publisher sends `RERENDER` with the current source time;
the receiver applies the reset state, renders every environment with
`repeat=True`, and the source takes only the reset rows. Locally rendered
cameras keep their frame from before the reset for that step.

Limits: the receiver renders all of its cameras for every published frame, while
the source updates only its due cameras. Runtime mount-pose changes and episode
or seed information are not forwarded. Images need the same same-GPU CUDA IPC as
state; CPU renderers are not supported.

## Batched object-root state

MuJoCo and Isaac expose writable `object_root_states` `[E, O, 13]` in
`object_state_names` order. Empty objects produce `[E, 0, 13]`. Robot roots and
joints remain separate existing SimState fields. Writes go through each backend's
actor writers; static objects take the pose and ignore velocity.

MJWarp resolves addresses once and gathers roots on GPU, converting body-local
angular velocity to world coordinates for free objects. Jointless objects use
world body poses and zero velocities. The classic backend retains its per-actor
CPU getter. Changing topology requires a new simulator and address plan.

Scene fixtures, process or container launchers, benchmarks and recordings are
left to callers; the APIs and bridge need no task subclass, scene patch, custom
state writer, stream fence or camera initialization helper.
