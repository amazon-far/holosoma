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
    task.simulator.set_kinematic_time(source_time)  # what simulator.time() reports
    task.simulator.apply_state(inputs)
    task.render_frame()
```

`RunSimConfig.create_render_task()` builds a `RenderTask`: normal startup with
`sim.kinematic_playback`, the backend's native placement where it has one
(Isaac's clone grid), camera preparation, and every sensor rendering each frame.
It has no physics step; `task.step()` raises.

`SimState.set_buffer_allocator()` selects SimState cache storage before the
first cached read. Plugins are installed before any backend reads state, so a
plugin can select it at install; the co-simulation plugin does.

`apply_state()` accepts complete writable SimState fields, such as
`robot_root_states` `[E, 13]`, `dof_pos` and `dof_vel` `[E, D]`, and
`object_root_states` `[E, O, 13]`. Roots use Holosoma world coordinates, xyzw
quaternions and world velocities. Joint order is `dof_names`; object columns follow
`object_state_names`. Inputs are copied into the receiver's SimState tensors and
reach the backend through its registered SimState writers at the next flush, so
any backend with those writers can receive state. Every environment is written
each frame, without reading the simulation first. Inputs are read, not retained.

`render_frame()` emits `FRAME_BEGIN`, calls the simulator's
`render_applied_state()`, and emits `FRAME_END` for sensor capture and consumers.
It returns after GPU work finishes. Order consumers after
`simulator.sensor_render_hook`. It does not emit physics `PRE_STEP`/`POST_STEP` or
integrate physics, and `frames_rendered` counts every call. Rendering does not
depend on time; `set_kinematic_time()` only sets what sensors and consumers read
from `simulator.time()`. Isaac
checks that its physics clock never moves after preparation; render failure
propagates and leaves its physics stopped. Isaac receivers render cameras only;
source notifications own cadence. A `RenderTask` starts without the terrain height
scanner; other tasks can disable it with `terrain.terrain_term.height_scanner=False`.

## Optional CUDA IPC / ZeroMQ bridge

`holosoma.cosim` is an optional same-host, same-GPU bridge. It shares persistent
SimState cache allocations, sends only metadata through ZeroMQ, and uses CUDA
ready/done events. Scenes, environment placement and process/container deployment
remain caller-owned. The bridge validates joint/object order and environment
origins at startup (after the renderer has matched them where it can), and that
both processes use the same physical GPU (by UUID, since `CUDA_VISIBLE_DEVICES`
can renumber devices).

From the command line, start the renderer with the scene's run_sim arguments, then run
the source with the `cosim-render` plugin. Configure the source's remote cameras as
ordinary cameras on the renderer:

```bash
python -m holosoma.cosim.render --endpoint ipc:///tmp/$USER/cosim.sock \
  simulator:isaacsim robot:g1-29dof terrain:terrain-locomotion-mix --training.num-envs 4 \
  sensor.overview:overview

python -m holosoma.run_sim \
  simulator:mjwarp robot:g1-29dof terrain:terrain-locomotion-mix --training.num-envs 4 \
  sensor.overview:overview --sensor.overview.remote True \
  plugin.cosim:cosim-render --plugin.cosim.endpoint ipc:///tmp/$USER/cosim.sock
```

The plugin installs the shared cache allocator, so it must be installed before anything
reads simulator state; list it first. Co-simulation plugins on one simulator share that
allocator. At startup the renderer adopts the source's environment origins: Isaac
moves its clone-grid environments to them, and MuJoCo re-places world-mounted sensors,
so world-frame state, world-mounted cameras and terrain agree.

Receiver process:

```python
with receiver_config.create_render_task() as task:
    from holosoma.cosim.bridge import Receiver
    from holosoma.cosim.render import RenderHandler
    Receiver(RenderHandler(task), endpoint=endpoint).run()
```

Source process, with the same scene and the plugin configured:

```python
from dataclasses import replace

from holosoma.config_types.plugin import CosimRenderPluginConfig

plugins = {"cosim": CosimRenderPluginConfig(endpoint=endpoint)}
with replace(source_config, plugin=plugins).create_task() as task:
    task.run()
```

The plugin owns when publication happens and what the source requests: it
publishes when a camera is due and fills remote cameras from the receiver's
images. `holosoma.cosim.bridge.Publisher` is the channel underneath:
the handshake at construction, then `publish()`, its mapped
`imports` and `close()`, called whenever its owner decides. On the receiving
side, `Receiver` maps the shared state and hands each frame to a handler;
`holosoma.cosim.render.RenderHandler` applies it to a `RenderTask`, renders it
and exports the requested camera outputs.

The source shares every non-empty SimState field, writable and derived. When the
connection opens, each side sends the buffers it exports (name, shape, dtype and
IPC handle) and the source also sends the buffers it requests from the receiver.
The receiving handler chooses the exported fields it reads; the receiver maps only
those, read-only, names them in its reply, and passes them to the handler each frame.
Each later exchange refreshes only those fields on the source. The render handler
reads the fields its simulator can write, rejects any whose shape or dtype differs
from its own, and applies them before rendering; other handlers may read any
exported field, derived ones such as `contact_forces` included, without writing
them anywhere. The
receiver rejects a read of a field the source does not share, and any scene layout
(joint, object and environment order) or GPU that differs. Import CUDA helpers
after provider startup when they use Warp; importing `CacheAllocator` itself loads
no GPU SDK.

Publication follows the source's camera capture: a frame is published whenever
any source camera is due. MJWarp orders Torch writers and readers around its Warp
physics, and every MuJoCo backend refreshes pending kinematics before rendering
sensors, so published state and source images match the latest step.

The source holds its caches while the receiver applies state, renders, publishes
cameras and completes GPU reads. Physics resumes after completion: serialized
lockstep, with no physics/render overlap or frame pipelining. Startup handles and
layout travel once; subsequent notifications carry the frame sequence,
simulation time and the source's hook phase they were sent from (`frame_end` for
camera frames, `episode_start` for re-renders after resets, `None` unless exactly
one phase is emitting). The receiver sets its kinematic clock to the frame's
time. Publisher close obtains an unmap acknowledgment before storage can be
released; after a failed exchange the publisher refuses further exchanges and
closes without contacting the receiver. Failures/timeouts propagate; reconnection
and scene changes are not implemented. Startup and the first frame exchange may
take up to five minutes (first-use compilation); later source exchanges time out
after 30 seconds. After the handshake the receiver waits for frames as long as
the source process (whose PID the handshake carries) runs, so both processes must
share a PID namespace. On a lost peer, retain exporting allocations until that peer has exited or
confirmed unmapping.

Warp's owning allocations use `ScopedMempool(..., False)` and remain GPU memory.
`shared_state()` keeps its `CacheAllocator` alive for the simulator's lifetime; a
caller that installs its own allocator must retain it while SimState or imported
tensors refer to that storage. Ordinary unselected caches continue using Torch
allocations. Normal Holosoma use does not require Warp IPC APIs; the export path
requires Warp's CUDA IPC arrays and events (Warp 1.15 to 1.19).

Sharing caches removes a separate publication copy and CPU payload transfers; the
native-to-unified SimState refresh and the native writes on the receiver remain.

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
`set_mount_pose()` raises for it. Sensor rendering is two FRAME_END hooks:
`simulator.sensor_capture_hook` renders the due local cameras, and
`simulator.sensor_render_hook` follows it once every camera buffer is final, so
consumers ordered after `sensor_render_hook` see remote images too. A due remote
camera that nothing filled raises at `sensor_render_hook`.

The `CosimRenderPlugin` fills the remote cameras from a hook ordered after
`sensor_capture_hook` and before `sensor_render_hook`. It requests one buffer per
camera output, rgb `uint8 [E, H, W, 3]` and depth
`float32 [E, H, W, 1]`, with the camera's mount, field of view, near/far range and
depth clipping. The render handler rejects any request that differs from its own
camera's before mapping state, then exports one persistent buffer per request and
copies its sensor buffers into them before signalling completion. The source
checks that the exports match its requests, then copies each due camera's image
into a new buffer, so earlier frames held by consumers stay valid. Publication follows the
remote cameras: a frame is published when one of them is due. Without remote
cameras the source publishes after capture, as above.

Sources reset environments after cameras render. On `EPISODE_START`, which fires
once per reset batch after the reset state is written, the plugin publishes a
frame at the current source time; the receiver applies the reset state and
renders every environment, and the source takes only the reset rows (the whole
image if a camera has none yet). A plugin that writes state at `EPISODE_START`
orders itself before this re-render with
`before=HookRef(<cosim slot>, CosimRenderHook.RERENDER)`, using
`holosoma.cosim.render.CosimRenderHook`. State
changes that start no episode, such as a motion-tracking command teleporting an
environment to a new clip, are not re-rendered: remote cameras keep their frame
from before the change for that step.

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
left to callers.
