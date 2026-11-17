# GPU co-simulation

One process simulates; another process on the same GPU renders its state, or consumes it some other
way. They share SimState buffers through CUDA IPC and step in lockstep. The code lives in
`holosoma.cosim`; its module docstrings document the protocol and the APIs.

## Render cameras in another simulator

Start the renderer with the scene's `run_sim` arguments, then run the source with the `cosim-render`
plugin. A source camera whose `remote` names the plugin's slot takes its images from the renderer, which
configures the same camera as an ordinary one:

```bash
python -m holosoma.cosim.render --endpoint ipc:///tmp/$USER/cosim.sock \
  simulator:isaacsim robot:g1-29dof terrain:terrain-locomotion-mix --training.num-envs 4 \
  sensor.overview:overview

python -m holosoma.run_sim \
  simulator:mjwarp robot:g1-29dof terrain:terrain-locomotion-mix --training.num-envs 4 \
  sensor.overview:overview --sensor.overview.remote cosim \
  plugin.cosim:cosim-render --plugin.cosim.endpoint ipc:///tmp/$USER/cosim.sock
```

The renderer exits when the source closes, including on Ctrl+C. Local and remote cameras can mix on one
source. For several renderers, give each its own plugin slot and endpoint, and set each remote camera to
the slot that supplies it.

Each time a camera is due, the source publishes its state after its local cameras render. The renderer
applies the state, renders, and returns the requested images, which fill the remote cameras before
`simulator.sensor_render_hook`, so observation terms and camera plugins read them unchanged. A
`cosim-render` plugin that no camera names still publishes when a local camera is due, for peers that
only read state.

## Requirements and limits

- Same host and same physical GPU (checked by UUID), CUDA devices on both sides, and a shared PID
  namespace: the renderer waits while the source process runs.
- The same scene: joint and object order must match. The renderer adopts the source's environment
  origins; Isaac Sim moves its clone grid, and MuJoCo re-places world-mounted sensors.
- List the plugin before any plugin that reads state; it selects where SimState caches are allocated.
- Lockstep: the source waits for each frame, with no overlap between physics and rendering.
- The renderer renders all of its cameras for every frame, and its sensors must not be decimated. An
  Isaac Sim renderer serves cameras only.
- After a reset, a remote camera keeps the reset environment's frame from before the reset until the
  next frame, as local cameras do.
- Remote cameras have no runtime `set_mount_pose()`.
- Failures propagate to both processes; there is no reconnection. Startup and the first frame may take
  minutes on first use; later frames time out after 30 seconds.

## Extending

- A renderer outside Holosoma: subclass `holosoma.cosim.base.BufferHandler` and run it under
  `holosoma.cosim.bridge.Receiver`. `holosoma.cosim.render.plugin` documents what the plugin sends and
  expects back.
- Another co-simulation: `holosoma.cosim.base.CosimPlugin` owns the source's side; a subclass chooses its
  requests, handshake metadata and publish hooks.
- Rendering supplied state without the bridge: `RenderTask`, in `holosoma.cosim.render.renderer`.
