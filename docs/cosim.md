# GPU co-simulation

One process, the head, simulates; a peer process on the same GPU consumes its state each frame and can answer with
buffers of its own. State and answers stay in shared CUDA memory, and the two step in lockstep. There are
three ways to use it, from least to most code:

| Use | Head side | Peer side |
|---|---|---|
| [Holosoma renders Holosoma](#render-cameras-in-another-holosoma-simulator) | `cosim-render` plugin | `python -m holosoma.cosim.render` |
| [Any renderer](#render-with-an-engine-outside-holosoma) | `cosim-render` plugin | a `PeerHandler` subclass |
| [Other co-simulations](#other-kinds-of-co-simulation) | a `HeadPlugin` subclass | a `PeerHandler` subclass |

The module docstrings in `holosoma.cosim` document the protocol (`bridge`), shared memory (`cuda`), the bases
(`base`) and the renderer contract (`render.plugin`).

## Render cameras in another Holosoma simulator

Simulate in one backend and render cameras in another, for example MJWarp physics with Isaac Sim images.
Start the renderer with the scene's `run_sim` arguments, then the head with the `cosim-render` plugin. A
head camera whose `remote` names the plugin's slot takes its images from the renderer, which configures
the same camera as an ordinary one:

```bash
python -m holosoma.cosim.render --endpoint ipc:///tmp/$USER/cosim.sock \
  simulator:isaacsim robot:g1-29dof terrain:terrain-locomotion-mix --training.num-envs 4 \
  sensor.overview:overview

python -m holosoma.run_sim \
  simulator:mjwarp robot:g1-29dof terrain:terrain-locomotion-mix --training.num-envs 4 \
  sensor.overview:overview --sensor.overview.remote cosim \
  plugin.cosim:cosim-render --plugin.cosim.endpoint ipc:///tmp/$USER/cosim.sock
```

Each time a camera is due, the head publishes after its local cameras render. The renderer applies the
state and renders its cameras straight into shared memory; the head's remote cameras read those images in
place (`camera/<name>/<rgb|depth>`), filled before `simulator.sensor_render_hook`, so observation terms and
camera plugins read them unchanged. Each camera renders with its own config on each side, so configure
them the same. The renderer exits when the head closes, including on Ctrl+C.

- Local and remote cameras mix on one head.
- Episode starts publish too: the renderer ends and starts the same environments' episodes, so its plugins
  (e.g. video recording) split episodes as the head does; its `episode_end` hooks see the reset state.
- `set_mount_pose()` works on a remote camera: the pose travels with the state, as `mount_pose/<camera>`.
  On the renderer, cameras the head also has follow the head's poses every frame.
- Several renderers: give each its own plugin slot and endpoint, and set each remote camera to the slot that
  supplies it. The remote cameras of one plugin share a decimation.
- `cosim-render` is an ordinary plugin, so any task that installs plugins can use it; the tested path is
  `run_sim`.

## Render with an engine outside Holosoma

Any process on the same GPU can serve `cosim-render` without a Holosoma simulator or task: subclass
`holosoma.cosim.base.PeerHandler` and run it under `holosoma.cosim.bridge.PeerBridge`.

```python
from holosoma.cosim.base import PeerHandler
from holosoma.cosim.bridge import PeerBridge


class EngineRenderer(PeerHandler):
    def accept(self, handshake):  # joint, body and object order; environment origins
        self.engine.load_layout(handshake)

    def reads(self, fields):  # head state fields to receive each exchange
        return ("rigid_body_pos_w", "rigid_body_quat_w", "object_root_states")

    def step(self, snapshot, state):
        if snapshot.env_ids is None:  # episode starts carry env_ids and are not rendered
            self.engine.render(state, time=snapshot.sim_time)


PeerBridge(EngineRenderer("cuda:0"), endpoint=endpoint).run()
```

The head names the fields the peer writes, one per remote-camera output, e.g. `camera/overview/rgb`.
`write_source(field)` says where each one's data comes from, once, at the handshake. It has two forms.

**The engine renders into a buffer you give it**: return a tensor shaped like the field. It is moved onto
the shared export, so the engine writes the head's image directly, with no copy. The Holosoma renderer
works this way.

```python
    def write_source(self, field):
        _, camera, data_type = field.name.split("/")
        image = torch.empty(field.shape, dtype=getattr(torch, field.dtype), device=self.device)
        self.engine.add_camera(camera, data_type, out=image)
        return image
```

**The engine hands back a frame each render**: return a callable. Its result is copied into the export after
every `step`.

```python
    def write_source(self, field):
        _, camera, data_type = field.name.split("/")
        self.engine.add_camera(camera, data_type)
        return lambda: self.engine.image(camera, data_type)
```

The head runs unchanged, as above. Body poses arrive in world coordinates, so the engine needs no
kinematics; the `holosoma.cosim.render.plugin` docstring lists the fields, conventions and image layouts it
must follow.

## Other kinds of co-simulation

`holosoma.cosim.base` assumes nothing about rendering. The handshake is by SimState field name: the head
offers its fields, the peer picks the shared ones it `reads`, and the head names the ones the peer `writes`.
`HeadPlugin` owns the head side, and a subclass chooses those writes, the handshake, and the hooks
that publish. `PeerHandler` is the peer. A written output (a camera image) serves the peer's memory; a
written writable field flushes into the head's simulation at its next mutation.

```python
import torch

from holosoma.cosim.base import PeerHandler, HeadPlugin
from holosoma.hooks import Phase
from holosoma.simulator.base_simulator.state import DERIVED, STATE


class ForcePlugin(HeadPlugin):  # head: before every physics substep, the peer writes the forces
    def __init__(self, cfg, simulator):
        forces = torch.zeros(simulator.num_envs, simulator.num_bodies, 3, device=simulator.sim_device)

        def apply(rows, values):  # flushed into the simulation before the next substep
            forces[rows] = values[rows]
            simulator.apply_external_force("robot", forces=values[rows], env_ids=rows)

        # Registered before the bridge offers the head's fields; forces last one substep, and the peer
        # writes them every exchange.
        simulator.state.register(
            "external_forces", lambda: forces, write=apply, stale_with=(STATE,), stales=(DERIVED,)
        )
        super().__init__(cfg, simulator)

    def writes(self):
        return ["external_forces"]

    def install(self, hooks):
        yield hooks.add(Phase.PRE_STEP, self.bridge.publish, name="forces.exchange")


class ForceSolver(PeerHandler):  # peer: read contacts, write forces
    def reads(self, fields):
        return ("contact_forces",)

    def write_source(self, field):  # a new tensor each step, so copied after it
        return lambda: self.forces

    def step(self, snapshot, state):
        self.forces = solve(state["contact_forces"])
```

Register the plugin's config like any other plugin (subclass `CosimPluginConfig`). A peer that only reads
state needs no new head code: a `cosim-render` plugin that no camera names still publishes whenever a
local camera is due. Several co-simulations on one head share its state buffers; each refreshes only the
fields its peer reads.

## Requirements and limits

- Same host and same physical GPU (checked by UUID), CUDA on both sides, and a shared PID namespace: the
  peer waits while the head process runs.
- The head's plugin selects where SimState caches are allocated, so list it before any plugin that reads
  state.
- A Holosoma renderer needs the same scene: joint and object order must match, and it adopts the head's
  environment origins (Isaac Sim moves its clone grid; MuJoCo re-places world-mounted sensors). It renders
  every camera each frame, so its sensors must not be decimated; an Isaac Sim renderer serves cameras only.
- Lockstep: the head waits for every exchange, with no overlap between physics and the peer's work.
- Each shared buffer is written only by the process that allocated it: the head's state and the peer's
  exports. The other reads it in place and fails the exchange if it wrote one; the Holosoma renderer serves
  the head's state as its own SimState this way, and the head serves the renderer's images as its remote
  cameras' outputs. Those are valid until the next exchange; code reading them on another CUDA stream
  synchronises first.
- After a reset, a remote camera keeps the reset environment's frame from before the reset until the next
  frame, as local cameras do.
- Failures propagate to both processes; there is no reconnection. The handshake and the first exchange time
  out after five minutes (`STARTUP_TIMEOUT_MS`, first use compiles); later exchanges after 30 seconds
  (`EXCHANGE_TIMEOUT_MS`). Both are `HeadBridge` arguments.
