# Isaac Sim backends

Backend names follow the **Isaac Sim major version**, separately from Isaac Lab.

| Preset | Runtime | Choose it for |
| --- | --- | --- |
| `simulator:isaacsim` | Isaac Sim 5.1 / Isaac Lab 2.3.2 (`isaacsim5`) | Existing experiments and checkpoints using the released integration. |
| `simulator:isaacsim6` | Isaac Sim 6.2 / Isaac Lab 3.0.0 (`isaacsim6`) | Selectable physics/renderers and standalone execution. |

Existing presets/configurations stay on `isaacsim5`; activation does not select a
backend. Migrate saved configs with `--simulator.config.isaacsim-backend=ISAACSIM6`.
The installed runtime must match. Changing engines can change policy performance.

## Isaac Sim 6: choose physics and rendering

Set `--simulator.config.sim.isaacsim6.physics-backend` independently of
`--simulator.config.sim.isaacsim6.renderer`. All six pairings below are supported.

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
bash scripts/setup_isaacsim6.sh
source scripts/source_isaacsim6_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac simulator:isaacsim6 logger:disabled \
    --simulator.config.sim.isaacsim6.physics-backend=newton \
    --simulator.config.sim.isaacsim6.renderer=ovrtx
```

Isaac Sim 5 uses `setup_isaacsim.sh` / `source_isaacsim_setup.sh`. Docker builds use
`docker/isaacsim.Dockerfile` or `docker/isaacsim6.Dockerfile` respectively.

## Keep shared settings in their existing locations

- **Solver:** Newton MJWarp reads `sim.mujoco` and `sim.mujoco.warp`; PhysX reads
  `sim.physx`. `sim.substeps` remains shared. Newton-only collision/execution
  controls use `sim.isaacsim6.newton`.
- **Objects and robot links:** use the existing `PhysicsConfig`,
  `robot.asset.link_physics` and ordered `link_physics_overrides`. Newton reuses
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
  `sim.isaacsim6.require_kit=True` with PhysX and a compatible renderer when using
  that event; detached physics rejects it.
- OVRTX uses the SDK's `1/60` native snapshot increment per submission, independently
  of physics time. Native output can retain prior material color; exact temporal
  reconstruction remains undocumented. A faster shared batch changes every
  camera's native capture history, including cameras with slower observation updates.
  All cameras and movies use ordinary shared SDK ownership and batching.
- Renderer quality is not interchangeable. Kit rendering presets apply to
  `isaac_rtx`; the typed `renderer_options` currently control Warp features.
  Camera `far` and `depth_clipping_behavior` remain authoritative for all three.
  Segmentation output is not implemented.
