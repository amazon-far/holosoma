# Isaac Sim backends

Both versions use the `IsaacSim` simulator, shared scene configuration,
state accessors, sensors, domain randomization, and hooks. Native API differences
live in `holosoma/simulator/isaacsim/backends`.

| Profile | Isaac Lab | Isaac Sim | Setup / activation | Dockerfile |
| --- | --- | --- | --- | --- |
| `isaaclab2` | 2.3.2 | 5.1 | `setup_isaacsim.sh` / `source_isaacsim_setup.sh` | `docker/isaacsim.Dockerfile` |
| `isaaclab3` | 3.0.0 | 6.2 | `setup_isaacsim3.sh` / `source_isaacsim3_setup.sh` | `docker/isaacsim3.Dockerfile` |

The Lab 3 profile uses Python 3.12, PyTorch 2.11 with CUDA 13.0,
Warp 1.17, and NumPy 2. The old profile retains its dependency constraints.
The new image uses Isaac Sim's bundled Python to create its virtual environment;
the standalone installer uses Lab's pinned Isaac Sim extra.
Lab 3 setup includes the locked video extra required for native headless rendering
and recording.

## Select a profile

```bash
bash scripts/setup_isaacsim3.sh
source scripts/source_isaacsim3_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac simulator:isaacsim3 logger:disabled
```

Activation sets `HOLOSOMA_ISAACSIM_BACKEND=isaaclab3`, so existing commands with
`simulator:isaacsim` also select the new backend. The explicit `isaacsim3`
preset works for training and `python -m holosoma.run_sim`.
Saved configurations lacking `isaacsim_backend` retain `isaaclab2`; to use one
with the new runtime, override `--simulator.config.isaacsim-backend=ISAACLAB3`.
An explicit backend that conflicts with the installed runtime fails at launch.

```bash
docker build -f docker/isaacsim3.Dockerfile -t holosoma-isaacsim3 .
docker run --rm --gpus all --ipc=host \
    -e HOLOSOMA_MULTIGPU=False holosoma-isaacsim3 \
    bash tests/ci/isaacsim_ci_tests.sh
```

This CI command checks the runtime and GPU before running the existing Isaac Sim
tests, then the pure Python suite. The workflows build, test, and promote the
new profile separately and include it in nightly training. Multi-GPU tests
require at least two visible devices.

## Lab 3 options

These settings are under `--simulator.config.sim.isaaclab3.*`:

| Option | Default | Effect |
| --- | --- | --- |
| `use-fabric` | `True` | Use native Fabric synchronization. |
| `create-stage-in-memory` | `False` | Author assets into the simulation's in-memory stage. |
| `use-newton-actuators` | `True` | Use the new actuator evaluation path with the PhysX solver. |
| `contact-force-mode` | `normal` | Preserve the prior normal-force reward input; `total` includes friction. |

Holosoma keeps XYZW state quaternions and WXYZ configuration quaternions.
The native adapter exposes explicit Torch views of Lab 3's proxy arrays,
uses indexed state writers, and composes world-frame forces through the native
wrench composer. Environment replication, camera rendering, and video cleanup
use Lab 3's lifecycle APIs.

The native backend restores the robot's Isaac Lab default pose, velocity,
and joint state after physics startup, before domain randomization runs.

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
