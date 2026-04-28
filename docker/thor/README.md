# Thor Docker images — `holosoma_inference`

Jetson Thor (JetPack 7.1, Ubuntu 24.04, CUDA 13, aarch64 SBSA).

Two targets ship from a single `Dockerfile`:

| Target | Image name | Includes | Use case |
|---|---|---|---|
| `inference` | `holosoma-thor-inference` | Policy + unitree_sdk2. No ROS. | Joystick or keyboard input, self-contained inference on Thor. |
| `inference-ros` | `holosoma-thor-inference-ros` | Policy + unitree_sdk2 + ROS 2 Jazzy (FastDDS for rclpy, unitree's bundled CycloneDDS for the SDK). | `Ros2Input` — subscribe to `/cmd_vel` from any ROS publisher. |

## Build

From the repo root:

```bash
# No-ROS variant (smaller image, works with joystick/keyboard input)
docker build --target inference -t holosoma-thor-inference -f docker/thor/Dockerfile .

# ROS-enabled variant (required for Ros2Input)
docker build --target inference-ros -t holosoma-thor-inference-ros -f docker/thor/Dockerfile .
```

Or via Docker Compose (builds the same images, uses `compose.yaml`):

```bash
docker compose -f docker/thor/compose.yaml build inference
docker compose -f docker/thor/compose.yaml build inference-ros
```

Or via the scoped `Makefile` (run from `docker/thor/`):

```bash
cd docker/thor
make inference        # no-ROS
make inference-ros    # ROS-enabled
make both             # both
```

## Run (launch scripts)

The `scripts/` directory has thin wrappers around `docker compose run` for
the common input-mode combinations (joystick+joystick, ros2+joystick, etc.).
See `scripts/README.md` for the full list. Quick start:

```bash
cd docker/thor/scripts
./run_joystick.sh                      # blind-loco via joystick (no ROS)
./run_ros2_joystick.sh                 # /cmd_vel from ROS + joystick state
./run_shuttle_publisher.sh             # (other terminal) drives /cmd_vel
```

## Run (Docker Compose — recommended)

The compose file wires up the runtime flags (`--runtime nvidia`, host
network, host IPC, `--privileged`, GPU env) and mounts a model directory
from the host. Copy `.env.example` to `.env` to override `MODEL_PATH` if
your ONNX models live somewhere other than `$HOME/models`.

```bash
# Walking demo on a Unitree G1 via joystick (no ROS needed)
docker compose -f docker/thor/compose.yaml run --rm inference \
  inference:g1-29dof-loco --task.interface eth0

# Receive /cmd_vel from another ROS container on the same network
docker compose -f docker/thor/compose.yaml run --rm inference-ros \
  inference:g1-29dof-loco \
    --task.velocity-input ros2 \
    --task.state-input interface \
    --task.interface eth0
```

Or via Makefile shortcuts (run from `docker/thor/`):

```bash
cd docker/thor
make run-inference ARGS='inference:g1-29dof-loco --task.interface eth0'
make run-inference-ros ARGS='inference:g1-29dof-loco --task.velocity-input ros2 --task.state-input interface --task.interface eth0'
```

## Run (direct docker, if you don't want compose)

```bash
docker run --rm --runtime nvidia --network=host --ipc=host --privileged -it \
  -v $HOME/models:/models:ro \
  holosoma-thor-inference \
  inference:g1-29dof-loco --task.interface eth0

docker run --rm --runtime nvidia --network=host --ipc=host --privileged -it \
  -v $HOME/models:/models:ro \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  holosoma-thor-inference-ros \
  inference:g1-29dof-loco \
    --task.velocity-input ros2 \
    --task.state-input interface \
    --task.interface eth0
```

`RMW_IMPLEMENTATION=rmw_fastrtps_cpp` is also the image default. Publisher
containers can use either FastDDS or CycloneDDS — cross-vendor DDS interop
for standard message types is validated on Jazzy.

`--network=host` is required so unitree_sdk2's UDP multicast can reach the
robot. `--runtime nvidia` gives the container access to the Thor GPU.
`--ipc=host` lets the container share POSIX shared memory with sibling
containers (e.g. a ZED depth pipeline).

## Layer structure

Ordered stable → volatile so day-to-day code changes only rebuild the top
layer:

```
l4t-cuda (CUDA 13 devel, Ubuntu 24.04)           ← never
 └─ python-base (python3.12, build tools, uv)    ← ~never
     ├─ long-deps (NVPL, cuDSS, TensorRT libs)   ← ~never
     │   └─ common-deps (numpy, scipy, pin, …)   ← weekly-ish
     │       └─ app-deps (unitree_sdk2 + src)    ← every commit
     │           └─ inference                    ← terminal target
     │
     └─ ros-jazzy (ros-jazzy-ros-base + FastDDS + CycloneDDS)   ← ~never
         └─ long-deps-ros (same as long-deps)         ← ~never
             └─ common-deps-ros (same as common-deps) ← weekly-ish
                 └─ app-deps-ros (unitree_sdk2 + src) ← every commit
                     └─ inference-ros                 ← terminal target
```

Both branches share `l4t-cuda` and `python-base`. Everything below forks
because the ROS install mutates system apt state and Python deps need to be
installed on top of ROS's system packages to avoid ABI surprises.

## Version pinning

- **CUDA**: `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` (matches JetPack 7.1
  + CUDA 13 on Thor).
- **ROS 2**: Jazzy (Noble native; compatible with common ROS nav stacks).
- **`unitree_sdk2`**: `0.1.3` (set via `--build-arg UNITREE_SDK2_VERSION=...`).
  Bump when a new amazon-far release is published with a cp312 aarch64 wheel.
- **Python deps**: unpinned by design — let `uv` resolve. Source of truth
  for the dep list is `src/holosoma_inference/setup.py`.

## Troubleshooting

**Build fails at `wget ... unitree_sdk2-0.1.3-cp312-cp312-linux_aarch64.whl`**

The wheel asset may not yet be uploaded to the release. Check
https://github.com/amazon-far/unitree_sdk2/releases and ping the maintainer.

**`libcudss.so.0: cannot open shared object file`**

cuDSS installs to `/usr/lib/aarch64-linux-gnu/libcudss/13/`, which the
Dockerfile adds via `ld.so.conf.d`. If you see this error, something has
overwritten `ld.so.cache`; run `ldconfig` inside the container.

**Robot doesn't respond to `/cmd_vel` (inference-ros)**

Most common cause: the policy isn't in Walking state. `--task.state-input
interface` means `START/WALK/STAND` come from the joystick, not ROS — press
the start + walk combo on the Unitree controller before expecting incoming
velocity commands to take effect.

DDS interop: this image's rclpy runs on FastDDS by default
(`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`). Cross-vendor interop with a
CycloneDDS publisher is hardware-validated for `TwistStamped` on Jazzy-to-
Jazzy (see `scripts/run_shuttle_publisher_cyclonedds.sh`). If you observe
silent data-delivery failures with a mixed setup, first confirm both sides
are on the same ROS 2 distro — our own testing has seen cross-distro
pairings (e.g. Humble ↔ Jazzy) fail silently while same-distro cross-vendor
pairings work.

Why FastDDS for rclpy (not CycloneDDS): unitree_sdk2's pybind11 binding
bundles CycloneDDS 0.10.2 C++ libraries, ABI-incompatible with Jazzy's
CycloneDDS 0.10.5. Loading both in one process crashes. FastDDS and
CycloneDDS have disjoint binary symbol spaces, so they coexist cleanly —
rclpy on FastDDS, unitree_sdk2 on its bundled CycloneDDS.
