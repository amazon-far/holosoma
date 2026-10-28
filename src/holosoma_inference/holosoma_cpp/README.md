# Holosoma Inference (C++)

A native C++ runtime for the [`holosoma_inference`](../README.md) policies. It
reads the same presets, accepts the same command line and implements the same control state
machine as `run_policy.py`, without Python in the control loop.

| Robot      | Locomotion | WBT |
|:----------:|:----------:|:---:|
| Unitree G1 | ✅         | ✅  |
| Booster T1 | ❌         | ❌  |

It talks to the robot through Unitree SDK2 (DDS), so the same binary drives the real G1 and the
holosoma MuJoCo simulator bridge (`run_sim.py`).

## Build

Linux x86_64 or aarch64 (e.g. the G1 Jetson), CMake 3.16+ and a C++17 compiler. ONNX Runtime,
Unitree SDK2, yaml-cpp, nlohmann_json, tinyxml2 and GoogleTest are taken from the system when
installed and are otherwise downloaded at pinned versions.

```bash
cd src/holosoma_inference/holosoma_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use `-DONNXRUNTIME_ROOT=/path/to/onnxruntime` for a local ONNX Runtime and
`-DHOLOSOMA_FETCH_DEPENDENCIES=OFF` to forbid downloads. On macOS only the core library and the
tests build (Unitree SDK2 ships Linux binaries only).

## Run

The binary takes the Python presets and flags (`inference:<preset>`, `--task.*`, `--robot.*`,
`--secondary.*`, `--secondary none`).

Sim-to-sim whole-body tracking (start `python src/holosoma/holosoma/run_sim.py robot:g1-29dof` first):

```bash
src/holosoma_inference/holosoma_cpp/build/holosoma_run_policy inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.no-action-scales-by-effort-limit-over-p-gain --task.policy-action-scale 1.0 \
    --task.use-sim-time \
    --task.interface lo
```

Real robot (Unitree wireless controller, robot on `eth0`):

```bash
src/holosoma_inference/holosoma_cpp/build/holosoma_run_policy inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.no-action-scales-by-effort-limit-over-p-gain --task.policy-action-scale 1.0 \
    --task.use-joystick \
    --task.interface eth0
```

`fastsac_g1_29dof_dancing.onnx` has no `action_scale` metadata, so without the two action-scale flags the
preset falls back to `robot.default_per_joint_action_scale`, and this model then falls in simulation. The
same applies to the Python runtime.

Locomotion uses `inference:g1-29dof-loco`. Keyboard and joystick controls, the stiff-hold prompt,
dual mode (`x` / X switches to the FastSAC locomotion policy) and model switching (`1`-`9`,
Select) are the same as in the [Python runtime](../README.md#policy-controls).

## Performance

### Contextual policy library

An optional `contextual` configuration section selects the depth-conditioned policy through `holosoma::Policy`. It uses the same `OnnxModel`, provider setup, buffers, profiling and factory as the other native policies. Configure `task.use_phase: false`, the runtime joint names/defaults/gains, and the contextual depth geometry. Models use float32 `obs` and `depth_image`, optional `time_step`, named `actions`, and optional `climb_pred` outputs. The contextual policy supplies zero for the legacy `time_step` input.

The existing API is preserved: locomotion and WBT callers still construct `Policy(config)` or `Policy(preset, checkpoint)` and call `step(robot_state)`. WBT's `set_reference(...)`, commands, reset and `JointCommand` output remain unchanged. Contextual inference adds the overload below; existing callers do not need to supply a `ContextualInput`. Passing the depth and command events together with the robot state keeps them scoped to one tick, including failed or inactive ticks, rather than leaving a pending sensor frame behind a separate setter.

```cpp
holosoma::Policy policy(config);
holosoma::ContextualInput input;
input.monotonic_time = now_seconds;
input.depth = normalized_depth_frame;
input.velocity = std::array<double, 3>{vx, vy, yaw_rate};
const auto& command = policy.step(robot_state, input);
```

Each inference tick requires its own input. Optional command fields are events: omitting them retains the current command rather than resetting a pending transition. Missing depth inserts one zero frame; the caller owns freshness and any startup gate (the Python `NativeContextualRunner` stops an engaged policy once its newest frame is older than `contextual.max_depth_age_s`, 0.25 s by default, and refuses a start until frames return). Climb arbitration can evaluate twice using the same observation/history tick. Reset clears temporal policy state. A zero or non-finite orientation is rejected with the same strict state validation as every other policy kind.

Gravity is computed in the IMU body frame, either from the orientation or from the optional corrected gravity in `RobotState::projected_gravity` (the same IMU-frame quantity the other policies read), and then rotated through the joint chain `contextual.imu_to_obs_joint_chain` (each joint's current angle about its axis, in order from the IMU's link) into the observation frame the model was trained on. The default chain is the G1 waist (`waist_yaw_joint` z, `waist_roll_joint` x, `waist_pitch_joint` y), giving torso-frame gravity from the pelvis IMU; set `[]` for IMU-frame gravity, or the chain for another robot. Every listed joint must be a runtime joint.

Contextual and live WBT share a ±20 raw-action ceiling before action scaling and previous-action feedback; live WBT can retain a tighter checkpoint limit. Observation clipping and research-policy behavior are independent. Nearest-primitive calibration is caller-supplied through `velocity_primitives`, distance scales and an optional in-place yaw threshold. Metadata joint permutations and configured-gain precedence are retained.

The generic Python `Policy` binding accepts the same `ContextualInput`, and `NativePolicy.step_contextual` forwards to it. There is no Python contextual inference implementation. The standalone SDK runner does not acquire depth frames: transports embed this API and supply their own sensor input, engagement lifecycle and command publication. Existing performance figures below concern the named research policies, not contextual deployment latency.

Measured in sim-to-sim with the MuJoCo bridge on a c6i.4xlarge (Intel Xeon 8375C), using the runtimes' own
per-tick latency reports:

| Model | Python, per tick | C++, per tick | CPU while running (Python / C++) |
|:--|--:|--:|--:|
| `fastsac_g1_29dof_dancing.onnx` (WBT) | 1.12 ms | 0.21 ms | 2.2 / 0.11 cores |
| `fastsac_g1_29dof.onnx` (locomotion) | 0.76 ms | 0.17 ms | 2.2 / 0.11 cores |

Both runtimes track the full dance clip equally well (mean joint error 0.125 vs 0.126 rad, with the action-scale
flags above) and follow the same locomotion commands (0.41 m/s for a 0.5 m/s forward command, over 5 runs each).

## Differences from the Python runtime

- **Inputs:** keyboard and the Unitree wireless controller. USB gamepads (evdev), ROS 2 inputs,
  depth sensors and `wandb://` model paths are not supported. Download models first.
- **Sim time:** `--task.use-sim-time` reads the simulator clock from `LowState.tick`, which the
  simulator bridge sets to the simulation time, instead of the ZMQ clock.
- **Startup:** the runtime waits for the first `LowState` before sending any command.
- **Safety:** if robot state is older than `--runtime.state-timeout-s` (default 0.5 s, 0
  disables), or a command would be non-finite, the runtime damps the robot
  (`kp = 0`, `kd = --runtime.damping-kd`) and exits. On exit (Ctrl-C, L1+R1) it damps for 0.5 s.
- **Model checks:** a model whose ONNX `dof_names` differ from `robot.dof_names` is rejected.
  Each model loaded with `--task.model-path` keeps its own gains when you switch between them.
- **kp level:** the operator kp level cannot go below 0.

## Keeping parity with Python

- `config/inference/*.yaml` is generated from the Python preset registry. After changing a Python
  preset, run `python src/holosoma_inference/holosoma_cpp/scripts/export_presets.py`. A no-sim test fails
  while the files are stale.
- `tests/golden/*.json` are reference runs of the Python policies: commands, observations and
  motion timesteps, tick by tick, for scripted robot states and operator input. The C++ tests
  replay them. Regenerate them from the repository root with
  `python src/holosoma_inference/holosoma_cpp/tests/gen_golden.py` in the `holosoma_inference` environment.
