# Native inference library

The C++ library owns model execution, observation history, action scaling and
policy state. Python binds that same implementation. Constructing a library
policy opens no robot, socket, keyboard listener or SDK interface.

## Python

The shared front door is `Policy(preset_path, checkpoint_path)`. Its `step(RobotState)` returns a `JointCommand` for locomotion, embedded-motion WBT or host-reference WBT. `set_reference(ReferenceWindow)` supplies the latter's reference. The preset/config factories below compose this same class.

Build the extension with a C++17 compiler and CMake:

```sh
uv pip install ./src/holosoma_inference
```

Dependencies are resolved by CMake from installed packages or pinned downloads.
Set `ONNXRUNTIME_ROOT` for an existing SDK, including a CUDA-enabled SDK. The
default source build uses CPU inference. Robot SDKs are optional extras for the
explicit command-line runner, not dependencies of the device-free API.

```python
from dataclasses import replace
import numpy as np
import holosoma_inference as inference

config = inference.get_config("g1-29dof-loco")
config = replace(config, task=replace(config.task, model_path="policy.onnx"))
policy = inference.get_policy_from_preset(config)
policy.reset()

# One already-packed actor observation, in the model's configured term order.
obs_config = config.observation
width = sum(obs_config.obs_dims[t] for t in obs_config.obs_dict["actor_obs"])
width *= obs_config.history_length_dict.get("actor_obs", 1)
targets = policy.step(np.zeros((1, width), dtype=np.float32))
```

`get_config` returns an independent copy and includes registered extension
presets. SDK-specific fields on robot-config subclasses remain available to the
Python transport owner; the native engine receives the common robot contract.
`InferenceConfig` and `LocomotionPolicy(config)` are also exported for
composition without a named preset. Model paths may be local or supported W&B
URLs, resolved before native construction. Packed observations must have shape
`(N,)` or `(1, N)` and contain finite values. The returned `(1, num_joints)` array
contains absolute joint-position targets, including defaults and calibration
offsets, not raw neural-network actions.

For native observation construction, `policy.step_state(state)` accepts the
existing joint-ordered flat state:
`[base_pos(3), quaternion_wxyz(4), joint_pos(N), base_lin_vel(3),
base_ang_vel(3), joint_vel(N)]`, optionally followed by corrected gravity (3).
It returns a command with `q`, `dq`, `tau`, `kp` and `kd`. Start with
`autostart=False` for explicit `policy.command("START")` / `"STOP"` / `"INIT"`
control. `velocity(x, y, yaw)` supplies a locomotion request.
`step_state` advances the gait phase once per call; an external loop that already
advances it passes `advance_phase=False`. `infer_offsets(state)` warms the native
model and history without changing engagement or publishing a command.

## Live-reference whole-body tracking

Embedded-motion policies read their reference outputs from the model. A
live-reference policy instead receives reference frames from its caller and
derives its observation layout from ONNX `experiment_config` metadata. These
contracts remain explicit; neither guesses the other's layout from its width.

SONIC-aligned layout names refer to NVIDIA's public
[GEAR-SONIC model family](https://github.com/NVlabs/GR00T-WholeBodyControl).
The parser retains the corresponding exporter metadata keys for compatibility.

```python
from holosoma_inference import Policy, RobotState, ReferenceWindow

policy = Policy.from_live_model("live-policy.onnx")
state = RobotState()
state.dof_pos = state.dof_vel = [0.0] * policy.num_dofs()
reference = ReferenceWindow()
slots = max(1, len(policy.lookahead_offsets))
# Replace these illustrative values with caller-buffered frames at the trained offsets.
reference.q = reference.dq = [0.0] * (slots * policy.num_dofs())
reference.quat = [[1.0, 0.0, 0.0, 0.0]] * slots
policy.set_reference(reference)
policy.set_elapsed_seconds(0.0)
command = policy.step(state)
```

The snippet illustrates the types, not a robot motion sequence. Live frames use
the checkpoint's metadata joint order, wxyz orientation, body-local angular velocity and SI
units. `reset_heading()` reanchors heading and motion phase while preserving
history; `reset()` starts a new episode and clears history/action feedback.

History initialization is a runtime setting, `task.wbt_history_initialization` (`--task.wbt-history-initialization` for `run_policy` and the C++ runner, `history_initialization=` for `Policy.from_live_model` and `LiveWbtPolicy`, `WbtPolicy::SetHistoryInitialization` in C++). `repeat_first_valid_obs`, the default, seeds each observation term's history with that term's first valid, fully built frame, then rolls forward oldest-to-newest. `zero_pad` fills missing frames with zeros. A full `reset()` reseeds from the next observation, while heading-only reset and reference-buffer replacement preserve history. Changing the setting resets the policy, so one episode never mixes the two. Exports may still declare `wbt_history_initialization` as `repeat_first` or `zero` in ONNX metadata. The declaration is validated and reported in the load log, but the runtime setting decides, and the log says when it overrides the checkpoint. Any other spelling fails, in the metadata and in the setting alike. The setting applies only to metadata-driven (live) WBT checkpoints; embedded-motion WBT and locomotion presets accept it and ignore it, because their history is always zero-padded.

Raw actions are clipped symmetrically to at most 20 (`kWbtActionCeiling`) before action scaling and action-history feedback. This is a bound on raw policy output, not a joint-angle limit: scaled targets remain subject to the ordinary joint and driver limits. A declared `experiment_config.robot.control.action_clip_value` tighter than 20 further restricts the bound. A missing value, a value above 20, or `clip_actions=false` leaves the bound at 20; a missing flag behaves like `true`. Declared values are validated even when the flag is false. Nonboolean flags and invalid declared clip values fail initialization. The load log reports the effective clip and whether it came from the ceiling or the checkpoint. This applies to every metadata-driven live WBT model. Observation clipping (`clip_observations`) and the separate embedded-motion and locomotion policies keep their existing contracts.

The convenience live preset defaults to canonical G1 joint names. For another 29-joint humanoid, set `config.robot.dof_names` to its checkpoint's ordered `dof_names` before constructing `Policy`. The library requires exact agreement, including order, and derives default positions and anchor FK from those names. State, reference, command, and gain arrays all follow this same order; transports must reorder named wire messages at their boundary.
With no explicit elapsed clock, the live API advances one model timestep per call. Update `set_elapsed_seconds(seconds)` before each tick to preserve motion phase across dropped ticks; pass `None` to return to tick counting. `set_control_interval(seconds)` sets the elapsed-clock conversion. Python retains only the currently supplied reference and returns independent command snapshots. `set_reference(None)` clears it. The former `LiveWbtPolicy` entry point remains available for compatibility.

The reference is offset-major, with one quaternion per frame. Window dimensions must match the checkpoint; malformed or non-finite input is rejected before inference. The first future sample is also the current reference unless `ReferenceWindow.current` explicitly supplies a `PolicyReferenceFrame`. The default 0p2s preset uses five offsets, but metadata-complete checkpoints may declare other layouts.

`set_reference` changes the borrowed buffer, not the episode. Replacing a window during continuous streaming preserves heading anchors, history and motion time. When switching to an independent clip with a different starting heading, call `reset()` before the first step with that clip. This clears history/action feedback and recaptures both the robot and reference heading at the new episode's first state; it keeps the supplied reference attached. Pointer identity alone cannot identify a clip boundary because streaming callers may replace buffers every tick.

```python
policy.set_reference(next_clip_window)
policy.reset()
command = policy.step(state)
```

The equivalent C++ sequence is `policy.set_reference(&next_clip_window); policy.reset(); policy.step(state);`. Merely setting a new pointer intentionally continues using the old heading anchors.

## C++

```sh
cmake -S src/holosoma_inference/holosoma_cpp -B build/native \
  -DHOLOSOMA_BUILD_RUNNER=OFF -DHOLOSOMA_BUILD_TESTS=OFF
cmake --build build/native -j
cmake --install build/native --prefix "$HOME/.local"
```

Consumers use `find_package(holosomaInference CONFIG REQUIRED)` and link `holosoma::inference`. No robot SDK or YAML header is needed by consumers:

```cpp
#include <holosoma_cpp/api.hpp>

holosoma::Policy policy("preset.yaml", "checkpoint.onnx");
holosoma::RobotState state;  // fill joint positions, velocities and base orientation
const holosoma::JointCommand& command = policy.step(state);
```

Relative model paths inside a preset resolve against its directory. An explicit checkpoint argument resolves against the caller's working directory. For metadata-complete live models, construct `Policy(live_policy_config(path, use_gpu, threads))`; supply a borrowed `ReferenceWindow` before stepping and keep it alive until replaced. `num_dofs()`, `dof_names()`, `config()` and the `policy()` escape hatch follow the same API for all three policy families. `OfflineInterface` is also exported for shadow-mode integrations.

`GeneralWBTPolicy` adapts the metadata-driven core rather than maintaining another observation implementation. It shares the policy slot's ONNX session, including CUDA/thread options. `InferenceEngine` is a compatibility alias. Configuration factories, packed observations and explicit engagement remain available as composition extensions. Each instance has one owner; serialize calls, including reference mutation. The high-level Python preset facade provides a lock; direct `Policy` bindings release the GIL during inference and require caller serialization.

## Command-line compatibility

The Python `python -m holosoma_inference.run_policy` entry point retains Tyro
arguments, presets, secondary policies, W&B resolution, SDK plugins and input
providers. Built-in policies use the native engine. The Python transport layer
supports explicit `interface`, `velocity_input`, `command_provider` and
`shared_hardware` injection instead of constructing hardware and then patching
it out. SDK command overrides remain in joint order; the SDK owns motor
permutation. A configured simulation clock continues to use the Python clock
subscriber, while the C++ engine owns timestep conversion.

Robot-specific adapters can override `transform_command(command, state)` to
adjust final joint-order targets or gains, for example a smooth startup hold,
without reimplementing inference. Invalid shapes, non-finite values and negative
gains are rejected before transmission. Embedded-motion policies expose
`initial_motion` for such initialization, and adapters receive a model-switch
hook after the native model changes.

For editable development, reinstall after changing C++ source. Container
dependency stages compile that source before the final Python-source overlay,
so a Python-only edit can reuse the native build and a C++ edit invalidates it.

The former Python policy implementations are retained only under
`holosoma_cpp/tests/python_reference` as independent replay oracles. They are
excluded from the installed package. Custom policies should compose the native
library and keep transport-specific logic outside the inference engine.

## Verification

```sh
uv pip install './src/holosoma_inference[dev]'
cmake -S src/holosoma_inference/holosoma_cpp -B build/native-tests \
  -DHOLOSOMA_WITH_UNITREE_SDK=OFF -DHOLOSOMA_BUILD_TESTS=ON \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/native-tests -j
ctest --test-dir build/native-tests --output-on-failure
python -m pytest --import-mode=importlib src/holosoma_inference/tests -q
```

Tests cover both model contracts, observation history, heading/FK, lookahead,
clocks, reload failure, native Python loading and SDK-boundary joint ordering.
Synthetic models and recorded golden vectors do not establish physical robot
stability; evaluate the selected checkpoint on the intended robot separately.
