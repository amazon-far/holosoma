# Robot SDK bridges

A bridge publishes the simulated robot's state on the channels the real robot's SDK uses and reads
commands back, so a policy or driver talks to the simulator through the interface it uses on
hardware. Enable it with `--simulator.config.bridge.enabled True` and select one with
`--robot.bridge.sdk-type`.

| `sdk_type` | |
|---|---|
| `unitree` | In-process Unitree SDK. The default. |
| `unitree_mp` | The Unitree SDK with its binding in a spawned child process. See [Unitree bridges](#unitree-bridges). |
| `booster` | Booster SDK. No base-state channel, so `publish_odom` is a no-op. |

Extensions register further bridges under the `holosoma.bridge` entry-point group.

## Tuning transport cost

The bridge is a common source of slowdown. It runs on the simulator's step thread, so everything it
does is paid per step, and with light physics — one robot on a plane — it can easily cost more than
the physics itself. If a run will not hold real time, check the bridge before you blame the scene.

Two settings are worth reaching for, and both trade something:

- **`simulator.config.bridge.transport-decimation`** defaults to `1`, running the SDK exchange on
  every physics step — 200 Hz at `sim.fps 200`, usually well above what the consumer needs. An
  integer runs it every Nth step; a frequency string (`"100Hz"`) resolves against the physics rate.
  You pay in state freshness: a consumer reading faster than the cadence you pick sees stale state.
  The torque-control cadence is unaffected and stays at the physics rate.
- **`simulator.config.bridge.publish-odom`** defaults to `True`, so a bridge whose SDK has a
  base-state channel publishes base odometry on it for parity with hardware. Set it `False` when
  nothing in the pipeline reads that channel. The cost is not the publish itself but the simulator
  state read and world→body conversion in front of it.

To size either for a given configuration, run with `--time-scale 0` and compare the
`Simulation performance:` lines the simulation loop logs, toggling one setting at a time.

### Unitree bridges

`unitree_mp` behaves identically to `unitree` but runs the SDK's C++/CycloneDDS binding in a spawned
child process, at the cost of a blocking parent/child round trip on every transport update. It is
required only when the simulator process also loads `rclpy` (a ROS2 egress plugin): the Unitree SDK
and ROS2 each bundle CycloneDDS, and the two heap-corrupt in one address space. Prefer `unitree`
whenever ROS2 is not in the picture.

`publish_odom` on these bridges writes `SportModeState` to `rt/odommodestate`, the channel the
robot's onboard sport mode publishes, which keeps a downstream `read_odom_state` →
`/telemetry/odom` identical in sim and on hardware.
