# Robot SDK bridges

A bridge publishes the simulated robot's state on the channels the real robot's SDK uses and reads
commands back, so a policy or driver talks to the simulator through the interface it uses on
hardware. Enable it with `--simulator.config.bridge.enabled True` and select one with
`--robot.bridge.sdk-type`.

| `sdk_type` | |
|---|---|
| `unitree` | In-process. The default, and the cheapest. |
| `unitree_mp` | Same behaviour with the SDK binding in a spawned child process. Required only when the simulator process also loads `rclpy` (a ROS2 egress plugin): the Unitree SDK and ROS2 each bundle CycloneDDS, and the two heap-corrupt in one address space. Costs one blocking parent/child round trip per transport update. |
| `booster` | Booster SDK. No base-state channel, so `publish_odom` is a no-op. |

Extensions register further bridges under the `holosoma.bridge` entry-point group.

## Transport cost

The bridge, not the physics, dominates a headless sim's step time. Three settings control it:

- **`robot.bridge.sdk-type`** — `unitree` avoids the round trip `unitree_mp` makes to its SDK child
  on every transport update. Only ROS2 coexistence requires `unitree_mp`.
- **`simulator.config.bridge.transport-decimation`** — defaults to `1`, running the SDK exchange on
  every physics step: 200 Hz at `sim.fps 200`. An integer runs it every Nth step; a frequency string
  (`"100Hz"`) resolves against the physics rate. A consumer reading faster than this cadence sees
  stale state. The torque-control cadence is unaffected and stays at the physics rate.
- **`simulator.config.bridge.publish-odom`** — defaults to `True`, so base odometry reaches
  `rt/odommodestate` as the robot's onboard sport mode publishes it, and a downstream
  `read_odom_state` → `/telemetry/odom` is identical in sim and on hardware. Set it `False` when
  nothing reads that channel. The DDS write is the cheap part; the simulator state read and
  world→body conversion in front of it are the cost.

To size any of this for a given configuration, run it with `--time-scale 0` and read the
`Simulation performance:` lines the simulation loop logs, toggling one setting at a time.
