# Thor launch scripts

Shell wrappers over `docker compose run --rm ...` for common policy invocations.
Build the image once (`cd docker/thor && make inference` or
`make inference-ros`), then launch with one of these scripts.

Each script passes through extra arguments to `run_policy.py`, e.g.:

```bash
./run_joystick.sh --task.model-path /models/my_custom.onnx
```

| Script | Target image | velocity_input | state_input | Policy |
|---|---|---|---|---|
| `run_joystick.sh` | `inference` | joystick | joystick | g1-29dof-loco |
| `run_keyboard.sh` | `inference` | keyboard | keyboard | g1-29dof-loco |
| `run_ros2_joystick.sh` | `inference-ros` | ros2 (`/cmd_vel`) | joystick | g1-29dof-loco |
| `run_ros2_keyboard.sh` | `inference-ros` | ros2 (`/cmd_vel`) | keyboard | g1-29dof-loco |
| `run_wbt_joystick.sh` | `inference` | joystick | joystick | g1-29dof-wbt |
| `run_shuttle_publisher.sh` | `inference-ros` | (publisher, not policy) | — | — |
| `run_shuttle_publisher_cyclonedds.sh` | `inference-ros` | (publisher, not policy) | — | — |

`run_shuttle_publisher.sh` publishes a 3s-forward / 3s-back shuttle pattern
on `/cmd_vel` to validate the Ros2Input cross-container DDS path. Launch
`run_ros2_joystick.sh` in another terminal first; transition the policy
to Walking via the joystick; then run the shuttle publisher.

`run_shuttle_publisher_cyclonedds.sh` is the same publisher but forces
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` — useful for validating cross-vendor
DDS interop when the policy container runs rclpy on FastDDS (the default)
and the publisher uses CycloneDDS.
