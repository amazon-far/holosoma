# Holosoma Inference Service

Service layer for deploying `holosoma` policies. The input/output API is ROS2 messages; a backend turns them into robot motion. The service is not required to run `holosoma_inference`, but can be helpful for integrating it into a larger system.

## Service API

Inputs:

| Topic | Type | Description | Exp. Rate |
|---|---|---|---|
| `/holosoma/smplh_command` | `CmdSMPLH.msg` | SMPL-H 24-joint pose targets, retargeted to dense before the policy | 50Hz |
| `/holosoma/dense_tracking_command` | `CmdDense.msg` | Dense per-joint 29-DoF target (q/dq + root quat) consumed directly by the policy | 50Hz |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | Velocity command for the locomotion policy | any (timeout-guarded) |
| `/holosoma/state_input` | `std_msgs/String` | Discrete state commands: `start` / `stop` / `init` / `walk` / `stand` / `switch_mode` / `kill` (in-band emergency kill) | on demand |
| `/holosoma/exoskeleton_command` | `CmdExoskeleton.msg` | Left/right arm joint targets (7+7) plus base twist for the split-body controller | 50Hz |
| `/holosoma/3pt_command` | `Cmd3pt.msg` | Head + wrist poses with grippers (not supported yet) | 50Hz |
| depth topics (configurable) | `sensor_msgs/Image` | Raw metric depth (`32FC1`, meters) for depth-conditioned policies; topics + preprocessing (resize, clip, `frame_delay_ms`) come from the preset's `task.depth` config | camera rate |

Outputs:

| Topic | Type | Description | Rate |
|---|---|---|---|
| `/holosoma/holosoma_executed_cmd` | `JointState.msg` | Executed joint command, always full 29-DoF | Policy Rate (typ. 50Hz)  |
| `/holosoma/heartbeat` | `Heartbeat.msg` | Liveness + status | every 10th control tick (5Hz at 50Hz) |

Note: for `/holosoma/holosoma_executed_cmd`, the policy backend fills all 29 values. The split-body backend fills the 14 arm joints (indices 15–28) and zeros the rest.

## Internal structure

One node — `policy_service_node` (`ServiceIONode`) — owns **all** ROS2 I/O and runs any holosoma policy (locomotion, WBT, …) the same way `run_policy.py` does, including runtime policy swapping (dual-mode). It replaces the former `holosoma_node`.

```bash
CmdSMPLH ─▶ retargeter_node ─┐
                             ├─ CmdDense ──────────▶ ┐
external publisher ──────────┘                       │
/cmd_vel (TwistStamped) ────────────────────────────▶ ├─ policy_service_node ─▶ G1
/holosoma/state_input (String: start/stop/kill) ───▶ │  (one node owns all I/O;
depth topics (sensor_msgs/Image, optional) ────────▶ ┘   executed_cmd + heartbeat out)

CmdExoskeleton ────────────────▶ unitree_split_controller ─▶ G1   (arm_sdk + loco)
```

- The policy is selected by the **inference preset** (tyro positional, e.g. `inference:g1-29dof-loco`); a WBT preset resolves to its policy class via `config.task.policy_type`.
- Velocity/state arrive through the node's `injected` input providers (`--task.velocity-input injected --task.state-input injected`, the launch-file default). Use `interface` instead for the native G1 joystick (L1+R1 kill).
- The Unitree SDK runs in a multiprocess proxy (`unitree_mp`, forced by the node), isolating its CycloneDDS from the node's rclpy.
- The retargeter is its own node (`retargeter_node`): `CmdSMPLH -> IK -> CmdDense`, decoupling variable-latency IK from the control loop. Implementations are discovered from the `holosoma.retargeter` entry-point group (`g1-smpl` built in; extensions register their own and select via `retargeter:=<name>`).

## Input Support across Modes

| mode \ input                                      | `CmdSMPLH` (24-joint)        | `CmdDense` (29-DOF)          | `/cmd_vel` + `state_input`   | `CmdExoskeleton` (arm q + twist) | `Cmd3pt` |
|---------------------------------------------------|------------------------------|------------------------------|------------------------------|----------------------------------|----------|
| **WBT policy** (`teleop_with_holosoma_policy`)    | ✅ `input_type:=smplh` (retargeter) | ✅ `input_type:=dense` (direct) | ✅ state only                | ❌                               | ❌       |
| **loco policy** (`teleop_with_loco_policy`)       | ❌                           | ❌                           | ✅                           | ❌                               | ❌       |
| **split-body** (`teleop_with_unitree_split_body`) | ❌                           | ❌                           | ❌                           | ✅                               | ❌       |

## Build & source

Build & source before launching; re-run after editing a `.msg`.

```bash
cd src/holosoma_inference/holosoma_inference_service
rm -rf build install log # for a clean build
colcon build && source install/setup.bash
```

## Run (locomotion policy)

```bash
ros2 launch holosoma_service teleop_with_loco_policy.launch.py \
    model_path:=<model.onnx> interface:=eth0
# preset defaults to inference:g1-29dof-loco; override with preset:=<inference-preset>
```

Drive it:

```bash
# velocity (e.g. the bundled demo publisher, from the repo root)
python demo_scripts/ros2_velocity_publisher.py --pattern shuttle
# state transitions / emergency kill
ros2 topic pub --once /holosoma/state_input std_msgs/String "{data: start}"
ros2 topic pub --once /holosoma/state_input std_msgs/String "{data: kill}"
```

## Run (WBT policy)

Whole-body tracking with SMPL-H teleop (retargeter on, the default):

```bash
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    preset:=<inference-preset> \
    urdf_path:=<fixed-base g1_29dof.urdf> \
    input_type:=smplh \
    model_path:=<model.onnx>
```

Whole body policy with `CmdDense` input (retargeter off, `urdf_path` not needed):

```bash
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    preset:=<inference-preset> \
    input_type:=dense \
    model_path:=<model.onnx>
```

Both policy launch files are fully parameterized (`velocity_input`, `state_input`, `interface`, `domain_id`, `node_name`, …) so a parent launch can compose and override them.

## Run (unitree split-body backend)

```bash
# arm_sdk + LocoClient. Robot must be standing in FSM-501.
ros2 run holosoma_service unitree_split_controller --iface eth0
ros2 run holosoma_service unitree_split_controller --iface eth0 --no-arms   # loco only
```

## Input publishers

A backend does nothing without an **input publisher**. For the WBT policy backend, the simplest one is the bundled NPZ replay script, which streams a reference-motion NPZ onto `CmdDense` as a live feed (pair with `input_type:=dense`):

```bash
python holosoma_service/scripts/publish_from_npz.py <motion.npz> --loop
```

Other publishers: your tracker / AVP / Pico, `demo_scripts/ros2_velocity_publisher.py` for `/cmd_vel`, or `ros2 run holosoma_service wasd_controller_node` for mocking `CmdExoskeleton.msg`.

## Tests

ROS2-native integration tests for the retargeter service API
(`holosoma_service/retargetting/test_retargeter_service_api.py`) run in CI inside a
minimal ROS2 Jazzy image:

```bash
# from the repo root
docker build -f src/holosoma_inference/holosoma_inference_service/docker/Dockerfile.test \
    -t holosoma-service-test .
docker run --rm holosoma-service-test
```

Host-side unit tests (`test_service_node_attach.py`, `test_depth_consumer_delay.py`,
`test_freejoint_strip.py`) skip cleanly when rclpy isn't importable and run with the
normal `pytest src/holosoma_inference/` suite.
