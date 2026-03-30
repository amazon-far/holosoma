# Real Robot Whole Body Tracking Workflow

> **See also:** [Inference & Deployment Guide](../../README.md) for all deployment options

This guide provides a complete workflow for running whole body tracking (WBT) policies on physical robot hardware.

## Overview

Deploy and run WBT policies on physical Unitree G1 robots to perform motion tracking and dynamic movements.

**Note**: Booster T1 is not supported for WBT yet

## Prerequisites

- Physical robot hardware (Unitree G1)
- Ethernet cable
- Unitree G1 remote controller
- Laptop with holosoma inference environment set up

## Hardware Setup

### 1. Prepare the Robot

- Hang the robot on the gantry
- Turn on the robot and the controller
- Connect the robot to your laptop with an Ethernet cable
- Put the robot in damping mode
- Press `L2+R2` on the controller to enter development mode

For detailed hardware setup instructions, see the [Unitree Quick Start page](https://support.unitree.com/home/en/G1_developer/quick_start).

### 2. Configure Network

Configure your laptop's network interface:
- IP Address: `192.168.123.224`
- Netmask: `255.255.255.0`

### 3. Find Your Network Interface

Identify which network interface is connected to the robot:

```bash
ifconfig
```

Look for the interface with IP `192.168.123.224`. Common names:
- `eth0` - Common Ethernet interface name
- `enp0s31f6` - Modern Linux Ethernet naming

## Running the Policy

### 1. Launch the Policy

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.use-joystick \
    --task.rl-rate 50 \
    --task.interface eth0
```

**Note**: Replace `eth0` with your network interface name (e.g., `enp0s31f6`). Find it using `ifconfig`.

### 2. Initialize Stiff Control Mode

Press `Enter` when prompted in the policy terminal. The robot enters stiff control mode and holds its initial pose.

### 3. Start the Policy

Press `A` button on joystick to activate the policy.

### 4. Start Motion Clip

Press `Start` button on joystick to start the motion clip. The robot will begin tracking the whole body motion.

---

## Policy Controls Reference

**Enter these commands in the policy terminal** (where you ran `run_policy.py`):

### General Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start the policy | `]` | A button |
| Stop the policy | `o` | B button |
| Set robot to default pose | `i` | Y button |
| Kill controller program | - | L1 (LB) + R1 (RB) |

### Whole Body Tracking Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start motion clip | `s` | Start button |

**Default pose**: Standing with raised arms

---

## Deployment Options

### Option 1: Run Offboard (Laptop)

This is the default setup described above. The policy runs on your laptop and communicates with the robot over Ethernet.

**Advantages:**
- Easier to debug and monitor
- No need to modify robot software
- Quick iteration

**Requirements:**
- Stable Ethernet connection
- Laptop with sufficient compute

### Option 2: Run Onboard (Jetson)

Run the policy directly on the robot's onboard Jetson computer for lower latency.

#### Setup Steps

1. Complete hardware setup (see above)

2. SSH to the onboard Jetson:
   ```bash
   ssh unitree@192.168.123.164
   # Default password: '123'
   ```

3. Set Jetson to maximum performance:
   ```bash
   sudo jetson_clocks
   ```

4. Set up the environment on Jetson:
   ```bash
   cd ~/holosoma
   bash scripts/setup_inference.sh
   source scripts/source_inference_setup.sh
   ```

5. Run the policy using `eth0` interface:
   ```bash
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
       --task.model-path <path-to-onnx> \
       --task.use-joystick \
       --task.rl-rate 50 \
       --task.interface eth0
   ```

**Advantages:**
- Lower latency
- No external laptop required after setup
- More portable

### Option 3: Run Inside Docker

Run the policy inside a Docker container (works both onboard and offboard).

#### Setup Steps

1. Complete hardware setup (see above)

2. Build the Docker image:
   ```bash
   bash holosoma/src/holosoma_inference/docker/build.sh
   ```

3. Create and enter the Docker container:
   ```bash
   bash holosoma/src/holosoma_inference/docker/run.sh
   ```

4. Run the policy inside the container:
   - Use `eth0` on Jetson
   - Use your interface name on laptop (check with `ifconfig`)

**Advantages:**
- Consistent environment
- Easier dependency management
- Reproducible setup

---

## Tips and Troubleshooting

### Network Issues

- **Cannot connect to robot**: Check that your IP is `192.168.123.224` and netmask is `255.255.255.0`
- **Wrong interface**: Use `ifconfig` to verify which interface is connected to the robot
- **Connection drops**: Ensure Ethernet cable is properly connected and not damaged

### Robot Behavior

- **Stiff mode**: The `Enter` prompt initializes stiff control mode - this is required for WBT policies to maintain balance before the policy starts
- **Emergency stop**: Press `L1 + R1` (LB + RB) on joystick to kill the controller
- **Default pose**: Press `Y` button to return the robot to standing pose with raised arms
- **Motion not starting**: Ensure you pressed `Start` button after activating the policy with `A`

### Control Issues

- **No joystick response**: Ensure `--task.use-joystick` flag is set
- **Keyboard control**: Remove `--task.use-joystick` flag and use `s` key to start motion clip
- **Mixed input**: Use `--task.velocity-input` and `--task.state-input` individually for mixed setups (e.g., ROS2 velocity + keyboard commands). See the [Input Sources](../../README.md#input-sources) section.
- **Low responsiveness**: Check network latency if running offboard
- **RL rate**: Always use `--task.rl-rate 50` for WBT policies (50 Hz control rate)

### Safety

- **Always** keep the robot on the gantry for initial testing
- Have the emergency stop ready (`L1 + R1`)
- WBT motions can be dynamic - ensure adequate clearance around the robot
- Test with simple motions before attempting complex choreography

### Motion Clip Issues

- **Clip not playing**: Ensure you pressed `Start` button after starting the policy
- **Unstable tracking**: Verify that stiff mode initialization completed successfully
