# Real Robot Locomotion Workflow

> **See also:** [Inference & Deployment Guide](../../README.md) for all deployment options

This guide provides a complete workflow for running locomotion policies on physical robot hardware.

## Overview

Deploy and run locomotion policies on physical Unitree G1 or Booster T1 robots with velocity tracking control.

## Prerequisites

- Physical robot hardware (Unitree G1 or Booster T1)
- Ethernet cable
- Controller (Default remote controller for robot or a custom joystick)
- Laptop with holosoma inference environment set up

---

## Unitree G1 (29-DOF)

### Hardware Setup

#### 1. Prepare the Robot

- Hang the robot on the gantry
- Turn on the robot and the controller
- Connect the robot to your laptop with an Ethernet cable
- Put the robot in damping mode
- Press `L2+R2` on the controller to enter development mode

For detailed hardware setup instructions, see the [Unitree Quick Start page](https://support.unitree.com/home/en/G1_developer/quick_start).

#### 2. Configure Network

Configure your laptop's network interface:
- IP Address: `192.168.123.224`
- Netmask: `255.255.255.0`

#### 3. Find Your Network Interface

Identify which network interface is connected to the robot:

```bash
ifconfig
```

Look for the interface with IP `192.168.123.224`. Common names:
- `eth0` - Common Ethernet interface name
- `enp0s31f6` - Modern Linux Ethernet naming

### Running the Policy

#### 1. Launch the Policy

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.use-joystick \
    --task.interface eth0
```

**Notes**:
- Replace `eth0` with your network interface name (e.g., `enp0s31f6`). Find it using `ifconfig`.
- For mixed input setups (e.g., ROS2 velocity + keyboard commands), see the [Input Sources](../../README.md#input-sources) section.

#### 2. Start the Policy

Press `A` button on joystick to activate the policy.

#### 3. Enter Walking Mode

Press `Start` button on joystick to enter walking mode.

#### 4. Control the Robot

- Use left joystick to move forward/backward/left/right
- Use right joystick to turn left/right

---

## Booster T1 (29-DOF)

### Hardware Setup

#### 1. Prepare the Robot

- Hang the robot on the gantry
- Turn on the robot and bring the default remote controller
- Connect the robot to your laptop with an Ethernet cable
- (Optional) Connect your custom joystick to the laptop if you want joystick control
- Put the robot in PREP mode (`RT + Y` with the default remote controller)

**Important - T1 Control Options**:
- **Keyboard control** (default): You can control the robot using keyboard commands in the policy terminal
- **Joystick control** (optional): If you prefer joystick control, you will need **two joysticks**:
  1. **Default Booster T1 remote controller**: Only for putting the robot in damping or PREP mode
  2. **Custom joystick connected to laptop**: For all policy control (starting policy, walking mode, velocity control)

For detailed hardware setup instructions, see the [Booster T1 Documentation](https://booster.feishu.cn/wiki/XAS3wv4lwiSiXXkDbMrceE6UnHc).

#### 2. Configure Network

Configure your laptop's network interface:
- IP Address: `192.168.10.10`
- Netmask: `255.255.255.0`
- Gateway: `192.168.10.1`

#### 3. Find Your Network Interface

Identify which network interface is connected to the robot:

```bash
ifconfig
```

Look for the interface with IP `192.168.10.10`. Common names:
- `eth0` - Common Ethernet interface name
- `enp0s31f6` - Modern Linux Ethernet naming

### Running the Policy

#### 1. Launch the Policy

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --task.use-joystick \
    --task.interface eth0
```

**Notes**:
- Replace `eth0` with your network interface name (e.g., `enp0s31f6`). Find it using `ifconfig`.
- **Joystick control**: If using joystick, use the custom joystick connected to the laptop (not the default Booster T1 remote controller) for all policy controls.
- **Keyboard control**: Remove `--task.use-joystick` flag to control the robot with keyboard commands instead.
- **Mixed input**: Use `--task.velocity-input` and `--task.state-input` individually for mixed setups (e.g., ROS2 velocity + keyboard commands). See the [Input Sources](../../README.md#input-sources) section for details.

#### 2. Start the Policy

Press `A` button on joystick to activate the policy.

#### 3. Enter Walking Mode

Press `Start` button on joystick to enter walking mode.

#### 4. Control the Robot

- Use left joystick to move forward/backward/left/right
- Use right joystick to turn left/right

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

### Locomotion Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Switch walking/standing | `=` | Start button |
| Adjust linear velocity | `w` `a` `s` `d` | Left stick |
| Adjust angular velocity | `q` `e` | Right stick |

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
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:<task-config> \
       --task.model-path <path-to-onnx> \
       --task.use-joystick \
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

- **Cannot connect to robot**: Verify your IP configuration matches your robot (G1: `192.168.123.224`, T1: `192.168.10.10`) with correct netmask
- **Wrong interface**: Use `ifconfig` to verify which interface is connected to the robot
- **Connection drops**: Ensure Ethernet cable is properly connected and not damaged

### Robot Behavior

- **Standing mode**: The robot starts in standing mode - press `Start` to switch to walking mode
- **Emergency stop**: Press `L1 + R1` (LB + RB) on joystick to kill the controller
- **Default pose**: Press `Y` button to return the robot to standing pose
- **Damping mode**: If robot becomes stiff or unresponsive, put it back in damping mode via the controller

### Control Issues

- **No joystick response**: Ensure `--task.use-joystick` flag is set
- **Keyboard control**: Remove `--task.use-joystick` flag to use keyboard instead
- **Low responsiveness**: Check network latency if running offboard

### Safety

- **Always** keep the robot on the gantry for initial testing
- Have the emergency stop ready (`L1 + R1`)
- Clear the area around the robot before walking
