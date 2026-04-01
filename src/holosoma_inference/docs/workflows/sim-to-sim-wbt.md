# Sim-to-Sim Whole Body Tracking Workflow

> **See also:** [Inference & Deployment Guide](../../README.md) for all deployment options

This guide provides a complete workflow for running whole body tracking (WBT) policies in MuJoCo simulation.

## Overview

The sim-to-sim workflow allows you to replay IsaacSim/IsaacGym-trained WBT checkpoints inside MuJoCo for evaluation and testing.

## Prerequisites

- MuJoCo environment set up (`scripts/source_mujoco_setup.sh`)
- Holosoma inference environment set up (`scripts/source_inference_setup.sh`)
- ONNX model checkpoint
- Keyboard for control

**Note:** Always use `--task.interface lo` (loopback) when inference and MuJoCo run on the same machine.

---

## Unitree G1 (29-DOF)

### 1. Start MuJoCo Environment

In one terminal, launch the MuJoCo environment:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

The robot will spawn in the simulator, hanging from a gantry.

### 2. Launch the Policy

In another terminal, run the policy inference:

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo
```

### 3. Initialize Stiff Control Mode

In policy terminal, press `Enter` when prompted. The robot enters stiff control mode and holds its initial pose.

### 4. Deploy the Robot

- In MuJoCo window, press `8` to lower the gantry until robot touches ground
- In MuJoCo window, press `9` to remove gantry
- Wait a few seconds for the stiff controller to stabilize the robot

### 5. Start the Policy

In policy terminal, press `]` to activate the policy.

### 6. Start Motion Clip

In policy terminal, press `m` to start the motion clip. The robot will begin tracking the whole body motion.

---

## MuJoCo Controls Reference

**Enter these commands in the MuJoCo window** (not the policy terminal):

### Gantry Controls

- `7`: Lift the gantry
- `8`: Lower the gantry
- `9`: Disable/remove the gantry

### General Controls

- `Backspace`: Reset simulation

---

## Policy Controls Reference

**Enter these commands in the policy terminal** (where you ran `run_policy.py`):

### General Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start the policy | `]` | A button |
| Stop the policy | `o` | B button |
| Set robot to default pose | `i` | Y button |

### Whole Body Tracking Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start motion clip | `m` | Select+A |

**Default pose**: Standing with raised arms

---

## Tips and Troubleshooting

- **Reset anytime**: Press `Backspace` in the MuJoCo window to reset the simulation
- **Interface**: Always use `lo` (loopback) for sim-to-sim on the same machine
- **Stiff mode**: The `Enter` prompt initializes stiff control mode - this is required for WBT policies to maintain balance before the policy starts
- **Stabilization**: Wait a few seconds after removing the gantry (step 3) before starting the policy to let the stiff controller stabilize
- **RL rate**: Use `--task.rl-rate 50` for WBT policies (50 Hz control rate)
- **Sim time**: Use `--task.use-sim-time` to synchronize with MuJoCo's simulation time
