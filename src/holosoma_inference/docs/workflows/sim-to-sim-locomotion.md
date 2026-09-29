# Sim-to-Sim Locomotion Workflow

> **See also:** [Inference & Deployment Guide](../../README.md) for all deployment options

This guide provides a complete workflow for running locomotion policies in MuJoCo simulation.

## Overview

The sim-to-sim workflow allows you to replay IsaacSim-trained locomotion checkpoints inside MuJoCo for evaluation and testing.

## Prerequisites

- MuJoCo environment set up (`scripts/source_mujoco_setup.sh`)
- Holosoma inference environment set up (`scripts/source_inference_setup.sh`)
- ONNX model checkpoint
- Keyboard or joystick for control

**Note:** Always use `--task.interface lo` (loopback) when inference and MuJoCo run on the same machine.

---

## Robot Workflows

### Unitree G1 (29-DOF)

#### 1. Start MuJoCo Environment

In one terminal, launch the MuJoCo environment:

**For keyboard control:**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

**For joystick control:**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    --simulator.config.bridge.enabled=True \
    --simulator.config.bridge.use-joystick=True
```

The robot will spawn in the simulator, hanging from a gantry.

#### 2. Launch the Policy

In another terminal, run the policy inference:

**For keyboard control:**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.no-use-joystick \
    --task.interface lo
```

**For joystick control:**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.use-joystick \
    --task.interface lo
```

#### 3. Deploy the Robot

- In MuJoCo window, press `8` to lower the gantry until robot touches ground
- In MuJoCo window, press `9` to remove gantry

#### 4. Start the Policy

In policy terminal, press `]` (or `A` button on joystick) to activate the policy.

#### 5. Enter Walking Mode

In policy terminal, press `=` (or `Start` button on joystick) to enter walking mode.

#### 6. Control the Robot

- Use keyboard (`w` `a` `s` `d` for linear, `q` `e` for angular) or joystick to control velocity
- Left joystick: Move forward/backward/left/right
- Right joystick: Turn left/right

---

### Booster T1 (29-DOF)

#### 1. Start MuJoCo Environment

In one terminal, launch the MuJoCo environment:

**For keyboard control:**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:t1-29dof-waist-wrist
```

**For joystick control:**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:t1-29dof-waist-wrist \
    --simulator.config.bridge.enabled=True \
    --simulator.config.bridge.use-joystick=True
```

The robot will spawn in the simulator, hanging from a gantry.

#### 2. Launch the Policy

In another terminal, run the policy inference:

**For keyboard control:**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --task.no-use-joystick \
    --task.interface lo
```

**For joystick control:**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --task.use-joystick \
    --task.interface lo
```

#### 3. Deploy the Robot

- In MuJoCo window, press `8` to lower the gantry until robot touches ground
- In MuJoCo window, press `9` to remove gantry

#### 4. Start the Policy

In policy terminal, press `]` to activate the policy.

#### 5. Enter Walking Mode

In policy terminal, press `=` to enter walking mode.

#### 6. Control the Robot

Use keyboard to control velocity:
- `w` `a` `s` `d` for linear movement (forward/backward/left/right)
- `q` `e` for angular movement (turn left/right)

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

### Locomotion Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Switch walking/standing | `=` | Start button |
| Adjust linear velocity | `w` `a` `s` `d` | Left stick |
| Adjust angular velocity | `q` `e` | Right stick |

---

## Tips and Troubleshooting

- **Reset anytime**: Press `Backspace` in the MuJoCo window to reset the simulation
- **Interface**: Always use `lo` (loopback) for sim-to-sim on the same machine
- **Standing mode**: The robot starts in standing mode - press `=` or `Start` to switch to walking mode
- **Default pose**: Press `i` or `Y` button to return the robot to standing pose
