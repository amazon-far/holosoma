# Sim-to-Sim Depth Locomotion Workflow

> **See also:** [Inference & Deployment Guide](../../README.md) for all deployment options

This guide runs a depth locomotion policy on the Unitree G1 (29-DOF) in MuJoCo
with a simulated RealSense D435i depth camera.

## Overview

Depth locomotion policies see the terrain ahead, so they can place footholds on
terrain rather than walking blind. A policy is a pair of ONNX models:

- **depth backbone** — encodes a depth image into a compact latent
- **student** — maps proprioception + a direction command + that latent to joint targets

The sim renders depth from a torso-mounted D435i camera and publishes it to shared memory;
the policy process reads it each control tick. The same policy code runs on hardware
against an on-robot depth server publishing the same format.

## Prerequisites

- MuJoCo environment set up (`scripts/source_mujoco_setup.sh`)
- Holosoma inference environment set up (`scripts/source_inference_setup.sh`)
- A D435i-compatible checkpoint **pair**: `depth_backbone.onnx` and `student.onnx`
- Keyboard and an X11 display for the MuJoCo viewer and hold-to-move controls

Run the commands from the Holosoma repository root. When connecting over SSH,
set `DISPLAY` to your graphical desktop's X server in both terminals.

**Note:** Always use `--task.interface lo` (loopback) when inference and MuJoCo run on
the same machine.

**Note:** Start the simulator **before** the policy — the sim creates the shared-memory
block the policy attaches to.

---

## Unitree G1 (29-DOF)

### 1. Start MuJoCo Environment

In one terminal, launch the simulator with the depth camera and the shared-memory
publisher:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    --simulator.config.sim.fps=500 \
    sensor.d435i_front_depth:g1-d435i-front-depth \
    plugin.depth:depth-shm-d435i \
    terrain:terrain-load-step \
    --robot.asset.xml-file g1/g1_29dof_halfspherehand.xml \
    --simulator.config.bridge.enabled=True
```

The robot will spawn in the simulator, hanging from a gantry. The log should show:

```
[DepthShmPlugin] created 'depth_img_shm' (20184 bytes) shape=(1, 1, 58, 87)
```

These presets render at 106×60, crop to 98×58,
and resize to the policy's 87×58 depth input. The simulator runs at 500 Hz.
The command uses the half-sphere-hand G1 model and the bundled terrain.
Use `terrain:terrain-locomotion-plane` in place of `terrain:terrain-load-step` for flat ground.

The [PHP repository](https://github.com/amazon-far/php_parkour) provides
`run_php_sim.sh` and `run_php_inference.sh` launchers for this workflow; see its
[sim2sim guide](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/DEPLOY.md).

### 2. Launch the Policy

In another terminal, run the policy inference:

```bash
source scripts/source_inference_setup.sh
CKPT=/absolute/path/to/checkpoint_directory
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-wbt-distillation-d435i \
    --task.interface lo \
    --task.model-path "['${CKPT}/depth_backbone.onnx','${CKPT}/student.onnx']"
```

Confirm the policy attached to the sim's depth stream:

```
[DepthShmSensor] attached to 'depth_img_shm' shape=(1, 1, 58, 87)
```

### 3. Deploy the Robot

- In MuJoCo window, press `8` repeatedly to lower the gantry until the feet
  touch the ground
- In MuJoCo window, press `9` to remove the gantry

The policy holds a stiff standing pose until started, so the robot stays upright while
the gantry comes down.

### 4. Start the Policy

In policy terminal, press `]` to activate the policy.

### 5. Control the Robot

In policy terminal, use `w` `a` `s` `d` `q` `e` to steer and `=` to change speed mode.
Each key selects a heading outright — see below.

---

## Policy Controls Reference

**Enter these commands in the policy terminal** (where you ran `run_policy.py`).

### General Controls

| Action | Keyboard |
|--------|----------|
| Start the policy | `]` |
| Damping mode (Kp=0, Kd>0) | `o` |
| Re-enter stiff hold | `i` |

`o` enters a damping mode rather than zeroing all gains, so the robot yields but
resists free-fall. `i` eases back to the startup pose over 2 s.

### Direction Controls

| Action | Keyboard |
|--------|----------|
| Forward / backward | **hold** `w` / `s` |
| 45° left / right | **hold** `a` / `d` |
| 90° left / right | **hold** `q` / `e` |
| Stand | release all direction keys, or `z` |
| Toggle speed mode (LOW ↔ HIGH) | `=` |

Direction keys are **momentary**: the robot walks while a key is held and returns to stand as soon as
you let go — like a gamepad d-pad. 

### MuJoCo Controls Reference

**Enter these in the MuJoCo window** (not the policy terminal):

- `7` / `8`: Raise / lower the gantry
- `9`: Disable/remove the gantry
- `Backspace`: Reset simulation
