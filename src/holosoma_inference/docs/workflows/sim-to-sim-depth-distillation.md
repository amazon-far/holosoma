# Sim-to-Sim Depth Distillation Workflow

> **See also:** [Inference & Deployment Guide](../../README.md) for all deployment options

This guide provides a complete workflow for running vision-based locomotion (depth
distillation) policies in MuJoCo simulation.

## Overview

Depth distillation policies see the terrain ahead, so they can place footholds on
stairs and rough ground rather than walking blind. A policy is a pair of ONNX models:

- **depth backbone** — encodes a depth image into a compact latent
- **student** — maps proprioception + a direction command + that latent to joint targets

The sim renders depth from a torso-mounted camera and publishes it to shared memory;
the policy process reads it each control tick. The same policy code runs on hardware
against an on-robot depth server publishing the same format.

## Prerequisites

- MuJoCo environment set up (`scripts/source_mujoco_setup.sh`)
- Holosoma inference environment set up (`scripts/source_inference_setup.sh`)
- A checkpoint **pair**: `<run>_depth_backbone.onnx` and `<run>_student.onnx`
- Keyboard for control

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
    sensor.stair_front_depth:g1-stair-front-depth \
    plugin.depth:depth-shm
```

The robot will spawn in the simulator, hanging from a gantry. The log should show:

```
[DepthShmPlugin] created 'depth_img_shm' (20184 bytes) shape=(1, 1, 58, 87)
```

`sensor.<key>:<preset>` and `plugin.<key>:<preset>` are positional declarations, not
`--flags`. The plugin's `camera` field must name the sensor key; the `depth-shm` preset
defaults to `stair_front_depth`, matching the key above.

### 2. Launch the Policy

In another terminal, run the policy inference:

```bash
source scripts/source_inference_setup.sh
CKPT=/path/to/checkpoints/2026-07-03_05-20-32_rfh-rvol-grid-fh5.0-rv10.0
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-wbt-distillation \
    --task.interface lo \
    --task.model-path "['${CKPT}_depth_backbone.onnx','${CKPT}_student.onnx']" \
    --task.record-dir stair \
    --task.record-label stair
```

Confirm the policy attached to the sim's depth stream:

```
[DepthShmSensor] attached to 'depth_img_shm' shape=(1, 1, 58, 87)
```

Both model paths go in **one quoted Python list literal**. `TaskConfig.model_path` is
typed `str | list[str]` and the CLI resolves that union to `str`, so space-separated
paths are parsed as a single token and fail. Comma-separated also works.

### 3. Deploy the Robot

- In MuJoCo window, press `8` repeatedly (~25 times) to lower the gantry until the feet
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

To exercise the policy control loop without a depth producer, append
`--task.depth-shm.no-required` to the inference command above; it will use zero-filled depth.

## Policy Controls Reference

**Enter these commands in the policy terminal** (where you ran `run_policy.py`).

### General Controls

| Action | Keyboard |
|--------|----------|
| Start the policy | `]` |
| Damping mode (Kp=0, Kd>0) | `o` |
| Re-enter stiff hold | `i` |
| Toggle motion recording | `c` |

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
you let go — like a gamepad d-pad. Holding two directions stacks them, so releasing the newer one
resumes the one still held. This makes letting go of the keyboard a reliable stop.

Unlike the blind locomotion policy, direction is a **discrete class**, not a continuous velocity —
that is how the command was represented during training, so one tap responds immediately and
reversing from forward to back is a single keypress rather than several.

Hold-to-move needs true key-up events, which a terminal does not deliver (it reports auto-repeat, not
releases), so this uses `pynput` and an X11 **DISPLAY**. Without one, the policy logs

```
Hold-to-move needs key-up events (pynput + a DISPLAY); direction keys will latch until another is pressed.
```

and each press *latches* until another direction or `z` is pressed. Note the robot then keeps walking
after you release the key — use `z` to stop.

Joystick and ROS2 still deliver continuous velocities; those are quantized to the
nearest direction sector, so all input sources work.

### MuJoCo Controls Reference

**Enter these in the MuJoCo window** (not the policy terminal):

- `7` / `8`: Raise / lower the gantry
- `9`: Disable/remove the gantry
- `Backspace`: Reset simulation

---

## Recording Motion Clips

Press `c` while the policy is running to start recording, `c` again to stop and save.
Stopping the policy (`o`) or re-entering stiff hold (`i`) autosaves an in-flight
recording, so a clip is never silently lost.

Clips are written to
`recorded_motion/<record-dir>/<record-label>_duration<X.X>s_motion.npz`, **relative to
the working directory**, containing:

| Key | Shape | Contents |
|-----|-------|----------|
| `qpos` | `[N, 36]` float32 | root quaternion `[0:4]` (wxyz), root position `[4:7]`, absolute joint positions `[7:36]` in robot order |
| `vel_cmd` | `[N, 15]` float32 | the one-hot direction command per frame |
| `fps` | scalar int64 | control rate (50) |

Note `qpos` is **quaternion-first**, which differs from the position-first ordering used
by MuJoCo-format training clips. The clip duration is part of the filename, so two clips
of the same length overwrite each other (the policy logs a warning when it does).

---

## Configuration

### Depth preprocessing

The simulator does the resize, clip and normalize so the policy reads a tensor it can
hand straight to the backbone — this keeps the two sides from disagreeing about it.
The values must match what the checkpoint was trained with:

```bash
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    --simulator.config.sim.fps=500 \
    sensor.stair_front_depth:g1-stair-front-depth plugin.depth:depth-shm \
    --plugin.depth.near-clip 0.1 \
    --plugin.depth.far-clip 2.0 \
    --plugin.depth.latency-frames 5
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--plugin.depth.near-clip` | `0.3` | Depth at or below this normalizes to `-0.5` |
| `--plugin.depth.far-clip` | `2.0` | Depth at or above this normalizes to `+0.5` (the training camera's `max_range`: 2.0 ZED, 3.0 D435i) |
| `--plugin.depth.crop-top` / `-bottom` / `-left` / `-right` | `0` | Border rows/cols dropped **before** the resize; the D435i preset uses `2/0/4/4` (training's `depth[2:, 4:-4]`) |
| `--plugin.depth.render-hz` | `10.0` | Rate of the plugin's own render thread. Independent of `fps` and of the camera's `update_decimation` |
| `--plugin.depth.latency-frames` | `0` | Publish a frame this many steps old, modeling real camera/transport delay. Counted in *rendered frames*, so one frame is `1/render_hz` (100 ms at the default) |
| `--plugin.depth.shm-name` | `depth_img_shm` | Shared-memory block name; must match `--task.depth-shm.name` |

The resize uses bicubic interpolation with antialiasing to match
`torchvision.transforms.Resize(..., BICUBIC)` as used in training. Bilinear resizing
shifts the depth statistics enough to degrade foothold placement.

### Camera resolution

If you change the render resolution, the resized output must still match the backbone's
input — the policy fails loud on a size mismatch rather than reading garbage:

```bash
--sensor.stair_front_depth.width 320 --sensor.stair_front_depth.height 180
```

Note the sensor's `near`/`far` are **not** the clip range. MuJoCo's near/far is a global view
frustum that *removes* geometry, so a near plane at `near_clip` would make a 0.2 m obstacle
invisible (revealing the background behind it) where training would report it at 0.3 m. Keep the
frustum permissive and let the plugin's `near_clip`/`far_clip` do the clamping.

### Render rate

The depth-shm plugin renders on **its own thread** at `--plugin.depth.render-hz` (default **10 Hz**,
the D435i's real publish rate), not on the physics thread:

```bash
--plugin.depth.render-hz 20
```

This is load-bearing for hitting the sim's target rate. A GL depth render costs ~0.25 ms on a GPU
context (and ~4.7 ms on a software one), which does not fit the 2 ms budget of a 500 Hz physics step.
Rendered inline, each frame lands on one unlucky step, blows its slot, and the rate limiter then
sprints to catch up — the sim reports 400/600 Hz swings and never settles. Off-thread it holds a flat
500.0 Hz, before and after a policy attaches.

The camera is taken off the control-loop render schedule when the plugin starts
(`SensorManager.claim_external_rendering`), so its `update_decimation` is unused and the physics loop
pays nothing for it. The policy polls shared memory at its own 50 Hz and simply re-reads the latest
frame, so a render rate below the control rate means repeated frames, not stalls.

`update_decimation` still governs any camera rendered *inline* by another consumer (ros2-image, viz,
video). Note it resolves against the **control** rate (`fps / control_decimation` = 125 Hz here),
where a bare `"50Hz"` is not an exact divisor and raises at startup — hence `">50Hz"` in the presets.

### Depth liveness

Shared memory alone cannot tell a live producer from a dead one: the last frame stays readable after
the producer stops. So the producer also publishes a small heartbeat after every frame write, on a zmq
socket at `ipc:///tmp/<shm-name>.heartbeat`, and the policy watches how long the frame has stopped
advancing:

| Frame age | Policy behavior |
|-----------|-----------------|
| below `--task.depth-shm.stale-warn-s` (0.25 s) | normal |
| above it | one warning per stall |
| above `--task.depth-shm.stale-stop-s` (0.5 s), or producer reports `FAILED` | enters stiff hold; press `]` to resume once depth recovers |

When a camera drops out, a producer keeps the last frame in shared memory and reports `RECONNECTING`
while it retries, rather than writing zeros: a normalized zero is a valid depth reading to the policy,
not "no data". The sim plugin follows this contract (`--plugin.depth.heartbeat`), and on-robot camera
daemons can use `holosoma_inference.sensors.depth_heartbeat.DepthHeartbeatPublisher`. If a producer
restarts, the policy re-attaches to its new shared-memory block. A producer that publishes no heartbeat
still works: the policy logs one warning and leaves liveness unchecked (`--task.depth-shm.no-heartbeat`
silences it).

### Stair / stepped terrain

`terrain:terrain-load-step` ships a stepped-block course (`holosoma/data/terrains/terrain.obj`):
an 80 m ground plane with a line of raised blocks of varying height along +X. Point it at a
different mesh with an absolute path:

```bash
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    --simulator.config.sim.fps=500 \
    sensor.stair_front_depth:g1-stair-front-depth plugin.depth:depth-shm \
    terrain:terrain-load-step \
    --terrain.terrain-term.obj-file-path <path>/chained_stairs_15.obj
```

Use `terrain:terrain-locomotion-plane` for flat ground.

---

## RealSense D435i variant

The presets above describe the ZED 2i rig. For checkpoints trained against the **D435i** rig
(`G1FlatRsD435iConfig`: 27° down torso mount at `(0.01, 0.01, 0.44)`, 106x60, `[0.3, 3.0]` m), use
the D435i triple instead — the depth tensor is 58x87 either way, so the ZED presets also *run*, just
with the wrong extrinsics and far clip. The [PHP repository](https://github.com/amazon-far/php_parkour)
owns the project-level `run_php_sim.sh` and `run_php_inference.sh` launchers; see its
[sim2sim guide](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/DEPLOY.md).
Those launchers wrap the following Holosoma commands:

```bash
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    --simulator.config.sim.fps=500 \
    sensor.d435i_front_depth:g1-d435i-front-depth \
    plugin.depth:depth-shm-d435i \
    terrain:terrain-load-step \
    --robot.asset.xml-file g1/g1_29dof_halfspherehand.xml \
    --simulator.config.bridge.enabled=True

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-wbt-distillation-d435i \
    --task.interface lo \
    --task.model-path "['<RUN>/model_19999/depth_backbone.onnx','<RUN>/model_19999/student.onnx']"
```

`--robot.asset.xml-file` picks the half-sphere-hand MJCF, matching the reference sim2sim rig. It is a
per-run flag rather than part of the `g1-29dof` preset because training ran on an Isaac backend and
read only `urdf_file`; `xml_file` is required but unused there, so it stayed at the plain
`g1_29dof.xml`. Omit the flag to get that plain model (rubber-hand mesh + capsule collider) — the
joint and actuator order are identical either way, so checkpoints load against both.

`<RUN>` may be a `wandb://<entity>/<project>/<run_id>` URI; checkpoints are cached under
`~/.cache/holosoma_inference/weights/<run_id>/`, so later runs are offline.

---

## How It Works

```
MuJoCo depth render (metric meters; 240x135 ZED rig / 106x60 D435i rig)
  └─ DepthShmPlugin: clip [near, far] -> crop -> resize 58x87 (bicubic+antialias)
     -> normalize [-0.5, 0.5]
       └─ /dev/shm/depth_img_shm : float32 (1, 1, 58, 87)
            └─ DepthShmSensor (policy process)
                 └─ depth_backbone.onnx -> latent [1, 32]
                      └─ student.onnx: obs [1, 140] -> actions [1, 29]
```

Step order matters and mirrors training exactly: the clip runs **before** the resize (the bicubic
kernel would otherwise smear MuJoCo's far-plane sentinel across its neighbours), and the crop runs
before the resize so the resize sees training's field of view. This is pinned by
`simulator/plugins/tests/test_depth_shm_plugin.py`, which reimplements the training math and asserts
the two agree.

Shared memory rather than a ROS2 topic because the consumer runs on the same host at
the control rate: no serialization, no broker, no per-frame allocation.

The student's 140-dim input is a **wire format**:

| Slice | Contents |
|-------|----------|
| `[0:29]` | previous `actions` (in **model** joint order) |
| `[29:32]` | `base_ang_vel` |
| `[32:61]` | `dof_pos` (model order) |
| `[61:90]` | `dof_vel` (model order) |
| `[90:93]` | `projected_gravity` — in the **torso** frame, not the pelvis |
| `[93:108]` | one-hot direction command |
| `[108:140]` | depth latent |

Three details there are load-bearing. Each would produce a plausible-looking but wrong
action rather than an error, so all three are pinned by tests:

- **Term order is training's `sorted()` order, not the ONNX metadata's order.** The
  training-side `ObservationManager` concatenates a `concatenate=True` group over
  `sorted(term_names)`, so the wire layout is alphabetical: `actions`, `base_ang_vel`,
  `dof_pos`, `dof_vel`, `projected_gravity`. The preset lists them already sorted and sets
  `sort_obs_terms=False` so the list is used verbatim.

  Do **not** "fix" this from the checkpoint's `observation_names` metadata: the exporter
  writes that field as the training config's *declaration* order (gravity first), which is
  not the order it concatenates. Trusting the metadata transposes gravity and actions.
- **Joint order may differ from the robot's.** The permutation is derived at load time from
  the checkpoint's `joint_names` metadata, so it adapts per checkpoint — it is the identity
  for checkpoints exported in the robot's own serial order, and a real reordering for
  IsaacLab-interleaved ones. `last_policy_action` is deliberately kept in *model* order,
  because it feeds back as the `actions` observation on the next tick.
- **Gravity is observed in the anchor (torso) frame.** On hardware the IMU sits in the
  torso, so the base quaternion already is the torso's; in MuJoCo the floating base is
  the pelvis, so the waist yaw/roll/pitch rotations are chained on.

Control gains come from the **robot config**, not the student's ONNX metadata. The
exporter writes per-joint `joint_stiffness`/`joint_damping` that differ from the
deployed values by a few percent (hip_pitch 38.869 vs 40.179); the reference deployment
runs the config values, so the config takes priority.

---

## Tips and Troubleshooting

- **`Depth shared memory 'depth_img_shm' not found`**: the simulator is not running, or
  exited (it unlinks the block on close). Start the sim first, or pass
  `--task.depth-shm.no-required` to run with zero-filled depth.
- **`Existing shared memory ... is N bytes but this camera needs M`**: a stale block from
  a run at a different resolution. Remove `/dev/shm/depth_img_shm` and retry.
- **`requires exactly 2 model paths`**: the two paths collapsed into one token. Quote
  them as a single Python list literal (see step 2).
- **Keyboard does nothing**: keyboard input needs a TTY. Without one the policy logs
  `No TTY — keyboard input disabled` and auto-starts, so it runs but cannot be steered.
- **Reset anytime**: press `Backspace` in the MuJoCo window.
- **Interface**: always use `lo` (loopback) for sim-to-sim on the same machine.
