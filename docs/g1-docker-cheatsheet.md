# G1 Docker Launch Cheatsheet

For G1 Jetson connection and network setup, see the [Real Robot Locomotion](../src/holosoma_inference/docs/workflows/real-robot-locomotion.md) or [Real Robot WBT](../src/holosoma_inference/docs/workflows/real-robot-wbt.md) workflow docs.

## Docker Setup

Build the image and create the container:
```bash
# Build (from the repo root)
bash src/holosoma_inference/docker/build.sh

# Create container (bind-mounts repo at /workspace/holosoma)
bash src/holosoma_inference/docker/run.sh

# Or recreate from scratch
bash src/holosoma_inference/docker/run.sh --new
```

## Docker Launch Commands

The inference runs inside a Docker container with ROS2 Humble and Python 3.10.
All commands assume `CONTAINER_NAME="holosoma-inference-container"`.

Start the container first (idempotent):
```bash
docker start "$CONTAINER_NAME"
```

### Locomotion Policy

**Keyboard velocity + keyboard state** (basic testing, no joystick needed):
```bash
docker exec -it "$CONTAINER_NAME" bash -c \
  "source /opt/ros/humble/setup.bash && cd /workspace/holosoma && \
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
     --task.velocity-input keyboard --task.state-input keyboard --task.interface eth0"
```

**Joystick velocity + joystick state** (full joystick control):
```bash
docker exec -it "$CONTAINER_NAME" bash -c \
  "source /opt/ros/humble/setup.bash && cd /workspace/holosoma && \
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
     --task.use-joystick --task.interface eth0"
```

**Joystick velocity + keyboard state**:
```bash
docker exec -it "$CONTAINER_NAME" bash -c \
  "source /opt/ros/humble/setup.bash && cd /workspace/holosoma && \
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
     --task.velocity-input interface --task.state-input keyboard --task.interface eth0"
```

**ROS2 cmd_vel + keyboard state**:
```bash
docker exec -it "$CONTAINER_NAME" bash -c \
  "source /opt/ros/humble/setup.bash && cd /workspace/holosoma && \
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
     --task.velocity-input ros2 --task.state-input keyboard --task.interface eth0"
```

**ROS2 cmd_vel + joystick state**:
```bash
docker exec -it "$CONTAINER_NAME" bash -c \
  "source /opt/ros/humble/setup.bash && cd /workspace/holosoma && \
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
     --task.velocity-input ros2 --task.state-input interface --task.interface eth0"
```

### Whole-Body Tracking (WBT)

**WBT with joystick** (Select+A to start motion clip):
```bash
docker exec -it "$CONTAINER_NAME" bash -c \
  "source /opt/ros/humble/setup.bash && cd /workspace/holosoma && \
   python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
     --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
     --task.use-joystick --task.rl-rate 50 --task.interface eth0"
```

### ROS2 Velocity Publisher (for testing)

Run in a **second SSH session** while a policy is running:
```bash
docker exec -it "$CONTAINER_NAME" bash -c \
  "source /opt/ros/humble/setup.bash && cd /workspace/holosoma && \
   python3 demo_scripts/ros2_velocity_publisher.py --pattern shuttle"
```

## Joystick Controls

| Action | Joystick | Keyboard |
|--------|----------|----------|
| Start policy (enable actions) | A | `]` |
| Stop policy | B | `o` |
| Default pose | Y | `i` |
| Walk/Stand toggle | Start | `=` |
| Zero velocity | L2 | - |
| Emergency kill | L1+R1 | - |
| Start motion clip (WBT) | Select+A | `m` |
| Switch dual-mode policy | X | `x` |
| Next model | Select | - |

## Workflow

1. SSH into the Jetson
2. Run a docker exec command above (first SSH session)
3. Press **A** on joystick to activate the policy
4. Press **Start** to enter walking mode
5. Use sticks (or ROS2 publisher) for velocity

## Tips

- The Docker container uses ROS2 Humble with Python 3.10
- Code is bind-mounted at `/workspace/holosoma` — no rebuild needed after code changes
- For ROS2 modes, run the velocity publisher in a separate SSH session
