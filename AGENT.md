## Lưu ý

AGENT luôn yêu cầu giao tiếp bằng tiếng việt và không được hỏi lại ngôn ngữ giao tiếp. Nếu có câu hỏi về yêu cầu, hãy trả lời bằng tiếng việt.

CHỉ Code khi được cho phép, luôn hỏi lại thông tin nếu chưa rõ phần nào trước khi bắt đầu viết code. Nếu có yêu cầu viết code, hãy hỏi lại về yêu cầu đó để đảm bảo hiểu đúng trước khi viết.

git push -u origin HEAD# Agent Handoff Notes

This file captures the current repo understanding and task state so a future agent can continue without rediscovering the same context.

## Current Objective

Research and train a Unitree G1 29-DOF locomotion policy where both arms are held in a forward "carry object" pose. The current implementation does not add payload mass yet. The first milestone is a PPO locomotion baseline with fixed arm pose.

Active training preset:

```text
exp:g1-29dof-arm-hold
```

Recommended algorithm for this task:

```text
PPO first, FastSAC only after a stable PPO baseline.
```

Reason: this task changes balance/posture and needs stable environment/reward debugging. PPO is less sensitive than FastSAC.

## Repo Map

Main packages:

```text
src/holosoma              Core training framework
src/holosoma_inference    ONNX inference/deployment
src/holosoma_retargeting  Motion retargeting and data conversion
```

Important entrypoints:

```text
src/holosoma/holosoma/train_agent.py
src/holosoma/holosoma/eval_agent.py
src/holosoma/holosoma/run_sim.py
src/holosoma_inference/holosoma_inference/run_policy.py
```

Training preset registry:

```text
src/holosoma/holosoma/config_values/experiment.py
```

G1 locomotion configs:

```text
src/holosoma/holosoma/config_values/loco/g1/
```

Robot config:

```text
src/holosoma/holosoma/config_values/robot.py
```

Action implementation:

```text
src/holosoma/holosoma/managers/action/terms/joint_control.py
```

## Current Code Changes

The task-specific code change is already implemented.

Added action term:

```text
FixedJointPositionActionTerm
```

Location:

```text
src/holosoma/holosoma/managers/action/terms/joint_control.py
```

Behavior:

```text
Policy still outputs 29 actions.
The 14 arm-joint actions are ignored/overridden.
Arm joints are held by PD position control at fixed targets.
Legs and waist are still controlled normally by the policy.
```

Reason for keeping 29 actions:

```text
Avoids larger changes to PPO actor output, observation action history, symmetry utilities, ONNX export, and robot_config.actions_dim assertions.
```

Added action preset:

```text
g1_29dof_arm_hold_joint_pos
```

Location:

```text
src/holosoma/holosoma/config_values/loco/g1/action.py
```

Added robot preset:

```text
g1_29dof_arm_hold
```

Location:

```text
src/holosoma/holosoma/config_values/robot.py
```

Added experiment preset:

```text
g1_29dof_arm_hold
```

CLI name:

```text
exp:g1-29dof-arm-hold
```

Location:

```text
src/holosoma/holosoma/config_values/loco/g1/experiment.py
src/holosoma/holosoma/config_values/experiment.py
```

Symmetry is disabled for this preset:

```text
use_symmetry=False
```

Reason: the requested left/right arm pose is not perfectly symmetric.

## Fixed Arm Pose

```python
ARM_HOLD_JOINT_ANGLES = {
    "left_shoulder_pitch_joint": -0.75,
    "left_shoulder_roll_joint": 0.35,
    "left_shoulder_yaw_joint": 0.08,
    "left_elbow_joint": 0.60,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.20,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": -0.75,
    "right_shoulder_roll_joint": -0.40,
    "right_shoulder_yaw_joint": 0.35,
    "right_elbow_joint": 0.65,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": -0.10,
    "right_wrist_yaw_joint": 0.0,
}
```

G1 29-DOF order:

```text
0  left_hip_pitch_joint
1  left_hip_roll_joint
2  left_hip_yaw_joint
3  left_knee_joint
4  left_ankle_pitch_joint
5  left_ankle_roll_joint
6  right_hip_pitch_joint
7  right_hip_roll_joint
8  right_hip_yaw_joint
9  right_knee_joint
10 right_ankle_pitch_joint
11 right_ankle_roll_joint
12 waist_yaw_joint
13 waist_roll_joint
14 waist_pitch_joint
15 left_shoulder_pitch_joint
16 left_shoulder_roll_joint
17 left_shoulder_yaw_joint
18 left_elbow_joint
19 left_wrist_roll_joint
20 left_wrist_pitch_joint
21 left_wrist_yaw_joint
22 right_shoulder_pitch_joint
23 right_shoulder_roll_joint
24 right_shoulder_yaw_joint
25 right_elbow_joint
26 right_wrist_roll_joint
27 right_wrist_pitch_joint
28 right_wrist_yaw_joint
```

## Environment

The user is running on:

```text
RTX 5060 Ti 16GB
SSH server
No display
IsaacSim headless
```

Use IsaacSim env:

```bash
cd /home/jkl0909/Code/rl/holosoma
unset CONDA_ENV_NAME
unset LD_LIBRARY_PATH
unset DISPLAY
source scripts/source_isaacsim_setup.sh
```

IsaacSim URDF importer needs extra libs on this machine:

```bash
export ASSET_CONVERTER_LIBS="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.asset_converter-5.0.17+107.3.1.lx64.r.cp311.u353/asset_converter_native_bindings/libs"
export OLD_LIBXML2="/home/jkl0909/.holosoma_deps/miniconda3/pkgs/libxml2-2.13.9-h2c43086_0/lib"
export ICU73="/home/jkl0909/.holosoma_deps/miniconda3/pkgs/icu-73.1-h6a678d5_0/lib"
export LD_LIBRARY_PATH="$ASSET_CONVERTER_LIBS:$OLD_LIBXML2:$ICU73:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Without this, expected errors include:

```text
libxml2.so.2: cannot open shared object file
libicuuc.so.73: cannot open shared object file
Failed to acquire interface: isaacsim::asset::importer::urdf::Urdf
```

`GLFW initialization failed` and `No Viewport Window` are usually tolerable warnings in SSH/headless mode when video is disabled.

## Train Command

Recommended tmux workflow:

```bash
tmux new -s g1-arm-hold
cd /home/jkl0909/Code/rl/holosoma
unset CONDA_ENV_NAME
unset LD_LIBRARY_PATH
unset DISPLAY
source scripts/source_isaacsim_setup.sh

export ASSET_CONVERTER_LIBS="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.asset_converter-5.0.17+107.3.1.lx64.r.cp311.u353/asset_converter_native_bindings/libs"
export OLD_LIBXML2="/home/jkl0909/.holosoma_deps/miniconda3/pkgs/libxml2-2.13.9-h2c43086_0/lib"
export ICU73="/home/jkl0909/.holosoma_deps/miniconda3/pkgs/icu-73.1-h6a678d5_0/lib"
export LD_LIBRARY_PATH="$ASSET_CONVERTER_LIBS:$OLD_LIBXML2:$ICU73:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

DISPLAY= python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-arm-hold \
  logger:wandb \
  simulator:isaacsim \
  --logger.video.enabled False
```

Detach tmux:

```text
Ctrl-b, then d
```

Attach:

```bash
tmux attach -t g1-arm-hold
```

## W&B

W&B login has worked on this machine.

Project:

```text
https://wandb.ai/12wuu115-post-and-telecommunication-institute-of-technology/hv-g1-manager
```

Example run:

```text
https://wandb.ai/12wuu115-post-and-telecommunication-institute-of-technology/hv-g1-manager/runs/8ohgu33d
```

`View project` shows all runs in project. `View run` shows one train run.

## How To Read Current Metrics

W&B group counts such as `Env 3`, `Episode 10`, `Loss 10` mean the number of charts in that group, not the number of environments/episodes.

Actual training env count is controlled by:

```text
training.num_envs
```

Default is usually 4096 unless overridden.

### Env

Important:

```text
Env/average_episode_length
Env/penalty_scale
Env/action_clip_frac
```

Interpretation:

```text
average_episode_length increasing is good.
action_clip_frac near 0 is good.
penalty_scale is curriculum scale for penalty terms.
```

### Episode

Important:

```text
Episode/rew_tracking_lin_vel
Episode/rew_tracking_ang_vel
Episode/rew_alive
Episode/rew_feet_phase
Episode/rew_pose
Episode/rew_penalty_orientation
Episode/rew_penalty_action_rate
```

Good signs:

```text
tracking rewards increase
rew_alive increases
rew_feet_phase increases
```

Watch:

```text
rew_pose becomes too negative
rew_penalty_orientation becomes too negative
rew_penalty_action_rate becomes too negative
```

For this arm-hold task, pose/orientation/action-rate penalties matter because the forward arms change balance and may induce unstable gait.

### RawEpisode

Difference:

```text
RawEpisode = raw reward terms before weights/curriculum scaling.
Episode    = weighted/scaled reward contribution used for optimization.
```

Use `Episode` for quick training quality. Use `RawEpisode` to debug why a term changed.

### Loss

Important:

```text
Loss/KL
Loss/Entropy
Loss/critic_loss
Loss/actor_loss
Loss/Value
Loss/Surrogate
```

Current expected signs:

```text
Loss/symmetry_actor_loss = 0
Loss/symmetry_critic_loss = 0
```

Because symmetry is disabled.

KL around 0.01-0.02 is reasonable. Losses do not need to monotonically decrease in PPO. Watch for NaN, exploding value loss, or KL spikes.

### Perf

Important:

```text
Perf/total_fps
Perf/learning_time
Perf/collection_time
```

On RTX 5060 Ti 16GB + IsaacSim, observed `total_fps` around 31k-33k looked OK. Collection time is usually larger than learning time, meaning simulation rollout is the bottleneck.

### Policy

Important:

```text
Policy/mean_noise_std
```

It measures policy exploration noise. Too low too early can mean premature convergence. Values around 0.4-1.0 during early training were not immediately concerning while rewards/episode length were improving.

### Train

Important:

```text
Train/num_samples
Train/mean_reward
Train/mean_episode_length
```

Good:

```text
num_samples increases linearly
mean_episode_length increases
mean_reward improves
```

If mean reward and mean episode length peak then drop, the policy is not necessarily dead; it may be curriculum/update instability. Continue watching whether it recovers.

## Checkpoint And ONNX

Logs/checkpoints:

```text
logs/hv-g1-manager/<timestamp>-g1_29dof_arm_hold_ppo-locomotion/
```

Find latest checkpoint:

```bash
find logs/hv-g1-manager -name 'model_*.pt' | sort | tail
```

Find latest ONNX:

```bash
find logs/hv-g1-manager -name '*.onnx' | sort | tail
```

Eval:

```bash
DISPLAY= python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=/path/to/model_xxxxx.pt \
  --training.max-eval-steps=1000 \
  --logger.video.enabled False
```

Export ONNX during eval:

```bash
DISPLAY= python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=/path/to/model_xxxxx.pt \
  --training.export-onnx True \
  --training.max-eval-steps=1000 \
  --logger.video.enabled False
```

## Next Technical Steps

1. Let PPO arm-hold train long enough to see if `mean_episode_length` and `mean_reward` stabilize.
2. Evaluate checkpoint in sim and visually/recorded inspect whether arms stay in carry pose.
3. If locomotion is stable, add payload/randomization during training, not only eval.
4. Use payload eval callback as a stress test, but do not expect eval to update policy.
5. Consider a smaller future refactor to make policy output 15 DOF instead of 29 if wasted arm actions become a real issue.
6. Only benchmark FastSAC after PPO baseline is stable.

## Important Constraints

Do not revert user changes. Current working tree includes several modified files for this task and `AGENT.md` is untracked/new.

Use `apply_patch` for manual edits. Prefer `rg` for search. Avoid destructive git commands.
