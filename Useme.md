# Holosoma G1 Arm-Hold PPO Runbook

Task hiện tại: train G1 29-DOF locomotion với hai tay được PD giữ ở tư thế bê vật. Chưa thêm payload/khối lượng.

Preset dùng:

```text
exp:g1-29dof-arm-hold
```

W&B:

```text
Project: https://wandb.ai/12wuu115-post-and-telecommunication-institute-of-technology/hv-g1-manager
Run:     https://wandb.ai/12wuu115-post-and-telecommunication-institute-of-technology/hv-g1-manager/runs/8ohgu33d
```

## 1. Kích Hoạt Môi Trường

```bash
cd /home/jkl0909/Code/rl/holosoma
unset CONDA_ENV_NAME
unset LD_LIBRARY_PATH
unset DISPLAY
source scripts/source_isaacsim_setup.sh
```

`unset DISPLAY` giúp IsaacSim không cố bám vào display ảo/cũ khi chạy qua SSH.

## 2. Fix IsaacSim URDF Importer Libs

IsaacSim trên máy này cần thêm `libxml2.so.2`, ICU 73 và asset converter libs vào `LD_LIBRARY_PATH`.

```bash
export ASSET_CONVERTER_LIBS="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.asset_converter-5.0.17+107.3.1.lx64.r.cp311.u353/asset_converter_native_bindings/libs"
export OLD_LIBXML2="/home/jkl0909/.holosoma_deps/miniconda3/pkgs/libxml2-2.13.9-h2c43086_0/lib"
export ICU73="/home/jkl0909/.holosoma_deps/miniconda3/pkgs/icu-73.1-h6a678d5_0/lib"
export LD_LIBRARY_PATH="$ASSET_CONVERTER_LIBS:$OLD_LIBXML2:$ICU73:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Kiểm tra nhanh:

```bash
ldd "$ASSET_CONVERTER_LIBS/libomniverse_asset_converter.so" | grep -E "libxml2.so.2|libicuuc.so.73|not found"
```

Không được còn:

```text
libxml2.so.2 => not found
libicuuc.so.73 => not found
```

## 3. Train Qua SSH Bằng Tmux

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

Thoát khỏi tmux nhưng giữ job chạy:

```text
Ctrl-b, rồi d
```

Quay lại session:

```bash
tmux attach -t g1-arm-hold
```

## 4. Kiểm Tra W&B

```bash
wandb status
```

Nếu chưa login:

```bash
wandb login
```

Khi train, mở `View run` để xem lần train hiện tại. Mở `View project` để so sánh nhiều run.

## 5. Metric Ưu Tiên Theo Dõi

Quan trọng nhất:

```text
Train/mean_episode_length
Train/mean_reward
Env/average_episode_length
Env/penalty_scale
Episode/rew_tracking_lin_vel
Episode/rew_tracking_ang_vel
Episode/rew_alive
Episode/rew_pose
Episode/rew_penalty_orientation
Episode/rew_penalty_action_rate
```

Dấu hiệu tốt:

```text
average_episode_length tăng
rew_alive tăng
tracking reward tăng
action_clip_frac gần 0
KL khoảng 0.01-0.02
FPS ổn định
```

Dấu hiệu cần chú ý:

```text
rew_pose tụt sâu
rew_penalty_orientation âm mạnh hơn
rew_penalty_action_rate âm mạnh hơn
mean_episode_length tụt lâu không hồi
loss NaN
GPU memory chạm trần
```

## 6. Checkpoint Và ONNX

Checkpoint và log được lưu trong:

```text
logs/hv-g1-manager/<timestamp>-g1_29dof_arm_hold_ppo-locomotion/
```

Tìm checkpoint mới nhất:

```bash
find logs/hv-g1-manager -name 'model_*.pt' | sort | tail
```

Tìm ONNX mới nhất:

```bash
find logs/hv-g1-manager -name '*.onnx' | sort | tail
```

## 7. Eval Checkpoint

```bash
DISPLAY= python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=/path/to/model_xxxxx.pt \
  --training.max-eval-steps=1000 \
  --logger.video.enabled False
```

Nếu cần export ONNX trong eval:

```bash
DISPLAY= python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=/path/to/model_xxxxx.pt \
  --training.export-onnx True \
  --training.max-eval-steps=1000 \
  --logger.video.enabled False
```

## 8. Ghi Chú Task

Task này đang giữ tay bằng PD target cố định. Policy vẫn output 29 action để tránh phá pipeline PPO/export, nhưng action của 14 khớp tay bị override bởi `FixedJointPositionActionTerm`.

Pose tay:

```text
left_shoulder_pitch_joint  = -0.75
left_shoulder_roll_joint   =  0.35
left_shoulder_yaw_joint    =  0.08
left_elbow_joint           =  0.60
left_wrist_roll_joint      =  0.00
left_wrist_pitch_joint     =  0.20
left_wrist_yaw_joint       =  0.00
right_shoulder_pitch_joint = -0.75
right_shoulder_roll_joint  = -0.40
right_shoulder_yaw_joint   =  0.35
right_elbow_joint          =  0.65
right_wrist_roll_joint     =  0.00
right_wrist_pitch_joint    = -0.10
right_wrist_yaw_joint      =  0.00
```

Chưa thêm payload. Bước tiếp theo sau khi locomotion ổn là thêm payload/randomization khi train, không chỉ eval.
