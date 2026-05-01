#!/bin/bash
# Stage-4-style RL finetune of the DAgger student using PPO with the standard
# motion-tracking reward (the same `g1_29dof_wbt_reward` the teacher was
# trained with). Mirrors VideoMimic's `train_stage_4_rl_finetune.sh`:
#
#   * Same observation as the DAgger student (so its weights drop in cleanly).
#   * Algo: PPO (no BC).
#   * Lower learning rate (3e-5, fixed schedule).
#   * --training.finetune True passes strict=False to PPO.load(); the modified
#     load() keeps the freshly random-init critic when the DAgger checkpoint
#     has none, so the actor weights load and the critic warms up from
#     scratch over the first few hundred PPO updates.
#
# Override STUDENT_CKPT to start from a different DAgger iter.

source scripts/source_isaacsim_setup.sh

MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned"
STUDENT_CKPT="${STUDENT_CKPT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260429_g1_unreal_student_DAgger/model_40000.pt}"

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-multi-terrain-videomimic-student-rl-finetune \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=Unreal_Engine \
    --logger.name=g1_unreal_student_RLfinetune \
    --logger.video.enabled False \
    --training.checkpoint="$STUDENT_CKPT" \
    --training.finetune True \
    --simulator.config.videomimic-height-map.map-height 11 \
    --simulator.config.videomimic-height-map.map-width 17 \
    --simulator.config.videomimic-height-map.resolution 0.1 \
    --simulator.config.videomimic-height-map.num-channels 3 \
    --terrain.terrain-term.obj-dir="$MOTION_DIR" \
    --terrain.terrain-term.num-rows 1 \
    --terrain.terrain-term.obj-max-faces-per-tile 0 \
    --terrain.terrain-term.obj-tile-gap 1.0 \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$MOTION_DIR" \
    --command.setup_terms.motion_command.params.motion_config.max_motions 200 \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.init_root_offset="[0.0, 0.0, 0.1]" \
    --training.num_envs 4096 \
    --simulator.config.scene.env_spacing=0.0 \
    --algo.config.save_interval 1000 \
    --algo.config.eval_interval 1000
