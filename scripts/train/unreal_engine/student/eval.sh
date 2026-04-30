#!/bin/bash
# Inference of the DAgger student over its train motions (no teacher in the
# loop). The env plays each motion clip and feeds the student its observed
# motion-driven command (motion_ref_pos_b/ori_b) plus heightmap each step,
# exactly the way VideoMimic's `play_terrain_policy.sh` does inference.
#
# Override CHECKPOINT to pick a different student iteration.

source scripts/source_isaacsim_setup.sh

CHECKPOINT="${CHECKPOINT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260428_g1_unreal_student_DAgger/model_31000.pt}"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned"

python src/holosoma/holosoma/eval_agent.py \
    --checkpoint="$CHECKPOINT" \
    --terrain.terrain-term.obj-dir="$MOTION_DIR" \
    --terrain.terrain-term.num-rows 1 \
    --terrain.terrain-term.obj-max-faces-per-tile 0 \
    --terrain.terrain-term.obj-max-tiles 1 \
    --terrain.terrain-term.obj-tile-gap 1.0 \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$MOTION_DIR" \
    --command.setup_terms.motion_command.params.motion_config.max_motions 1 \
    --command.setup_terms.motion_command.params.motion_config.pool_size 1 \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.num_envs 1 \
    --training.headless False \
    --simulator.config.scene.env_spacing=0.0
