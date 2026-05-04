#!/bin/bash
# Single-motion Isaac Sim playback for the RL-finetuned student on a small
# synthetic stair (h=10cm, w=71cm, ascending, 12 steps). Runs with
# headless=False so the Isaac Sim window opens and you can watch the
# student receive the +x command and try to walk forward.
#
# Override SYNTH_DIR to point at a directory with a different single
# synthetic stair .npz.

source scripts/source_isaacsim_setup.sh

CHECKPOINT="${CHECKPOINT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260501_g1_unreal_student_RLfinetune/model_27000.pt}"
SYNTH_DIR="${SYNTH_DIR:-/tmp/synth_one_stair}"

python src/holosoma/holosoma/eval_agent.py \
    --checkpoint="$CHECKPOINT" \
    --terrain.terrain-term.obj-dir="$SYNTH_DIR" \
    --terrain.terrain-term.num-rows 1 \
    --terrain.terrain-term.obj-max-faces-per-tile 0 \
    --terrain.terrain-term.obj-max-tiles 1 \
    --terrain.terrain-term.obj-tile-gap 1.0 \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$SYNTH_DIR" \
    --command.setup_terms.motion_command.params.motion_config.max_motions 1 \
    --command.setup_terms.motion_command.params.motion_config.pool_size 1 \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.num_envs 1 \
    --training.headless False \
    --simulator.config.scene.env_spacing=0.0
