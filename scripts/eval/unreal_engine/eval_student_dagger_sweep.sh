#!/bin/bash
# Sweep all DAgger student checkpoints on the train motion set, measure
# success rate at 0.5m / 0.25m, and save plot + JSON.

source scripts/source_isaacsim_setup.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="${LOG_DIR:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260428_g1_unreal_student_DAgger}"
MOTION_DIR="${MOTION_DIR:-src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned}"
NUM_ENVS="${NUM_ENVS:-200}"
MAX_MOTIONS="${MAX_MOTIONS:-200}"

EXTRA_ARGS=()
if [ "${SKIP_EXISTING:-0}" = "1" ]; then
    EXTRA_ARGS+=("--skip_existing")
fi

python src/holosoma/holosoma/eval_success_rate_sweep.py \
    --log_dir="$LOG_DIR" \
    --motion_dir="$MOTION_DIR" \
    --num_envs="$NUM_ENVS" \
    --max_motions="$MAX_MOTIONS" \
    "${EXTRA_ARGS[@]}"
