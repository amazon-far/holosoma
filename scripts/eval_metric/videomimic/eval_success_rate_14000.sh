#!/bin/bash
# Per-motion success rate evaluation for the videomimic multi-terrain checkpoint.
# Runs every .npz under videomimic_captures_29dof (130 motions) and writes a
# per-motion OK/FAIL report to the eval log directory.

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="/home/kyungminlee/work/holosoma/logs/MotionTracking/20260422_g1_videomimic_multi_terrain_future_motion/model_14000.pt"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof"

export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 25))        # 32 MB
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 24))               # 16 MB
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 17))     # 128 K
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 16))   # 64 K
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 15))       # 32 K

python src/holosoma/holosoma/eval_success_rate.py \
    --checkpoint="$CHECKPOINT" \
    --motion_dir="$MOTION_DIR" \
    --num_envs=130
