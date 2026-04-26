#!/bin/bash
# Per-motion success rate evaluation for the videomimic multi-terrain checkpoint.
# Runs every .npz under videomimic_captures_29dof (130 motions) and writes a
# per-motion OK/FAIL report to the eval log directory.
#
# Outputs:
#   - Overall success rate at 0.5m and 0.25m thresholds (logged to stdout)
#   - Per-motion results: <eval_log_dir>/success_rate_iter<iter>.txt
#     (eval_log_dir is printed near the top of the run as "Saving eval logs to …")
#
# PhysX GPU buffer overrides below let this eval coexist with a running
# training job on the same GPU. Training needs 2 GB collision stack for
# 4096 envs × 130 tiles; with num_envs=130 the contact count is ~30× lower
# and 256 MB easily suffices.

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="/home/kyungminlee/work/holosoma/logs/MotionTracking/20260422_g1_videomimic_multi_terrain_future_motion/model_10000.pt"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof"

# Shrink PhysX GPU buffers aggressively so eval fits alongside a running
# training job on the same GPU. With num_envs=130 (1 robot per tile) the
# expected contact count is a few thousand; these minima are still 10× over.
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 25))        # 32 MB  (default 2 GB)
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 24))               # 16 MB  (default 256 MB)
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 17))     # 128 K  (default 32 M)
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 16))   # 64 K   (default 8 M)
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 15))       # 32 K   (default 2 M)

python src/holosoma/holosoma/eval_success_rate.py \
    --checkpoint="$CHECKPOINT" \
    --motion_dir="$MOTION_DIR" \
    --num_envs=130
