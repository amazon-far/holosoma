#!/bin/bash
# Evaluate success rate of model_05000.pt on all 200 aligned UE staircase motions.
# Reuses the same MOTION_DIR / num_envs as the training run so motion-to-tile
# binding is identical. No split file (eval on the full training set).

source scripts/source_isaacsim_setup.sh

# Reduce CUDA fragmentation; eval re-uses the training pool size of 200 so this
# is mostly precautionary.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT="/home/kyungminlee/work/holosoma/logs/MotionTracking/20260426_g1_ue_multi_terrain_stair_200_future_motion_heightmap_xyz/model_05000.pt"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned"
# Match num_envs to num_motions: SuccessRateCallback only uses the first
# min(num_envs, num_motions) envs, so 4096 wastes ~95% of GPU memory.
NUM_ENVS=200

python src/holosoma/holosoma/eval_success_rate.py \
    --checkpoint="${CHECKPOINT}" \
    --motion_dir="${MOTION_DIR}" \
    --num_envs="${NUM_ENVS}"
