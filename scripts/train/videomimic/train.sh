#!/bin/bash
# Train the PhUMA multi-motion policy on VideoMimic captures with per-motion
# scene meshes laid out in an n_rows x n_motions grid.
#
# Each .npz in MOTION_DIR contributes one tile column (scene mesh + motion).
# `n_rows` just duplicates that column strip vertically -- it controls the
# terrain layout, NOT the number of envs. At reset, MultiMotionCommand samples
# a motion per env and then picks a random row within that motion's column,
# rebinding the env_origin to the matching tile. So num_envs is independent
# from n_rows and n_motions (follow the PhUMA convention, e.g. 4096).
set -euo pipefail

source scripts/source_isaacsim_setup.sh

MOTION_DIR="${MOTION_DIR:-src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof}"
N_AVAILABLE="$(find "$MOTION_DIR" -type f -name '*.npz' | wc -l)"
if [[ "$N_AVAILABLE" -eq 0 ]]; then
    echo "No .npz files found under $MOTION_DIR" >&2
    exit 1
fi

# Small-scale defaults proven to fit in GPU memory and finish USD mesh import
# quickly. Once you confirm the pipeline runs end-to-end, scale up by setting
# MAX_MOTIONS / NUM_ENVS / MAX_FACES_PER_TILE via env vars.
MAX_MOTIONS="${MAX_MOTIONS:-20}"
if [[ "$MAX_MOTIONS" -gt 0 && "$MAX_MOTIONS" -lt "$N_AVAILABLE" ]]; then
    N_MOTIONS="$MAX_MOTIONS"
else
    N_MOTIONS="$N_AVAILABLE"
fi

# num_envs is independent of the terrain grid.
NUM_ENVS="${NUM_ENVS:-512}"
# n_rows: duplicates the column strip along y. MUST be large enough that
# n_rows * n_motions >= num_envs, otherwise round-robin init spawns multiple
# robots at the same world transform and PhysX contact buffer overflows.
# Default = ceil(num_envs / n_motions) so each env init-lands on a unique cell.
N_ROWS_DEFAULT="$(( (NUM_ENVS + N_MOTIONS - 1) / N_MOTIONS ))"
N_ROWS="${N_ROWS:-$N_ROWS_DEFAULT}"

echo "VideoMimic multi-mesh training:"
echo "  motion_dir = $MOTION_DIR"
echo "  n_motions  = $N_MOTIONS of $N_AVAILABLE available (cap = MAX_MOTIONS=$MAX_MOTIONS)"
echo "  n_rows     = $N_ROWS"
echo "  num_envs   = $NUM_ENVS"

LOCAL_RANK=${LOCAL_RANK:-1} python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma \
    terrain:terrain-load-obj \
    logger:disabled \
    --terrain.terrain-term.obj-dir="$MOTION_DIR" \
    --terrain.terrain-term.num-rows "$N_ROWS" \
    --terrain.terrain-term.obj-max-tiles "$MAX_MOTIONS" \
    --terrain.terrain-term.obj-max-faces-per-tile "${MAX_FACES_PER_TILE:-3000}" \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$MOTION_DIR" \
    --command.setup_terms.motion_command.params.motion_config.pool_size "$N_MOTIONS" \
    --command.setup_terms.motion_command.params.motion_config.pool_ordered True \
    --command.setup_terms.motion_command.params.motion_config.bind_terrain_tiles True \
    --command.setup_terms.motion_command.params.motion_config.max_motions "$MAX_MOTIONS" \
    --command.setup_terms.motion_command.params.motion_config.resample_interval 0 \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.num_envs "$NUM_ENVS" \
    --simulator.config.scene.env_spacing=0.0 \
    --algo.config.save_interval 1000 \
    --algo.config.eval_interval 0 \
    --training.headless False \
