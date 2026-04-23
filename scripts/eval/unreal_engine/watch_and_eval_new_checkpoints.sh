#!/bin/bash
# Watch training log dirs for new checkpoints, run eval_agent.py + visualize.py
# per checkpoint, and save the rendered mp4 under
#   videos/unreal_engine_eval/<wandb_name>/checkpoint_XXXXX.mp4
#
# Usage:
#   bash scripts/eval/unreal_engine/watch_and_eval_new_checkpoints.sh          # poll loop (default 60s)
#   POLL_INTERVAL=30 bash scripts/eval/unreal_engine/watch_and_eval_new_checkpoints.sh
#   bash scripts/eval/unreal_engine/watch_and_eval_new_checkpoints.sh --once   # single pass then exit
#
# Per-checkpoint motion data is kept at
#   <log_dir>/motion_data/ref_motion_XXXXX.npy  (so new checkpoints don't
# clobber each other; eval_agent.py itself always writes ref_motion.npy, we
# rename it after each eval).

set -u
source scripts/source_isaacsim_setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

LOGS_DIR="$PROJECT_ROOT/logs/MotionTracking"
VIDEOS_DIR="$PROJECT_ROOT/videos/unreal_engine_eval"
MOTION_DATA_DIR="$PROJECT_ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/unreal_engine"
SCENE_XML="$PROJECT_ROOT/src/holosoma/holosoma/data/robots/g1/g1_29dof.xml"

POLL_INTERVAL="${POLL_INTERVAL:-60}"
# Skip checkpoints whose mtime is newer than this (still being written). Seconds.
CHECKPOINT_STABILITY_SEC="${CHECKPOINT_STABILITY_SEC:-15}"

mkdir -p "$VIDEOS_DIR"

# Map experiment log dir -> terrain npz filename (in MOTION_DATA_DIR)
declare -A TERRAIN
TERRAIN["20260419_g1_stair_30depth_attention_heightmap"]="g1_stair_30depth_20260419.npz"
TERRAIN["20260419_g1_stair_30depth_with_blendspace_motion"]="g1_stair_30depth_20260419.npz"
TERRAIN["20260419_g1_stair_30depth_with_videomimic_heightmap"]="g1_stair_30depth_20260419.npz"
TERRAIN["20260419_g1_stair_30depth_with_videomimic_heightmap_foot_contact"]="g1_stair_30depth_20260420_foot_contact.npz"
TERRAIN["20260419_g1_stair_30depth_with_videomimic_heightmap_real"]="g1_stair_30depth_20260419.npz"
TERRAIN["20260419_g1_stair_60depth_with_blendspace_motion"]="g1_stair_60depth_20260419.npz"
TERRAIN["20260419_g1_stair_60depth_with_videomimic_heightmap"]="g1_stair_60depth_20260419.npz"
TERRAIN["20260419_g1_stair_60depth_with_videomimic_heightmap_real"]="g1_stair_60depth_20260419.npz"

ONCE=0
if [ "${1:-}" = "--once" ]; then
    ONCE=1
fi

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

process_checkpoint() {
    local log_dir="$1"       # absolute path
    local ckpt_basename="$2" # e.g. model_00500.pt
    local wandb_name="$3"
    local terrain_npz_name="$4"

    # Extract the padded number, e.g. model_00500.pt -> 00500
    local ckpt_num_padded
    ckpt_num_padded="${ckpt_basename#model_}"
    ckpt_num_padded="${ckpt_num_padded%.pt}"

    local out_dir="$VIDEOS_DIR/$wandb_name"
    local out_mp4="$out_dir/checkpoint_${ckpt_num_padded}.mp4"

    if [ -f "$out_mp4" ]; then
        return 0  # already processed
    fi

    local ckpt_path="$log_dir/$ckpt_basename"
    # Stability check — don't touch a checkpoint that was written seconds ago.
    local now
    now=$(date +%s)
    local mtime
    mtime=$(stat -c '%Y' "$ckpt_path" 2>/dev/null || echo 0)
    if [ $((now - mtime)) -lt "$CHECKPOINT_STABILITY_SEC" ]; then
        return 0
    fi

    mkdir -p "$out_dir"

    local motion_data_dir="$log_dir/motion_data"
    local motion_data_per_ckpt="$motion_data_dir/ref_motion_${ckpt_num_padded}.npy"
    local motion_data_default="$motion_data_dir/ref_motion.npy"

    # Run eval if per-checkpoint npy is missing.
    if [ ! -f "$motion_data_per_ckpt" ]; then
        log "EVAL  $wandb_name / $ckpt_basename"
        mkdir -p "$motion_data_dir"
        # Remove the default path so we can detect whether eval_agent.py wrote a new one.
        rm -f "$motion_data_default"
        if python src/holosoma/holosoma/eval_agent.py \
                --checkpoint="$ckpt_path" \
                --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
                --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
                --training.num_envs 1
        then
            if [ -f "$motion_data_default" ]; then
                mv "$motion_data_default" "$motion_data_per_ckpt"
            else
                log "  WARN: eval produced no $motion_data_default — skipping"
                return 1
            fi
        else
            log "  WARN: eval_agent.py failed for $ckpt_path — skipping"
            return 1
        fi
    fi

    log "RENDER $out_mp4"
    if ! python visualize.py \
            --input "$motion_data_per_ckpt" \
            --output "$out_mp4" \
            --scene "$SCENE_XML" \
            --terrain_npz "$MOTION_DATA_DIR/$terrain_npz_name" \
            --input_fps 50
    then
        log "  WARN: visualize.py failed for $motion_data_per_ckpt"
        rm -f "$out_mp4"
        return 1
    fi
}

scan_once() {
    local log_name log_dir wandb_name terrain_npz_name ckpt_path ckpt_basename
    for log_name in "${!TERRAIN[@]}"; do
        log_dir="$LOGS_DIR/$log_name"
        [ -d "$log_dir" ] || continue
        wandb_name="${log_name#*_}"   # drop the leading "20260419_"
        terrain_npz_name="${TERRAIN[$log_name]}"

        shopt -s nullglob
        local ckpt_files=("$log_dir"/model_*.pt)
        shopt -u nullglob
        [ "${#ckpt_files[@]}" -gt 0 ] || continue

        # Sort to process oldest first.
        IFS=$'\n' sorted_ckpts=($(printf '%s\n' "${ckpt_files[@]}" | sort))
        unset IFS
        for ckpt_path in "${sorted_ckpts[@]}"; do
            ckpt_basename=$(basename "$ckpt_path")
            process_checkpoint "$log_dir" "$ckpt_basename" "$wandb_name" "$terrain_npz_name"
        done
    done
}

if [ "$ONCE" = "1" ]; then
    scan_once
    exit 0
fi

log "Watching for new checkpoints every ${POLL_INTERVAL}s (Ctrl+C to stop)"
while true; do
    scan_once
    sleep "$POLL_INTERVAL"
done
