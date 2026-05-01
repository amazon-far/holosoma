#!/bin/bash
# Render 10 success + 10 failure per-motion mp4s for the videomimic
# diverse-action-scale checkpoint at iter 26000.
#
# Pipeline:
#   1. eval_success_rate.py — full 126-motion eval, writes per-motion OK/FAIL
#      to <eval_log_dir>/success_rate_iter26000.txt.
#   2. Sample 10 OK + 10 FAIL (0.5m threshold), symlink them into a filtered
#      motion dir.
#   3. render_per_motion.py on the filtered dir → per-motion mp4 in
#      videos/<run_dir>_model_26000/.

cd /home/kyungminlee/work/holosoma

source scripts/source_isaacsim_setup.sh

CHECKPOINT="/home/kyungminlee/work/holosoma/logs/MotionTracking/20260428_g1_videomimic_multi_terrain_future_motion_heightmap_xyz_diverse_action_scale/model_26000.pt"
RUN_DIR="$(basename "$(dirname "$CHECKPOINT")")"
CKPT_NAME="$(basename "$CHECKPOINT" .pt)"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof"

WORK_DIR="${WORK_DIR:-/tmp/dagger_sweep/${RUN_DIR}_${CKPT_NAME}}"
FILTERED_DIR="${FILTERED_DIR:-${WORK_DIR}/filtered_motions}"
OUTPUT_DIR="${OUTPUT_DIR:-videos/${RUN_DIR}_${CKPT_NAME}}"
mkdir -p "$WORK_DIR" "$FILTERED_DIR" "$OUTPUT_DIR"

# --- (sandbox-friendly) shrink PhysX GPU buffers so eval coexists with a running
# training job on the same GPU. Same values as the existing
# eval_success_rate_10000.sh.
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 25))
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 24))
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 17))
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 16))
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 15))

# How many motions are actually on disk — 130 .npz files in the
# videomimic_captures_29dof tree (each lives in its own subdirectory under a
# common ``motion_for_holosoma_with_mesh.npz`` name). MotionLibrary uses a
# recursive glob, so we count via find. The checkpoint was trained with
# pool_size=8192 (default), so we need to override max_motions/pool_size to
# match the discovered count.
N_MOTIONS=$(find "$MOTION_DIR" -maxdepth 2 -name "*.npz" | wc -l)
echo "Discovered $N_MOTIONS motions under $MOTION_DIR"

# --- Phase 1: success rate eval ---
if [ ! -f "$WORK_DIR/per_motion.tsv" ]; then
    EVAL_LOG="$WORK_DIR/eval.log"
    echo "Running eval_success_rate.py (log: $EVAL_LOG)"
    python src/holosoma/holosoma/eval_success_rate.py \
        --checkpoint="$CHECKPOINT" \
        --motion_dir="$MOTION_DIR" \
        --max_motions="$N_MOTIONS" \
        --num_envs="$N_MOTIONS" > "$EVAL_LOG" 2>&1

    SR_TXT=$(grep -oE "/[^ ]+/success_rate_iter[0-9]+\.txt" "$EVAL_LOG" | tail -1)
    if [ -z "$SR_TXT" ] || [ ! -f "$SR_TXT" ]; then
        echo "Failed to find success_rate report in eval log:" >&2
        tail -20 "$EVAL_LOG" >&2
        exit 1
    fi
    echo "Eval report: $SR_TXT"

    # Extract Per-Motion [0.5m] section into a TSV: status\tpath
    awk '
        /^--- Per-Motion \[0\.5m\]/ { in_block=1; next }
        /^--- / && !/0\.5m/ { in_block=0 }
        in_block && /^  OK:/ { sub(/^  OK: /, ""); print "OK\t" $0 }
        in_block && /^  FAIL:/ { sub(/^  FAIL: /, ""); print "FAIL\t" $0 }
    ' "$SR_TXT" > "$WORK_DIR/per_motion.tsv"
fi

OK_COUNT=$(grep -c '^OK' "$WORK_DIR/per_motion.tsv")
FAIL_COUNT=$(grep -c '^FAIL' "$WORK_DIR/per_motion.tsv")
echo "Per-motion: $OK_COUNT OK, $FAIL_COUNT FAIL"

# --- Phase 2: sample 10 + 10 and copy subdirs into filtered_motions ---
# Each motion lives in its own subdirectory like
# ``<motion_dir>/<subdir>/motion_for_holosoma_with_mesh.npz``. We *copy*
# (not symlink) the subdirectories because both MotionLibrary and the
# terrain manager use ``Path.rglob`` to discover .npz files, and rglob does
# not follow symlinks. Each motion is only ~4MB, so 20 copies cost ~80MB.
if [ -z "$(ls -A "$FILTERED_DIR" 2>/dev/null)" ]; then
    grep '^OK' "$WORK_DIR/per_motion.tsv" | shuf -n 10 --random-source="$WORK_DIR/per_motion.tsv" \
        > "$WORK_DIR/selected_ok.tsv"
    grep '^FAIL' "$WORK_DIR/per_motion.tsv" | shuf -n 10 --random-source="$WORK_DIR/per_motion.tsv" \
        > "$WORK_DIR/selected_fail.tsv"

    i=0
    while IFS=$'\t' read -r status path; do
        # path is <motion_dir>/<subdir>/motion_for_holosoma_with_mesh.npz
        # (relative to repo root). Get the subdir's absolute path.
        subdir_abs=$(dirname "/home/kyungminlee/work/holosoma/$path")
        subdir_name=$(basename "$subdir_abs")
        label=$(echo "$status" | tr '[:upper:]' '[:lower:]')
        cp -r "$subdir_abs" "$FILTERED_DIR/$(printf '%02d' $i)_${label}_${subdir_name}"
        i=$((i+1))
    done < <(cat "$WORK_DIR/selected_ok.tsv" "$WORK_DIR/selected_fail.tsv")
fi
echo "Filtered dir has $(ls "$FILTERED_DIR" | wc -l) motion subdirs ($(find "$FILTERED_DIR" -maxdepth 2 -name '*.npz' | wc -l) .npz total)"

# --- Phase 3: roll out the 20 selected motions in one IsaacSim session,
# saving per-motion policy_<motion>.npy. Use train-level GPU buffers.
export HOLOSOMA_GPU_COLLISION_STACK_SIZE=$((1 << 30))
export HOLOSOMA_GPU_HEAP_CAPACITY=$((1 << 26))
export HOLOSOMA_GPU_MAX_RIGID_CONTACT_COUNT=$((1 << 22))
export HOLOSOMA_GPU_FOUND_LOST_PAIRS_CAPACITY=$((1 << 20))
export HOLOSOMA_GPU_MAX_RIGID_PATCH_COUNT=$((1 << 19))

ROLLOUT_DIR="${WORK_DIR}/rollouts"
mkdir -p "$ROLLOUT_DIR"
python scripts/eval/unreal_engine/rollout_single_policy.py \
    --checkpoint="$CHECKPOINT" \
    --motion_dir="$FILTERED_DIR" \
    --out_dir="$ROLLOUT_DIR"

# --- Phase 4: render with visualize.py — policy as main, motion .npz as
# ghost (visualize.py reads .npz directly: extracts pelvis pose + joint angles
# and converts wxyz quat → xyzw internally). ---
VIZ_PY="${VIZ_PY:-/home/kyungminlee/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
mkdir -p "$OUTPUT_DIR"
for sub in "$FILTERED_DIR"/*/; do
    sub_name=$(basename "$sub")
    policy_npy="$ROLLOUT_DIR/policy_${sub_name}.npy"
    npz_path=$(find "$sub" -maxdepth 2 -name "*.npz" | head -1)
    out_mp4="$OUTPUT_DIR/${sub_name}.mp4"

    if [ ! -f "$policy_npy" ] || [ ! -f "$npz_path" ]; then
        echo "Skipping $sub_name (missing inputs)"
        continue
    fi

    "$VIZ_PY" visualize.py \
        --input "$policy_npy" \
        --ref_motion "$npz_path" \
        --terrain_npz "$npz_path" \
        --output "$out_mp4" \
        --ghost_alpha 0.45 \
        --fps 30
done

echo "Done. mp4s in $OUTPUT_DIR"
