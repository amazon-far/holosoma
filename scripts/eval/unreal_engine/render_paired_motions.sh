#!/bin/bash
# Phase 1: roll out student + teacher on the 20 selected motions.
# Phase 2: render each motion as an mp4 with student=main, teacher=ghost,
#          terrain pulled from the motion .npz (mesh_vertices/mesh_faces).

source scripts/source_isaacsim_setup.sh

STUDENT_CKPT="${STUDENT_CKPT:-/home/kyungminlee/work/holosoma/logs/MotionTracking/20260429_g1_unreal_student_DAgger/model_40000.pt}"
FILTERED_DIR="${FILTERED_DIR:-/tmp/dagger_sweep/filtered_motions}"
OUT_DIR="${OUT_DIR:-/tmp/dagger_sweep/paired_renders}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${OUT_DIR}/rollouts}"
# Auto-name a per-run subfolder under videos/ from the run dir + iteration so
# different student checkpoints don't overwrite each other's outputs. Override
# with VIDEO_DIR=... to put them somewhere else.
RUN_NAME="$(basename "$(dirname "$STUDENT_CKPT")")"
CKPT_NAME="$(basename "$STUDENT_CKPT" .pt)"
VIDEO_DIR="${VIDEO_DIR:-/home/kyungminlee/work/holosoma/videos/${RUN_NAME}_${CKPT_NAME}}"

mkdir -p "$ROLLOUT_DIR" "$VIDEO_DIR"

# --- Phase 1: paired rollouts ---
if [ "${SKIP_ROLLOUT:-0}" != "1" ]; then
    python scripts/eval/unreal_engine/rollout_paired_motions.py \
        --student_ckpt="$STUDENT_CKPT" \
        --motion_dir="$FILTERED_DIR" \
        --out_dir="$ROLLOUT_DIR"
fi

# --- Phase 1.5: trim/pad rollouts at first 0.5m motion deviation ---
VIZ_PY="${VIZ_PY:-/home/kyungminlee/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
"$VIZ_PY" scripts/eval/unreal_engine/trim_rollouts_at_failure.py \
    --rollout_dir "$ROLLOUT_DIR" \
    --motion_dir "$FILTERED_DIR" \
    --threshold 0.5

# --- Phase 2: relabel student rollouts based on fresh inference result ---
# Generates a JSON mapping motion_basename -> "ok"/"fail" using the trimmed
# student rollout (so the labels reflect what we just inferenced, not the
# training-time eval labels). Used to name the mp4 files below.
LABEL_JSON="${OUT_DIR}/fresh_labels.json"
"$VIZ_PY" scripts/eval/unreal_engine/relabel_from_rollout.py \
    --rollout_dir "$ROLLOUT_DIR" \
    --motion_dir "$FILTERED_DIR" \
    --out "$LABEL_JSON"

# --- Phase 3: render videos (student vs teacher only) ---
for motion_path in "$FILTERED_DIR"/*.npz; do
    base=$(basename "$motion_path" .npz)
    student_npy="$ROLLOUT_DIR/student_${base}.npy"
    teacher_npy="$ROLLOUT_DIR/teacher_${base}.npy"

    if [ ! -f "$student_npy" ] || [ ! -f "$teacher_npy" ]; then
        echo "Skipping $base (rollout missing)"
        continue
    fi

    # Pull fresh label ("ok" / "fail") from the JSON; default to "unknown" if
    # the relabel script couldn't decide.
    fresh_label=$("$VIZ_PY" -c "import json; print(json.load(open('$LABEL_JSON')).get('$base', 'unknown'))")
    out_mp4="$VIDEO_DIR/${fresh_label}_${base}.mp4"

    "$VIZ_PY" visualize.py \
        --input "$student_npy" \
        --ref_motion "$teacher_npy" \
        --terrain_npz "$motion_path" \
        --output "$out_mp4" \
        --ghost_alpha 0.45 \
        --fps 30
done

echo "Done. Videos in $VIDEO_DIR"
