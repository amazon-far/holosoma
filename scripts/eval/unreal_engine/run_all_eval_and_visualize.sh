#!/bin/bash
# Run inference (eval) for all 6 UE scene experiments, then visualize with terrain.
#
# Usage:
#   bash scripts/eval/unreal_engine/run_all_eval_and_visualize.sh
#
# This script:
#   1. Runs eval_agent.py for each experiment (saves .npy motion data)
#   2. Runs visualize.py with terrain mesh for each experiment (saves .mp4)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

LOGS_DIR="$PROJECT_ROOT/logs/MotionTracking"
VIDEOS_DIR="$PROJECT_ROOT/videos/unreal_engine_eval"
MOTION_DATA_DIR="$PROJECT_ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/unreal_engine"
SCENE_XML="$PROJECT_ROOT/src/holosoma/holosoma/data/robots/g1/g1_29dof.xml"

mkdir -p "$VIDEOS_DIR"

# Define experiments: name, checkpoint, terrain_npz
declare -A EXPERIMENTS
declare -A CHECKPOINTS
declare -A TERRAIN_NPZ

EXPERIMENTS[huddle]="20260315_g1_huddle"
CHECKPOINTS[huddle]="$LOGS_DIR/20260315_g1_huddle/model_04000.pt"
TERRAIN_NPZ[huddle]="$MOTION_DATA_DIR/huddle_walk.npz"

EXPERIMENTS[stair_30depth]="20260315_g1_stair_30depth"
CHECKPOINTS[stair_30depth]="$LOGS_DIR/20260315_g1_stair_30depth/model_04000.pt"
TERRAIN_NPZ[stair_30depth]="$MOTION_DATA_DIR/stair_walk.npz"

EXPERIMENTS[stair_60depth]="20260315_g1_stair_60depth"
CHECKPOINTS[stair_60depth]="$LOGS_DIR/20260315_g1_stair_60depth/model_04000.pt"
TERRAIN_NPZ[stair_60depth]="$MOTION_DATA_DIR/stair_walk_60depth.npz"

EXPERIMENTS[huddle_heightmap]="20260316_g1_huddle_heightmap"
CHECKPOINTS[huddle_heightmap]="$LOGS_DIR/20260316_g1_huddle_heightmap/model_03000.pt"
TERRAIN_NPZ[huddle_heightmap]="$MOTION_DATA_DIR/huddle_walk.npz"

EXPERIMENTS[stair_60depth_heightmap]="20260316_g1_stair_60depth_heightmap"
CHECKPOINTS[stair_60depth_heightmap]="$LOGS_DIR/20260316_g1_stair_60depth_heightmap/model_03000.pt"
TERRAIN_NPZ[stair_60depth_heightmap]="$MOTION_DATA_DIR/stair_walk_60depth.npz"

EXPERIMENTS[stair_with_motion]="20260316_g1_stair_with_motion"
CHECKPOINTS[stair_with_motion]="$LOGS_DIR/20260316_g1_stair_with_motion/model_04000.pt"
TERRAIN_NPZ[stair_with_motion]="$MOTION_DATA_DIR/stair_walk_with_motion.npz"

ORDER="huddle stair_30depth stair_60depth huddle_heightmap stair_60depth_heightmap stair_with_motion"

# ============================================================
# Step 1: Run eval_agent.py for each experiment
# ============================================================
echo "====================================="
echo "Step 1: Running inference (eval) for all experiments"
echo "====================================="

for name in $ORDER; do
    log_dir="${EXPERIMENTS[$name]}"
    checkpoint="${CHECKPOINTS[$name]}"

    echo ""
    echo "--- Evaluating: $name (checkpoint: $checkpoint) ---"

    # Check if motion data already exists
    motion_data_dir="$(dirname "$checkpoint")/motion_data"
    if [ -f "$motion_data_dir/ref_motion.npy" ]; then
        echo "  Motion data already exists at $motion_data_dir/ref_motion.npy, skipping eval."
        continue
    fi

    bash "$SCRIPT_DIR/eval_${name}.sh"
    echo "  Done evaluating $name"
done

# ============================================================
# Step 2: Visualize with terrain mesh
# ============================================================
echo ""
echo "====================================="
echo "Step 2: Visualizing inference motion with terrain"
echo "====================================="

for name in $ORDER; do
    checkpoint="${CHECKPOINTS[$name]}"
    terrain_npz="${TERRAIN_NPZ[$name]}"

    motion_data_dir="$(dirname "$checkpoint")/motion_data"
    input_npy="$motion_data_dir/ref_motion.npy"
    output_mp4="$VIDEOS_DIR/${name}_eval.mp4"

    if [ ! -f "$input_npy" ]; then
        echo "  WARNING: Motion data not found at $input_npy, skipping visualization for $name"
        continue
    fi

    echo ""
    echo "--- Visualizing: $name ---"
    echo "  Input: $input_npy"
    echo "  Terrain: $terrain_npz"
    echo "  Output: $output_mp4"

    python visualize.py \
        --input "$input_npy" \
        --output "$output_mp4" \
        --scene "$SCENE_XML" \
        --terrain_npz "$terrain_npz" \
        --input_fps 50

    echo "  Done: $output_mp4"
done

echo ""
echo "====================================="
echo "All done! Videos saved in $VIDEOS_DIR/"
echo "====================================="
ls -la "$VIDEOS_DIR/"
