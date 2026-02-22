#!/bin/bash

# Script for running retargeting, data conversion, and whole-body tracking training
# Requires Ubuntu/Linux OS (IsaacSim is not supported on Mac)

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Detect operating system and check if it's supported
OS="$(uname -s)"
case "${OS}" in
    Linux*)
        MACHINE=Linux
        echo "Detected Linux OS - proceeding..."
        ;;
    Darwin*)
        echo "Error: Mac OS is not supported. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
    CYGWIN*|MINGW*)
        echo "Error: Windows is not supported. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
    *)
        echo "Error: Unsupported operating system: ${OS}. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
esac

# Source retargeting setup script (for retargeting and data conversion)
echo "Sourcing retargeting setup..."
source "$PROJECT_ROOT/scripts/source_retargeting_setup.sh"

# Change to retargeting directory
cd "$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/"

# Step 0: Download and process LAFAN data if needed
echo "Checking LAFAN data availability..."
LAFAN_DATA_DIR="demo_data/lafan"
LAFAN_TEMP_DIR="demo_data/lafan_temp"
LAFAN_ZIP="demo_data/lafan1.zip"
DATA_UTILS_DIR="data_utils"

# Check if processed LAFAN data already exists
if [ -d "$LAFAN_DATA_DIR" ] && [ "$(ls -A $LAFAN_DATA_DIR/*.npy 2>/dev/null)" ]; then
    echo "LAFAN data already processed. Skipping download and processing."
else
    echo "LAFAN data not found. Downloading and processing..."

    # Create demo_data directory if it doesn't exist
    mkdir -p demo_data

    # Download lafan1.zip if it doesn't exist
    if [ ! -f "$LAFAN_ZIP" ]; then
        echo "Downloading lafan1.zip..."
        curl -L -o "$LAFAN_ZIP" "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip"
    else
        echo "lafan1.zip already exists. Skipping download."
    fi

    # Uncompress lafan1.zip to temp directory
    if [ ! -d "$LAFAN_TEMP_DIR" ] || [ -z "$(ls -A $LAFAN_TEMP_DIR/*.bvh 2>/dev/null)" ]; then
        echo "Uncompressing lafan1.zip..."
        mkdir -p "$LAFAN_TEMP_DIR"
        unzip -q -o "$LAFAN_ZIP" -d "$LAFAN_TEMP_DIR"
        # Handle different zip structures - move BVH files to top level
        if [ -d "$LAFAN_TEMP_DIR/lafan1/lafan" ]; then
            # Structure: lafan1/lafan/*.bvh
            mv "$LAFAN_TEMP_DIR/lafan1/lafan"/* "$LAFAN_TEMP_DIR/" 2>/dev/null || true
            rm -rf "$LAFAN_TEMP_DIR/lafan1" 2>/dev/null || true
        elif [ -d "$LAFAN_TEMP_DIR/lafan1" ]; then
            # Structure: lafan1/*.bvh
            mv "$LAFAN_TEMP_DIR/lafan1"/* "$LAFAN_TEMP_DIR/" 2>/dev/null || true
            rmdir "$LAFAN_TEMP_DIR/lafan1" 2>/dev/null || true
        fi
    else
        echo "LAFAN BVH files already extracted. Skipping extraction."
    fi

    # Ensure lafan1 processing code is available in data_utils
    if [ ! -d "$DATA_UTILS_DIR/lafan1" ]; then
        echo "Cloning ubisoft-laforge-animation-dataset for processing code..."
        cd "$DATA_UTILS_DIR"
        if [ ! -d "ubisoft-laforge-animation-dataset" ]; then
            git clone -q https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
        fi
        if [ -d "ubisoft-laforge-animation-dataset/lafan1" ] && [ ! -d "lafan1" ]; then
            mv ubisoft-laforge-animation-dataset/lafan1 .
        fi
        cd ..
    else
        echo "lafan1 processing code already available."
    fi

    # Convert BVH files to .npy format
    echo "Converting BVH files to .npy format..."
    cd "$DATA_UTILS_DIR"
    python extract_global_positions.py --input_dir "../$LAFAN_TEMP_DIR" --output_dir "../$LAFAN_DATA_DIR"
    cd ..

    echo "LAFAN data processing complete!"
fi

# Step 1: Run retargeting
echo "Running retargeting..."
python examples/robot_retarget.py --data_path demo_data/lafan --task-type robot_only --task-name dance2_subject1 --data_format lafan --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/lafan --retargeter.foot-sticking-tolerance 0.02

# Step 2: Run data conversion
echo "Running data conversion..."
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz --output_fps 50 --output_name converted_res/robot_only/dance2_subject1_mj_fps50.npz --data_format lafan --object_name "ground" --once

# Step 3: Source IsaacSim setup script (for whole-body tracking training)
echo "Sourcing IsaacSim setup..."
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

# Step 4: Run whole-body tracking training
echo "Running whole-body tracking training..."
CONVERTED_FILE="$PROJECT_ROOT/src/holosoma_retargeting/converted_res/robot_only/dance2_subject1_mj_fps50.npz"
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE

echo "Done!"
