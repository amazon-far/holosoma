#!/usr/bin/env bash

# Script for running retargeting and data conversion for all LAFAN demo tasks
# Requires Ubuntu/Linux OS

set -e  # Exit on error

# figure out where this file is located even if it is being run from another location
# or as a symlink
SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect operating system and check if it's supported
OS="$(uname -s)"
case "${OS}" in
    Linux*)
        MACHINE=Linux
        echo "Detected Linux OS - proceeding..."
        ;;
    Darwin*)
        echo "Error: Mac OS is not supported. This script requires Ubuntu/Linux."
        exit 1
        ;;
    CYGWIN*|MINGW*)
        echo "Error: Windows is not supported. This script requires Ubuntu/Linux."
        exit 1
        ;;
    *)
        echo "Error: Unsupported operating system: ${OS}. This script requires Ubuntu/Linux."
        exit 1
        ;;
esac

# Source retargeting setup script
echo "Sourcing retargeting setup..."
source "$PROJECT_ROOT/scripts/source_retargeting_setup.sh"

# Ensure holosoma_retargeting is installed with correct dependencies (e.g. numpy version)
pip install -e "$PROJECT_ROOT/src/holosoma_retargeting" --quiet

# Change to retargeting directory
RETARGET_DIR="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting"
cd "$RETARGET_DIR"

# Step 0: Download and process LAFAN data if needed
echo "Checking LAFAN data availability..."
LAFAN_DATA_DIR="$RETARGET_DIR/demo_data/lafan"
LAFAN_TEMP_DIR="$RETARGET_DIR/demo_data/lafan_temp"
LAFAN_ZIP="$RETARGET_DIR/demo_data/lafan1.zip"
DATA_UTILS_DIR="$RETARGET_DIR/data_utils"

# Check if processed LAFAN data already exists
if [ -d "$LAFAN_DATA_DIR" ] && [ "$(ls -A $LAFAN_DATA_DIR/*.npy 2>/dev/null)" ]; then
    echo "LAFAN data already processed. Skipping download and processing."
else
    echo "LAFAN data not found. Downloading and processing..."

    # Create demo_data directory if it doesn't exist
    mkdir -p "$RETARGET_DIR/demo_data"

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
        cd "$RETARGET_DIR"
    else
        echo "lafan1 processing code already available."
    fi

    # Convert BVH files to .npy format
    echo "Converting BVH files to .npy format..."
    cd "$DATA_UTILS_DIR"
    python extract_global_positions.py --input_dir "$LAFAN_TEMP_DIR" --output_dir "$LAFAN_DATA_DIR"
    cd "$RETARGET_DIR"

    echo "LAFAN data processing complete!"
fi

# Step 1: Run retargeting and data conversion for every processed LAFAN task
echo "Running retargeting for all LAFAN tasks..."
RETARGET_SAVE_DIR="demo_results/g1/robot_only/lafan"
CONVERTED_DIR="converted_res/robot_only"
mkdir -p "$RETARGET_SAVE_DIR" "$CONVERTED_DIR"

shopt -s nullglob
TASK_FILES=("$LAFAN_DATA_DIR"/*.npy)
shopt -u nullglob

if [ "${#TASK_FILES[@]}" -eq 0 ]; then
    echo "Error: No processed LAFAN .npy files found in $LAFAN_DATA_DIR"
    exit 1
fi

for TASK_FILE in "${TASK_FILES[@]}"; do
    TASK_NAME="$(basename "$TASK_FILE" .npy)"
    RETARGETED_FILE="./$RETARGET_SAVE_DIR/$TASK_NAME.npz"
    CONVERTED_FILE="$CONVERTED_DIR/${TASK_NAME}_mj_fps50.npz"

    echo "Retargeting task: $TASK_NAME"
    python examples/robot_retarget.py \
        --data_path "$LAFAN_DATA_DIR" \
        --task-type robot_only \
        --task-name "$TASK_NAME" \
        --data_format lafan \
        --task-config.ground-range -10 10 \
        --save_dir "$RETARGET_SAVE_DIR" \
        --retargeter.foot-sticking-tolerance 0.02

    echo "Converting task: $TASK_NAME"
    python data_conversion/convert_data_format_mj.py \
        --input_file "$RETARGETED_FILE" \
        --output_fps 50 \
        --output_name "$CONVERTED_FILE" \
        --data_format lafan \
        --object_name "ground" \
        --no-visualize \
        --once
done

echo "Done!"
