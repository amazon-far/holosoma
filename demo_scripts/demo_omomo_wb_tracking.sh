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

# Step 1: Run retargeting
echo "Running retargeting..."
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh

# Step 2: Run data conversion
echo "Running data conversion..."
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

# Step 3: Source IsaacSim setup script (for whole-body tracking training)
echo "Sourcing IsaacSim setup..."
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

# Step 4: Run whole-body tracking training
echo "Running whole-body tracking training..."
CONVERTED_FILE="$PROJECT_ROOT/src/holosoma_retargeting/converted_res/robot_only/sub3_largebox_003_mj_fps50.npz"
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE

echo "Done!"
