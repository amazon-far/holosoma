#!/bin/bash
# Activation script for the uv-based MuJoCo environment
# Usage: source scripts/source_mujoco_uv_setup.sh

# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

WORKSPACE_DIR=$HOME/.holosoma_deps
VENV_DIR=$WORKSPACE_DIR/venvs/hsmujoco

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: uv MuJoCo environment not found at $VENV_DIR"
    echo "Run 'bash scripts/setup_mujoco_via_uv.sh' first."
    return 1 2>/dev/null || exit 1
fi

source "$VENV_DIR/bin/activate"

# Set MuJoCo-specific environment variables
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$VENV_DIR/lib

# Validate environment
if python -c "import mujoco" 2>/dev/null; then
    echo "MuJoCo uv environment activated successfully"
    echo "MuJoCo version: $(python -c 'import mujoco; print(mujoco.__version__)')"

    if python -c "import torch" 2>/dev/null; then
        echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    fi

    if python -c "import mujoco_warp" 2>/dev/null; then
        MUJOCO_WARP_COMMIT=$(git -C ${WORKSPACE_DIR}/mujoco_warp rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "MuJoCo Warp commit: ${MUJOCO_WARP_COMMIT}"
    fi
else
    echo "Warning: MuJoCo environment activation may have issues"
    echo "Try running 'bash scripts/setup_mujoco_via_uv.sh' to reinstall"
fi
