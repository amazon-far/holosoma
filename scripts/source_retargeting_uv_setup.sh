#!/usr/bin/env bash
# Activation script for the uv-based retargeting environment
# Usage: source scripts/source_retargeting_uv_setup.sh

# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

ROOT_DIR=$(dirname "$SCRIPT_DIR")
VENV_DIR=${RETARGETING_UV_VENV_DIR:-$ROOT_DIR/.venv/hsretargeting}

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: uv retargeting environment not found at $VENV_DIR"
    echo "Run 'bash scripts/setup_retargeting_via_uv.sh' first."
    return 1 2>/dev/null || exit 1
fi

source "$VENV_DIR/bin/activate"

# Ensure venv bin dir is first in PATH (version managers like mise can override it)
case ":$PATH:" in
    *":$VENV_DIR/bin:"*) ;;
    *) export PATH="$VENV_DIR/bin:$PATH" ;;
esac

if python -c "import holosoma_retargeting, mujoco, numpy" 2>/dev/null; then
    echo "Retargeting uv environment activated successfully"
    echo "Python version: $(python -c 'import sys; print(sys.version.split()[0])')"
    echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"
    echo "MuJoCo version: $(python -c 'import mujoco; print(mujoco.__version__)')"
else
    echo "Warning: Retargeting environment activation may have issues"
    echo "Try running 'bash scripts/setup_retargeting_via_uv.sh --reinstall' to reinstall"
fi
