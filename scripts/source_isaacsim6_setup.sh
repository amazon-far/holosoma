# Source this file to activate the hssim6 environment (Isaac Sim 6.1 + Isaac Lab 3.0, Python 3.12).
#
# This lane is a uv venv, not a conda environment -- Isaac Lab 3.0 rejects conda (see
# scripts/setup_isaacsim6.sh). Use scripts/source_isaacsim_setup.sh for the 5.1 lane.

# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

# Hardcoded, not ${VAR:-default}: a CONDA_ENV_NAME or VENV_NAME exported for another lane must
# not be able to redirect this one into the wrong environment.
VENV_NAME=hssim6

# Separate workspace from the 5.1 lane. Set before sourcing source_common.sh so WORKSPACE_DIR
# and CONDA_ROOT derive from it.
export HOLOSOMA_DEPS_DIR=$HOME/.holosoma_deps_sim6
source "${SCRIPT_DIR}/source_common.sh"

VENV_ROOT=$HOLOSOMA_DEPS_DIR/$VENV_NAME
if [ ! -f "$VENV_ROOT/bin/activate" ]; then
    echo "hssim6 not found at $VENV_ROOT. Run: bash scripts/setup_isaacsim6.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# Isaac Lab 3.0 refuses to combine a downloaded Isaac Sim with conda, so make sure no conda
# environment is left active from a previous lane in this shell.
if [ -n "${CONDA_PREFIX}" ]; then
    echo "A conda environment is active ($CONDA_PREFIX). Deactivate it before using hssim6:" >&2
    echo "  conda deactivate" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$VENV_ROOT/bin/activate"
export OMNI_KIT_ACCEPT_EULA=1

# Keep bytecode out of the shared holosoma source tree: both Isaac Sim lanes install the same
# worktree editable, and under Docker the tree is bind-mounted, so concurrent lanes would
# otherwise write __pycache__ into it side by side.
export PYTHONPYCACHEPREFIX=$HOLOSOMA_DEPS_DIR/pycache

echo "hssim6 active (Isaac Sim ${ISAACSIM6_VERSION:-6.1.0} / Isaac Lab 3.0, $(python --version 2>&1))"
