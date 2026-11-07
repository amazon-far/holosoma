#!/usr/bin/env bash
set -eo pipefail

ISAACSIM3_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ISAACSIM3_ROOT_DIR=$(dirname "$ISAACSIM3_SCRIPT_DIR")
source "$ISAACSIM3_SCRIPT_DIR/source_common.sh"
source "$ISAACSIM3_SCRIPT_DIR/versions.sh"
export ISAACLAB3_DIR=${ISAACLAB3_DIR:-$WORKSPACE_DIR/IsaacLab3}
export ISAACSIM3_ENV_ROOT=${ISAACSIM3_ENV_ROOT:-$WORKSPACE_DIR/envs/hssim3}
ISAACSIM3_SENTINEL=$WORKSPACE_DIR/.env_setup_finished_hssim3_${ISAACSIM3_SETUP_ID}

isaacsim3_validate_runtime() {
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        if [[ ! -f $ISAACSIM_ROOT_PATH/VERSION || ! -f $ISAACSIM_ROOT_PATH/setup_python_env.sh ]]; then
            echo "Isaac Sim runtime not found at $ISAACSIM_ROOT_PATH." >&2
            return 1
        fi
        local version
        version=$(head -n 1 "$ISAACSIM_ROOT_PATH/VERSION")
        if [[ $version != "$ISAACSIM3_VERSION"* ]]; then
            echo "Expected Isaac Sim $ISAACSIM3_VERSION; found $version." >&2
            return 1
        fi
    fi
}

isaacsim3_prepare_isaaclab() {
    if [[ ! -d $ISAACLAB3_DIR ]]; then
        git clone --depth 1 --branch "$ISAACLAB3_REF" \
            https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB3_DIR"
    elif [[ $(git -C "$ISAACLAB3_DIR" rev-parse HEAD) != $(git -C "$ISAACLAB3_DIR" rev-parse "$ISAACLAB3_REF") ]]; then
        echo "IsaacLab3 is not at $ISAACLAB3_REF; use a separate ISAACLAB3_DIR." >&2
        return 1
    fi
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        ln -sfn "$ISAACSIM_ROOT_PATH" "$ISAACLAB3_DIR/_isaac_sim"
    fi
}

isaacsim3_setup_main() {
    isaacsim3_validate_runtime
    if [[ -f $ISAACSIM3_SENTINEL ]]; then
        return
    fi
    mkdir -p "$WORKSPACE_DIR"
    if ! command -v uv >/dev/null; then
        curl -LsSf "https://astral.sh/uv/${ISAACSIM3_UV_VERSION}/install.sh" | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    isaacsim3_prepare_isaaclab

    # Downloaded Isaac Sim must use its bundled interpreter, including for a
    # venv. Standalone pip installations use a regular Python 3.12 environment.
    local python_exe=3.12
    local extras=(--extra video --extra ov --extra importers)
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        python_exe=$ISAACSIM_ROOT_PATH/kit/python/bin/python3
        export UV_PYTHON_PREFERENCE=only-system
    else
        extras+=(--extra isaacsim)
    fi
    uv venv --seed --allow-existing --python "$python_exe" "$ISAACSIM3_ENV_ROOT"
    export UV_PROJECT_ENVIRONMENT=$ISAACSIM3_ENV_ROOT
    export UV_PYTHON=$ISAACSIM3_ENV_ROOT/bin/python
    (
        cd "$ISAACLAB3_DIR"
        # Honor Lab's dependency overrides (constraints alone cannot reconcile
        # Isaac Sim's older exact torch/USD pins). Include video encoding,
        # detached OV runtimes, and the standalone URDF/MJCF importers.
        uv sync --frozen --inexact "${extras[@]}"
    )
    source "$ISAACSIM3_SCRIPT_DIR/source_isaacsim3_setup.sh"
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        # Match Lab's Docker setup for extension launch paths that load the
        # bundled torch before importing isaaclab.
        "$ISAACLAB3_DIR/isaaclab.sh" -p -c \
            'from isaaclab.cli.commands.install import _repoint_prebundle_packages; _repoint_prebundle_packages()'
    fi
    uv pip install --python "$ISAACSIM3_ENV_ROOT/bin/python" \
        -c "$ISAACSIM3_SCRIPT_DIR/isaacsim3_constraints.txt" \
        -e "$ISAACSIM3_ROOT_DIR/src/holosoma[unitree,booster]"
    touch "$ISAACSIM3_SENTINEL"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    isaacsim3_setup_main "$@"
fi
