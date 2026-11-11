#!/usr/bin/env bash
set -eo pipefail

ISAACSIM6_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ISAACSIM6_ROOT_DIR=$(dirname "$ISAACSIM6_SCRIPT_DIR")
source "$ISAACSIM6_SCRIPT_DIR/source_common.sh"
source "$ISAACSIM6_SCRIPT_DIR/versions.sh"
export ISAACSIM6_ISAACLAB_DIR=${ISAACSIM6_ISAACLAB_DIR:-$WORKSPACE_DIR/IsaacLab3}
export ISAACSIM6_ENV_ROOT=${ISAACSIM6_ENV_ROOT:-$WORKSPACE_DIR/envs/hssim6}
ISAACSIM6_SENTINEL=$WORKSPACE_DIR/.env_setup_finished_hssim6_${ISAACSIM6_SETUP_ID}

isaacsim6_validate_runtime() {
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        if [[ ! -f $ISAACSIM_ROOT_PATH/VERSION || ! -f $ISAACSIM_ROOT_PATH/setup_python_env.sh ]]; then
            echo "Isaac Sim runtime not found at $ISAACSIM_ROOT_PATH." >&2
            return 1
        fi
        local version
        version=$(head -n 1 "$ISAACSIM_ROOT_PATH/VERSION")
        if [[ $version != "$ISAACSIM6_VERSION"* ]]; then
            echo "Expected Isaac Sim $ISAACSIM6_VERSION; found $version." >&2
            return 1
        fi
    fi
}

isaacsim6_prepare_isaaclab() {
    if [[ ! -d $ISAACSIM6_ISAACLAB_DIR ]]; then
        git clone --depth 1 --branch "$ISAACSIM6_ISAACLAB_REF" \
            https://github.com/isaac-sim/IsaacLab.git "$ISAACSIM6_ISAACLAB_DIR"
    elif [[ $(git -C "$ISAACSIM6_ISAACLAB_DIR" rev-parse HEAD) != $(git -C "$ISAACSIM6_ISAACLAB_DIR" rev-parse "$ISAACSIM6_ISAACLAB_REF") ]]; then
        echo "Isaac Lab checkout for isaacsim6 is not at $ISAACSIM6_ISAACLAB_REF; use a separate ISAACSIM6_ISAACLAB_DIR." >&2
        return 1
    fi
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        ln -sfn "$ISAACSIM_ROOT_PATH" "$ISAACSIM6_ISAACLAB_DIR/_isaac_sim"
    fi
}

isaacsim6_setup_main() {
    isaacsim6_validate_runtime
    mkdir -p "$WORKSPACE_DIR"
    isaacsim6_prepare_isaaclab
    if [[ -f $ISAACSIM6_SENTINEL && -x $ISAACSIM6_ENV_ROOT/bin/python && -f $ISAACSIM6_ENV_ROOT/bin/activate ]]; then
        return
    fi
    # A removed/relocated environment must complete setup before reusing its stamp.
    rm -f "$ISAACSIM6_SENTINEL"
    if ! command -v uv >/dev/null; then
        curl -LsSf "https://astral.sh/uv/${ISAACSIM6_UV_VERSION}/install.sh" | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

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
    uv venv --seed --allow-existing --python "$python_exe" "$ISAACSIM6_ENV_ROOT"
    export UV_PROJECT_ENVIRONMENT=$ISAACSIM6_ENV_ROOT
    export UV_PYTHON=$ISAACSIM6_ENV_ROOT/bin/python
    (
        cd "$ISAACSIM6_ISAACLAB_DIR"
        # Honor Lab's dependency overrides (constraints alone cannot reconcile
        # Isaac Sim's older exact torch/USD pins). Include video encoding,
        # detached OV runtimes, and the standalone URDF/MJCF importers.
        uv sync --frozen --inexact "${extras[@]}"
    )
    source "$ISAACSIM6_SCRIPT_DIR/source_isaacsim6_setup.sh"
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        # Match Lab's Docker setup for extension launch paths that load the
        # bundled torch before importing isaaclab.
        "$ISAACSIM6_ISAACLAB_DIR/isaaclab.sh" -p -c \
            'from isaaclab.cli.commands.install import _repoint_prebundle_packages; _repoint_prebundle_packages()'
    fi
    uv pip install --python "$ISAACSIM6_ENV_ROOT/bin/python" \
        -c "$ISAACSIM6_SCRIPT_DIR/isaacsim6_constraints.txt" \
        -e "$ISAACSIM6_ROOT_DIR/src/holosoma[unitree,booster]"
    touch "$ISAACSIM6_SENTINEL"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    isaacsim6_setup_main "$@"
fi
