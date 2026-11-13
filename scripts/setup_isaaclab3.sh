#!/usr/bin/env bash
set -eo pipefail

ISAACLAB3_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ISAACLAB3_ROOT_DIR=$(dirname "$ISAACLAB3_SCRIPT_DIR")
source "$ISAACLAB3_SCRIPT_DIR/source_common.sh"
source "$ISAACLAB3_SCRIPT_DIR/versions.sh"
export ISAACLAB3_DIR=${ISAACLAB3_DIR:-$WORKSPACE_DIR/IsaacLab3}
export ISAACLAB3_ENV_ROOT=${ISAACLAB3_ENV_ROOT:-$WORKSPACE_DIR/envs/hslab3}
ISAACLAB3_SENTINEL=$WORKSPACE_DIR/.env_setup_finished_hslab3_${ISAACLAB3_SETUP_ID}

isaaclab3_validate_runtime() {
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        if [[ ! -f $ISAACSIM_ROOT_PATH/VERSION || ! -f $ISAACSIM_ROOT_PATH/setup_python_env.sh ]]; then
            echo "Isaac Sim runtime not found at $ISAACSIM_ROOT_PATH." >&2
            return 1
        fi
        local version
        version=$(head -n 1 "$ISAACSIM_ROOT_PATH/VERSION")
        if [[ $version != "$ISAACLAB3_ISAACSIM_VERSION"* ]]; then
            echo "Expected Isaac Sim $ISAACLAB3_ISAACSIM_VERSION; found $version." >&2
            return 1
        fi
    fi
}

isaaclab3_prepare_isaaclab() {
    if [[ ! -d $ISAACLAB3_DIR ]]; then
        git clone --depth 1 --branch "$ISAACLAB3_REF" \
            https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB3_DIR"
    elif [[ $(git -C "$ISAACLAB3_DIR" rev-parse HEAD) != $(git -C "$ISAACLAB3_DIR" rev-parse "$ISAACLAB3_REF") ]]; then
        echo "Isaac Lab checkout for isaaclab3 is not at $ISAACLAB3_REF; use a separate ISAACLAB3_DIR." >&2
        return 1
    fi
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        ln -sfn "$ISAACSIM_ROOT_PATH" "$ISAACLAB3_DIR/_isaac_sim"
    fi
}

isaaclab3_setup_main() {
    isaaclab3_validate_runtime
    mkdir -p "$WORKSPACE_DIR"
    isaaclab3_prepare_isaaclab
    if [[ -f $ISAACLAB3_SENTINEL && -x $ISAACLAB3_ENV_ROOT/bin/python && -f $ISAACLAB3_ENV_ROOT/bin/activate ]]; then
        if [[ -n ${ISAACSIM_ROOT_PATH:-} || -f $ISAACLAB3_ENV_ROOT/isaacsim-runtime/isaacsim/__init__.py ]]; then
            return
        fi
    fi
    # A removed/relocated environment must complete setup before reusing its stamp.
    rm -f "$ISAACLAB3_SENTINEL"
    if ! command -v uv >/dev/null; then
        curl -LsSf "https://astral.sh/uv/${ISAACLAB3_UV_VERSION}/install.sh" | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Downloaded Isaac Sim must use its bundled interpreter, including for a
    # venv. Standalone pip installations use a regular Python 3.12 environment.
    local python_exe=3.12
    local extras=(--extra video --extra ov --extra importers)
    local runtime_requirements=$ISAACLAB3_ENV_ROOT/isaacsim-runtime-requirements.txt
    local runtime_packages=()
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        python_exe=$ISAACSIM_ROOT_PATH/kit/python/bin/python3
        export UV_PYTHON_PREFERENCE=only-system
    else
        extras+=(--extra isaacsim)
    fi
    uv venv --seed --allow-existing --python "$python_exe" "$ISAACLAB3_ENV_ROOT"
    export UV_PROJECT_ENVIRONMENT=$ISAACLAB3_ENV_ROOT
    export UV_PYTHON=$ISAACLAB3_ENV_ROOT/bin/python
    if [[ -z ${ISAACSIM_ROOT_PATH:-} ]]; then
        # Keep pip's regular isaacsim root outside the standalone namespace.
        # Resolve their shared dependencies together, but install the full
        # runtime wheels separately, with the versions/hashes from Lab's lock.
        "$UV_PYTHON" - "$ISAACLAB3_DIR/uv.lock" "$runtime_requirements" <<'PY'
import sys
import tomllib
from pathlib import Path

lock = tomllib.loads(Path(sys.argv[1]).read_text())
packages = [package for package in lock["package"]
            if package["name"].startswith("isaacsim") and package["name"] != "isaacsim-asset-isolated"]
Path(sys.argv[2]).write_text("".join(
    f'{package["name"]}=={package["version"]}'
    + "".join(f' --hash={wheel["hash"]}' for wheel in package["wheels"]) + "\n"
    for package in packages
))
PY
        while IFS= read -r requirement; do
            local package=${requirement%%==*}
            # The standalone importer also needs this codeless schema wheel.
            if [[ $package != isaacsim-robot-schema ]]; then
                extras+=(--no-install-package "$package")
                runtime_packages+=("$package")
            fi
        done < "$runtime_requirements"
        # Also repair environments created by this PR's earlier mixed layout.
        uv pip uninstall --python "$UV_PYTHON" "${runtime_packages[@]}"
    fi
    (
        cd "$ISAACLAB3_DIR"
        # Honor Lab's dependency overrides (constraints alone cannot reconcile
        # Isaac Sim's older exact torch/USD pins). Include video encoding,
        # detached OV runtimes, and the standalone URDF/MJCF importers.
        uv sync --frozen --inexact "${extras[@]}"
    )
    if [[ -z ${ISAACSIM_ROOT_PATH:-} ]]; then
        uv pip install --python "$UV_PYTHON" --no-deps --require-hashes \
            --index https://pypi.nvidia.com --target "$ISAACLAB3_ENV_ROOT/isaacsim-runtime" \
            --requirement "$runtime_requirements"
    fi
    source "$ISAACLAB3_SCRIPT_DIR/source_isaaclab3_setup.sh"
    if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
        # Match Lab's Docker setup for extension launch paths that load the
        # bundled torch before importing isaaclab.
        "$ISAACLAB3_DIR/isaaclab.sh" -p -c \
            'from isaaclab.cli.commands.install import _repoint_prebundle_packages; _repoint_prebundle_packages()'
    fi
    uv pip install --python "$ISAACLAB3_ENV_ROOT/bin/python" \
        -c "$ISAACLAB3_SCRIPT_DIR/isaaclab3_constraints.txt" \
        -e "$ISAACLAB3_ROOT_DIR/src/holosoma[unitree,booster]"
    touch "$ISAACLAB3_SENTINEL"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    isaaclab3_setup_main "$@"
fi
