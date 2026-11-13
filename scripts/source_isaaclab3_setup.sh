#!/usr/bin/env bash
# Lab 3 uses a venv built on Kit's Python in images, rather than a Conda interpreter.
if [ -n "${BASH_SOURCE[0]}" ]; then
    ISAACLAB3_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    ISAACLAB3_SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

source "$ISAACLAB3_SCRIPT_DIR/source_common.sh"
export ISAACLAB3_ENV_ROOT=${ISAACLAB3_ENV_ROOT:-$WORKSPACE_DIR/envs/hslab3}
export ISAACLAB3_DIR=${ISAACLAB3_DIR:-$WORKSPACE_DIR/IsaacLab3}
source "$ISAACLAB3_ENV_ROOT/bin/activate"
export OMNI_KIT_ACCEPT_EULA=1

if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
    ISAACLAB3_ORIGINAL_PYTHONPATH=${PYTHONPATH:-}
    export ISAAC_PATH=$ISAACSIM_ROOT_PATH
    export CARB_APP_PATH=$ISAACSIM_ROOT_PATH/kit
    export EXP_PATH=$ISAACSIM_ROOT_PATH/apps
    # Native library lookup must be prepared before Python starts. Merely
    # exposing these directories does not import Kit or a physics engine.
    source "$ISAACSIM_ROOT_PATH/setup_python_env.sh"
    ISAACLAB3_SITE_PACKAGES=$("$ISAACLAB3_ENV_ROOT/bin/python" -c 'import site; print(site.getsitepackages()[0])')
    # Defer Kit's Python packages until the configured backend requires Kit.
    # Keep the locked venv ahead of Kit's older prebundled dependencies.
    export ISAACLAB3_KIT_PYTHONPATH="$ISAACLAB3_SITE_PACKAGES:${PYTHONPATH:-}"
    export PYTHONPATH=$("$ISAACLAB3_ENV_ROOT/bin/python" - "$ISAACSIM_ROOT_PATH" "$ISAACLAB3_ORIGINAL_PYTHONPATH" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
runtime_paths = [root / path for path in ("python_packages", "kit/kernel", "kit/plugins", "exts", "extscache")]
print(os.pathsep.join(
    path for path in sys.argv[2].split(os.pathsep)
    if path and not any(Path(path).resolve().is_relative_to(prefix) for prefix in runtime_paths)
))
PY
)
    unset ISAACLAB3_ORIGINAL_PYTHONPATH
else
    # Pip's full runtime lives outside the standalone namespace directory.
    export ISAACLAB3_KIT_PYTHONPATH="$ISAACLAB3_ENV_ROOT/isaacsim-runtime"
fi
