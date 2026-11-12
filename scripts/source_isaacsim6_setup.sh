#!/usr/bin/env bash
# Lab 3 uses a venv built on Kit's Python in images, rather than a Conda interpreter.
if [ -n "${BASH_SOURCE[0]}" ]; then
    ISAACSIM6_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    ISAACSIM6_SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

source "$ISAACSIM6_SCRIPT_DIR/source_common.sh"
export ISAACSIM6_ENV_ROOT=${ISAACSIM6_ENV_ROOT:-$WORKSPACE_DIR/envs/hssim6}
export ISAACSIM6_ISAACLAB_DIR=${ISAACSIM6_ISAACLAB_DIR:-$WORKSPACE_DIR/IsaacLab3}
source "$ISAACSIM6_ENV_ROOT/bin/activate"
export OMNI_KIT_ACCEPT_EULA=1

if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
    ISAACSIM6_ORIGINAL_PYTHONPATH=${PYTHONPATH:-}
    export ISAAC_PATH=$ISAACSIM_ROOT_PATH
    export CARB_APP_PATH=$ISAACSIM_ROOT_PATH/kit
    export EXP_PATH=$ISAACSIM_ROOT_PATH/apps
    # Native library lookup must be prepared before Python starts. Merely
    # exposing these directories does not import Kit or a physics engine.
    source "$ISAACSIM_ROOT_PATH/setup_python_env.sh"
    ISAACSIM6_SITE_PACKAGES=$("$ISAACSIM6_ENV_ROOT/bin/python" -c 'import site; print(site.getsitepackages()[0])')
    # Defer Kit's Python packages until the configured backend requires Kit.
    # Keep the locked venv ahead of Kit's older prebundled dependencies.
    export ISAACSIM6_KIT_PYTHONPATH="$ISAACSIM6_SITE_PACKAGES:${PYTHONPATH:-}"
    export PYTHONPATH=$("$ISAACSIM6_ENV_ROOT/bin/python" - "$ISAACSIM_ROOT_PATH" "$ISAACSIM6_ORIGINAL_PYTHONPATH" <<'PY'
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
    unset ISAACSIM6_ORIGINAL_PYTHONPATH
else
    # Pip's full runtime lives outside the standalone namespace directory.
    export ISAACSIM6_KIT_PYTHONPATH="$ISAACSIM6_ENV_ROOT/isaacsim-runtime"
fi
