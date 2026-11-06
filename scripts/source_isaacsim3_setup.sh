#!/usr/bin/env bash
# Lab 3 uses a venv built on Kit's Python in images, rather than a Conda interpreter.
if [ -n "${BASH_SOURCE[0]}" ]; then
    ISAACSIM3_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    ISAACSIM3_SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

source "$ISAACSIM3_SCRIPT_DIR/source_common.sh"
export ISAACSIM3_ENV_ROOT=${ISAACSIM3_ENV_ROOT:-$WORKSPACE_DIR/envs/hssim3}
export ISAACLAB3_DIR=${ISAACLAB3_DIR:-$WORKSPACE_DIR/IsaacLab3}
source "$ISAACSIM3_ENV_ROOT/bin/activate"
export OMNI_KIT_ACCEPT_EULA=1

if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
    export ISAAC_PATH=$ISAACSIM_ROOT_PATH
    export CARB_APP_PATH=$ISAACSIM_ROOT_PATH/kit
    export EXP_PATH=$ISAACSIM_ROOT_PATH/apps
    source "$ISAACSIM_ROOT_PATH/setup_python_env.sh"
    # Kit's prebundle paths contain older torch/typing_extensions. Match the
    # upstream isaaclab.sh activation order so the locked venv packages win.
    ISAACSIM3_SITE_PACKAGES=$("$ISAACSIM3_ENV_ROOT/bin/python" -c 'import site; print(site.getsitepackages()[0])')
    export PYTHONPATH="$ISAACSIM3_SITE_PACKAGES:${PYTHONPATH:-}"
fi
