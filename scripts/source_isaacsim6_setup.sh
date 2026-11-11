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
    export ISAAC_PATH=$ISAACSIM_ROOT_PATH
    export CARB_APP_PATH=$ISAACSIM_ROOT_PATH/kit
    export EXP_PATH=$ISAACSIM_ROOT_PATH/apps
    source "$ISAACSIM_ROOT_PATH/setup_python_env.sh"
    # Kit's prebundle paths contain older torch/typing_extensions. Match the
    # upstream isaaclab.sh activation order so the locked venv packages win.
    ISAACSIM6_SITE_PACKAGES=$("$ISAACSIM6_ENV_ROOT/bin/python" -c 'import site; print(site.getsitepackages()[0])')
    export PYTHONPATH="$ISAACSIM6_SITE_PACKAGES:${PYTHONPATH:-}"
fi
