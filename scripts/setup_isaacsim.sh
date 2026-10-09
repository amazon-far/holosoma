#!/usr/bin/env bash
# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/setup_isaacsim_delta.sh"

if [[ -n ${ISAACSIM_ROOT_PATH:-} ]]; then
  isaacsim_delta_main "$@"
  exit
fi

# The fresh-environment path installs Isaac Sim, then invokes the same delta
# used by images that already contain the simulator.
echo "conda environment name is set to: $CONDA_ENV_NAME"
echo "SENTINEL_FILE: $ISAACSIM_SENTINEL_FILE"

if isaacsim_setup_complete; then
  exit 0
fi

isaacsim_ensure_miniconda
isaacsim_ensure_standalone_conda_environment
isaacsim_install_environment_dependencies
isaacsim_activate_environment

# Follow Isaac Lab's supported pip installation for a standalone environment.
python -m pip install --upgrade pip
python -m pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install pyperclip
python -m pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
  --extra-index-url https://pypi.nvidia.com

isaacsim_prepare_isaaclab
isaacsim_install_delta
