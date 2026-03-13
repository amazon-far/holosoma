#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

if ! command -v sudo &> /dev/null; then
  # in docker build sudo isn't avaiable, but its ok
  echo "Warning: sudo could not be found, you may need to run this script with sudo"
  function sudo { "$@"; }
  export -f sudo
fi
# Set up conda environment
source $CONDA_ROOT/etc/profile.d/conda.sh


cd "$SCRIPT_DIR"
chmod +x setup_isaacsim.sh setup_isaacgym.sh setup_mujoco.sh setup_inference.sh setup_retargeting.sh
OMNI_KIT_ACCEPT_EULA=1 ./setup_isaacsim.sh
./setup_isaacgym.sh
./setup_mujoco.sh --no-warp
./setup_inference.sh
./setup_retargeting.sh
