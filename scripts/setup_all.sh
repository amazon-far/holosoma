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

# Each setup script defaults its own environment name, but none of them scopes CONDA_ENV_NAME,
# so a value exported by the caller would redirect every environment below into that one name.
unset CONDA_ENV_NAME

# setup_isaacsim6.sh is intentionally excluded: it builds a
# uv venv rather than a conda env, and the combined image cannot host a second Isaac Sim runtime
# (one ISAACSIM_ROOT_PATH, ~10-20 GB per runtime). Run it directly for the 6.1 lane.
chmod +x setup_isaacsim.sh setup_mujoco.sh setup_inference.sh setup_retargeting.sh
OMNI_KIT_ACCEPT_EULA=1 ./setup_isaacsim.sh
./setup_mujoco.sh --skip-driver-check  # build hosts have no GPU; mjwarp needs one only at run time
./setup_inference.sh
./setup_retargeting.sh
