# Remove the isaacsim environment so one can re-run the setup script
# Exit on error, and print commands

# Ask for confirmation
read -p "Are you sure you want to reset the isaacsim environment? This will remove the environment and IsaacLab. [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/hssim
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_isaacsim

rm -rf $ENV_ROOT
rm -rf $WORKSPACE_DIR/IsaacLab
rm -f $SENTINEL_FILE
