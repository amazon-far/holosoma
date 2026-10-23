#!/usr/bin/env bash
# Remove the hssim6 environment so one can re-run scripts/setup_isaacsim6.sh.
#
# Scoped to the 6.1 lane's own workspace. The 5.1 lane lives under a different
# HOLOSOMA_DEPS_DIR, so this cannot delete its IsaacLab checkout -- but the path is printed
# and confirmed explicitly, because an operator who exported HOLOSOMA_DEPS_DIR for the other
# lane would otherwise wipe it without warning.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# MUST precede source_common.sh, which defaults HOLOSOMA_DEPS_DIR to the 5.1 lane's
# ~/.holosoma_deps and exports it. Setting it afterwards is a silent no-op -- and in a script
# that runs `rm -rf`, that would delete the 5.1 lane's IsaacLab checkout.
export HOLOSOMA_DEPS_DIR=${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps_sim6}

source "${SCRIPT_DIR}/source_common.sh"
source "${SCRIPT_DIR}/versions.sh"

WORKSPACE_DIR=$HOLOSOMA_DEPS_DIR
VENV_NAME=${VENV_NAME:-hssim6}
VENV_ROOT=$WORKSPACE_DIR/$VENV_NAME
ISAACLAB_DIR=$WORKSPACE_DIR/IsaacLab
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_${VENV_NAME}_${ISAACSIM6_SETUP_ID}

echo "This will remove:"
echo "  venv     : $VENV_ROOT"
echo "  IsaacLab : $ISAACLAB_DIR"
echo "  sentinel : $SENTINEL_FILE"
echo
read -p "Reset the hssim6 environment at the paths above? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

set -ex

rm -rf "$VENV_ROOT"
rm -rf "$ISAACLAB_DIR"
rm -f "$SENTINEL_FILE"
