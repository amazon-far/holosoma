#!/bin/bash
# Source this file to activate the hsmujoco_py312 conda env (Python 3.12 + ROS2 Jazzy).
# Usage:  source holosoma/scripts/source_mujoco_py312_setup.sh

# Resolve the directory of this script, following symlinks (works in bash and zsh).
_resolve_script_dir() {
  local source="${BASH_SOURCE[0]:-${(%):-%x}}"
  while [ -h "$source" ]; do
    local dir="$( cd -P "$( dirname "$source" )" >/dev/null && pwd )"
    source="$(readlink "$source")"
    [[ $source != /* ]] && source="$dir/$source"
  done
  cd -P "$( dirname "$source" )" >/dev/null && pwd
}
SCRIPT_DIR="$(_resolve_script_dir)"
unset -f _resolve_script_dir

CONDA_ENV_NAME=hsmujoco_py312

source ${SCRIPT_DIR}/source_common.sh

# Source ROS2 Jazzy first (provides rclpy for Python 3.12)
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    echo "ROS2 Jazzy sourced"
else
    echo "Warning: /opt/ros/jazzy/setup.bash not found — bridge imports will fail"
fi

source ${CONDA_ROOT}/bin/activate $CONDA_ENV_NAME

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT}/envs/$CONDA_ENV_NAME/lib

# Prepend unitree_interface's bundled CycloneDDS so it's resolved before the
# ROS2 Jazzy version (which mismatches and aborts with `free(): invalid
# pointer` when the C++ binding initializes DDS). Required for both the
# simulator bridge (run_sim.py) and inference (run_policy.py) when ROS2 is
# sourced. No-op if unitree_interface isn't installed.
_unitree_dir=$(python -c 'import unitree_interface,pathlib;print(pathlib.Path(unitree_interface.__file__).parent)' 2>/dev/null || true)
if [ -n "$_unitree_dir" ]; then
    export LD_LIBRARY_PATH=${_unitree_dir}:${LD_LIBRARY_PATH}
fi
unset _unitree_dir

# Validate
if python -c "import mujoco" 2>/dev/null; then
    echo "✅ $CONDA_ENV_NAME activated (Python $(python --version 2>&1 | cut -d' ' -f2))"
    echo "   MuJoCo $(python -c 'import mujoco; print(mujoco.__version__)')"
    echo "   PyTorch $(python -c 'import torch; print(torch.__version__)')"
    python -c "import rclpy; print('   rclpy OK')" 2>/dev/null || echo "   ⚠ rclpy not available"
else
    echo "Warning: MuJoCo environment activation may have issues"
fi
