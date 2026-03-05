#!/bin/bash
# Exit on error, and print commands
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# MuJoCo Warp version to install
MUJOCO_WARP_COMMIT="09ec1da"

# Parse command-line arguments
INSTALL_WARP=true
INSTALL_ROBOT_SDKS=true
PYTHON_VERSION=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-warp)
      INSTALL_WARP=false
      echo "MuJoCo Warp (GPU) installation disabled - CPU-only mode"
      shift
      ;;
    --no-robot-sdks)
      INSTALL_ROBOT_SDKS=false
      echo "Robot SDK installation disabled (unitree, booster)"
      shift
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--no-warp] [--no-robot-sdks] [--python VERSION]"
      echo ""
      echo "Options:"
      echo "  --no-warp         Skip MuJoCo Warp installation (CPU-only)"
      echo "  --no-robot-sdks   Skip unitree/booster SDK installation"
      echo "  --python VERSION  Python version to use (e.g., 3.10, 3.12). Default: system python"
      echo "  --help, -h        Show this help message"
      echo ""
      echo "Default: GPU-accelerated installation (WarpBackend + ClassicBackend)"
      echo ""
      echo "Note: GPU acceleration requires NVIDIA driver >= 550.54.14"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-warp] [--no-robot-sdks] [--python VERSION]"
      exit 1
      ;;
  esac
done

WORKSPACE_DIR=$HOME/.holosoma_deps
VENV_DIR=${WORKSPACE_DIR}/venvs/hsmujoco
SENTINEL_FILE=${WORKSPACE_DIR}/.env_uv_setup_finished_hsmujoco
WARP_SENTINEL_FILE=${WORKSPACE_DIR}/.env_uv_setup_finished_hsmujoco_warp

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install uv if not present
  if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  echo "uv version: $(uv --version)"

  # Create venv
  if [[ ! -d $VENV_DIR ]]; then
    echo "Creating virtual environment..."
    if [[ -n "$PYTHON_VERSION" ]]; then
      echo "Using Python $PYTHON_VERSION"
      uv venv --python "$PYTHON_VERSION" $VENV_DIR
    else
      echo "Using system Python"
      uv venv $VENV_DIR
    fi
  fi

  source $VENV_DIR/bin/activate

  # Install MuJoCo and related packages
  echo "Installing MuJoCo Python bindings..."
  uv pip install 'mujoco>=3.0.0' mujoco-python-viewer

  # Install Holosoma packages
  if [[ "$INSTALL_ROBOT_SDKS" == "true" ]]; then
    uv pip install -e "$ROOT_DIR/src/holosoma[unitree,booster]"
  else
    uv pip install -e "$ROOT_DIR/src/holosoma"
  fi

  # Validate MuJoCo installation
  echo "Validating MuJoCo installation..."
  python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
  python -c "import mujoco_viewer; print('MuJoCo viewer imported successfully')"

  touch $SENTINEL_FILE
  echo ""
  echo "=========================================="
  echo "Base MuJoCo environment setup completed!"
  echo "=========================================="
  echo ""
  echo "Activate with: source scripts/source_mujoco_uv_setup.sh"
  echo "=========================================="
fi

# Separate Warp installation
if [[ "$INSTALL_WARP" == "true" ]] && [[ ! -f $WARP_SENTINEL_FILE ]]; then
  echo ""
  echo "Installing MuJoCo Warp (GPU acceleration)..."

  source $VENV_DIR/bin/activate

  # Check NVIDIA driver version
  MIN_DRIVER_VERSION="550.54.14"
  DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)

  if [ -z "$DRIVER_VERSION" ] || [[ "$DRIVER_VERSION" < "$MIN_DRIVER_VERSION" ]]; then
    echo ""
    echo "ERROR: NVIDIA driver not found or too old!"
    echo ""
    if [ -z "$DRIVER_VERSION" ]; then
      echo "Status: No NVIDIA driver detected"
    else
      echo "Current driver:  $DRIVER_VERSION"
    fi
    echo "Minimum required: $MIN_DRIVER_VERSION (for CUDA 12.4+)"
    echo ""
    echo "After driver installation, re-run this script"
    echo "(or use --no-warp for CPU-only installation)"
    exit 1
  fi

  echo "NVIDIA driver version: $DRIVER_VERSION (meets minimum $MIN_DRIVER_VERSION)"

  if [[ ! -d $WORKSPACE_DIR/mujoco_warp ]]; then
    git clone https://github.com/google-deepmind/mujoco_warp.git $WORKSPACE_DIR/mujoco_warp && \
      git -C $WORKSPACE_DIR/mujoco_warp checkout ${MUJOCO_WARP_COMMIT}
  fi
  uv pip install -e "$WORKSPACE_DIR/mujoco_warp[dev,cuda]"

  touch $WARP_SENTINEL_FILE

  echo ""
  echo "=========================================="
  echo "MuJoCo Warp installation completed!"
  echo "=========================================="
  echo ""
  echo "Activate with: source scripts/source_mujoco_uv_setup.sh"
  echo "=========================================="
fi

echo ""
if [[ -f $WARP_SENTINEL_FILE ]]; then
  echo "MuJoCo environment ready with GPU acceleration (ClassicBackend + WarpBackend)"
elif [[ "$INSTALL_WARP" == "false" ]] && [[ -f $SENTINEL_FILE ]]; then
  echo "MuJoCo environment ready (CPU-only ClassicBackend)"
else
  echo "MuJoCo environment ready."
fi
echo "Use 'source scripts/source_mujoco_uv_setup.sh' to activate."
