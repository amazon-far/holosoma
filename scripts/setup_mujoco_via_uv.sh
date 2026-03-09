#!/bin/bash
# Exit on error, and print commands
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Venv configuration
VENV_DIR=$ROOT_DIR/.venv/hsmujoco

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
      echo "  --no-warp          Skip MuJoCo Warp installation (CPU-only)"
      echo "  --no-robot-sdks    Skip robot SDK installation (unitree, booster)"
      echo "  --python VERSION   Python version to use (e.g., 3.10, 3.11)"
      echo "  --help, -h         Show this help message"
      echo ""
      echo "Default: GPU-accelerated installation with robot SDKs"
      echo ""
      echo "Examples:"
      echo "  # Full setup (default: with GPU acceleration + robot SDKs)"
      echo "  $0"
      echo ""
      echo "  # Setup without GPU acceleration (CPU-only)"
      echo "  $0 --no-warp"
      echo ""
      echo "  # Setup with specific Python version, no robot SDKs"
      echo "  $0 --python 3.10 --no-robot-sdks"
      echo ""
      echo "Note: GPU acceleration requires NVIDIA driver >= 555.58.02"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-warp] [--no-robot-sdks] [--python VERSION]"
      echo "Use --help for more information"
      exit 1
      ;;
  esac
done

# Sentinel files
SENTINEL_FILE=${VENV_DIR}/.env_uv_setup_finished_hsmujoco
WARP_SENTINEL_FILE=${VENV_DIR}/.env_uv_setup_finished_hsmujoco_warp

# Install uv if not present
if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Source the env so uv is available in this session
  source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Base installation
if [[ ! -f $SENTINEL_FILE ]]; then
  OS_NAME="$(uname -s)"

  # Create venv
  echo "Creating virtual environment at $VENV_DIR..."
  UV_PYTHON_FLAG=""
  if [[ -n "$PYTHON_VERSION" ]]; then
    UV_PYTHON_FLAG="--python $PYTHON_VERSION"
  fi
  uv venv $UV_PYTHON_FLAG "$VENV_DIR"

  # Activate venv
  source "$VENV_DIR/bin/activate"

  # Install MuJoCo and related packages
  echo "Installing MuJoCo Python bindings..."
  uv pip install 'mujoco>=3.0.0'
  uv pip install mujoco-python-viewer

  # Install Holosoma packages
  echo "Installing Holosoma packages..."
  if [[ "$INSTALL_ROBOT_SDKS" == "true" && "$OS_NAME" == "Linux" ]]; then
    uv pip install -e "$ROOT_DIR/src/holosoma[unitree,booster]"
  else
    if [[ "$INSTALL_ROBOT_SDKS" == "true" && "$OS_NAME" == "Darwin" ]]; then
      echo "Note: Robot SDK wheels (unitree, booster) are not available for macOS."
      echo "Installing holosoma without robot SDK extras."
    fi
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
  echo "  MuJoCo CPU backend (ClassicBackend) installed"
  echo ""
  echo "Activate with: source scripts/source_mujoco_uv_setup.sh"
  echo "=========================================="
fi

# Separate Warp installation (can be run independently after base install)
if [[ "$INSTALL_WARP" == "true" ]] && [[ ! -f $WARP_SENTINEL_FILE ]]; then
  echo ""
  echo "Installing MuJoCo Warp (GPU acceleration)..."

  # Ensure venv is activated
  source "$VENV_DIR/bin/activate"

  # Check NVIDIA driver version (required for CUDA 12.4+)
  MIN_DRIVER_VERSION="555.58.02"
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
    echo "Minimum required: $MIN_DRIVER_VERSION (for CUDA 12.4+ support)"
    echo ""
    echo "MuJoCo Warp requires:"
    echo "  - NVIDIA GPU (CUDA-capable)"
    echo "  - NVIDIA driver >= $MIN_DRIVER_VERSION (for CUDA 12.4+)"
    echo ""
    echo "Install/Upgrade NVIDIA driver:"
    echo "  1. Check available drivers: ubuntu-drivers devices"
    echo "  2. Install recommended:    sudo ubuntu-drivers install"
    echo "  3. Or install specific:    sudo ubuntu-drivers install nvidia:590"
    echo "  4. Reboot:                 sudo reboot"
    echo ""
    echo "Reference: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/"
    echo ""
    echo "After driver installation, re-run this script"
    echo "(or use --no-warp for CPU-only installation)"
    exit 1
  fi

  echo "NVIDIA driver version: $DRIVER_VERSION (meets minimum $MIN_DRIVER_VERSION)"

  uv pip install 'mujoco-warp[cuda]'

  touch $WARP_SENTINEL_FILE

  echo ""
  echo "=========================================="
  echo "MuJoCo Warp installation completed!"
  echo "=========================================="
  echo ""
  echo "  GPU acceleration enabled (WarpBackend)"
  echo "  Both backends now available: ClassicBackend (CPU) + WarpBackend (GPU)"
  echo ""
  echo "Activate with: source scripts/source_mujoco_uv_setup.sh"
  echo "=========================================="
fi

echo ""
if [[ -f $WARP_SENTINEL_FILE ]]; then
  echo "MuJoCo environment ready with GPU acceleration (ClassicBackend + WarpBackend)"
elif [[ "$INSTALL_WARP" == "false" ]] && [[ -f $SENTINEL_FILE ]]; then
  echo "MuJoCo environment ready (CPU-only ClassicBackend)"
  echo ""
  echo "To add GPU acceleration later, run:"
  echo "  bash scripts/setup_mujoco_via_uv.sh"
else
  echo "MuJoCo environment ready."
fi
echo "Use 'source scripts/source_mujoco_uv_setup.sh' to activate."
