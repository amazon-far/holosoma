#!/bin/bash
# Exit on error, and print commands
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Venv configuration (override via HS_INFER_VENV)
VENV_DIR="${HS_INFER_VENV:-$ROOT_DIR/.venv/hsinference}"

# Parse command-line arguments
INSTALL_ROBOT_SDKS=true
PYTHON_VERSION=""
REINSTALL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-robot-sdks)
      INSTALL_ROBOT_SDKS=false
      echo "Robot SDK installation disabled (unitree, booster)"
      shift
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --reinstall)
      REINSTALL=true
      echo "Reinstall requested — existing environment will be removed"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--no-robot-sdks] [--python VERSION] [--reinstall]"
      echo ""
      echo "Options:"
      echo "  --no-robot-sdks    Skip robot SDK installation (unitree, booster)"
      echo "  --python VERSION   Python version to use (e.g., 3.10, 3.12)"
      echo "  --reinstall        Remove existing environment and reinstall from scratch"
      echo "  --help, -h         Show this help message"
      echo ""
      echo "Python auto-detection (when --python is not specified):"
      echo "  Ubuntu 22.04 → Python 3.10 (ROS2 Humble compatible)"
      echo "  Ubuntu 24.04 → Python 3.12 (ROS2 Jazzy compatible)"
      echo "  Other         → system default Python"
      echo ""
      echo "Note: ROS2 is optional. The environment works standalone."
      echo "      Use 'source scripts/source_inference_uv_setup.sh' to activate"
      echo "      (it will source ROS2 automatically if installed)."
      echo ""
      echo "Examples:"
      echo "  # Full setup (default: with robot SDKs)"
      echo "  $0"
      echo ""
      echo "  # Setup with specific Python version, no robot SDKs"
      echo "  $0 --python 3.12 --no-robot-sdks"
      echo ""
      echo "  # Force clean reinstall"
      echo "  $0 --reinstall"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-robot-sdks] [--python VERSION] [--reinstall]"
      echo "Use --help for more information"
      exit 1
      ;;
  esac
done

# Sentinel file
SENTINEL_FILE=${VENV_DIR}/.env_uv_setup_finished_hsinference

# Reinstall: remove existing venv and sentinel so install runs fresh
if [[ "$REINSTALL" == "true" ]] && [[ -d "$VENV_DIR" ]]; then
  echo "Removing existing environment at $VENV_DIR..."
  rm -rf "$VENV_DIR"
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Source the env so uv is available in this session
  source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Auto-detect Python version from Ubuntu release if not explicitly set.
# This ensures the venv matches the system Python used by ROS2:
#   Ubuntu 22.04 (Jammy)  → Python 3.10 → ROS2 Humble
#   Ubuntu 24.04 (Noble)  → Python 3.12 → ROS2 Jazzy
if [[ -z "$PYTHON_VERSION" ]]; then
  OS_NAME_DETECT="$(uname -s)"
  if [[ "$OS_NAME_DETECT" == "Linux" && -f /etc/os-release ]]; then
    UBUNTU_VERSION=$(grep '^VERSION_ID=' /etc/os-release | cut -d'"' -f2)
    case "$UBUNTU_VERSION" in
      22.04)
        PYTHON_VERSION="3.10"
        echo "Detected Ubuntu 22.04 → using Python 3.10 (ROS2 Humble compatible)"
        ;;
      24.04)
        PYTHON_VERSION="3.12"
        echo "Detected Ubuntu 24.04 → using Python 3.12 (ROS2 Jazzy compatible)"
        ;;
      *)
        echo "Ubuntu $UBUNTU_VERSION detected — no default Python version mapped, using system default"
        ;;
    esac
  fi
fi

# Base installation
if [[ ! -f $SENTINEL_FILE ]]; then
  OS_NAME="$(uname -s)"
  ARCH="$(uname -m)"

  # Create venv
  echo "Creating virtual environment at $VENV_DIR..."
  UV_PYTHON_FLAG=""
  if [[ -n "$PYTHON_VERSION" ]]; then
    UV_PYTHON_FLAG="--python $PYTHON_VERSION"
  fi
  uv venv $UV_PYTHON_FLAG "$VENV_DIR"

  # Activate venv
  source "$VENV_DIR/bin/activate"

  # Install holosoma_inference
  # Robot SDK wheels (unitree, booster) are only available on Linux aarch64/x86_64.
  echo "Installing holosoma_inference..."
  if [[ "$INSTALL_ROBOT_SDKS" == "true" && "$OS_NAME" == "Linux" ]]; then
    uv pip install -e "$ROOT_DIR/src/holosoma_inference[unitree,booster]"
  else
    if [[ "$INSTALL_ROBOT_SDKS" == "true" && "$OS_NAME" == "Darwin" ]]; then
      echo "Note: Robot SDK wheels (unitree, booster) are not available for macOS."
      echo "Installing holosoma_inference without robot SDK extras."
    fi
    uv pip install -e "$ROOT_DIR/src/holosoma_inference"
  fi

  # Pinocchio (rigid-body dynamics) is required by WBT policies.
  # Installed separately because it's not in setup.py install_requires.
  echo "Installing pinocchio for WBT policy support..."
  uv pip install 'pin>=3.8.0'

  # Jetson power mode bump for aarch64 (best-effort, non-fatal)
  if [[ "$OS_NAME" == "Linux" && "$ARCH" == "aarch64" ]]; then
    sudo nvpmodel -m 0 2>/dev/null || true
  fi

  # Validate inference imports
  echo "Validating holosoma_inference installation..."
  python -c "import holosoma_inference; print('holosoma_inference imported successfully')"
  python -c "import pinocchio as pin; print(f'pinocchio version: {pin.__version__}')"

  touch $SENTINEL_FILE
  echo ""
  echo "=========================================="
  echo "holosoma_inference environment setup completed!"
  echo "=========================================="
  echo ""
  echo "Activate with: source scripts/source_inference_uv_setup.sh"
  echo "=========================================="
fi

echo ""
echo "holosoma_inference environment ready."
echo "Use 'source scripts/source_inference_uv_setup.sh' to activate."
