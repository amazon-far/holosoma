#!/usr/bin/env bash
# Exit on error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Venv configuration
VENV_DIR=${RETARGETING_UV_VENV_DIR:-$ROOT_DIR/.venv/hsretargeting}

PYTHON_VERSION=${PYTHON_VERSION:-3.11}
REINSTALL=false
INSTALL_DEV=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --reinstall)
      REINSTALL=true
      echo "Reinstall requested; existing environment will be removed"
      shift
      ;;
    --dev)
      INSTALL_DEV=true
      echo "Development dependencies will be installed"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--python VERSION] [--reinstall] [--dev]"
      echo ""
      echo "Options:"
      echo "  --python VERSION   Python version to use (default: ${PYTHON_VERSION})"
      echo "  --reinstall        Remove existing environment and reinstall from scratch"
      echo "  --dev              Install holosoma-retargeting development dependencies"
      echo "  --help, -h         Show this help message"
      echo ""
      echo "Environment:"
      echo "  RETARGETING_UV_VENV_DIR   Override venv location (default: $ROOT_DIR/.venv/hsretargeting)"
      echo "  PYTHON_VERSION            Override default Python version"
      echo ""
      echo "Examples:"
      echo "  $0"
      echo "  $0 --python 3.10"
      echo "  $0 --reinstall --dev"
      echo ""
      echo "Activate with: source scripts/source_retargeting_uv_setup.sh"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--python VERSION] [--reinstall] [--dev]"
      echo "Use --help for more information"
      exit 1
      ;;
  esac
done

BASE_SENTINEL_FILE=${VENV_DIR}/.env_uv_setup_finished_hsretargeting
DEV_SENTINEL_FILE=${VENV_DIR}/.env_uv_setup_finished_hsretargeting_dev

if [[ "$REINSTALL" == "true" ]] && [[ -d "$VENV_DIR" ]]; then
  echo "Removing existing environment at $VENV_DIR..."
  rm -rf "$VENV_DIR"
fi

if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"
echo "Retargeting uv environment: $VENV_DIR"
echo "Python version request: $PYTHON_VERSION"

if [[ ! -f "$BASE_SENTINEL_FILE" ]]; then
  mkdir -p "$(dirname "$VENV_DIR")"

  echo "Creating virtual environment at $VENV_DIR..."
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

  source "$VENV_DIR/bin/activate"

  echo "Installing holosoma-retargeting and runtime dependencies..."
  uv pip install -e "$ROOT_DIR/src/holosoma_retargeting"

  touch "$BASE_SENTINEL_FILE"
else
  source "$VENV_DIR/bin/activate"
  echo "Base retargeting uv environment is already installed."
fi

if [[ "$INSTALL_DEV" == "true" ]] && [[ ! -f "$DEV_SENTINEL_FILE" ]]; then
  echo "Installing holosoma-retargeting development dependencies..."
  uv pip install -e "$ROOT_DIR/src/holosoma_retargeting[dev]"
  touch "$DEV_SENTINEL_FILE"
fi

echo "Validating holosoma-retargeting imports..."
python - <<'PY'
import sys

import cvxpy
import holosoma_retargeting
import igl
import imageio
import imageio_ffmpeg
import mujoco
import numpy
import torch
import viser
import yourdfpy

def version(module):
    return getattr(module, "__version__", "unknown")

print(f"Python: {sys.version.split()[0]}")
print(f"holosoma_retargeting: {holosoma_retargeting.__file__}")
print(f"numpy: {version(numpy)}")
print(f"torch: {version(torch)}")
print(f"mujoco: {version(mujoco)}")
print(f"viser: {version(viser)}")
print(f"cvxpy: {version(cvxpy)}")
print(f"imageio: {version(imageio)}")
print(f"imageio-ffmpeg: {version(imageio_ffmpeg)}")
print(f"yourdfpy: {version(yourdfpy)}")
print(f"libigl module: {igl.__name__}")
PY

echo ""
echo "=========================================="
echo "Retargeting uv environment setup completed!"
echo "=========================================="
echo ""
echo "Activate with: source scripts/source_retargeting_uv_setup.sh"
echo "Environment: $VENV_DIR"
echo "=========================================="
