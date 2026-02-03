#!/bin/bash
# Exit on error, and print commands
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# MuJoCo Warp version to install -- the repo is missing version tags and branches
# Arbitrarily chosen from mainline at the time we've ~tested against
MUJOCO_WARP_COMMIT="09ec1da"

# Parse command-line arguments
INSTALL_WARP=true  # Default: install warp (GPU-accelerated)

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-warp)
      INSTALL_WARP=false
      echo "MuJoCo Warp (GPU) installation disabled - CPU-only mode"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--no-warp]"
      echo ""
      echo "Options:"
      echo "  --no-warp      Skip MuJoCo Warp installation (CPU-only)"
      echo "  --help, -h     Show this help message"
      echo ""
      echo "Default: GPU-accelerated installation (WarpBackend + ClassicBackend)"
      echo ""
      echo "Examples:"
      echo "  # Initial setup (default: with GPU acceleration)"
      echo "  $0"
      echo ""
      echo "  # Setup without GPU acceleration (CPU-only)"
      echo "  $0 --no-warp"
      echo ""
      echo "Note: GPU acceleration requires NVIDIA driver >= 550.54.14"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-warp]"
      echo "Use --help for more information"
      exit 1
      ;;
  esac
done

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/hsmujoco
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_mujoco
WARP_SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_mujoco_warp

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda (reuse existing logic)
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n hsmujoco python=3.10 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate hsmujoco

  # Install system dependencies for MuJoCo
  # Note: These may require sudo access - document this requirement
  echo "Installing system dependencies for MuJoCo..."
  # sudo apt-get update
  # sudo apt-get install -y libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev

  # Install libstdcxx-ng to fix potential GLIBCXX issues
  conda install -c conda-forge -y libstdcxx-ng

  # Install ffmpeg for video encoding (consistent with other envs)
  conda install -c conda-forge -y ffmpeg

  # Install MuJoCo and related packages
  echo "Installing MuJoCo Python bindings..."
  pip install --upgrade pip

  # Core MuJoCo packages
  pip install 'mujoco>=3.0.0'
  pip install mujoco-python-viewer
  # Optional: Gymnasium MuJoCo environments (if needed for compatibility)
 # pip install "gymnasium[mujoco]"

  # Scientific computing stack (ensure compatibility)
  #pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  #pip install numpy scipy matplotlib

  # Install Holosoma packages
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma[unitree,booster]

  # Validate MuJoCo installation
  echo "Validating MuJoCo installation..."
  python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
  python -c "import mujoco_viewer; print('MuJoCo viewer imported successfully')"

  # Create validation script for later testing
  cat > $WORKSPACE_DIR/validate_mujoco.py << 'EOF'
#!/usr/bin/env python3
"""Validation script for MuJoCo installation."""

import sys
import mujoco
import numpy as np

def validate_mujoco():
    """Validate MuJoCo installation with basic functionality test."""
    print(f"MuJoCo version: {mujoco.__version__}")

    # Test basic model creation
    xml_string = """
    <mujoco>
      <worldbody>
        <body name="box" pos="0 0 1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint type="free"/>
        </body>
      </worldbody>
    </mujoco>
    """

    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)

        # Test simulation step
        mujoco.mj_step(model, data)

        print("✓ Basic MuJoCo functionality validated")
        print(f"✓ Model has {model.nbody} bodies, {model.nq} DOFs")
        return True

    except Exception as e:
        print(f"✗ MuJoCo validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_mujoco()
    sys.exit(0 if success else 1)
EOF

  # Run validation
  python $WORKSPACE_DIR/validate_mujoco.py

  touch $SENTINEL_FILE
  echo ""
  echo "=========================================="
  echo "Base MuJoCo environment setup completed!"
  echo "=========================================="
  echo ""
  echo "✓ MuJoCo CPU backend (ClassicBackend) installed"
  echo ""
  echo "Activate with: source scripts/source_mujoco_setup.sh"
  echo "=========================================="
fi

# Separate Warp installation (can be run independently after base install)
if [[ "$INSTALL_WARP" == "true" ]] && [[ ! -f $WARP_SENTINEL_FILE ]]; then
  echo ""
  echo "Installing MuJoCo Warp (GPU acceleration)..."

  # Ensure conda environment is activated
  source $CONDA_ROOT/bin/activate hsmujoco

  # Check NVIDIA driver version (required for CUDA 12.4+)
  MIN_DRIVER_VERSION="550.54.14"
  DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)

  # Check if driver exists and meets minimum version
  if [ -z "$DRIVER_VERSION" ] || [[ "$DRIVER_VERSION" < "$MIN_DRIVER_VERSION" ]]; then
    echo ""
    echo "❌ ERROR: NVIDIA driver not found or too old!"
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
    echo "  3. Or install specific:    sudo ubuntu-drivers install nvidia:550"
    echo "  4. Reboot:                 sudo reboot"
    echo ""
    echo "Reference: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/"
    echo ""
    echo "After driver installation, re-run this script"
    echo "(or use --no-warp for CPU-only installation)"
    exit 1
  fi

  echo "✓ NVIDIA driver version: $DRIVER_VERSION (meets minimum $MIN_DRIVER_VERSION)"

  if [[ ! -d $WORKSPACE_DIR/mujoco_warp ]]; then
    git clone https://github.com/google-deepmind/mujoco_warp.git $WORKSPACE_DIR/mujoco_warp && \
      git -C $WORKSPACE_DIR/mujoco_warp checkout ${MUJOCO_WARP_COMMIT}
  fi
  pip install uv
  uv pip install -e $WORKSPACE_DIR/mujoco_warp[dev,cuda]

  touch $WARP_SENTINEL_FILE

  echo ""
  echo "=========================================="
  echo "MuJoCo Warp installation completed!"
  echo "=========================================="
  echo ""
  echo "✓ GPU acceleration enabled (WarpBackend)"
  echo "✓ Both backends now available: ClassicBackend (CPU) + WarpBackend (GPU)"
  echo ""
  echo "Activate with: source scripts/source_mujoco_setup.sh"
  echo "=========================================="
fi

echo ""
if [[ -f $WARP_SENTINEL_FILE ]]; then
  echo "MuJoCo environment ready with GPU acceleration (ClassicBackend + WarpBackend)"
elif [[ "$INSTALL_WARP" == "false" ]] && [[ -f $SENTINEL_FILE ]]; then
  echo "MuJoCo environment ready (CPU-only ClassicBackend)"
  echo ""
  echo "To add GPU acceleration later, run:"
  echo "  bash scripts/setup_mujoco.sh"
else
  echo "MuJoCo environment ready."
fi
echo "Use 'source scripts/source_mujoco_setup.sh' to activate."
