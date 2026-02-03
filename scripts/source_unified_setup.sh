# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

# [중요] ROOT_DIR 변수 정의 (holosoma 폴더 위치)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# 1. Load common variables
source ${SCRIPT_DIR}/source_common.sh

# 2. Activate the UNIFIED environment
# (앞서 setup_unified.sh에서 만든 환경 이름 'hunified' 사용)
source ${CONDA_ROOT}/bin/activate hunified

# 3. Export Library Paths
# Python 3.10 기준 (Inference 호환성을 위해 3.10을 선택했으므로 경로도 이에 맞춤)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT}/envs/hunified/lib/python3.10/site-packages/lib

# 4. [핵심 수정] PYTHONPATH 추가
# 아까 수동으로 입력했던 해결책을 여기에 영구 적용합니다.
# holosoma/src 폴더를 파이썬 경로에 추가하여 retargeting 모듈을 찾게 합니다.
export PYTHONPATH=${PYTHONPATH}:${ROOT_DIR}/src

# 5. Check UFW status (Crucial for Robot Communication)
if command -v ufw >/dev/null 2>&1; then
    if sudo ufw status | grep -q "Status: inactive"; then
        echo "✓ UFW disabled (Ready for connection)"
    else
        echo "⚠️  Warning: UFW is currently enabled."
        echo "   Robot connection might fail. Consider running: sudo ufw disable"
    fi
fi

echo "Environment 'hunified' activated. (PYTHONPATH updated)"