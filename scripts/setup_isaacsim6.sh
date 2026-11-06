#!/usr/bin/env bash
# Install Lab 3 in an isolated environment; the Sim 5.1 installation is independent.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/versions.sh"
ISAACSIM6_ROOT=${ISAACSIM6_ROOT:-${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}/isaacsim6}
LAB_DIR="$ISAACSIM6_ROOT/IsaacLab"
ENV_DIR="$ISAACSIM6_ROOT/hssim6"

command -v uv >/dev/null || { echo "Install uv before running setup_isaacsim6.sh." >&2; exit 1; }
mkdir -p "$ISAACSIM6_ROOT"
if [[ ! -d "$LAB_DIR" ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "$LAB_DIR"
    git -C "$LAB_DIR" checkout --detach "$ISAACLAB3_COMMIT"
fi
if [[ "$(git -C "$LAB_DIR" rev-parse HEAD)" != "$ISAACLAB3_COMMIT" ]] ||
   [[ -n "$(git -C "$LAB_DIR" status --porcelain --untracked-files=no)" ]]; then
    echo "Expected a clean Isaac Lab checkout at $ISAACLAB3_COMMIT: $LAB_DIR" >&2
    exit 1
fi

export UV_PROJECT_ENVIRONMENT="$ENV_DIR"
# Use the release's lock and dependency overrides, including its matched USD ABI.
uv sync --project "$LAB_DIR" --frozen --extra isaacsim --extra importers \
    --extra ov --extra viser --extra video --extra test

"$ENV_DIR/bin/python" - "$LAB_DIR" "$ISAACSIM6_ROOT" <<'PY'
import importlib.metadata
import pathlib
import sys
import tomllib

lab, root = map(pathlib.Path, sys.argv[1:])
project = tomllib.loads((lab / "pyproject.toml").read_text())
(root / "overrides.txt").write_text("\n".join(project["tool"]["uv"]["override-dependencies"]) + "\n")
# Keep engine/renderer versions fixed. Resolve the analysis/telemetry
# dependencies with Holosoma: it requires pandas <3 and W&B requires protobuf <7.
pins = sorted(
    f"{d.metadata['Name']}=={d.version}"
    for d in importlib.metadata.distributions()
    if not d.metadata["Name"].startswith("isaaclab")
    and d.metadata["Name"].lower() not in {"pandas", "protobuf"}
)
(root / "constraints.txt").write_text("\n".join(pins) + "\n")
PY
uv --no-config pip install --python "$ENV_DIR/bin/python" --default-index https://pypi.org/simple \
    --constraint "$ISAACSIM6_ROOT/constraints.txt" --overrides "$ISAACSIM6_ROOT/overrides.txt" \
    -e "$SCRIPT_DIR/../src/holosoma[unitree,booster]"

"$ENV_DIR/bin/python" - "$ISAACSIM6_VERSION" <<'PY'
from importlib.metadata import version
import sys

assert version("isaacsim") == sys.argv[1]
assert version("isaaclab").startswith("28.")
print("Installed Isaac Sim", version("isaacsim"), "/ Isaac Lab", version("isaaclab"))
PY
echo "Activate with: source $SCRIPT_DIR/source_isaacsim6_setup.sh"
