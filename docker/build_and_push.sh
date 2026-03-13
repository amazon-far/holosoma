#!/bin/bash
#
# build_and_push.sh — Build and push all Holosoma Docker images to ECR.
#
# Usage:
#   bash docker/build_and_push.sh                  # build & push all images
#   bash docker/build_and_push.sh mujoco retarget  # only matching images
#   bash docker/build_and_push.sh --no-push        # build only, skip push
#   bash docker/build_and_push.sh --dry-run        # print commands, do nothing
#
# Environment variables:
#   ECR_REPO   — ECR registry URI (default: 982423663241.dkr.ecr.us-west-2.amazonaws.com)
#   IMAGE_TAG  — override the date-based tag (default: YYYY_MMDD_HHMM)

set -euo pipefail

ROOT_REPO="$(realpath "$(dirname "$0")/..")"
ECR_REPO="${ECR_REPO:-982423663241.dkr.ecr.us-west-2.amazonaws.com}"
TAG="${IMAGE_TAG:-$(date +%Y_%m%d_%H%M)}"

# ── Image definitions ────────────────────────────────────────────────
# Each entry: "key|dockerfile|image_name|json_key"
#   key        — short name used for positional filtering
#   dockerfile — path relative to repo root
#   image_name — ECR image name (without registry prefix)
#   json_key   — key in docker_images.json
IMAGES=(
  "holosoma|docker/Dockerfile|holosoma|holosoma"
  "isaacsim|docker/isaacsim.Dockerfile|holosoma-isaacsim|hs-isaacsim"
  "isaacgym|docker/isaacgym.Dockerfile|holosoma-isaacgym|hs-isaacgym"
  "mujoco|docker/mujoco.Dockerfile|holosoma-mujoco|hs-mujoco"
  "retargeting|src/holosoma_retargeting/docker/Dockerfile|holosoma-retargeting|hs-retargeting"
  "inference|src/holosoma_inference/docker/Dockerfile|holosoma-inference|hs-inference"
)

# ── Parse flags ──────────────────────────────────────────────────────
NO_PUSH=false
DRY_RUN=false
FILTERS=()

for arg in "$@"; do
  case "$arg" in
    --no-push)  NO_PUSH=true ;;
    --dry-run)  DRY_RUN=true ;;
    --help|-h)
      sed -n '2,/^$/{ s/^# \?//; p }' "$0"
      exit 0
      ;;
    -*)
      echo "Unknown flag: $arg" >&2; exit 1 ;;
    *)
      FILTERS+=("$arg") ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────
run() {
  echo "+ $*"
  if ! $DRY_RUN; then
    "$@"
  fi
}

matches_filter() {
  local key="$1"
  if [[ ${#FILTERS[@]} -eq 0 ]]; then
    return 0  # no filter → build everything
  fi
  for f in "${FILTERS[@]}"; do
    if [[ "$key" == *"$f"* ]]; then
      return 0
    fi
  done
  return 1
}

# ── Preflight checks ────────────────────────────────────────────────
for cmd in docker; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: $cmd is not installed" >&2; exit 1
  fi
done

if ! $NO_PUSH && ! $DRY_RUN; then
  if ! command -v aws &>/dev/null; then
    echo "Error: aws CLI is not installed (required for push)" >&2; exit 1
  fi
fi

# ── ECR authentication ──────────────────────────────────────────────
if ! $NO_PUSH; then
  echo "==> Configuring ECR credential helper for ${ECR_REPO}"
  if ! $DRY_RUN; then
    # Ensure docker config dir exists
    mkdir -p ~/.docker

    # Add ecr-login credential helper for our registry if not already present
    if [ ! -f ~/.docker/config.json ]; then
      echo '{}' > ~/.docker/config.json
    fi

    if ! jq -e ".credHelpers[\"${ECR_REPO}\"]" ~/.docker/config.json &>/dev/null; then
      tmp=$(mktemp)
      jq --arg repo "$ECR_REPO" '.credHelpers[$repo] = "ecr-login"' ~/.docker/config.json > "$tmp" \
        && mv "$tmp" ~/.docker/config.json
      echo "    Added ecr-login credential helper for ${ECR_REPO}"
    else
      echo "    ECR credential helper already configured"
    fi
  fi
fi

# ── Build & push loop ───────────────────────────────────────────────
SUCCEEDED=()
FAILED=()

for entry in "${IMAGES[@]}"; do
  IFS='|' read -r key dockerfile image_name json_key <<< "$entry"

  if ! matches_filter "$key"; then
    continue
  fi

  full_image="${ECR_REPO}/${image_name}"
  echo ""
  echo "==> Building ${image_name} from ${dockerfile}"

  # tags: date tag + latest
  tags=(-t "${full_image}:latest" -t "${full_image}:${TAG}")

  if run env DOCKER_BUILDKIT=1 docker build "${tags[@]}" -f "${dockerfile}" "${ROOT_REPO}"; then
    echo "    Built: ${image_name}"

    if ! $NO_PUSH; then
      echo "    Pushing ${image_name}..."
      push_ok=true
      if ! run docker push "${full_image}:latest"; then
        push_ok=false
      fi
      if ! run docker push "${full_image}:${TAG}"; then
        push_ok=false
      fi

      if $push_ok; then
        SUCCEEDED+=("$image_name")
      else
        echo "    FAILED to push: ${image_name}" >&2
        FAILED+=("$image_name")
      fi
    else
      SUCCEEDED+=("$image_name")
    fi
  else
    echo "    FAILED to build: ${image_name}" >&2
    FAILED+=("$image_name")
  fi
done

# ── Update docker_images.json ───────────────────────────────────────
JSON_FILE="${ROOT_REPO}/docker/docker_images.json"
if [[ -f "$JSON_FILE" ]] && [[ ${#SUCCEEDED[@]} -gt 0 ]] && ! $DRY_RUN; then
  echo ""
  echo "==> Updating ${JSON_FILE}"
  for entry in "${IMAGES[@]}"; do
    IFS='|' read -r key _ image_name json_key <<< "$entry"

    if ! matches_filter "$key"; then
      continue
    fi

    # Only update images that succeeded
    for s in "${SUCCEEDED[@]}"; do
      if [[ "$s" == "$image_name" ]]; then
        new_val="${image_name}:${TAG}"
        tmp=$(mktemp)
        jq --arg k "$json_key" --arg v "$new_val" '.images[$k] = $v' "$JSON_FILE" > "$tmp" \
          && mv "$tmp" "$JSON_FILE"
        echo "    ${json_key} → ${new_val}"
      fi
    done
  done
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
  echo "  Succeeded: ${SUCCEEDED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  FAILED:    ${FAILED[*]}" >&2
  exit 1
fi
if [[ ${#SUCCEEDED[@]} -eq 0 ]] && [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "  No images matched the given filter(s): ${FILTERS[*]}"
  exit 1
fi
echo "  Tag: ${TAG}"
$NO_PUSH && echo "  (push skipped — --no-push)"
$DRY_RUN && echo "  (dry run — nothing was executed)"
echo "  Done."
