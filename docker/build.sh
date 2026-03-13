#!/bin/bash
#
# build.sh - Script to build the Holosoma Docker image
#
# This script builds the Docker image using the Dockerfile in the docker/ directory.
# It uses Docker BuildKit for improved build performance and caching.
#
# Usage: ECR_REPO=your-ecr-repo bash docker/build.sh

# Enable command echo for debugging
# set -x

# ECR repository for the Docker image (must be provided as environment variable)
if [ -z "$ECR_REPO" ]; then
    echo "Error: ECR_REPO environment variable must be set"
    echo "Usage: ECR_REPO=your-ecr-repo bash docker/build.sh"
    exit 1
fi

ROOT_REPO="$(realpath "$(dirname "$0")/..")"

# Use IMAGE_TAG if set, otherwise use date format YYYY_MMDD_HHMM
TAG=${IMAGE_TAG:-$(date +%Y_%m%d_%H%M)}

cd "$ROOT_REPO"

# Build the Docker image using BuildKit
# - Uses the Dockerfile in the docker/ directory
# - Tags the image as ${ECR_REPO}/holosoma
# - Build context is the repository root
DOCKER_BUILDKIT=1 docker buildx build -t ${ECR_REPO}/holosoma:${TAG} -f docker/Dockerfile .

# Print the name of the built image
echo "Built Docker image: ${ECR_REPO}/holosoma:${TAG}"

rm -rf build_context
