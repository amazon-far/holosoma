#!/bin/bash

# Build the Docker image using the holosoma directory as context
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # holosoma/src/holosoma_inference/docker
SRC_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")" # holosoma/src
IMAGE_NAME="holosoma-inference"
ECR_IMAGE="982423663241.dkr.ecr.us-west-2.amazonaws.com/holosoma-inference:latest"

docker build "$SRC_DIR" -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE_NAME" -t "$ECR_IMAGE"

[[ "$1" == "--push" ]] && docker push "$ECR_IMAGE"

rm -f "$SCRIPT_DIR"/*.whl
