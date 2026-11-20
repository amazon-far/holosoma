#!/usr/bin/env bash
# Fetch the prebuilt ONNX Runtime that holosoma_cpp links against and unpack it
# so DEST holds include/ and lib/ (the ONNX Runtime release tarball layout).
#
# github.com/microsoft/onnxruntime and its release assets have returned 404
# since 2026-11-17, so CI cannot let CMake download them. CI reads this mirror
# from the holosoma assets bucket instead and points ONNXRUNTIME_ROOT at DEST.
# The mirror's PROVENANCE.md records where the files came from: the shared
# libraries are byte-identical to the ones in images built from the original
# release tarballs.
#
# Needs AWS credentials that can read the bucket (the holosoma CodeBuild runner
# roles can). Uses the aws CLI when installed, else a pinned one through uvx.
#
# Usage: scripts/fetch_onnxruntime.sh DEST [linux-x64|linux-aarch64]
set -euo pipefail

# Keep in step with HOLOSOMA_ONNXRUNTIME_VERSION in
# src/holosoma_inference/holosoma_cpp/cmake/HolosomaDependencies.cmake.
ORT_VERSION=1.30.0
ORT_S3_PREFIX=s3://far-holosoma-assets/third_party/onnxruntime
ORT_S3_REGION=us-west-2
AWSCLI_VERSION=1.46.3

dest=${1:?usage: $0 DEST [linux-x64|linux-aarch64]}
platform=${2:-}
if [[ -z $platform ]]; then
  case $(uname -m) in
    x86_64 | amd64) platform=linux-x64 ;;
    aarch64 | arm64) platform=linux-aarch64 ;;
    *)
      echo "unsupported architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
fi
case $platform in
  linux-x64) sha256=43b5e1e8aa9d0d3fe7899aad64a9832c6ff1eb98faf1753b66c7691dc92ab953 ;;
  linux-aarch64) sha256=94593c1664190612a7db5f292b29bea2e993d1d5a2020702e5b821dcc78e5fc6 ;;
  *)
    echo "unsupported platform: $platform" >&2
    exit 1
    ;;
esac

aws_cli() {
  if command -v aws > /dev/null 2>&1; then
    aws "$@"
  else
    uvx --from "awscli==${AWSCLI_VERSION}" aws "$@"
  fi
}

mkdir -p "$dest"
if [[ -n $(ls -A "$dest") ]]; then
  echo "$dest is not empty" >&2
  exit 1
fi

archive=onnxruntime-${platform}-${ORT_VERSION}.tgz
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
aws_cli s3 cp --only-show-errors --region "$ORT_S3_REGION" \
  "${ORT_S3_PREFIX}/${ORT_VERSION}/${archive}" "$tmp/$archive"
echo "${sha256}  $tmp/$archive" | sha256sum -c -
tar -xzf "$tmp/$archive" -C "$dest" --strip-components=1
test -f "$dest/include/onnxruntime_cxx_api.h"
test -e "$dest/lib/libonnxruntime.so"
echo "ONNX Runtime ${ORT_VERSION} (${platform}) unpacked in ${dest}"
