# syntax=docker/dockerfile:1.7
#
# Thor Docker image for holosoma_inference on Jetson Thor (JetPack 7.1).
#
# Platform: aarch64 SBSA, Ubuntu 24.04 (Noble), Python 3.12, CUDA 13.
# Purpose: run policy inference on a real Thor without needing the full
# public Isaac Sim base image (which is x86_64).
#
# Build (default `inference` target — no ROS):
#   docker build -t holosoma-thor-inference -f docker/thor/Dockerfile .
#
# Build ROS-enabled target (for Ros2Input cmd_vel bridge):
#   docker build --target inference-ros -t holosoma-thor-inference-ros \
#     -f docker/thor/Dockerfile .
#
# Run:
#   docker run --rm --runtime nvidia --network=host -it \
#     holosoma-thor-inference inference:g1-29dof-loco --task.interface eth0
#
# Layer strategy (stable → volatile, so code edits rebuild only the top):
#   l4t-cuda       — CUDA 13 devel on Ubuntu 24.04 aarch64. Rebuilds ~never.
#   python-base    — python3.12, build tools, uv. Rebuilds ~never.
#   long-deps      — NVPL, cuDSS, TensorRT runtime libs. Rebuilds ~never.
#   common-deps    — image-only Python dependencies.
#   app-deps       — holosoma_inference metadata + unitree_sdk2.
#   inference      — linked source layer + terminal entrypoint.
#
# The -ros branch mirrors long-deps/common-deps/app-deps on top of a
# ros-jazzy layer, so the Jazzy apt install (~1.5 GB) is also cached stably.

# ─── Stage: l4t-cuda ──────────────────────────────────────────────────────────
# CUDA 13 devel so nvcc + headers are available for any torch extension compile.
# The SBSA apt repo is already configured in this base image.
FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 AS l4t-cuda

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# ─── Stage: python-base ───────────────────────────────────────────────────────
# System Python + build tools + uv. Output venv lives at /opt/venv so it
# survives across layer boundaries and can be PATH-prepended.
FROM l4t-cuda AS python-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    curl \
    git \
    wget \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image (pinning to :latest — bump explicitly in
# this file if you want a deterministic version).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# ─── Stage: long-deps ─────────────────────────────────────────────────────────
# NVIDIA math + inference libs that rarely change.
FROM python-base AS long-deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    # NVPL math libs (required by CUDA torch wheel on aarch64 SBSA)
    libnvpl-blas0 \
    libnvpl-lapack0 \
    # cuDSS sparse solver (required by torch on CUDA 13)
    libcudss0-cuda-13 \
    # TensorRT runtime (policy ONNX runtime may dlopen libnvinfer)
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvonnxparsers10 \
    && rm -rf /var/lib/apt/lists/*

# cuDSS installs to a versioned path not in the default linker search path.
RUN echo "/usr/lib/aarch64-linux-gnu/libcudss/13" > /etc/ld.so.conf.d/cudss-13.conf \
    && ldconfig


# ─── Stage: common-deps ───────────────────────────────────────────────────────
# Image-only dependencies not declared by holosoma_inference.
FROM long-deps AS common-deps

RUN uv pip install "pin>=3.8.0"

COPY docker/apt/zzzz-holosoma-security-updates /etc/apt/apt.conf.d/
RUN apt-get update \
    && apt-get install -y --no-install-recommends unattended-upgrades \
    && unattended-upgrade --verbose \
    && apt-get purge -y unattended-upgrades \
    && rm -rf /var/lib/apt/lists/*


# ─── Stage: app-deps ──────────────────────────────────────────────────────────
# Install project dependencies and editable metadata at the final source path.
FROM common-deps AS app-deps

# Install against a placeholder module so source-only changes remain an
# independent linked layer in the final target.
COPY src/holosoma_inference/pyproject.toml src/holosoma_inference/README.md \
    /opt/holosoma-src/src/holosoma_inference/
RUN mkdir -p /opt/holosoma-src/src/holosoma_inference/holosoma_inference && \
    touch /opt/holosoma-src/src/holosoma_inference/holosoma_inference/__init__.py
WORKDIR /opt/holosoma-src
RUN uv pip install -e "src/holosoma_inference[unitree]"


# ─── Stage: inference ─────────────────────────────────────────────────────────
# Default target: terminal image with policy entrypoint, no ROS.
# Override CMD with the policy config + task flags at `docker run` time.
FROM app-deps AS inference

ARG HOLOSOMA_COMMIT=unknown
ARG HOLOSOMA_SOURCE_TREE=unknown
LABEL org.holosoma.commit="${HOLOSOMA_COMMIT}" \
      org.holosoma.source-tree="${HOLOSOMA_SOURCE_TREE}"

ENTRYPOINT ["python3", "src/holosoma_inference/holosoma_inference/run_policy.py"]
CMD ["--help"]

COPY --link src/holosoma_inference /opt/holosoma-src/src/holosoma_inference


# ═════════════════════════════════════════════════════════════════════════════
# ROS 2 Jazzy branch — mirrors no-ROS branch but layered on ros-jazzy.
# Built with:  docker build --target inference-ros ...
# ═════════════════════════════════════════════════════════════════════════════

# ─── Stage: ros-jazzy ─────────────────────────────────────────────────────────
# ROS 2 Jazzy from upstream packages.ros.org (not NVIDIA's L4T repo — we want
# the distribution mainline). Adds ~1.5 GB but is stable across code changes.
FROM python-base AS ros-jazzy

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Add ROS 2 apt repo + key.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
    > /etc/apt/sources.list.d/ros2.list

# Install Jazzy base + FastDDS + CycloneDDS.
#
# Why both RMWs: unitree_sdk2's pybind11 binding bundles CycloneDDS 0.10.2
# (ABI-incompatible with Jazzy's CycloneDDS 0.10.5). Loading both in one
# process crashes. Running rclpy on FastDDS sidesteps the conflict entirely
# — FastDDS and CycloneDDS have disjoint binary symbol spaces, so the two
# middlewares coexist cleanly in one process. unitree_sdk2 keeps its bundled
# CycloneDDS; rclpy uses FastDDS.
#
# Cross-container /cmd_vel interop: FastDDS ↔ CycloneDDS is validated on
# Jazzy for standard message types (TwistStamped tested on aarch64). Pick
# whichever RMW your publisher container uses; no alignment required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-base \
    ros-jazzy-rmw-fastrtps-cpp \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Source ROS on every shell so /opt/ros/jazzy bins are on PATH.
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc


# ─── Stage: long-deps-ros ─────────────────────────────────────────────────────
# NVIDIA math/inference libs on top of the ROS base.
FROM ros-jazzy AS long-deps-ros

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvpl-blas0 \
    libnvpl-lapack0 \
    libcudss0-cuda-13 \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvonnxparsers10 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "/usr/lib/aarch64-linux-gnu/libcudss/13" > /etc/ld.so.conf.d/cudss-13.conf \
    && ldconfig


# ─── Stage: common-deps-ros ───────────────────────────────────────────────────
FROM long-deps-ros AS common-deps-ros

RUN uv pip install "pin>=3.8.0"

COPY docker/apt/zzzz-holosoma-security-updates /etc/apt/apt.conf.d/
RUN apt-get update \
    && apt-get install -y --no-install-recommends unattended-upgrades \
    && unattended-upgrade --verbose \
    && apt-get purge -y unattended-upgrades \
    && rm -rf /var/lib/apt/lists/*


# ─── Stage: app-deps-ros ──────────────────────────────────────────────────────
FROM common-deps-ros AS app-deps-ros

COPY src/holosoma_inference/pyproject.toml src/holosoma_inference/README.md \
    /opt/holosoma-src/src/holosoma_inference/
RUN mkdir -p /opt/holosoma-src/src/holosoma_inference/holosoma_inference && \
    touch /opt/holosoma-src/src/holosoma_inference/holosoma_inference/__init__.py
WORKDIR /opt/holosoma-src
RUN uv pip install -e "src/holosoma_inference[unitree]"

# Wrap the ROS entrypoint in the dependency image so runtime source remains an
# independent linked layer.
#
# DDS coexistence: rclpy uses FastDDS (RMW_IMPLEMENTATION=rmw_fastrtps_cpp),
# unitree_sdk2 uses its bundled CycloneDDS 0.10.2. FastDDS and CycloneDDS
# have disjoint symbol spaces, so they coexist in one process. Sourcing ROS
# puts Jazzy's libddsc 0.10.5 on LD_LIBRARY_PATH, which would win when
# unitree's bundled libddscxx resolves its libddsc dependency and cause an ABI
# mismatch. Prepend unitree's bundle so its matching libddsc wins; rclpy does
# not touch libddsc because it uses FastDDS.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'source /opt/ros/jazzy/setup.bash' \
    'export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/unitree_interface:$LD_LIBRARY_PATH' \
    'exec python3 src/holosoma_inference/holosoma_inference/run_policy.py "$@"' \
    > /usr/local/bin/run_policy_ros \
    && chmod +x /usr/local/bin/run_policy_ros


# ─── Stage: inference-ros ─────────────────────────────────────────────────────
# ROS-enabled terminal target. Entrypoint sources Jazzy then runs the policy
# so Ros2Input can use rclpy + TwistStamped to receive /cmd_vel.
FROM app-deps-ros AS inference-ros

ARG HOLOSOMA_COMMIT=unknown
ARG HOLOSOMA_SOURCE_TREE=unknown
LABEL org.holosoma.commit="${HOLOSOMA_COMMIT}" \
      org.holosoma.source-tree="${HOLOSOMA_SOURCE_TREE}"

ENTRYPOINT ["/usr/local/bin/run_policy_ros"]
CMD ["--help"]

COPY --link src/holosoma_inference /opt/holosoma-src/src/holosoma_inference
# demo_scripts/ros2_velocity_publisher.py is used by the run_shuttle_publisher.sh
# helper to drive /cmd_vel for cross-container DDS tests.
COPY --link demo_scripts /opt/holosoma-src/demo_scripts
