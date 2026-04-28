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
#   common-deps    — Python deps that rarely bump (numpy, scipy, pin, etc.).
#   app-deps       — holosoma_inference + unitree_sdk2. Rebuilds every commit.
#   inference      — entrypoint + cmd (terminal, no-ROS target).
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
# Python dependencies of holosoma_inference that rarely change.
# Keep this layer separate from app source so edits to holosoma code don't
# invalidate these installs.
FROM long-deps AS common-deps

# Core deps from src/holosoma_inference/setup.py install_requires, plus
# pinocchio for WBT policy support (not in setup.py — installed separately,
# matching scripts/setup_inference_via_uv.sh).
#
# DRIFT RISK: this list duplicates setup.py's install_requires because we
# want the layered-cache benefit (deps install ~once, source rebuilds
# invalidate only the final layer). When setup.py deps change, this list
# must be updated too. Source of truth: src/holosoma_inference/setup.py.
RUN uv pip install \
    "pydantic" \
    "loguru" \
    "netifaces" \
    "onnx" \
    "onnxruntime" \
    "scipy" \
    "sshkeyboard" \
    "termcolor" \
    "pyyaml" \
    "tyro>=0.10.0a4" \
    "wandb" \
    "zmq" \
    "defusedxml" \
    "evdev" \
    "pin>=3.8.0"


# ─── Stage: app-deps ──────────────────────────────────────────────────────────
# Install amazon-far unitree_sdk2 wheel + copy holosoma_inference source.
# This is the only layer that changes on every commit.
FROM common-deps AS app-deps

# unitree_sdk2 Python 3.12 aarch64 wheel from amazon-far's GitHub release.
# Bump UNITREE_SDK2_VERSION in one place when a new release is published.
ARG UNITREE_SDK2_VERSION=0.1.3
ARG UNITREE_SDK2_WHL=unitree_sdk2-${UNITREE_SDK2_VERSION}-cp312-cp312-linux_aarch64.whl
RUN wget -q -O "/tmp/${UNITREE_SDK2_WHL}" \
    "https://github.com/amazon-far/unitree_sdk2/releases/download/${UNITREE_SDK2_VERSION}/${UNITREE_SDK2_WHL}" \
    && uv pip install "/tmp/${UNITREE_SDK2_WHL}" \
    && rm "/tmp/${UNITREE_SDK2_WHL}"

# holosoma_inference source (editable install, --no-deps since common-deps
# handled the runtime deps above).
COPY src/holosoma_inference /opt/holosoma-src/src/holosoma_inference
WORKDIR /opt/holosoma-src
RUN uv pip install --no-deps -e src/holosoma_inference


# ─── Stage: inference ─────────────────────────────────────────────────────────
# Default target: terminal image with policy entrypoint, no ROS.
# Override CMD with the policy config + task flags at `docker run` time.
FROM app-deps AS inference

WORKDIR /opt/holosoma-src

ENTRYPOINT ["python3", "src/holosoma_inference/holosoma_inference/run_policy.py"]
CMD ["--help"]


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
# Same Python deps as common-deps (kept verbatim so this stage caches
# independently — cross-branch cache sharing is not possible with different
# parent layers).
FROM long-deps-ros AS common-deps-ros

RUN uv pip install \
    "pydantic" \
    "loguru" \
    "netifaces" \
    "onnx" \
    "onnxruntime" \
    "scipy" \
    "sshkeyboard" \
    "termcolor" \
    "pyyaml" \
    "tyro>=0.10.0a4" \
    "wandb" \
    "zmq" \
    "defusedxml" \
    "evdev" \
    "pin>=3.8.0"


# ─── Stage: app-deps-ros ──────────────────────────────────────────────────────
FROM common-deps-ros AS app-deps-ros

ARG UNITREE_SDK2_VERSION=0.1.3
ARG UNITREE_SDK2_WHL=unitree_sdk2-${UNITREE_SDK2_VERSION}-cp312-cp312-linux_aarch64.whl
RUN wget -q -O "/tmp/${UNITREE_SDK2_WHL}" \
    "https://github.com/amazon-far/unitree_sdk2/releases/download/${UNITREE_SDK2_VERSION}/${UNITREE_SDK2_WHL}" \
    && uv pip install "/tmp/${UNITREE_SDK2_WHL}" \
    && rm "/tmp/${UNITREE_SDK2_WHL}"

COPY src/holosoma_inference /opt/holosoma-src/src/holosoma_inference
# demo_scripts/ros2_velocity_publisher.py is used by the run_shuttle_publisher.sh
# helper to drive /cmd_vel for cross-container DDS tests.
COPY demo_scripts /opt/holosoma-src/demo_scripts
WORKDIR /opt/holosoma-src
RUN uv pip install --no-deps -e src/holosoma_inference


# ─── Stage: inference-ros ─────────────────────────────────────────────────────
# ROS-enabled terminal target. Entrypoint sources Jazzy then runs the policy
# so Ros2Input can use rclpy + TwistStamped to receive /cmd_vel.
FROM app-deps-ros AS inference-ros

WORKDIR /opt/holosoma-src

# Wrap entrypoint to source ROS before execing python.
#
# DDS coexistence: rclpy uses FastDDS (RMW_IMPLEMENTATION=rmw_fastrtps_cpp),
# unitree_sdk2 uses its bundled CycloneDDS 0.10.2. FastDDS and CycloneDDS
# have disjoint symbol spaces, so they coexist in one process — BUT sourcing
# ROS puts Jazzy's libddsc 0.10.5 on LD_LIBRARY_PATH, which would win when
# unitree's bundled libddscxx resolves its libddsc dep, causing an ABI
# mismatch (unitree built against 0.10.2 headers). Prepend unitree's bundle
# dir to LD_LIBRARY_PATH so the bundled libddsc wins for unitree's use.
# rclpy doesn't touch libddsc at all (it's on FastDDS), so no conflict.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'source /opt/ros/jazzy/setup.bash' \
    'export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/unitree_interface:$LD_LIBRARY_PATH' \
    'exec python3 src/holosoma_inference/holosoma_inference/run_policy.py "$@"' \
    > /usr/local/bin/run_policy_ros \
    && chmod +x /usr/local/bin/run_policy_ros

ENTRYPOINT ["/usr/local/bin/run_policy_ros"]
CMD ["--help"]
