FROM ros:humble

ENV ROS_DISTRO=humble
ENV LD_LIBRARY_PATH="/usr/lib/python3.10/dist-packages/lib/"

WORKDIR /workspace/

# Patch base-image packages (openssl et al.) to pick up security fixes, then
# install dev tooling. apt-get upgrade keeps the ros:humble base current with
# Ubuntu security updates rather than shipping whatever was baked into the base.
RUN apt-get update && apt-get upgrade -y && apt-get install -y python3-pip vim \
    && rm -rf /var/lib/apt/lists/*

# Install ohmybash + dev tooling
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"

# Install uv (the project's build backend). holosoma_inference declares
# `build-backend = "uv_build"` in pyproject.toml, so the editable install below
# is driven by uv's build backend rather than setuptools. Pinning to :latest —
# bump explicitly here for a deterministic version.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install holosoma_inference (as the last step, since the cache of this layer will get invalidated frequently)
# NOTE: This assumes the build context is holosoma/src/ (see build.sh)
COPY src/holosoma_inference /workspace/holosoma/src/holosoma_inference/
# `uv pip install --system` builds via the uv_build backend declared in
# pyproject.toml and installs into the ROS image's system interpreter.
RUN uv pip install --system -e "/workspace/holosoma/src/holosoma_inference[unitree]"
RUN uv pip install --system -e "/workspace/holosoma/src/holosoma_inference[booster]"
RUN uv pip install --system pin

WORKDIR /workspace/holosoma
