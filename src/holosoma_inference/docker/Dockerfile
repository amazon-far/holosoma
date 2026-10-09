# syntax=docker/dockerfile:1.7
# Inference dependencies are installed from metadata at the final source path;
# Python-only changes add one linked source layer.
FROM ros:humble AS dependencies

ENV ROS_DISTRO=humble
ENV LD_LIBRARY_PATH="/usr/lib/python3.10/dist-packages/lib/"

WORKDIR /workspace/

# Install development tooling.
RUN apt-get update && apt-get install -y python3-pip vim \
    && rm -rf /var/lib/apt/lists/*

# Install ohmybash + dev tooling
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"

# Install uv (the project's build backend). holosoma_inference declares
# `build-backend = "uv_build"` in pyproject.toml, so the editable install below
# is driven by uv's build backend rather than setuptools. Pinning to :latest —
# bump explicitly here for a deterministic version.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install editable metadata against a placeholder at the final source path.
COPY src/holosoma_inference/pyproject.toml src/holosoma_inference/README.md \
    /workspace/holosoma/src/holosoma_inference/
RUN mkdir -p /workspace/holosoma/src/holosoma_inference/holosoma_inference && \
    touch /workspace/holosoma/src/holosoma_inference/holosoma_inference/__init__.py
RUN uv pip install --system -e "/workspace/holosoma/src/holosoma_inference[unitree,booster]"
RUN uv pip install --system pin

COPY docker/apt/zzzz-holosoma-security-updates /etc/apt/apt.conf.d/
RUN apt-get update \
    && apt-get install -y --no-install-recommends unattended-upgrades \
    && unattended-upgrade --verbose \
    && apt-get purge -y unattended-upgrades \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/holosoma

LABEL org.holosoma.layer="dependencies"

FROM dependencies AS runtime

ARG HOLOSOMA_COMMIT=unknown
ARG HOLOSOMA_SOURCE_TREE=unknown
LABEL org.holosoma.commit="${HOLOSOMA_COMMIT}" \
      org.holosoma.source-tree="${HOLOSOMA_SOURCE_TREE}" \
      org.holosoma.layer="runtime"

COPY --link src/holosoma_inference /workspace/holosoma/src/holosoma_inference
