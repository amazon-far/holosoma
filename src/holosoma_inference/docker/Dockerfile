FROM ros:humble

ENV ROS_DISTRO=humble
ENV LD_LIBRARY_PATH="/usr/lib/python3.10/dist-packages/lib/"

WORKDIR /workspace/

# Install ohmybash + dev tooling
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"
RUN apt-get update && apt-get install -y python3-pip vim

# Install holosoma_inference (as the last step, since the cache of this layer will get invalidated frequently)
# NOTE: This assumes the build context is holosoma/src/ (see build.sh)
COPY src/holosoma_inference /workspace/holosoma/src/holosoma_inference/
RUN pip3 install -e "/workspace/holosoma/src/holosoma_inference[unitree]"
RUN pip3 install -e "/workspace/holosoma/src/holosoma_inference[booster]"
RUN pip3 install pin

WORKDIR /workspace/holosoma
