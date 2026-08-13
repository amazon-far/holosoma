# Make targets for the Thor (Jetson AGX, JetPack 7.1) holosoma_inference
# Docker images. Thin wrappers over docker compose so the common commands
# are a single token.
#
# Usage (from docker/thor/):
#   make                       # help
#   make inference             # build no-ROS image
#   make inference-ros         # build ROS image
#   make both                  # build both
#   make run-inference ARGS='inference:g1-29dof-loco --task.interface eth0'
#   make run-inference-ros ARGS='inference:g1-29dof-loco --task.velocity-input ros2 --task.state-input interface --task.interface eth0'
#   make clean                 # remove both images

.PHONY: help inference inference-ros both run-inference run-inference-ros clean

COMPOSE := docker compose -f compose.yaml

help:
	@echo "Thor Docker targets (run from docker/thor/):"
	@echo "  make inference              Build holosoma-thor-inference (no ROS)"
	@echo "  make inference-ros          Build holosoma-thor-inference-ros (ROS 2 Jazzy)"
	@echo "  make both                   Build both images"
	@echo "  make run-inference          Run no-ROS image (pass ARGS='...')"
	@echo "  make run-inference-ros      Run ROS image (pass ARGS='...')"
	@echo "  make clean                  Remove both images"

inference:
	$(COMPOSE) build inference

inference-ros:
	$(COMPOSE) build inference-ros

both:
	$(COMPOSE) build

run-inference:
	$(COMPOSE) run --rm inference $(ARGS)

run-inference-ros:
	$(COMPOSE) run --rm inference-ros $(ARGS)

clean:
	-docker rmi holosoma-thor-inference holosoma-thor-inference-ros
