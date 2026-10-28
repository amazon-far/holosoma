#!/bin/bash
# CI runs this inside the combined holosoma image on a GPU runner.
#
# End-to-end: train a mini policy, export it, then run it under hsinference. It needs BOTH the
# training env and the inference env in one image, which only the combined image has. Training
# happens in hsmujoco on the MuJoCo-Warp backend (GPU); the test re-sources hsinference itself.
set -ex

cd /workspace/holosoma

# Headless GL for the classic MuJoCo camera renderer; see mujoco_ci_tests.sh for the full rationale.
export MUJOCO_GL=egl

source scripts/source_mujoco_setup.sh
pip install -e 'src/holosoma[unitree,booster]'
pip install -e src/holosoma_inference

# requires_inference is the only marker that selects the train -> export -> run_policy tests; the
# training backend comes from the pinned `simulator:` token in the test, not from this env.
pytest -s --strict-markers --ignore=thirdparty --ignore=src/holosoma_inference -m "requires_inference"
