#!/bin/bash
# Sim-to-sim: PhUMA checkpoint on a Front Kick motion.
#
# STEP 1 — In a SEPARATE terminal, start the MuJoCo simulator:
#   source scripts/source_mujoco_setup.sh
#   python src/holosoma/holosoma/run_sim.py robot:g1-29dof
#
# STEP 2 — In THIS terminal, run this script:
#   bash scripts/sim_to_sim/sim_to_sim_phuma_kick.sh
#
# STEP 3 — Controls:
#   1. Press Enter when prompted (stiff hold mode)
#   2. In MuJoCo window: press 8 to lower gantry until robot touches ground
#   3. In MuJoCo window: press 9 to remove gantry
#   4. Wait a few seconds for stabilization
#   5. In this terminal: press ] to start the policy
#   6. In this terminal: press s to start the motion clip

source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-future-motion \
    --task.model-path ./pretrained/phuma_g1_29dof/model_106000.onnx \
    --task.motion-file-path ./selected_motions/PHUMA/Horizontal/humanml_007110_chunk_0000.npz \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo \
    --task.save-debug
