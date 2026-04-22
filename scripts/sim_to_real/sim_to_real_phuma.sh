source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-future-motion \
    --task.model-path ./pretrained/phuma_g1_29dof/model_106000.onnx \
    --task.motion-file-path ./selected_motions/PHUMA/Horizontal/humanml_007110_chunk_0000.npz \
    --task.use-joystick \
    --task.rl-rate 50 \
    --task.interface eth0
