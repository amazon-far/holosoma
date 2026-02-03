source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path /home/kyungminlee/work/holosoma/logs/WholeBodyTracking/20260130_092605-g1_29dof_wbt_manager-unreal_engine/exported/model_28000.onnx \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo