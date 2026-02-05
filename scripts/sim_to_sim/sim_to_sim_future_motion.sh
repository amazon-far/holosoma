source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-future-motion \
    --task.model-path /home/kyungminlee/work/holosoma/logs/WholeBodyTracking/20260203_130104-g1_29dof_wbt_future_motion-ablation/exported/model_20000.onnx \
    --task.motion-file-path /home/kyungminlee/work/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/g1_walk_45_50fps.npz \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo --task.save-debug