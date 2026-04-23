source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=/home/nas4_user/kyungminlee/work/holosoma/logs/MotionTracking/20260315_g1_huddle/model_04000.pt \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.num_envs 1 \
