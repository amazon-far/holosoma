source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=/home/nas4_user/kyungminlee/work/holosoma/logs/WholeBodyTracking/20260205_134525-g1_29dof_wbt-ablation_baseball_future_motion/model_12000.pt \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.headless True