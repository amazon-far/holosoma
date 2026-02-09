source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=/home/nas4_user/kyungminlee/work/holosoma/logs/WholeBodyTracking/20260209_014018-statue_v0_1_wbt_stiff-statue_wbt/model_06000.pt \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.headless True