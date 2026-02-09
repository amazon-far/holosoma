source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_metric.py \
    --checkpoint=/home/nas4_user/kyungminlee/work/holosoma/logs/WholeBodyTracking/20260205_135615-g1_29dof_wbt_manager-ablation-jump_penalty_dof_acc/model_12000.pt \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.headless True
