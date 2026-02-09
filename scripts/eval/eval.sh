source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=/home/kyungminlee/work/holosoma/logs/WholeBodyTracking/20260130_092605-g1_29dof_wbt_manager-unreal_engine/model_28000.pt \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.headless True