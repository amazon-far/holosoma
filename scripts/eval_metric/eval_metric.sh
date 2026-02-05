# Isaac Sim headless often fails with "Failed to create simulation view backend".
# Run without headless (GUI) when possible. On headless servers, try: xvfb-run -a ./scripts/eval_metric/eval_metric.sh
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_metric.py \
    --checkpoint=/home/kyungminlee/work/holosoma/logs/WholeBodyTracking/20260130_092605-g1_29dof_wbt_manager-unreal_engine/model_20000.pt \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.max_eval_steps -1 \
    --training.headless True
