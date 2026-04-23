source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=/home/kyungminlee/work/holosoma/logs/WholeBodyTracking/20260203_130104-g1_29dof_wbt_future_motion-ablation/model_20000.pt \
    --command.setup_terms.motion_command.params.motion_config.motion_file "holosoma/data/motions/g1_29dof/whole_body_tracking/g1_walk_45_50fps.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.save-debug True