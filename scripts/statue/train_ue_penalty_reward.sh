source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=ablation \
    --logger.name=holosoma_wbt_penalty_dof_acc \
    --logger.video.enabled False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/statue_v0_1/whole_body_tracking/walk_chunk_0000_mj.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --reward.terms.penalty_dof_acc.weight=-3e-7 \

