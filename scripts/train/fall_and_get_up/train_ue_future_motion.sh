source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-future-motion \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=ablation \
    --logger.name=holosoma_wbt_future_motion_fall_and_getup \
    --logger.video.enabled False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/fall_and_getup.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \

