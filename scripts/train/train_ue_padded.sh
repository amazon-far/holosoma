source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project holosoma \
    --logger.group=unreal_engine \
    --logger.name=g1_ue_walk_45_padded_4090 \
    --logger.video.enabled False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/g1_walk_45_50fps_padded.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \

