source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:statue-v1-rmd-wbt-future-motion \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=statue_wbt \
    --logger.name=g1_ramp \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.02]" \
    --terrain.terrain-term.tile-mesh False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/statue_v1_rmd/whole_body_tracking/unreal_engine/walk_statue_v1_rmd_50fps.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --simulator.config.scene.env_spacing=0.0 \
    --training.num_envs 10 \
    --training.headless False \
