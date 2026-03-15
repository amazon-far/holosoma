source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-future-motion-heightmap \
    terrain:terrain-load-obj \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=statue_wbt \
    --logger.name=g1_huddle_heightmap \
    --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/unreal_engine/huddle_walk.npz" \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.05]" \
    --terrain.terrain-term.tile-mesh False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/unreal_engine/huddle_walk.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --termination.terms.bad_tracking.params.bad_ref_pos_threshold 0.5 \
    --termination.terms.bad_tracking.params.bad_ref_ori_threshold 0.8 \
    --termination.terms.bad_tracking.params.bad_motion_body_pos_threshold 0.5 \
    --simulator.config.scene.env_spacing=0.0 \
    --training.num_envs 10 \
    --training.headless False \
    robot:g1-29dof-diverse-action-scale
