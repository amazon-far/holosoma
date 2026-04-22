source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-future-motion \
    simulator:isaacsim \
    terrain:terrain-load-obj \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=VideoMimic \
    --logger.name=g1_videomimic_stair \
    --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof/holosoma_stairs_cam01_frame_0_140_subsample_1/motion_for_holosoma_with_mesh.npz" \
    --terrain.terrain-term.tile-mesh False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof/holosoma_stairs_cam01_frame_0_140_subsample_1/motion_for_holosoma_with_mesh.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.init_root_offset="[0.0, 0.0, 0.1]" \
    --termination.terms.bad_tracking.params.bad_ref_pos_threshold 0.5 \
    --termination.terms.bad_tracking.params.bad_ref_ori_threshold 0.8 \
    --termination.terms.bad_tracking.params.bad_motion_body_pos_threshold 0.5 \
    --simulator.config.scene.env_spacing=0.0 \
    --logger.video.enabled False \
    --algo.config.save_interval 500 \
    robot:g1-29dof-diverse-action-scale
