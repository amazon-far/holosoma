source scripts/source_isaacsim_setup.sh

MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof"

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-multi-terrain-future-motion-heightmap-videomimic-add \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=statue_wbt \
    --logger.name=g1_videomimic_multi_terrain_future_motion_heightmap_xyz_add_finetune_2048 \
    --logger.video.enabled False \
    --training.checkpoint="pretrained/phuma_g1_29dof/model_106000.pt" \
    --training.finetune True \
    --algo.config.load_optimizer False \
    --simulator.config.videomimic-height-map.map-height 11 \
    --simulator.config.videomimic-height-map.map-width 17 \
    --simulator.config.videomimic-height-map.resolution 0.1 \
    --simulator.config.videomimic-height-map.num-channels 3 \
    --terrain.terrain-term.obj-dir="$MOTION_DIR" \
    --terrain.terrain-term.num-rows 1 \
    --terrain.terrain-term.obj-max-faces-per-tile 0 \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$MOTION_DIR" \
    --command.setup_terms.motion_command.params.motion_config.max_motions 130 \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.init_root_offset="[0.0, 0.0, 0.1]" \
    --training.num_envs 2048 \
    --simulator.config.scene.env_spacing=0.0 \
    --algo.config.save_interval 1000 \
    --algo.config.eval_interval 1000 \
    robot:g1-29dof-diverse-action-scale
