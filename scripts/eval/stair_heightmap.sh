source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=/home/kyungminlee/work/holosoma/logs/MotionTracking/20260315_g1_stair_30depth_heightmap/model_04000.pt \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --training.num_envs 1 \
    --training.headless False \
    --simulator.config.sim.render_interval 4
