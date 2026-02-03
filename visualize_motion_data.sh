source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1