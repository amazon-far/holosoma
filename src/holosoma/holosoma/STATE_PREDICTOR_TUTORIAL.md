# Single-Action State Predictor

This guide describes the first-pass transition model workflow for Holosoma.

## Goal

Train a supervised predictor that maps:

$$
(s_t, a_t) \rightarrow s_{t+1}
$$

For the current first pass, the state comes from `actor_obs`, the action comes from the loaded policy checkpoint, and the target state is the post-action observation. When an episode terminates, the training target uses `final_observations` when available so the label stays aligned with the executed action.

## Source Checkpoint

Use the existing FastSAC policy checkpoint as the rollout policy source:

`logs/WholeBodyTracking/20260329_023617-t1_23dof_wbt_fast_sac_manager-locomotion/model_0400000.pt`

The trainer loads the checkpoint through the same `load_checkpoint()` / `load_saved_experiment_config()` path used by the existing evaluation flow.

## Environment Setup

Before launching collection, source the IsaacSim environment and set the Vulkan ICD:

```bash
source scripts/source_isaacsim_setup.sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

## Training Command

```bash
source scripts/source_isaacsim_setup.sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
python src/holosoma/holosoma/train_state_predictor.py \
    --checkpoint logs/WholeBodyTracking/20260329_023617-t1_23dof_wbt_fast_sac_manager-locomotion/model_0400000.pt \
    --training.headless \
    --training.num-envs 1 \
    --data.rollout-steps 8192 \
    --data.validation-ratio 0.1 \
    --optim.epochs 30 \
    --optim.batch-size 256 \
    simulator:isaacsim
```

For better predictor quality, this recommended setting uses more rollout samples and a longer training run. If you need a faster smoke test, reduce `--data.rollout-steps` back to 4096 and `--optim.epochs` back to 20.

## Data Format

The cached dataset stores:

- `states`: current state tensor
- `actions`: action tensor from the policy checkpoint
- `next_states`: target post-action state tensor
- `dones`: termination flags from the simulator

The cache is written next to the predictor experiment logs by default, and the script reuses it unless `--data.force-collect True` is set.

## Outputs

The training run writes:

- `state_predictor_config.yaml`: predictor config snapshot
- `state_predictor_transitions.pt`: cached rollout data
- `best.pt`: best validation checkpoint
- `model_*.pt`: periodic and final predictor checkpoints

The current trainer keeps the collected rollout data and training loop in-process, and it intentionally skips the IsaacSim shutdown path at the end of collection to avoid a known headless hang. Seeing the simulator logs stop without an explicit shutdown message is expected as long as the process exits with code 0 and the output files above are written.

## Extending to More Actions

The current structure keeps the data source, predictor model, and training config separate, so adding more action-conditioned predictors later only requires swapping the collected action source and expanding the dataset metadata.