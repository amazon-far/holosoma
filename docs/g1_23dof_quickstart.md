# G1 23-DOF Quick Start

Commands for training, evaluating, and running sim-to-sim with the Unitree G1 (23 DOF configuration).

## Training

### Minimal

```bash
# FastSAC (recommended)
python -m holosoma.train_agent exp:g1-23dof-fast-sac simulator:mjwarp

# PPO
python -m holosoma.train_agent exp:g1-23dof simulator:mjwarp
```

### With Logging

```bash
python -m holosoma.train_agent exp:g1-23dof-fast-sac simulator:mjwarp logger:wandb
```

### Common Options

```bash
python -m holosoma.train_agent exp:g1-23dof-fast-sac simulator:mjwarp logger:wandb \
    --training.num-envs=4096 \
    --training.seed=42

# Flat terrain (faster, good for initial debugging)
python -m holosoma.train_agent exp:g1-23dof-fast-sac simulator:mjwarp \
    terrain:terrain-locomotion-plane

# IsaacGym instead of MJWarp
python -m holosoma.train_agent exp:g1-23dof-fast-sac simulator:isaacgym logger:wandb
```

## Evaluation

### Minimal

```bash
# Local checkpoint
python -m holosoma.eval_agent --checkpoint path/to/model.pt

# W&B checkpoint
python -m holosoma.eval_agent --checkpoint wandb://entity/project/run_id/model.pt
```

### Common Options

```bash
# Override simulator for evaluation
python -m holosoma.eval_agent simulator:mjwarp --checkpoint path/to/model.pt

# Export ONNX for deployment
python -m holosoma.eval_agent --checkpoint path/to/model.pt --training.export-onnx=True

# Non-headless (with viewer)
python -m holosoma.eval_agent --checkpoint path/to/model.pt --training.headless=False
```

### Keyboard Controls (during evaluation)

| Key | Action |
|-----|--------|
| `w/s` | Forward/backward velocity |
| `a/d` | Left/right velocity |
| `q/e` | Turn left/right |
| `z` | Zero velocity |

## Sim-to-Sim (MuJoCo)

Run a trained policy in standalone MuJoCo for deployment testing.

### Terminal 1: MuJoCo Environment

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-23dof
```

### Terminal 2: Policy Inference

```bash
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-23dof-loco \
    --task.model-path path/to/model.onnx \
    --task.no-use-joystick \
    --task.interface lo
```

### Deployment Steps

1. In MuJoCo window: Press `8` to lower gantry until robot touches ground
2. In MuJoCo window: Press `9` to remove gantry
3. In policy terminal: Press `]` to activate policy
4. In policy terminal: Press `=` to enter walking mode
5. Control with `w/a/s/d` (linear) and `q/e` (angular)
