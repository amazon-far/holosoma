# Arm Tracking with Standing Balance - Implementation

This implementation provides a complete arm-tracking controller with standing balance using residual RL for the G1 humanoid robot in IsaacSim.

## Architecture

The system uses **residual RL** where:
1. A base controller provides nominal actions (standing + arm tracking)
2. An RL policy learns residual corrections on top of the base controller
3. Final actions = base_controller_output + scale × rl_residuals

## Files Created

```
src/holosoma/holosoma/
├── envs/arm_tracking/
│   ├── __init__.py
│   └── arm_tracking_manager.py          # Main task manager
├── managers/
│   ├── action/terms/
│   │   └── arm_tracking.py              # Residual action term
│   ├── command/terms/
│   │   └── arm_tracking.py              # Arm target commands
│   ├── observation/terms/
│   │   └── arm_tracking.py              # Observations
│   ├── reward/terms/
│   │   └── arm_tracking.py              # Reward functions
│   └── termination/terms/
│       └── arm_tracking.py              # Termination conditions
└── config_values/arm_tracking/g1/
    ├── __init__.py
    ├── action.py
    ├── command.py
    ├── curriculum.py
    ├── experiment.py
    ├── observation.py
    ├── randomization.py
    ├── reward.py
    └── termination.py
```

## Quick Start

### Training

```bash
# Setup IsaacSim environment
source scripts/source_isaacsim_setup.sh

# Train with PPO
python src/holosoma/holosoma/train_agent.py \
    exp:g1-arm-tracking \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1

# Train with FastSAC
python src/holosoma/holosoma/train_agent.py \
    exp:g1-arm-tracking-fast-sac \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1
```

## Implementing Your Base Controller

The current implementation uses a **placeholder controller** that:
- Uses default standing pose for legs
- Uses commanded targets for arms

To implement your own controller, edit `managers/action/terms/arm_tracking.py`:

```python
class ResidualArmTrackingAction(ActionTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # ... existing code ...
        
        # REPLACE THIS:
        self.base_controller = None
        
        # WITH YOUR CONTROLLER:
        self.base_controller = YourStandingBalanceController(env)
    
    def setup(self):
        # ... existing code ...
        
        # ADD THIS:
        if self.base_controller is not None:
            self.base_controller.setup()
    
    def process_actions(self, actions: torch.Tensor) -> None:
        # ... existing code ...
        
        # REPLACE THIS:
        # arm_command = self._env.command_manager.get_state("arm_command")
        # if arm_command:
        #     self.base_actions[:] = self._env.default_dof_pos
        #     arm_indices = self._env.arm_dof_indices
        #     self.base_actions[:, arm_indices] = arm_command.target_arm_pos
        
        # WITH THIS:
        if self.base_controller is not None:
            self.base_actions[:] = self.base_controller.compute_actions()
        else:
            # fallback to default behavior
            # ... existing code ...
```

### Controller Template

```python
class YourStandingBalanceController:
    """Your base controller for standing + arm tracking."""
    
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        
    def setup(self):
        """Initialize controller buffers."""
        # Your initialization here
        pass
    
    def compute_actions(self) -> torch.Tensor:
        """
        Compute base actions for standing + arm tracking.
        
        Returns:
            torch.Tensor: [num_envs, num_dof] joint position targets
        """
        actions = torch.zeros((self.num_envs, self.env.num_dof), device=self.device)
        
        # 1. Compute leg joint targets for standing balance
        leg_targets = self._compute_standing_balance()
        actions[:, self.env.leg_dof_indices] = leg_targets
        
        # 2. Compute arm joint targets for tracking
        arm_command = self.env.command_manager.get_state("arm_command")
        actions[:, self.env.arm_dof_indices] = arm_command.target_arm_pos
        
        return actions
    
    def _compute_standing_balance(self) -> torch.Tensor:
        """
        Compute leg joint targets to maintain standing balance.
        
        Implement your control strategy here:
        - PD control to default pose
        - CoM/ZMP control
        - Whole-body optimization
        """
        # YOUR IMPLEMENTATION HERE
        # For now, return default leg pose
        return self.env.default_dof_pos[:, self.env.leg_dof_indices]
```

## Configuration

### Reward Weights

Edit `config_values/arm_tracking/g1/reward.py`:

```python
g1_arm_tracking_reward = RewardManagerCfg(
    terms={
        "arm_tracking": RewardTermCfg(weight=1.0),      # Arm tracking accuracy
        "standing_balance": RewardTermCfg(weight=0.5),  # Upright orientation
        "base_height": RewardTermCfg(weight=0.3),       # Standing height
        "action_smoothness": RewardTermCfg(weight=0.1), # Small residuals
    }
)
```

### Residual Action Scale

Edit `config_values/arm_tracking/g1/action.py`:

```python
g1_arm_tracking_action = ActionManagerCfg(
    terms={
        "residual_arm_tracking": ActionTermCfg(
            params={
                "action_scale": 0.1,   # How much influence RL has
                "clip_actions": 1.0,   # Limit residual magnitude
            },
        )
    }
)
```

### Termination Thresholds

Edit `managers/termination/terms/arm_tracking.py`:

```python
def base_height_termination(env, **kwargs):
    min_height = 0.3  # Adjust for your robot
    # ...

def base_orientation_termination(env, **kwargs):
    tipped = base_quat[:, 3] < 0.7  # Adjust threshold
    # ...
```

## Observations

The policy receives:
- Base orientation (4D quaternion)
- Base angular velocity (3D)
- Joint positions (num_dof)
- Joint velocities (num_dof)
- Arm targets (num_arm_dof)

Total dimension: 4 + 3 + 2×num_dof + num_arm_dof

## Customizing Arm Targets

Edit `managers/command/terms/arm_tracking.py`:

```python
class ArmTrackingCommand(CommandTermBase):
    def reset(self, env_ids: torch.Tensor | None) -> None:
        """Sample new arm targets."""
        # Current: Random sampling within joint limits
        # Customize: Fixed poses, trajectories, etc.
        
        # Example: Fixed target pose
        # self.target_arm_pos[env_ids] = torch.tensor([...], device=self.device)
        
        # Example: Sinusoidal trajectory
        # phase = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        # self.target_arm_pos[env_ids] = torch.sin(phase).unsqueeze(-1) * amplitude
```

## Debugging Tips

1. **Test base controller alone**: Set `action_scale=0.0` to disable RL
2. **Visualize in IsaacSim**: Check robot maintains standing and tracks arms
3. **Monitor metrics**: Watch `arm_tracking_error`, base height, residual magnitude
4. **Verify DOF indices**: Print `env.arm_dof_indices` and `env.leg_dof_indices`

## DOF Identification

The system identifies arm/leg joints by keywords:

**Arm joints**: `["shoulder", "elbow", "wrist", "arm"]`
**Leg joints**: `["hip", "knee", "ankle", "leg"]`

If your robot uses different naming, edit `envs/arm_tracking/arm_tracking_manager.py`:

```python
def _get_arm_dof_indices(self):
    arm_keywords = ["your", "custom", "keywords"]  # Customize here
    # ...

def _get_leg_dof_indices(self):
    leg_keywords = ["your", "custom", "keywords"]  # Customize here
    # ...
```

## Next Steps

1. **Implement your base controller** in `managers/action/terms/arm_tracking.py`
2. **Tune hyperparameters** (reward weights, action scale, termination thresholds)
3. **Customize arm targets** in `managers/command/terms/arm_tracking.py`
4. **Test incrementally** (base controller → small residuals → full RL)
5. **Deploy to real robot** using the inference pipeline

## References

- Main planner: `planner.md`
- Locomotion example: `envs/locomotion/locomotion_manager.py`
- Whole-body tracking: `envs/wbt/wbt_manager.py`
- Training guide: `src/holosoma/README.md`
