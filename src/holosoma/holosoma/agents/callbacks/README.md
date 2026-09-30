# Eval Callbacks

Grid-based multi-condition evaluation sweep system. Runs a policy across a Cartesian product of parameters (velocities × push forces × payload masses) in a single eval, one env per condition, recording everything to NPZ.

## Architecture

```
EvalCallbacksConfig (eval_callback.py)
├── RecordingCallbackConfig → EvalRecordingCallback
│   └── owns GridConditionManager
├── GridEvalVelocityCallbackConfig → GridEvalVelocityCallback
├── GridEvalPushCallbackConfig → GridEvalPushCallback
└── GridEvalPayloadCallbackConfig → GridEvalPayloadCallback
```

## How It Works

### Lifecycle

1. `eval_agent.py` calls `EvalCallbacksConfig.collect_active_callbacks()` → instantiates enabled callbacks in field-declaration order.
2. **`on_pre_evaluate_policy`**: Recording creates a `GridConditionManager`. Grid callbacks register their sweep axes with it.
3. **First env step** (`ensure_finalized()`): The condition manager expands the Cartesian product of all axes, validates `num_envs >= num_conditions`, and assigns each env index to one condition.
4. **Each step**: Velocity callback overrides commands, push/payload callbacks inject forces, recording captures state for all condition envs.
5. **`on_post_evaluate_policy`**: Recording saves everything to a compressed NPZ.

### Deferred Setup Pattern

Grid callbacks can't fully initialize during `on_pre_evaluate_policy` because they need the finalized condition list (which requires all axes to be registered first). Instead:
- `on_pre_evaluate_policy`: register axes with the condition manager
- First env step: call `recording_cb.ensure_finalized()`, then read `condition_manager.conditions` to set up per-env state

### Termination Suppression

Recording disables env resets (`_check_termination` is monkey-patched to no-op). Fallen robots stay down so the full trajectory is captured for every condition.

## Files

| File | Purpose |
|------|---------|
| `base_callback.py` | `RLEvalCallback` base class with lifecycle hooks and helpers (`_get_env`, `_require_recording_cb`) |
| `grid_conditions.py` | `GridConditionManager` — Cartesian product of sweep axes, per-env condition assignment |
| `recording.py` | `EvalRecordingCallback` — central hub: owns condition manager, buffers, metadata, NPZ output |
| `grid_eval_velocity.py` | Per-env velocity command injection with warmup period |
| `grid_eval_push.py` | Deterministic push sweeps (body × direction × gait_phase × force) with substep force injection |
| `grid_eval_payload.py` | Payload sweeps (body_group × mass) with constant downward forces |
| `gait_analysis.py` | `GaitAnalyser` — foot-height-based gait cycle counting and phase detection |
| `sim_utils.py` | Body resolution and external force helpers for IsaacSim |

## Usage Examples

Velocity sweep (4 conditions):
```bash
python -m holosoma.eval_agent ... \
    --recording.config.enabled \
    --grid-velocity.config.enabled \
    --grid-velocity.config.lin-vel-x 0.25 0.5 0.75 1.0 \
    --training.num-envs 4
```

Push sweep (4 vel × 2 bodies × 4 dirs × 4 phases = 128 conditions):
```bash
python -m holosoma.eval_agent ... \
    --recording.config.enabled \
    --grid-velocity.config.enabled \
    --grid-velocity.config.lin-vel-x 0.5 \
    --grid-push.config.enabled \
    --grid-push.config.body-names torso_link pelvis \
    --grid-push.config.body-labels torso pelvis \
    --grid-push.config.directions forward backward left right \
    --grid-push.config.gait-phases swing_to_stance stance_to_swing mid_stance mid_swing \
    --grid-push.config.force-n 150.0 \
    --training.num-envs 128
```

Payload sweep (4 vel × 2 groups × 3 masses = 24 conditions):
```bash
python -m holosoma.eval_agent ... \
    --recording.config.enabled \
    --grid-velocity.config.enabled \
    --grid-velocity.config.lin-vel-x 0.25 0.5 0.75 1.0 \
    --grid-payload.config.enabled \
    --grid-payload.config.body-groups pelvis "left_wrist_yaw_link,right_wrist_yaw_link" \
    --grid-payload.config.body-labels torso wrists \
    --grid-payload.config.mass-kg 0.0 1.0 2.0 \
    --training.num-envs 24
```

## Constraints

- `grid_payload` and `grid_push` cannot both be enabled (both monkey-patch `_apply_force_in_physics_step`).
- `--training.num-envs` must be >= the total number of conditions (Cartesian product size).
- Push and payload callbacks require IsaacSim backend.
- Recording callback must always be enabled when using any grid callback.

## NPZ Output Format

Arrays have shape `[T, num_conditions, ...]`. Metadata is stored as a JSON string in `_metadata_json` and includes:
- Simulation params (`dt`, `fps`, `dof_names`, `body_names`, limits)
- Grid conditions (`grid_conditions` list of per-env condition dicts)
- Callback-specific metadata (push timing, payload groups, gait analysis results)
