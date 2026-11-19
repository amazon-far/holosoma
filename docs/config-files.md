# Running from a config file

`run_sim` launches every workflow. The document's `task` (or a `task:<name>` token on the
command line) selects a workflow task from `TASK_REGISTRY`; every task reads the same config type,
`ExperimentConfig`, where `env_class` names the environment the task builds and drives:

| `task` | Runs | Environment class | Shortcut |
|---|---|---|---|
| `sim` (default) | the simulation loop, no policy | `SimulationTask` | — |
| `train` | training | `PolicyTask` | `train_agent` |
| `eval` | checkpoint evaluation | `PolicyTask` | `eval_agent` |
| `replay` | motion replay | `WholeBodyTrackingManager` | `replay` |

`train_agent`, `eval_agent` and `replay` are shortcuts for `run_sim task:<name>` and accept the
same arguments and documents. Extensions add tasks through the `holosoma.config.task` entry-point
group. Each entry in `arguments` maps to the token you would type on the command line, so any
preset from an installed extension or an `--import-file` works the same way here.

`sim` defaults to the interactive simulator presets `sim-mujoco`, `sim-mjwarp` and
`sim-isaacsim`: the training presets with the robot bridge and virtual gantry on, and MuJoCo
stepping at 2000 Hz (MJWarp 400 Hz). `simulator:mujoco` and the other plain names select the
training presets.

```yaml
# @holosoma/examples/configs/g1_lidar.yaml
version: 1
task: sim
arguments:
  simulator: sim-mujoco          # <group>:<preset>
  robot: g1-29dof
  sensor:                        # <family>.<key>:<preset>
    scan: g1-pelvis-lidar
  overrides:                     # --<dotted.path>=<value>
    training.headless: true
    robot.init_state.pos: [0.0, 0.0, 0.8]
    sensor.scan:                 # nested mappings expand to one flag per leaf
      update_decimation: 10Hz
```

Example documents ship with the package in
[`src/holosoma/holosoma/examples/configs/`](../src/holosoma/holosoma/examples/configs/) and can be
passed as `@holosoma/examples/configs/<name>.yaml`; copy one as a starting point.

## Invocations

```bash
# Simulate
python -m holosoma.run_sim --config @holosoma/examples/configs/g1_lidar.yaml
python -m holosoma.run_sim --config @holosoma/examples/configs/g1_lidar.yaml terrain:terrain-locomotion-mix --training.seed 7

# Train; the document sets `task: train` and selects the experiment with `exp:`
python -m holosoma.run_sim --config @holosoma/examples/configs/g1_locomotion_train.yaml
python -m holosoma.run_sim --config @holosoma/examples/configs/g1_locomotion_train.yaml \
    simulator:mjwarp --training.num_envs 1024

# Evaluate; the checkpoint supplies the experiment and the document overrides it
python -m holosoma.run_sim task:eval --task.checkpoint=<CHECKPOINT_PATH> --config eval.yaml

# Replay a whole-body-tracking motion in the Isaac Sim viewer
python -m holosoma.run_sim --config replay.yaml
```

```yaml
# eval.yaml
version: 1
arguments:
  overrides:
    training.num_envs: 1
    training.max_eval_steps: 1000
    training.export_onnx: true
```

```yaml
# replay.yaml
version: 1
task: replay
arguments:
  exp: g1-29dof-wbt
  simulator: isaacsim
  overrides:
    training.num_envs: 1
    training.headless: false
```

## Rules

- **Experiment.** For `train` and `replay`, `exp: <preset>` is passed first wherever it
  appears. `eval` takes its experiment from the checkpoint (`--task.checkpoint`, also accepted
  as `--checkpoint`), so its documents hold overrides and other selections only; the
  checkpoint can also be set as the override `task.checkpoint`.
- **Precedence.** Command-line tokens win. A group the command line selects (`simulator:…`,
  `sensor.scan:…`) replaces the document's selection, and a repeated flag takes the last value.
  The document's overrides under a replaced group still apply to the new selection.
- **Values.** Scalars follow YAML 1.2: only `true`/`false` are booleans, `010` is ten, and dates
  stay strings. YAML lists become Python lists. Write tuple-typed values, and dict-typed fields
  set as a whole, as quoted literals (`"('a', 'b')"`, `'{"gravity": [0, 0, -2]}'`); `{}` clears
  a dict field.
- **Embedding.** A document can sit inside a larger file. When the root has no `version` but
  has a `holosoma_sim` mapping, that mapping is read; `--config-path a.b` selects any other
  nested mapping. Only `version` and `arguments` are allowed in the document, so settings for
  other tools belong next to it, not inside it.
- **Paths.** `--config` accepts the same path forms as `--import-file`. Pass it once per run.
