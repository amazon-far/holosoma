# Running from a config file

The `holosoma` command (also `python -m holosoma`) launches every workflow. The `task:<name>`
selection (on the command line, or `task: <name>` under a document's `arguments`) picks an entry
from `TASK_REGISTRY`. Every task reads the same config type, `HolosomaConfig`. The task entry
names the runner class that drives the workflow and the environment base that the config's
`env_class` must be or subclass:

| `task` | Runs | Runner | Environment base | Alias |
|---|---|---|---|---|
| `sim` (default) | the simulation loop, no policy | `SimTask` | `SimulationTask` | `run_sim` |
| `train` | training | `TrainTask` | `PolicyTask` | `train_agent` |
| `eval` | checkpoint evaluation | `EvalTask` | `PolicyTask` | `eval_agent` |
| `replay` | motion replay | `ReplayTask` | `WholeBodyTrackingManager` | `replay` |

`train` and `eval` drive the same kind of environment, so eval can rebuild the one a checkpoint
was trained in from its saved config.

The aliases (`python -m holosoma.run_sim`, `holosoma.train_agent`, `holosoma.eval_agent`,
`holosoma.replay`, or the same files run as scripts) are `holosoma` with that task selected and
accept the same arguments and documents. Extensions add tasks through the `holosoma.config.task` entry-point
group. Each entry in `arguments` maps to the token you would type on the command line, so any
preset from an installed extension or an `--import-file` works the same way here.

`sim` starts from its own base config (`holosoma.config_values.experiment.sim`): a `SimulationTask` on
`run:interactive` and the `sim-mujoco` simulator. The `sim-mujoco`, `sim-mjwarp` and
`sim-isaacsim` presets are the training presets with the robot bridge and virtual gantry on, and
MuJoCo stepping at 2000 Hz (MJWarp 400 Hz). `simulator:mujoco` and the other plain names select the
training presets.

```yaml
# @holosoma/examples/configs/g1_lidar.yaml
version: 1
arguments:
  task: sim
  simulator: sim-mujoco          # <group>:<preset>
  robot: g1-29dof
  sensor:                        # <family>.<key>:<preset>
    scan: g1-pelvis-lidar
  overrides:                     # --<dotted.path>=<value>
    run.headless: true
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
holosoma --config @holosoma/examples/configs/g1_lidar.yaml
holosoma --config @holosoma/examples/configs/g1_lidar.yaml terrain:terrain-locomotion-mix --run.seed 7

# Train; the document selects `task: train` and the experiment with `exp:`
holosoma --config @holosoma/examples/configs/g1_locomotion_train.yaml
holosoma --config @holosoma/examples/configs/g1_locomotion_train.yaml \
    simulator:mjwarp --run.num_envs 1024

# Evaluate; the checkpoint supplies the experiment and the document overrides it
holosoma --config eval.yaml
holosoma task:eval --task.checkpoint=<CHECKPOINT_PATH> run:batch --run.num_envs 64  # headless

# Print the equivalent command line, or turn a command line into a document, instead of running it
holosoma --config eval.yaml --print-args
holosoma task:train exp:g1-29dof simulator:mjwarp --run.seed 1 --to-config > train.yaml

# Replay a whole-body-tracking motion in the Isaac Sim viewer
holosoma --config replay.yaml
```

```yaml
# eval.yaml
version: 1
arguments:
  task: eval                             # task:eval
  overrides:
    task.checkpoint: <CHECKPOINT_PATH>   # the eval task's own field: --task.checkpoint
    task.max_eval_steps: 1000
    task.export_onnx: true
```

```yaml
# replay.yaml
version: 1
arguments:
  task: replay
  exp: g1-29dof-wbt
  simulator: isaacsim
```

## Experiment, task and run

`HolosomaConfig` is an `ExperimentConfig` plus what this invocation does with it:

- **Experiment** (`ExperimentConfig`, chosen with `exp:`): what is run. That is the environment
  class, robot, simulator, terrain, scene, sensors, plugins, managers and `algo`, and the
  `project` and `name` the run is filed under. Eval restores it from the checkpoint.
- **Task** (`task:`): what is done with it, plus that task's own fields.
- **Run** (`run:`): how it executes. `logger:` sits next to it.

Every field keeps its flat path (`--robot…`, `--task.checkpoint`, `--run.seed`):

| Block | Fields |
|---|---|
| top level (experiment) | `project`, `name` |
| `run` | `num_envs`, `headless`, `seed`, `torch_deterministic`, `device`, `time_scale`, `viewer_update_decimation` |
| `task` (train) | `resume_from`, `teacher_checkpoint`, `multigpu` |
| `task` (eval) | `checkpoint`, `max_eval_steps`, `export_onnx`, `callbacks` |
| `task` (sim, replay) | none of their own |

`run` is a selection like any other block. `run:batch` (4096 headless envs, unpaced) is the
experiments' default and what `train` uses. `run:interactive` (one env in the viewer at real time)
is what `sim` starts from, and `replay` selects it unless the arguments select a `run:`. `eval`
defaults to `run:interactive` with the checkpoint's `seed` and `torch_deterministic`; selecting a
`run:` replaces that too. `--run.*` flags apply on top of either.

Settings that used to live under `training` still work under their old spelling, with a warning
naming the new one: `training.{num_envs, headless, seed, torch_deterministic}` and
top-level `device`, `time_scale` and `viewer_*` → `run.*`; `training.{project, name}` and
`logger.project` → `project`/`name`; `logger.name` (the W&B run name) → `logger.display_name`;
`training.checkpoint` → `task.resume_from`; `training.{teacher_checkpoint, multigpu, max_eval_steps, export_onnx}` →
`task.*`; `--checkpoint` → `--task.checkpoint`. Checkpoints saved with the old layout load
unchanged. `eval_overrides.*` is ignored with a warning: eval applies its defaults (the
interactive run, logger off, deterministic spawn) to the checkpoint's config before the command
line, so set the field itself instead.

## Rules

- **Task arguments.** The selected task's own fields are `task.<field>`, in a document's
  overrides and on the command line alike (`--task.checkpoint`). A field the selected task does
  not have is rejected.
- **Experiment.** For `train` and `replay`, `exp: <preset>` is passed first wherever it
  appears. `eval` takes its experiment from the checkpoint (`task.checkpoint`, also accepted
  as `--checkpoint`), so its documents hold overrides and other selections only. `sim` starts
  from its own base config and takes no `exp:`.
- **Errors.** When the command line parser rejects a value that came from a document, the
  message ends with the file and key it came from.
- **Tooling.** `--print-args` prints the command line a run expands to and `--to-config` prints
  it as a document, without running it. Training and evaluation write that document as
  `holosoma_run.yaml` next to `holosoma_config.yaml`, so `holosoma --config holosoma_run.yaml`
  repeats the run.
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
