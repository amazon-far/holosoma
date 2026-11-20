# Configs, the `holosoma` command and run documents

## The command

`holosoma` (also `python -m holosoma`) runs every workflow:
- simulation with no policy;
- motion replay;
- training;
- evaluation.

```bash
holosoma                                                           # simulate: one G1 on MuJoCo, in the viewer
holosoma exp:g1-29dof env.simulator:mjwarp --run.num_envs 1024     # train the g1-29dof experiment
holosoma --task=holosoma.envs.locomotion.locomotion_manager:LeggedRobotLocomotionManager.evaluate \
    --eval.checkpoint=<CHECKPOINT_PATH>                            # evaluate a checkpoint
holosoma --task=holosoma.envs.simulation_task:SimulationTask.replay exp:g1-29dof-wbt   # replay its motion
holosoma --config run.yaml                                         # run a document
```

The older scripts set their workflow for you and accept the same arguments:
- `python -m holosoma.run_sim` (simulate);
- `holosoma.train_agent` (train);
- `holosoma.eval_agent` (evaluate the class the checkpoint was trained with);
- `holosoma.replay` (replay).

## What runs: `task`

Every config has a `task` field naming a task class and the classmethod that runs the config, as
`module:Class.function`. The class is the environment the workflow builds:

| Workflow | `task` | Config type | Parts |
|---|---|---|---|
| simulate | `holosoma.envs.simulation_task:SimulationTask.simulate` (the default) | `SimulationExperimentConfig` | `env`, `run` |
| replay | `holosoma.envs.simulation_task:SimulationTask.replay` | `ReplayExperimentConfig` | `env`, `motion`, `run` |
| train | `…:LeggedRobotLocomotionManager.train`, `…:WholeBodyTrackingManager.train` | `TrainingExperimentConfig` | `env`, `mdp`, `algo`, `run` |
| evaluate | the same classes' `evaluate` | `EvaluationExperimentConfig` | `env`, `mdp`, `eval`, `run` |

`task` comes from, in order:
1. `--task` on the command line, or a document's `overrides.task`;
2. the `exp:` preset's `task` (experiment presets train);
3. the default, `SimulationTask.simulate`.

The function's parameter annotation is the config type the command line builds, so a task class
can add workflows of its own (see [writing-extensions.md](writing-extensions.md#tasks--custom-workflows)).

## Parts

Each config type holds only the parts its workflow uses. Flags and selections are spelled by part.

| Part | Holds | Spelled |
|---|---|---|
| `env` | robot, simulator, terrain, scene, sensors, plugins, randomization, termination | `env.simulator:mjwarp`, `env.sensor.<key>:<preset>`, `--env.robot.init_state.pos=[0, 0, 0.8]` |
| `mdp` | observation, action, reward, command, curriculum | `mdp.reward:<preset>`, `--mdp.reward.terms.<term>.weight=2.5` |
| `algo` | the algorithm and its hyperparameters, plus `resume_from`, `teacher_checkpoint`, `multigpu`, `nightly` | `algo:fast-sac`, `--algo.config.num_learning_iterations=10`, `--algo.resume_from=<ckpt>` |
| `eval` | `checkpoint`, `max_eval_steps`, `export_onnx`, `callbacks` | `--eval.checkpoint=<ckpt>` |
| `run` | `num_envs`, `headless`, `seed`, `torch_deterministic`, `device`, `time_scale`, `viewer_update_decimation`, `project`, `name`, `logger` | `--run.num_envs=64`, `run.logger:wandb` |
| `motion` | the motion a replay plays (`motion_file` or `motion_dir`) | `--motion.motion_file=<npz>` |

`run.project` and `run.name` name the run's log directory (`<base_dir>/<project>/<time>-<name>-…`)
and its W&B project; `run.logger.display_name` overrides the W&B run name.

## Each workflow's starting config

The command line applies on top of a starting config that each config type builds:

- **Simulate.** The interactive defaults: `sim-mujoco` with the bridge and virtual gantry on, and
  one env in the viewer at real time. With `exp:<preset>`, it is that experiment's world on the
  interactive variant of its simulator.
- **Replay.** The `exp:` preset's world with kinematic playback on, and the motion its motion
  command tracks. It plays once in the viewer at real time, then stops.
  - The experiment's randomization and terminations are dropped, since playback is kinematic.
- **Train.** The `exp:` preset.
- **Evaluate.** The checkpoint's saved environment and MDP, with evaluation's defaults:
  - one env in the viewer;
  - logger off;
  - deterministic spawn;
  - no episode time limit;
  - the checkpoint's seed.

  The algorithm is the one the checkpoint trained, and `task` must name the class it trained with.

The `sim-isaacsim`, `sim-mujoco` and `sim-mjwarp` presets are the training presets with the robot
bridge and virtual gantry on, at the interactive physics rate.

## Run documents

```yaml
# holosoma --config run.yaml
version: 1
arguments:
  exp: g1-29dof                # the starting config (exp:g1-29dof); optional
  env.simulator: mjwarp        # a selection, by path (env.simulator:mjwarp)
  env.sensor:                  # named slots: env.sensor.scan:g1-pelvis-lidar
    scan: g1-pelvis-lidar
  overrides:                   # values: --<dotted.path>=<value>
    run.num_envs: 64
    algo.config.num_learning_iterations: 10
```

Each entry maps to one command-line token, so any preset from an installed extension or an
`--import-file` works the same way. Examples ship in
[`src/holosoma/holosoma/examples/configs/`](../src/holosoma/holosoma/examples/configs/) and can be
passed as `@holosoma/examples/configs/<name>.yaml`.

- **Precedence is by path.** A selection applies before the selections and values below its path:
  `exp` sets the whole config, `env.simulator:…` replaces one block of it, and values apply last.
  At the same path, the command line wins over the document.
  - A command-line `exp:` replaces the document's, and the document's other entries still apply
    on top of it; holosoma warns when that happens.
- **Tooling.**
  - `--print-args` prints the command line a run expands to.
  - `--to-config` prints it as a document.
  - Training and evaluation write that document as `holosoma_run.yaml` next to
    `holosoma_config.yaml`, so `holosoma --config holosoma_run.yaml` repeats the run.
- **Errors.** When the parser rejects a value that came from a document, the message ends with the
  file and key it came from.
- **Values.**
  - Scalars follow YAML 1.2: only `true`/`false` are booleans, `010` is ten, and dates stay
    strings.
  - Write tuple-typed values, and dict-typed fields set as a whole, as quoted literals.
- **Embedding.**
  - A document can sit inside a larger file. When the root has no `version` but has a
    `holosoma_sim` mapping, that mapping is read; `--config-path a.b` selects any other mapping.
  - Only `version` and `arguments` are allowed in a document.

## Old spellings and saved configs

The earlier flat spellings keep working, each with a one-time warning naming the new one:
- **The world and MDP fields:** `robot:`, `simulator:`, `--terrain.*`, `sensor.*`, `plugin.*`,
  `randomization:`, `termination:` move under `env`, and `observation:`, `--reward.*`,
  `--command.*`, `curriculum:` move under `mdp`.
- **`training.*`:**
  - `num_envs`, `headless`, `seed`, `torch_deterministic`, `project` and `name` → `run.*`;
  - `checkpoint` → `algo.resume_from`;
  - `teacher_checkpoint` and `multigpu` → `algo.*`;
  - `max_eval_steps` and `export_onnx` → `eval.*`.
- **Other moved fields:**
  - `--device`, `--time_scale` and `--viewer_*` → `run.*`;
  - `logger:`/`--logger.*` → `run.logger`, with `logger.project` → `run.project` and
    `logger.name` → `run.logger.display_name`;
  - `--checkpoint` → `--eval.checkpoint`;
  - `--env_class=X` → `--task=X.train`;
  - `--eval_overrides.*` → the field each one set.
- **`run_sim`'s simulators:** in `python -m holosoma.run_sim`, `simulator:mujoco` (and `isaacsim`,
  `mjwarp`) still means the interactive preset.

Checkpoints and `holosoma_config.yaml` files saved with the flat layout load into the parts
unchanged.
