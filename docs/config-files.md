# Running from a config file

Every entrypoint that parses its arguments with `parse_config` — `run_sim`, `train_agent`,
`eval_agent`, `replay` — reads them from a YAML or JSON document with `--config`. Each entry in
the document maps to the token you would type on the command line, so any preset from an
installed extension or an `--import-file` works the same way here.

```yaml
# @holosoma/examples/configs/g1_lidar.yaml
version: 1
arguments:
  simulator: mujoco              # <group>:<preset>
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

# Train; the document selects the experiment with `exp:`
python -m holosoma.train_agent --config @holosoma/examples/configs/g1_locomotion_train.yaml
python -m holosoma.train_agent --config @holosoma/examples/configs/g1_locomotion_train.yaml \
    simulator:mjwarp --training.num_envs 1024

# Evaluate; the checkpoint supplies the experiment and the document overrides it
python -m holosoma.eval_agent --checkpoint=<CHECKPOINT_PATH> --config eval.yaml

# Replay a whole-body-tracking motion in the Isaac Sim viewer
python -m holosoma.replay --config replay.yaml
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
arguments:
  exp: g1-29dof-wbt
  simulator: isaacsim
  overrides:
    training.num_envs: 1
    training.headless: false
```

## Rules

- **Experiment.** For `train_agent` and `replay`, `exp: <preset>` is passed first wherever it
  appears. `eval_agent` takes its experiment from the checkpoint, so its documents hold overrides
  and other selections only.
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
