# Running from a config file

`run_sim`, `train_agent`, `eval_agent` and `replay` read arguments from a YAML or JSON document
with `--config`:

```bash
python -m holosoma.run_sim --config g1_lidar.yaml
python -m holosoma.run_sim --config g1_lidar.yaml terrain:terrain-locomotion-mix --training.seed 7
```

```yaml
# g1_lidar.yaml
version: 2
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

Each entry maps to the token you would type on the command line, so any preset from an
installed extension or an `--import-file` works the same way here. For `train_agent` and
`replay`, select the experiment with `exp: <preset>`; it is passed first.

- **Precedence.** Command-line tokens win. A group the command line selects (`terrain:…`,
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
