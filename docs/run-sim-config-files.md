# Running `run_sim` from a config file

`run_sim` reads its arguments from a YAML or JSON document with `--config`:

```bash
python -m holosoma.run_sim --config g1_headless.yaml
python -m holosoma.run_sim --config g1_headless.yaml robot:t1-29dof-waist-wrist --training.headless False
```

```yaml
# g1_headless.yaml
version: 2
backend: mujoco                  # optional; becomes simulator:mujoco
arguments:
  robot: g1-29dof                # <group>:<preset>
  sensor:                        # <family>.<key>:<preset>
    scan: g1-pelvis-lidar
  plugin:
    clock: clock_publish
  overrides:                     # --<dotted.path> <value>
    training.headless: true
    robot.init_state.pos: [0.0, 0.0, 0.8]
    plugin.clock:                # nested mappings expand to one flag per leaf
      publish_every: 100Hz
```

Each entry maps to the token you would type on the command line, so any preset from an
installed extension or an `--import-file` works the same way here.

- **Precedence.** Command-line tokens win. A group the command line selects (`robot:…`,
  `sensor.scan:…`) replaces the document's selection; a repeated flag takes the last value.
- **Values.** Booleans and `null` encode as `True`, `False` and `None`; YAML lists become Python
  lists. Write tuple-typed values as quoted tuple literals, e.g. `"('a', 'b')"`.
- **Simulator.** Set it with `backend` or `arguments.simulator`, not both.
- **Embedding.** A document can sit inside a larger file. When the root has no `version` but
  has a `holosoma_sim` mapping, that mapping is read; `--config-path a.b` selects any other
  nested mapping. Only `version`, `backend` and `arguments` are allowed in the document, so
  settings for other tools belong next to it, not inside it.
