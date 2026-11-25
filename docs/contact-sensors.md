# Contact sensors

Contact sensors measure forces between selected bodies on Classic MuJoCo, MuJoCo Warp, and Isaac Sim.
Declare each sensor under its own key in the Holosoma `sensor` dictionary.

A source receives the force. A counterpart applies it.
Use these selectors in either group, except that terrain is a counterpart only:

| `target_kind` | Selection | `target` |
| --- | --- | --- |
| `robot` | The whole robot | Empty |
| `robot_link` | One robot body | Link name |
| `actor` | One registered scene actor | Actor name |
| `terrain` | The terrain | Empty |

The following example measures each foot against terrain and a box.
Replace the link and actor names with names from your scene.

```python
from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig

sensor = {
    "feet": ContactSensorConfig(
        sources=[
            ContactTargetConfig(target_kind="robot_link", target="left_ankle_roll_link"),
            ContactTargetConfig(target_kind="robot_link", target="right_ankle_roll_link"),
        ],
        counterparts=[
            ContactTargetConfig(target_kind="terrain"),
            ContactTargetConfig(target_kind="actor", target="box"),
        ],
    ),
    "box_robot": ContactSensorConfig(
        sources=[ContactTargetConfig(target_kind="actor", target="box")],
        counterparts=[ContactTargetConfig(target_kind="robot")],
        force_mode="total",
    ),
}
```

The default `force_mode="normal"` excludes friction.
Use `"total"` to include friction, with explicit counterparts.
An empty counterpart list selects all contacts in one column.
Groups sum their body forces, so opposing forces inside a group can cancel.
Explicit counterpart groups must not overlap.
The CLI presets `robot-contacts` and `robot-terrain-contacts` each use one whole-robot source row.

## Reading forces

Forces act on the source bodies, in world coordinates, Newtons, and `torch.float32`.

| Output | Shape | Meaning |
| --- | --- | --- |
| `forces` | `[N, S, 3]` | Sum over counterparts |
| `force_matrix` | `[N, S, T, 3]` | Force for each source and counterpart |

`N` counts environments. `S` and `T` follow the configured source and counterpart order.
The accessor also accepts `env_ids` and `device`.

```python
forces = sim.get_contact_data("feet")                 # [N, 2, 3]
matrix = sim.get_contact_data("feet", "force_matrix")  # [N, 2, 2, 3]
in_contact = (matrix.norm(dim=-1) > 1.0).any(dim=-1)   # Your threshold, per foot.
```

By default, sensors update after every solved physics step, before `POST_STEP` hooks.
Set `update_decimation=4` for steps 4, 8, and 12.
You can also use a frequency such as `"50Hz"` against the physics rate.
Skipped updates keep the prior measurement.

Resets clear selected environments until their next update.
Rendering does not update contact forces.
Reset clearing applies even when `sim.num_rerenders_on_reset` is zero and does not restart the physics schedule.

Consumers can keep history through the existing hooks.
Use the default decimation of one to retain brief contacts.
Clone samples because returned tensors are shared and read-only.

```python
from holosoma.simulator.base_simulator.hooks import Phase

samples = []
sim.hooks.add(Phase.FRAME_BEGIN, samples.clear, name="feet.begin")
sim.hooks.add(
    Phase.POST_STEP,
    lambda: samples.append(sim.get_contact_data("feet").clone()),
    name="feet.sample",
)
```

The default `contact_sensor` provides one row per entry in `sim.body_names`.
Existing `sim.contact_forces`, `sim.contact_recorder`, and rewards keep their current behavior.
The name `contact_sensor` is reserved.
