# Contact sensors

Contact sensors measure forces between selected collision bodies.
Declare each sensor under its own key in the Holosoma `sensor` dictionary.
The same configuration works with Classic MuJoCo, MuJoCo Warp, and Isaac Sim.
Read sensor outputs through `sim.get_contact_data(name, data_type)`.

A source is a group of bodies that receives a force.
A counterpart is a group of bodies that applies that force.
Each source selector produces one output row.
Each explicit counterpart selector produces one output column.
The sensor sums forces over all bodies in each group.
An empty counterpart list selects all contacts in one column.

The selectors use the existing Holosoma namespaces:

| `target_kind` | Selection | `target` |
| --- | --- | --- |
| `robot` | All robot collision bodies | Empty |
| `robot_link` | The native rigid body that owns one robot link | Unprefixed link name |
| `actor` | One registered scene actor and its descendants | Actor name |
| `terrain` | The terrain collider, as a counterpart | Empty |

Fixed links follow their owning rigid body after import or compilation.
Equivalent link selections require matching imported body boundaries across robot assets.
An actor includes its own geometry even when MuJoCo merges its static body into the world.
Scene-file actors use their registered names, such as `scene_free_box`.
Unknown names and missing terrain fail during setup.
Counterpart groups must remain disjoint after native body resolution.
Source groups can overlap because each row is an independent measurement.

## Configuration

The following configuration measures each foot against the terrain and a box.
It also measures the box against the whole robot.
Replace the link and actor names with names from your robot and scene.

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
        update_decimation="25Hz",
    ),
    "box_robot": ContactSensorConfig(
        sources=[ContactTargetConfig(target_kind="actor", target="box")],
        counterparts=[ContactTargetConfig(target_kind="robot")],
        force_mode="total",
    ),
}
```

The `robot-contacts` preset sums all robot contact forces into one source row.
The `robot-terrain-contacts` preset restricts that row to terrain contact.
Use separate `robot_link` source entries when you need one measurement per body.
The CLI supports per-key overrides of the selectors in these presets.

```shell
python -m holosoma.run_sim \
  simulator:mujoco \
  sensor.ground:robot-terrain-contacts \
  --sensor.ground.sources.0.target-kind=robot_link \
  --sensor.ground.sources.0.target=left_ankle_roll_link
```

## Output contract

All forces act on the source bodies.
Forces use world coordinates, Newtons, and `torch.float32`.
The opposite sensor reports the opposite force for the same pair.
The sensor reports solved contact forces from the latest physics substep.
It reads existing physics contacts and does not enable disabled collision pairs.
It does not include joint forces, external forces, or contact torque.

| `data_type` | Shape | Meaning |
| --- | --- | --- |
| `forces` | `[N, S, 3]` | Sum over selected counterpart groups |
| `force_matrix` | `[N, S, T, 3]` | Force for each source and counterpart group |
| `in_contact` | `[N, S]` | Any column exceeds `force_threshold`, in Newtons |
| `substep_forces` | `[N, H, S, 3]` | Every physics sample in the current frame, when `record_substeps=True` |

`N` is the number of environments.
`S` follows the source order in the configuration.
`T` follows the counterpart order, or equals one for an unfiltered sensor.
`H` counts completed physics substeps in the current control frame, oldest first.
`sim.sensor_manager.get_contact(name)` exposes these labels through `source_names` and `counterpart_names`.
Returned tensors are shared and read-only.
The accessor also accepts `env_ids` and `device`, like the camera and LiDAR accessors.

The default `force_mode="normal"` excludes friction.
The `"total"` mode includes tangential friction and requires explicit counterparts on every backend.
Force directions and units match across engines, but their solvers and collision geometry can produce different numerical values.

The contact flag uses the norm of each column's resultant force.
Opposing forces in different columns still produce a contact flag.
Opposing forces within one group can cancel.
Internal contacts cancel when both bodies belong to the same aggregated source and counterpart groups.
Use separate source or counterpart groups when that distinction matters.

## Updates, history, and resets

Contact sensors update after solved physics substeps, before `POST_STEP` hooks.
`update_decimation` counts completed physics steps, or accepts a frequency string against the physics rate.
The first update occurs after one full interval.
For example, `update_decimation=4` updates after steps 4, 8, and 12.
With 200 Hz physics and four substeps per control frame, `"50Hz"` gives the final sample of each frame.
`"25Hz"` updates at the end of every second frame.
Skipped updates retain the prior measurement.
All outputs start at zero.

Set `record_substeps=True` to retain every physics sample of the current control frame.
The history runs at the full physics rate, independently of output decimation.
Read it with `sim.get_contact_data(name, "substep_forces")`.
`FRAME_BEGIN` starts an empty history, and each solved substep adds one sample.
The public `simulate_at_each_physics_step()` API also records these samples without a `POST_STEP` hook.
Kinematic playback adds no physics samples.
The sensor does not average forces.
Use the history to detect brief contacts that disappear before frame end.

Episode resets and `rerender_sensors(env_ids)` clear the selected contact rows.
They also clear those rows in the configured history.
Measurements remain zero until an output update after a physics step.
History resumes at the next solved substep.
Other environments keep their measurements.
Resets and `FRAME_BEGIN` do not restart the physics update counter.
Contact sensors have no mount pose because they select bodies directly.

The manager also provides the default robot contact sensor as `contact_sensor`.
Its rows follow `sim.body_names`, and its named outputs use the shared normal-force contract.
The existing `sim.contact_forces` and `sim.contact_recorder` APIs read the same managed contact source.
They preserve their native force conventions and existing `POST_STEP` recorder behavior.
Existing rewards and body-selection configuration need no changes.
The legacy MuJoCo force tensor includes friction.
Use named sensors when you need the same force contract across backends.
The name `contact_sensor` and names that start with `__holosoma_contact_` are reserved.

## Native configuration

`isaacsim.max_contact_data_count_per_prim` controls native contact storage per source body and environment.
The default is 64.
Increase it for scenes with more simultaneous contact points.
`isaacsim.debug_vis` enables native contact markers for the configured sensor.

Contact sensors need no camera or rendering pass.
