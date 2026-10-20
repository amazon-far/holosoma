# Migrate a pre-refactor `RobotConfig`

Use this checklist to migrate a robot preset, its task presets, overrides, and direct consumers from the
pre-2026-09 schema. The field definitions in `holosoma/config_types/robot.py` are the source of truth. For the
current contract, see [Adding a robot](writing-extensions.md#adding-a-robot-robotconfig).

## Completion criteria

The migration is complete when:

1. No production preset, generator, override, or consumer references a removed field.
2. Every migrated robot and experiment constructs without validation errors or unexplained warnings.
3. The repository's native test suite passes, followed by a smoke run on each supported backend.
4. If legacy checkpoints matter, one representative checkpoint loads and its migration warnings have been reviewed.

## 1. Inventory the migration surface

Run these searches from the repository root so configs, scripts, examples, and text overrides are all covered:

```bash
rg -n 'from holosoma\.config_types\.robot import|RobotConfig\(|RobotInitState\(|RobotForceControlConfig' .
rg -n 'init_state\.(pos|rot|default_joint_angles)|robot\.init-state\.(pos|rot|default-joint-angles)|randomize_link_body_names|terminate_after_contacts_on|contact_indices_attr|floating_base|robot_fixed_base|control\.(stiffness|damping)\[' .
rg -n '\breplace\(' .
```

Inspect every robot-related `replace(...)` call and every generator that emits robot configs. Exclude vendored or
generated trees only after confirming that they are not shipped. Do not edit serialized checkpoint dumps; section
6 describes their load-time migration.

## 2. Migrate each robot preset

Apply the rows below in order. Import the preset after each group of edits. Because every `Robot*` config uses
`extra="forbid"`, a remaining or misspelled field raises a validation error at construction.

| Old field or behavior | Migration |
|---|---|
| `num_bodies`, `dof_obs_size`, `algo_obs_dim_dict`, `actions_dim`, `policy_obs_dim`, `critic_obs_dim`, `key_bodies`, `num_feet`, `knee_name`, `has_torso`, `has_upper_body_dof`, `penalize_contacts_on`, `knee_joint_min_threshold`, `lidar_height_offset`, `termination_close_to_dof_pos_limit`, `object`, `asset.armature`, `asset.mesh_root` | Delete. |
| `RobotInitState(pos=..., rot=[x, y, z, w], ...)` | Rename to `position` and `orientation`, and reorder the quaternion to `[w, x, y, z]`. A missed reorder is silent because both layouts can contain a unit quaternion. Zero velocities and identity orientation already have defaults. |
| `init_state.default_joint_angles` | Rename to `default_joint_positions`. Values are radians for hinges and metres for slides; omitted joints default to zero. Keep exact-name and tag keys as authored; checkpoint migration renames this field without expanding it. Conflicting old and new maps are rejected. |
| `foot_body_name`, `foot_height_name`, `torso_name` | Replace with exact-name body tags such as `#feet`, `#feet_height`, and `#base`. Expand the old substring against `body_names`, inspect the matches, and remove accidental matches. |
| `upper_*`, `lower_*`, arm, ankle, knee, hip, or waist DOF groups | Delete unused groups. Preserve groups still used by consumers as `joint_tags`. |
| `symmetry_joint_names`, `flip_sign_joint_names` | Declare `#mirror_left`, `#mirror_right`, and `#mirror_flip_sign`. For a reciprocal pair `{L: R, R: L}`, add `L` and `R` once at the same index and skip self-maps. The left and right lists must be non-empty, equal in length, disjoint, and free of duplicates. |
| `robot.control` | Move controller type, stiffness, damping and torque clipping to the joint action term's `controller`; move action scaling and clipping to that action term. |
| Substring keys in `control.stiffness` or `control.damping` | Replace each key with an exact DOF name or a `#joint_tag` on the destination controller. Every controlled DOF must be covered exactly once under `P` or `V` control. If old keys overlapped, preserve the old rule that the last matching dict entry won. |
| `soft_dof_pos_limit` | Move a non-default value to `MotionConfig.noise_to_initial_pose.soft_dof_pos_limit`. Do not copy it into `limits_dof_pos` rewards: those terms already owned their own parameter before the refactor. |
| `apply_dof_armature_in_isaacgym` | Delete. All backends apply the `dof_armature` map. If the old value was `False`, use `None` to keep asset values or name the intended overrides. |
| `contact_pairs_multiplier` | Set `PhysxConfig.max_gpu_contact_pairs` on the IsaacGym simulator preset to `contact_pairs_multiplier * 1024 * 1024`. Preserve any intentional non-default value, whether smaller or larger than the 16 Mi default. |
| `terminate_after_contacts_on` | Move the body specification to the contact termination term. |
| `randomize_link_body_names` | Move the body specification to the mass-randomization term. |
| `dof_*_limit_list`, `dof_armature_list`, `dof_joint_friction_list` | Replace with `dof_pos_limits`, `dof_vel_limits`, `dof_effort_limits`, `dof_armature`, and `dof_joint_friction`: name/tag maps. Combine lower/upper limits into `(lower, upper)` pairs. Missing entries keep asset values. MuJoCo reports velocity limits without enforcing them. Joint-friction units remain backend-dependent. |
| `RobotForceControlConfig` | Remove the import. Core did not consume this class; move any live force-control settings to the code that applies the force. |
| Unitree/Booster bridge with empty `controlled_dof_names` or only `excluded_dof_names` | Set `controlled_dof_names` to canonical DOF names in SDK motor order, including for whole-robot control. G1/T1 presets supply this tuple. |

Subclasses may keep additional declared fields. Only undeclared fields are rejected.

Audit inherited data whenever a derived preset replaces the articulation names. Replacing `body_names` requires
rebuilding `body_tags`. Replacing `dof_names` requires rebuilding joint tags,
default positions, gains, bridge DOF names, and physical-property maps. Reordering canonical names alone no longer
changes a property assignment.

Keep SDK motor order when reordering canonical DOFs; rename entries if the canonical names change. An inherited
`controlled_dof_names` tuple takes precedence over `excluded_dof_names`, so derived SDK presets must state their
complete intended motor tuple.

Declare every tag referenced by the selected task, action and algorithm presets. Do not assume that tags with
similar roles identify the same body.

WBT's `MotionConfig.root_body=None` discovers the loaded articulation's physical root. That body must appear in
`robot.body_names` and the motion data. An explicit name/tag must identify that same root; `body_name_ref` remains
the independent tracking reference.

For saved configs without this field, checkpoint migration records the former first body when its identity
can be recovered from the saved order. This preserves old behavior without guessing a physical root.
Pre-refactor MuJoCo and unknown simulator dumps require an explicit `root_body`, because their runtime body
order may differ from the saved list. Check this selection when correcting an old config's root identity.

## 3. Move task settings to their terms

| Old use | New term parameter |
|---|---|
| `contact_forces_exceeded` with `contact_indices_attr` | Set `params["bodies"]` to exact body names or tags, then remove `contact_indices_attr`. |
| `randomize_mass_startup` reading robot fields | Set `params["link_bodies"]` and `params["base_body"]`. The default base is `#base`. |
| `randomize_base_com_startup` reading `torso_name` | Set `params["body"]`. The default is `#base`. |
| Feet reward terms reading `env.feet_indices` | Use the term's `feet` parameter, whose default is `("#feet",)`. `penalty_close_feet_xy` requires exactly two feet. Set `sole_axes` for `penalty_feet_ori` when the sole plane is not the default. |
| Terrain foot-height rays | Declare `#feet_height` or set `TerrainTermCfg.feet_height_bodies`. |
| `replace(g1_29dof, robot=other)` with shared locomotion presets | Declare every tag used by those presets on `other`, or provide robot-specific termination and randomization presets. |

## 4. Update direct consumers

| Old use | Replacement |
|---|---|
| `env.feet_indices`, `env.torso_index`, `env.termination_contact_indices` | `env.robot_names.resolve_body_indices([...])`; use `env.robot_names.resolve_body_names(...)` when names are needed. |
| Direct gain dict lookup or local substring matching | Resolve gains from the active action term or plugin controller. Runtime consumers should read the controller's resolved tensors so exports and bridges match the controller actually in use. |
| Direct default-position dict lookup | `RobotConfig.default_dof_positions()` returns a validated exact-name map in `dof_names` order. Use `list(defaults.values())` when a vector is needed. |
| Reading a config limit list at runtime | Read the simulator's `torque_limits`, `dof_pos_limits`, or `dof_vel_limits`; these include asset values for joints omitted from the config maps. MuJoCo's velocity limits are reported only. |
| `robot_config.floating_base`, `simulator.robot_fixed_base` | `robot_config.asset.fix_base_link`. |
| `init_state.pos`, `init_state.rot` | `init_state.position`; use `orientation_xyzw` or `root_state_list()` for runtime tensor layout. |
| `env.dim_obs`, `env.dim_critic_obs` | Remove. `env.dim_actions` remains available. |
| A generator emitting the old schema | Emit exact names, tags, and the new init-state fields. Physical-property maps may be sparse or `None`. |

Simulator limit tensors represent **hard** limits. Reward and reset terms derive their own soft intervals from
those tensors using `soft_limits`; this avoids contracting an interval twice and handles unbounded joints
without `inf - inf` NaNs. Cached simulator limits are setup metadata, not a live readback of randomization.

## 5. Rewrite CLI and YAML overrides

Overrides are not migrated automatically:

- `robot.init-state.pos` becomes `robot.init-state.position`.
- `robot.init-state.rot: [x, y, z, w]` becomes `robot.init-state.orientation: [w, x, y, z]`.
- `robot.init-state.default-joint-angles` becomes `robot.init-state.default-joint-positions`.
- Controller gain overrides target the action term or plugin that owns the controller. For example:
  `action:g1-29dof-joint-pos --action.terms.joint-control.controller.stiffness.#hip-yaw 50`.

## 6. Verify legacy checkpoint loading

`migrate_legacy_experiment_config` migrates `experiment_config` stored in a `.pt` file and the neighboring
`holosoma_config.yaml`. It renames fields, converts positional properties using the saved DOF order, derives tags,
moves task parameters, moves `robot.control` into the joint action term, and expands substring gain keys with the
old matching rule. Unknown robot keys are dropped with a warning.

Gain matching preserves the last substring match even when every key is a complete DOF name.

The current schema contains no legacy property-list fields.

These legacy inputs need attention:

- Conflicting old lists and new property maps, or lists with the wrong length, fail clearly.
- `apply_dof_armature_in_isaacgym: false` raises because no backend-independent conversion can preserve it.
- A one-way `symmetry_joint_names` pair raises while symmetry is enabled. If symmetry is disabled, the loader drops
  that pair with a warning.
- Mixing substring gain keys with `#tag` keys is rejected. Migrate the complete gain map to exact names/tags;
  otherwise a legacy substring pass could silently discard the tag assignments.

Symmetry sides must preserve their authored pairing order without repeated joints, including repetitions hidden
inside nested tags. Sign flips must include both or neither member of each pair. The same validation runs at
config construction and when building the runtime mirror.

Evaluate one representative old checkpoint and review every migration warning.

Checkpoint migration does not infer SDK motor order. Before using Unitree or Booster with a loaded config,
provide `bridge.controlled_dof_names` in SDK order.

## 7. Verify the migrated source

Rerun the inventory searches from section 1. Every remaining production hit must use the current API; compatibility
tests may retain old names intentionally.

```bash
# Construct the robot. This validates names, tags and property-map assignments.
python -c "from my_ext.config_values.robot import my_robot; print(my_robot.asset.robot_type)"

# Check each experiment's CLI options. Experiment construction also validates controllers and symmetry.
python -m holosoma.train_agent exp:my-exp --help

# Run a short train on each supported backend to resolve task tags and exercise their consumers.
python -m holosoma.train_agent exp:my-exp \
  --training.num-envs 64 \
  --training.headless True \
  --algo.config.num-learning-iterations 3
```

The commands above assume a direct Python environment. Use your project's native test runner and run the smoke
trains through that environment.
