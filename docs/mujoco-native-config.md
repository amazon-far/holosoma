# Native MuJoCo configuration

All entry points use `simulator.config.sim.mujoco`: a frozen Pydantic dataclass,
`MujocoConfig`, with MJWarp settings in its `warp: MujocoWarpConfig` field. Both
types are defined in `src/holosoma/holosoma/config_types/mujoco.py`. Configuration
application belongs to core world compilation and backend initialization. No contact
plugin, plugin registration, or external extension is needed.

The explicit surface is deliberately small: `integrator`, `solver`, `cone`,
`jacobian`, `impratio`, `iterations`, `tolerance`, `ls_iterations`, `ls_tolerance`,
`noslip_iterations`, `gravity`, `disableflags`, and `enableflags`. None preserves
asset/native behavior. Selecting a preset or adding this feature does not change
the integrator or contact defaults of existing simulations.

Warp-specific fields under `mujoco.warp` are `nconmax_per_env`, `njmax_per_env`,
`naconmax`, `graph_conditional`, and one `extras` mapping. Classic MuJoCo retains
them in the config without applying them, so a Warp preset or saved config
can switch to classic with `--simulator.config.mujoco-backend CLASSIC`, without
unsetting resource or execution settings. Other simulators ignore the entire MuJoCo block
as they do other inactive backend configuration.

## CLI, YAML and presets

```bash
python -m holosoma.run_sim simulator:mujoco \
  --simulator.config.sim.fps 200 \
  --simulator.config.sim.mujoco.integrator implicitfast \
  --simulator.config.sim.mujoco.cone elliptic \
  --simulator.config.sim.mujoco.impratio 10 \
  --simulator.config.sim.mujoco.noslip-iterations 1
```

Flag collections use Holosoma's Python collection syntax:
`--simulator.config.sim.mujoco.disableflags "('contact', 'gravity')"`.
An empty tuple explicitly clears the requested mask. Unknown flag names fail
against the installed upstream enums during core setup.

The same values in a saved simulator config are:

```yaml
sim:
  fps: 200
  # Other existing engine fields retained here.
  mujoco:
    integrator: implicitfast
    cone: elliptic
    impratio: 10.0
    noslip_iterations: 1
    extra:
      ccd_iterations: 50
```

Construct presets with `MujocoConfig(...)`, `MujocoWarpConfig(...)`, and
`dataclasses.replace`, as for other Holosoma configs. Construction and replacement
run Pydantic validators; YAML dictionaries can be validated with
`TypeAdapter(MujocoConfig).validate_python(...)`. Existing dataclass serialization
works unchanged. Training, standalone simulation, inference and teleop use the
same simulator type. Inactive MuJoCo settings
are removed before handing the engine block to IsaacGym's native parser.

`examples/mujoco_native.py` is a bounded, headless core-simulator example with
an empty plugin configuration. Its command line is the same RunSimConfig CLI:

```bash
PYTHONPATH=src/holosoma python examples/mujoco_native.py simulator:mujoco \
  --simulator.config.sim.mujoco.integrator implicitfast \
  --simulator.config.sim.mujoco.cone elliptic \
  --simulator.config.sim.mujoco.impratio 10 \
  --simulator.config.sim.mujoco.noslip-iterations 1

PYTHONPATH=src/holosoma python examples/mujoco_native.py simulator:mjwarp \
  --training.num-envs 2 \
  --simulator.config.sim.mujoco.integrator rk4 \
  --simulator.config.sim.mujoco.warp.nconmax-per-env 128 \
  --simulator.config.sim.mujoco.warp.naconmax 300 \
  --simulator.config.sim.mujoco.warp.graph-conditional False \
  --simulator.config.sim.mujoco.warp.extras '{"nccdmax": 64}'
```

The first command works on CPU. The second requires CUDA and the pinned Warp
setup. No tuning plugin is selected by either command.

## Uncommon native settings

`mujoco.extra` accepts the remaining writable `MjOption` properties, for example
`extra: {noslip_tolerance: 0.00001, o_solref: [-1000, -10]}` on classic.
`mujoco.warp.extras` accepts other MJWarp-only `Option` fields and `make_data`
allocation arguments. Names are bare upstream attributes, without `option.` or
`data.` prefixes. On the CLI, pass one JSON/Python dictionary token:

```bash
--simulator.config.sim.mujoco.extra '{"ccd_iterations": 12}'
--simulator.config.sim.mujoco.warp.extras '{"nccdmax": 64, "run_collision_detection": false}'
```

The equivalent Warp block in YAML is:

```yaml
sim:
  mujoco:
    integrator: rk4
    extra:
      ccd_iterations: 12
    warp:
      nconmax_per_env: 128
      naconmax: 300
      graph_conditional: false
      extras:
        nccdmax: 64
        run_collision_detection: false
```

Names, public writability, enum names, scalar types, finite values, and vector
lengths are checked against the installed supported API. Validation does not
maintain an allowlist of accepted native attribute names or enum members.
Native compilation/conversion provides additional model-dependent checks.
Routing uses upstream definitions and signatures. If a future dependency introduces
an ambiguous option/allocation name, setup rejects it and the completeness check
requires a compatibility review.

The application targets are fixed; their public fields and argument names are
discovered from the supported APIs:

| Input mapping | Native definition and example keys | Application stage |
| --- | --- | --- |
| `mujoco.extra` | Writable `MjOption` properties: `ccd_iterations`, `noslip_tolerance` | Final specification before compilation; supported shared settings also reach Warp upload |
| `mujoco.warp.extras` | Public non-derived MJWarp-only `Option` fields: `run_collision_detection`, `broadphase` | Uploaded Warp model before graph capture |
| `mujoco.warp.extras` | Public `make_data()` arguments: `nccdmax`, `njmax_nnz` | Warp data allocation |

A shared option belongs only in `mujoco`, even when also present in MJWarp.
Putting it in `warp.extras` fails with its shared configuration path. A name
matching unrelated option and allocation interfaces is rejected as ambiguous;
unknown names also fail during active backend setup.

Classic ignores the entire `warp` block, including `extras`, without importing
MJWarp. Active Warp setup validates it against installed definitions and
signatures. The completeness test compares the checked-in inventory with those
definitions. Backend-specific ranges and combinations are checked when that
backend is selected.

Use explicit fields for their corresponding options; supplying them in `extra`
is an error. `timestep` is not exposed: `sim.fps` owns `sim_dt=1/fps`, control
and render decimation, and simulation time. A timestep in `extra` is an
explanatory error. `training.num_envs` similarly owns Warp world count.
Native allocation aliases `nconmax` and `njmax` are already owned by the
`warp.nconmax_per_env` and `warp.njmax_per_env` fields and cannot be supplied in
`warp.extras`. Private/derived fields and unsupported active-backend settings
fail with the configuration path. Arbitrary model arrays, transient state,
kernel block sizes, process-wide compiler settings, per-field model batching
and capture orchestration are outside this focused physics/resource interface.
Core backend setup retains their existing behavior.

Warp's effective capacity defaults are unchanged: unset `nconmax_per_env` resolves
to 96, and
`njmax_per_env=None` resolves to `max(6*nconmax_per_env, 4*nv)`.
Optional Warp fields use None to preserve native/default behavior through full
YAML serialization.
`nconmax_per_env` is an average allocation: all worlds share the contact pool,
and one world can exceed it. `njmax_per_env` is a hard constraint capacity per
world. `naconmax` overrides the **total** contact pool. `warp.extras.naccdmax`
is a total CCD allocation; `warp.extras.nccdmax` is its per-world allocation input.
`warp.extras.njmax_nnz` is sparse Jacobian nonzeros per world, and `warp.extras.nvmax` is
compacted active-DOF capacity per world. Explicit zero values are preserved.
The pinned API checks CCD capacities against the contact pool.

The old `simulator.config.mujoco_warp` location fails with an error naming
`simulator.config.sim.mujoco.warp`. Move its fields into that block; capacity field names
and defaults are unchanged. Old CLI paths likewise fail with the replacement.
No silent migration or deprecation alias is provided.

## Contacts and asset authoring

Existing `PhysicsConfig.mujoco` remains the scene/robot contact interface. It
exposes friction (sliding/torsional/rolling), solref, solimp, condim, collision
masks, priority, mixing, margin and gap. None keeps the asset. Zero mixing/masks
are intentional overrides. Masks use the native signed 32-bit representation;
priority and mixing retain native signed semantics. Negative solref pairs use native direct stiffness
and damping semantics; they are supported. Its `extra` supports other public
scalar/vector geom attributes, validated from MjsGeom. Shared mass/density and
scene identity/render groups retain their existing owners.

These are per-geom settings. Global contact overrides (`o_*` plus the `override`
enable flag) belong to `mujoco.extra`/`enableflags`; explicit pair attributes,
joint/tendon/equality solver parameters, compiler/resource/visual settings and
native sensors remain authored in existing MJCF asset paths (`robot.asset.xml_file`,
scene XML files, or rigid-object XML files). Camera/LiDAR and lighting settings
retain the existing shared sensor/scene configuration. This change does not
introduce a second schema for the entire MJCF model-authoring surface.

Precedence is: authored assets, existing scene/robot physics overrides, then
native simulator settings. Each setting has one configuration source; there is
no second Warp override of a common option. Presets
supply ordinary config values; CLI/YAML overrides select the final config.
Inactive backend blocks follow existing cross-simulator preset behavior.
Geom overrides apply before compilation; final global options apply after scene
assembly; upload, allocation and kernel settings all precede capture.

## Supported versions and lifecycle

The setup scripts pin MuJoCo 3.10.0, MJWarp commit
`ecaef88917a3c90cd238bf76681ca770f58033df`, and Warp 1.15.0. Native override
application checks these versions; the unconfigured classic path retains legacy
behavior. A VCS install at another MJWarp commit fails. Non-VCS wheels with version
3.10.0 cannot prove the source revision; use the setup script's VCS pin.

The pinned MJWarp API supports all four integrators, CG/Newton, and both cones.
It has no no-slip solver and rejects the contact-override enable flag, unsupported
flags, and CPU options absent from MJWarp Option. The classic teleop tuning
`elliptic/10/1` is therefore not a Warp preset. Use no-slip zero on Warp.
Upstream conversion forces its experimental island solver off and raises the
unset solver tolerance to at least 1e-6. Explicit supported tolerance values are
restored to the uploaded device array before capture.
Zero impedance ratio is accepted; Warp's inverse-square-root representation
uses the same native `mjMINVAL` floor as upstream conversion.

Settings apply during initialization. Reset preserves model settings and graph
bindings; reconstruct the simulator to change configuration.

[`mujoco-parameter-inventory.json`](mujoco-parameter-inventory.json)
records the supported runtime API, defaults, paths, stages, limits and reasons for
excluding authoring/execution infrastructure from this focused configuration.
`scripts/inventory_mujoco.py` regenerates it from the
installed public definitions and signatures. The completeness test fails if
those definitions change. Review compatibility when changing dependency pins;
regenerating the inventory alone does not validate a new backend version.

See `native-mujoco-verification.md` for actual test results and GPU limits.
