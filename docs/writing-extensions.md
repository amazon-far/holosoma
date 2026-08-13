# Adding Config Values to Holosoma

Two ways to add a preset (robot, reward, simulator, inference policy, …): a **`--import-file`** for local/one-off use, or a packaged **entry point** for anything you ship. Both land in the same menu and work across `run_sim`, `train_agent`, `eval_agent`, `replay`, `run_policy`.

Each config family is a `ConfigRegistry` with an entry-point group. Presets are type-checked on the way in; a wrong type or a broken plugin is skipped.

## `--import-file` — no packaging

A plain `.py` file that calls `REGISTRY.add(...)`:

```python
# my_presets.py  (anywhere on disk)
from dataclasses import replace
from holosoma.config_values.robot import ROBOT_REGISTRY, g1_29dof

ROBOT_REGISTRY.add("g1_stiff", replace(g1_29dof, control=replace(g1_29dof.control, action_scale=0.1)))
```

```bash
python -m holosoma.run_sim --import-file my_presets.py robot:g1-stiff   # repeatable; selectable like a built-in
```

`.add(name, value)` type-checks `value` and returns it, so `x = REGISTRY.add(...)` also keeps a normal module attribute.

## Entry point — packaged extension

The value must be a config instance of the family's type. Compose higher-level presets from lower-level ones:

```python
# holosoma_inference_ext_quadruped/config_values/robot.py
from holosoma_inference.config.config_types.robot import RobotConfig
go2_12dof = RobotConfig(robot_type="go2_12dof", robot="go2", num_motors=12, num_joints=12)  # ...

# holosoma_inference_ext_quadruped/config_values/inference.py
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import task          # reuse core presets
from holosoma_inference_ext_quadruped.config_values import observation, robot
go2_12dof_loco = InferenceConfig(robot=robot.go2_12dof, observation=observation.loco_go2_12dof, task=task.locomotion)
```

Declare one entry point per preset — `<name> = "<module>:<attr>"`, group picks the registry:

```toml
# pyproject.toml
[project]
dependencies = ["holosoma_inference"]        # or "holosoma" for training-side presets
[tool.setuptools.packages.find]
include = ["holosoma_inference_ext_quadruped*"]   # ship your config_values modules

[project.entry-points."holosoma.config.robot"]
go2-12dof = "holosoma_inference_ext_quadruped.config_values.robot:go2_12dof"
[project.entry-points."holosoma.config.inference"]
go2-12dof-loco = "holosoma_inference_ext_quadruped.config_values.inference:go2_12dof_loco"
```

```bash
pip install -e .
python -m holosoma_inference.run_policy inference:go2-12dof-loco   # discovered automatically, no registration code
```

## Adding a preset in the core repo

Edit the family's `config_values` module and register with `.add()`:

```python
# holosoma/config_values/robot.py
my_robot = ROBOT_REGISTRY.add("my_robot", RobotConfig(...))
```

## Naming

Register hyphen-case (`go2-12dof`). The CLI token is `<field>:<key>` and accepts both forms — `robot:g1_29dof` and `robot:g1-29dof` both work.

## Config families

Publish an entry point under the group whose config type matches your preset. Training and inference share some group names on purpose — the type check routes each preset to the right registry.

**Training (`holosoma`)** — `robot` `RobotConfig` · `simulator` `SimulatorConfig` · `run_sim` `SimulatorConfig` · `terrain` `TerrainManagerCfg` · `scene` `SceneConfig` · `algo` `PPOAlgoConfig`/`FastSACAlgoConfig` · `observation` `ObservationManagerCfg` · `action` `ActionManagerCfg` · `reward` `RewardManagerCfg` · `termination` `TerminationManagerCfg` · `randomization` `RandomizationManagerCfg` · `command` `CommandManagerCfg` · `curriculum` `CurriculumManagerCfg` · `logger` `DisabledLoggerConfig`/`WandbLoggerConfig` · `plugin` `PluginConfig` · `experiment` `ExperimentConfig` (top-level `exp:`)

The `plugin` family runs custom per-step behavior — see [Plugins](#plugins) below.

**Inference (`holosoma_inference`)** — `robot` `RobotConfig` · `observation` `ObservationConfig` · `task` `TaskConfig` · `inference` `InferenceConfig` (top-level `inference:`)

Group string is `holosoma.config.<family>`, registry var is `<FAMILY>_REGISTRY` (e.g. `holosoma.config.reward` → `REWARD_REGISTRY`).

## Plugins — custom per-step behavior

A **plugin** is a bundle of behavior — a set of lifecycle hooks plus whatever state they need — that runs your code at points in the simulator loop, without subclassing a backend. It's built on the hook system (`simulator.hooks`, `Phase`) and can depend on other simulator contracts (the virtual gantry, the clock, …). Write a class taking `(cfg, simulator)` that registers hooks in `__init__`, and pair it with a `PluginConfig` (its CLI-visible knobs). There is no base class to inherit.

```python
# my_plugins.py — log the robot's height, sampled at a fixed rate
from dataclasses import dataclass
from loguru import logger
from holosoma.config_types.plugin import PluginConfig
from holosoma.config_values.plugin import PLUGIN_REGISTRY
from holosoma.simulator.base_simulator.hooks import Phase

@dataclass(frozen=True)
class LogHeightPluginConfig(PluginConfig):
    every: str = "2Hz"                       # int decimation or frequency string
    def get_cls(self):
        return LogHeightPlugin               # import lazily here if the impl is heavy

class LogHeightPlugin:
    def __init__(self, cfg: LogHeightPluginConfig, simulator):
        self.cfg, self.simulator = cfg, simulator
        # register one hook per phase you care about:
        simulator.hooks.add(Phase.FRAME_END, self.log, name="log_height", every=cfg.every)

    def log(self):                           # signature = the phase's payload (below)
        logger.info(f"height = {self.simulator.robot_root_states[0, 2]:.3f}")

PLUGIN_REGISTRY.add("log_height", LogHeightPluginConfig())
```

```bash
python -m holosoma.run_sim --import-file my_plugins.py plugin.h:log_height --plugin.h.every=5Hz
```

Select plugins as the dynamic-dict `plugin.<key>:<preset>` (per-key leaf overrides via `--plugin.<key>.<field>=…`); each is constructed once in `BaseSimulator.__init__`. Ship them packaged via the `holosoma.config.plugin` entry-point group like any other preset.

### Phases

A plugin's hooks attach to lifecycle phases via `add(phase, callback, *, name=None, every=1)`. Per control frame the loop emits, in order:

| Phase | When | Callback args | Cadence |
|---|---|---|---|
| `FRAME_BEGIN` | before the physics substeps (push commands to the sim here) | — | control rate |
| `PRE_STEP` | before each physics substep | — | physics `fps` |
| `POST_STEP` | after each physics substep (freshest sim time) | — | physics `fps` |
| `FRAME_END` | after the substeps + state-tensor refresh (read state here) | — | control rate |
| `EPISODE_START` / `EPISODE_END` | on reset boundaries | `env_id` | per event |
| `CLOSE` | teardown (release resources); runs in reverse registration order | — | once |

### Cadence

`every` sub-samples a phase: an int runs the callback every Nth emission; a frequency string (`"100Hz"`, `">100Hz"`, `"<100Hz"`) is resolved against the phase's base rate (`fps` for the per-substep `*_STEP` phases, `fps/control_decimation` for the per-frame `FRAME_*` phases). Frequency strings are periodic-phases only; `CLOSE` always fires once. The registry sub-samples natively — hooks never write their own counters.

Built-ins to copy: the dependency-free `none` no-op in `holosoma/simulator/shared/builtin_plugins.py`; `clock_publish` / `gantry_control` / `odometry` (ROS2, `rclpy` imported lazily so core stays ROS-free) in `ros2_plugins.py`. Camera-frame egress ships as plugins too — `ros2-image` / `ros2-stereo` / `ros2-waist-depth*` publish rendered cameras over ROS2, `viz` / `viz-record` tile them into a live window or an mp4; their impls live in `holosoma/simulator/plugins/` and subclass the shared `CameraConsumerPlugin` base (which self-serves each step's fresh frames via `get_camera_data`). Select `plugin.<key>:none` to disable a slot.

## Don't

```python
# ❌ Snapshot merge — misses presets registered later; visible only via this module.
DEFAULTS = {**CORE_DEFAULTS, "x1_25dof": x1_25dof}
# ❌ In-place mutation — import-side-effect global; bypasses the type check.
CORE_DEFAULTS.update({"elf3_29dof": elf3_29dof})
# ❌ Hand-rolled ep loop — no error isolation; one bad plugin crashes the CLI.
for ep in entry_points(group="holosoma.config.inference"): all_defaults[ep.name] = ep.load()
```

`module.DEFAULTS` / `get_defaults()` still work but warn. Use an entry point, or `.add()` / `--import-file`.
