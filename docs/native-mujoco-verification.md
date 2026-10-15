# Native MuJoCo configuration: verification

Verified on macOS ARM64 on 2026-10-15, based on public `main` at `b633c2fc`.

## Environment and results

The setup scripts specify MuJoCo **3.10.0**, MJWarp
**ecaef88917a3c90cd238bf76681ca770f58033df**, and Warp **1.15.0**.
Those versions were installed in an isolated test environment, reusing the
existing Holosoma environment's other dependencies through symlinks. External
extension packages and their distribution metadata were excluded. The integration
tests construct the actual core simulator with an empty plugin configuration.
CLI validation used Tyro **1.0.13** and Pydantic **2.13.4**.

Commands below run from the repository root with that pinned environment active:

```bash
PYTHONPATH=src/holosoma python -m pytest -q \
  src/holosoma/holosoma/config_types/tests/test_native_mujoco_config.py \
  src/holosoma/tests/simulators/mujoco/test_native_config.py \
  src/holosoma/tests/simulators/mujoco/test_native_inventory.py \
  --tb=short -rs
```

**70 passed, 1 skipped**, no errors or expected failures. The skip is
`test_core_warp_actual_upload_allocation_capture_and_reset`: this Mac has no
CUDA device.

```bash
PYTHONPATH=src/holosoma python -m pytest -q \
  src/holosoma/holosoma/config_types/tests \
  src/holosoma/tests/simulators/mujoco/test_native_config.py \
  src/holosoma/tests/simulators/mujoco/test_native_inventory.py \
  src/holosoma/tests/simulators/mujoco/test_link_physics_apply_mujoco.py \
  src/holosoma/tests/simulators/mujoco/test_physics_clock.py \
  --tb=short -rs
```

**490 passed, 10 failed, 3 skipped** out of 503 collected; no tests were
deselected. The skips are the native Warp stepping test, the existing Warp clock
test, and the existing CUDA quaternion tensor test.

All ten failures are existing `test_rotation_cli_preserves_literal_syntax`
parameterizations. With these installed CLI dependencies, invalid quaternions
raise Pydantic `ValidationError`, while those tests expect `SystemExit`.
The same ten failures were reproduced on unchanged public `main` at `b633c2fc`
in a separate checkout using the same environment. They are not counted as
passing checks.

All commit hooks passed, including Ruff, formatting, isolated mypy, and
whitespace checks. The full `mypy .` check could not complete on this host:
`warp.context` has no installed stub, and the configured Python 3.8 checker
cannot parse pattern matching in the installed Matplotlib. Both blockers were
also reproduced on unchanged public `main`.

The bounded classic example in
`examples/mujoco_native.py` ran with `implicitfast`, `elliptic`, `impratio=10`,
and `noslip_iterations=1`, with no plugins. Twenty steps advanced native time
to **0.010000 s** at the standalone preset's default 2000 Hz.

A second bounded example started from `simulator:mjwarp` and selected
`--simulator.config.mujoco-backend CLASSIC`, retaining RK4, capacities,
`warp.graph_conditional=False`, and common/Warp extras. Twenty real CPU steps
advanced native time to **0.050000 s** at the retained Warp preset's 400 Hz.
The integration test checks that shared options reach the compiled model while
Warp-only settings remain unused on classic.

The inventory is checked-in documentation, with no runtime dependency or package
metadata change. Classic ignores the `warp` block without optional imports;
active Warp validation uses installed definitions and signatures.

The real MJWarp allocator and conversion ran on Warp's **CPU device**.
They verify pooled allocation dimensions, uploaded array values, tolerance
restoration, and flags. They provide no evidence about CUDA kernels or graph
execution. The CUDA test performs actual core initialization, device readback,
allocation, graph capture, twenty free-fall steps with an analytic RK4 trajectory,
and state reinitialization. It must pass on the supported CUDA stack before
claiming GPU validation. No mocked conversion or stepping results substitute
for it.

A separate check with the installed legacy MuJoCo **3.9.0** environment
confirmed that unconfigured classic setup still compiles and uses `1/fps`;
an explicit global native override fails with the version requirement and
`simulator.config.sim.mujoco` path.

## Test audit

Applied the `test-audit` skill. **BS Meter: 8/10** for this change's evidence:
71 focused cases, 70 honest executed cases, 1 not assessed on its required
GPU platform, no theatrical cases identified. No production application or
stepping logic is mocked.

| Finding | Evidence / limit |
| --- | --- |
| CPU application is constrained by observable behavior | Every remaining public MjOption uses a non-default value through `extra` and is checked on the compiled model. All four integrators step a real model; RK4 and Euler trajectories differ. Contacts enabled/disabled change the real trajectory. |
| Configuration owners and failure behavior are covered | Shared training/standalone parsing for both backend presets, preset/YAML round trips, experiment serialization, explicit zero/false/empty masks, unknown native names/types, unsupported Warp combinations, and relocated-capacity errors are exercised. Warp fields remain inactive on classic, including explicit zero/false values. A saved Warp config overrides only its backend, then initializes, steps and resets the actual classic simulator with empty plugins. |
| Inactive extras need no optional backend imports | A fresh subprocess blocks Warp/MJWarp imports and applies real classic options to a minimal model, compiles it and steps the RK4 trajectory. Only optional dependency discovery is altered; configuration application and stepping remain production code. |
| Warp extras reach the real public API | Actual Warp CPU allocation verifies contact/CCD pool sizes. Actual CPU-device conversion verifies graph settings and a named broadphase flag combination. Structural/generated and shared-option duplicates are rejected. |
| Core setup needs no tuning plugin | Actual G1 core setup is reconstructed twice with empty plugins, then stepped and reset. Signed contact parameters, masks, mixing, margins and gaps reach compiled geometry. |
| GPU evidence remains unavailable | The actual CUDA test is transparently skipped. CPU Warp conversion cannot establish that captured GPU execution consumes the settings. |
| Upstream coverage is guarded | Inventory comparison covers option definitions, enum choices, allocation/conversion/capture signatures, batchable fields, kernel block fields and compilation options. Changing a supported dependency requires compatibility review. |

Break-it checks were **reasoned, not executed mutations**: ignoring the
integrator would fail the RK4 trajectory; ignoring contact flags would fail the
enabled/disabled dynamics test; dropping `extra` application would fail the
non-default compiled-option checks; skipping uploaded tolerance restoration
would fail the real CPU-device array assertion. GPU kernel application remains
unassessed until the CUDA test runs. Full policy training, rendering and teleop
end-to-end execution were not performed.

The native schema follows Holosoma's frozen Pydantic dataclass pattern.
`MujocoConfig` contains common physics choices and `extra: {native_name: value}`;
its `warp: MujocoWarpConfig` contains four named settings and one `extras`
mapping for uncommon native options/allocation inputs. There is no prefix parser
or second Warp override of a shared physics option. The inventory also guards
option/allocation name collisions. Model batching, kernel block sizes,
process-wide compiler settings and capture orchestration retain core behavior.

IsaacGym/IsaacSim execution was unavailable on this Mac. The inactive MuJoCo
block is filtered from IsaacGym's native parameter input. Existing dataclass
experiment serialization and training logging remain unchanged.
