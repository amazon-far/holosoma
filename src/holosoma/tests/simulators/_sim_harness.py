"""Shared boot/step helpers for the standalone cross-backend assertion harnesses.

``behavior_assert`` / ``static_move_assert`` / ``subset_write_assert`` / ``scene_spawn_assert`` each
build a sim, step it, and check physical outcomes. The truly common pieces — the sim-time -> step
count conversion, the per-step advance (+ optional render), and the bridge/gantry-disabled
``RunSimConfig`` build — live here so they are defined once. The scenario-specific ``main`` flows
(which scenes, which asserts) stay in each harness; only the duplicated plumbing is shared.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from collections.abc import Callable
from typing import Any, NoReturn

import tyro

from holosoma.config_types.run_sim import RunSimConfig

SIMULATOR_CHOICES = (
    "mujoco",
    "mjwarp",
    "isaacsim",
    "isaacsim6",
    "isaacsim6-newton",
    "isaacsim6-ovphysx",
)
"""Every ``--simulator`` a harness accepts, matching the ``RUN_SIM_REGISTRY`` preset names.

Shared rather than repeated per harness: the same literal was previously inlined in 18 files, so
adding a backend meant editing all 18 and silently skipping any that was missed -- the harnesses
are the cross-backend drift detector, so a backend absent from one of them is a gap in exactly
the safety net that is supposed to catch divergence.
"""

ISAAC_SIM_LANES = frozenset({"isaacsim", "isaacsim6", "isaacsim6-newton", "isaacsim6-ovphysx"})
"""The preset names that select an Isaac Sim backend, one per Isaac Sim generation."""


def is_isaacsim(simulator: str) -> bool:
    """Whether a ``--simulator`` name selects any Isaac Sim generation.

    Harnesses branch on the backend to express Isaac-specific expectations (prim-path checks,
    ``/World/envs`` naming, RTX-sensor behaviour). Those branches are about Isaac Sim as a family,
    not about one generation, so comparing against the bare string ``"isaacsim"`` would route the
    6.1 lane into the ``else`` -- skipping the Isaac assertions, or applying another backend's,
    while still reporting a pass. That is a silent hole, so the family test is named once here.
    """
    return simulator in ISAAC_SIM_LANES


def run_and_hard_exit(main: Callable[[], int]) -> NoReturn:
    """Run a harness ``main`` and terminate the process immediately with its return code.

    Standalone IsaacSim harnesses MUST end this way instead of ``sys.exit(main())``. On IsaacSim,
    the interpreter's normal atexit teardown deadlocks in ``carbOnPluginShutdown`` while unwinding
    the ``omni.syntheticdata`` / OmniGraph render-product graph a TiledCamera creates, so the
    process never exits — the parent's ``subprocess`` timeout then SIGKILLs it, turning a PASS
    (verdict already written to ``--result-file``) into a spurious timeout failure. Flushing then
    ``os._exit`` skips the atexit teardown entirely; rendering itself is fine, only clean shutdown
    hangs. (behavior_assert / scene_spawn_assert use the same escape for the MuJoCo-viewer race.)

    Use in a harness entrypoint::

        if __name__ == "__main__":
            run_and_hard_exit(main)
    """
    try:
        rc = main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        import traceback

        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def steps_for_seconds(sim: Any, seconds: float) -> int:
    """Number of physics steps that advance ``seconds`` of SIM-TIME on this backend.

    Every backend advances exactly one ``sim.sim_dt`` physics step per
    ``simulate_at_each_physics_step`` / ``backend.step()`` call, so a wall of sim-time is a simple
    per-backend step count — one ``SECONDS`` constant means the same physical duration on all four
    backends instead of a step count tuned to the fastest one.
    """
    return max(1, math.ceil(seconds / sim.sim_dt))


def step(sim: Any, n: int, *, render: bool = False) -> None:
    """Advance ``n`` physics steps. With ``render=True`` also drive the render path each step.

    Every harness sim exposes ``simulate_at_each_physics_step``. ``render`` is on for the recording
    harness (so a video captures the motion) and off for the headless physics-only checks.
    """
    for _ in range(n):
        sim.refresh_sim_tensors()
        sim.simulate_at_each_physics_step()
        if render:
            sim.render()


def build_run_sim_config(
    simulator: str,
    scene: str,
    robot: str,
    terrain: str,
    *,
    sensors: dict[str, Any] | None = None,
    record_dir: str | None = None,
    video_camera: Any = None,
    show_command_overlay: bool = True,
    visualizers: str | None = None,
) -> RunSimConfig:
    """Parse a ``RunSimConfig`` as the CLI would, with the bridge + virtual gantry disabled.

    The bridge binds a ZMQ clock publisher that lingers between rapid subprocess runs, and the gantry
    applies a robot-only force tensor that errors once a scene adds bodies — neither is relevant to a
    spawn/behavior check, so both are turned off.

    ``sensors`` is a mounted-camera dict (``{name: CameraSensorConfig}``) injected directly onto the
    parsed config's ``sensor`` field — the harness builds rigs in Python rather than through the
    per-key ``--sensor`` CLI (a multi-camera rig can't be named by one token). When ``record_dir`` is
    set, enable video recording every episode to that dir (mp4, no wandb). ``video_camera`` overrides
    the camera config (e.g. an object-framing camera); ``show_command_overlay`` toggles the
    robot-command text overlay (off for object scenarios where it is irrelevant).
    """
    from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg

    argv = [f"simulator:{simulator}"]
    if visualizers is not None:
        # The renderer is the second independent axis on Isaac Sim 6.2, and the harnesses could not
        # reach it at all: they take a preset name through plain argparse, so `isaac_visualizers`
        # could not be varied and the rendering dimension went untested. Overridden here rather
        # than by adding a preset per visualizer, since those combinations multiply with the
        # physics backends.
        #
        # Two placement rules, both of which tyro enforces by rejecting the command outright:
        # the option must directly follow its own `simulator:` subcommand ("arguments are applied
        # to the directly preceding subcommand"), and the flag and its values must be separate
        # argv entries because the field is a tuple and so is parsed as variadic.
        argv.append("--simulator.config.isaac-visualizers")
        argv.extend(visualizers.split(","))
    argv += [f"robot:{robot}", f"terrain:{terrain}", f"scene:{scene}"]
    config = tyro.cli(RunSimConfig, args=argv)
    if sensors is not None:
        config = dataclasses.replace(config, sensor=dict(sensors))
    sim_cfg = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(
            config.simulator.config,
            bridge=BridgeConfig(enabled=False),
            virtual_gantry=VirtualGantryCfg(enabled=False),
        ),
    )
    config = dataclasses.replace(
        config, simulator=sim_cfg, training=dataclasses.replace(config.training, headless=True)
    )

    if record_dir is not None:
        video_kwargs: dict[str, Any] = {
            "enabled": True,
            "interval": 1,
            "save_dir": record_dir,
            "output_format": "mp4",
            "upload_to_wandb": False,
            "show_command_overlay": show_command_overlay,
        }
        if video_camera is not None:
            video_kwargs["camera"] = video_camera
        video = dataclasses.replace(config.logger.video, **video_kwargs)
        config = dataclasses.replace(config, logger=dataclasses.replace(config.logger, video=video))
    return config


ISAAC_LANE_PACKAGES = {
    "isaacsim": "holosoma.simulator.isaacsim",
    "isaacsim6": "holosoma.simulator.isaacsim6",
    # The physics-engine variants are the same lane; they differ only in `isaac_physics_backend`.
    # Listed explicitly rather than derived by prefix so adding a lane cannot silently inherit
    # another's package -- and listed at all because omitting them made every lane-module lookup
    # fall through to the 5.1 path, which then failed with a PhysX-only AttributeError that looked
    # like a Newton bug rather than a missing table entry.
    "isaacsim6-newton": "holosoma.simulator.isaacsim6",
    "isaacsim6-ovphysx": "holosoma.simulator.isaacsim6",
}
"""Backend package for each Isaac Sim preset name. Engine variants share their lane's package."""


def isaac_lane_module(simulator: str, submodule: str):
    """Import ``submodule`` from the Isaac backend package that ``simulator`` names.

    The Isaac-specific harnesses (``isaacsim_articulation_cfg_assert``,
    ``object_spawner_cfg_assert``) exercise code that lives inside a backend package rather than
    behind ``BaseSimulator``, so they have to name one. Hard-coding the 5.1 package made them
    silently untestable against 6.1 -- worse, running them under the 6.1 environment *imports the
    5.1 lane*, which then fails on an Isaac Sim 5.1-only module and looks like a 6.1 bug.

    Unlike :func:`holosoma.simulator.isaac_lane.isaac_lane_submodule`, this keys off the
    ``--simulator`` string, because these harnesses check configuration builders and never
    construct a simulator to inspect.

    Raises
    ------
    ValueError
        If ``simulator`` is not an Isaac Sim preset. These harnesses are Isaac-only by nature, so
        a MuJoCo name is a usage error rather than something to skip.
    """
    import importlib

    if simulator not in ISAAC_LANE_PACKAGES:
        raise ValueError(
            f"{simulator!r} is not an Isaac Sim backend; this harness checks Isaac-specific "
            f"configuration builders. Choose one of {sorted(ISAAC_LANE_PACKAGES)}."
        )
    return importlib.import_module(f"{ISAAC_LANE_PACKAGES[simulator]}.{submodule}")


def class_type_is(class_type, expected: type) -> bool:
    """Whether a cfg's ``class_type`` designates ``expected``, on either Isaac Lab generation.

    Isaac Lab 2.3.2 stores the class object itself, so ``cfg.class_type is SomeClass`` works.
    3.0 stores a lazy ``"module:QualName"`` reference and resolves it on use: reading
    ``cfg.class_type`` on an *instance* yields an ``isaaclab.utils.string.ResolvableString``,
    which is callable (so construction still works) but is **not** the class, so an ``is``
    comparison silently becomes False and the assertion fails for the wrong reason.

    Comparing the resolved text is exact rather than approximate: 3.0's value is precisely
    ``f"{cls.__module__}:{cls.__qualname__}"``, so this does not fall back to matching bare class
    names, which would conflate same-named classes from different backends (and 3.0 has one
    actuator implementation per physics backend).
    """
    if class_type is expected:
        return True
    return str(class_type) == f"{expected.__module__}:{expected.__qualname__}"
