import dataclasses

from holosoma.config_types.simulator import (
    IsaacPhysicsBackend,
    MujocoBackend,
    PhysxConfig,
    SimEngineConfig,
    SimulatorConfig,
    SimulatorInitConfig,
)
from holosoma.utils.config_registry import ConfigRegistry

SIMULATOR_REGISTRY = ConfigRegistry(SimulatorConfig, group="holosoma.config.simulator")

isaacsim = SIMULATOR_REGISTRY.add(
    "isaacsim",
    SimulatorConfig(
        _target_="holosoma.simulator.isaacsim.isaacsim.IsaacSim",
        _recursive_=False,
        config=SimulatorInitConfig(
            name="isaacsim",
            sim=SimEngineConfig(
                fps=200,
                control_decimation=4,
                substeps=1,
                physx=PhysxConfig(
                    solver_type=1,
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    bounce_threshold_velocity=0.5,
                ),
                render_mode="human",
                render_interval=4,
            ),
            contact_sensor_history_length=3,
        ),
    ),
)


isaacsim6 = SIMULATOR_REGISTRY.add(
    "isaacsim6",
    SimulatorConfig(
        # Isaac Sim 6.1 / Isaac Lab 3.0, implemented in its own backend package. The two
        # generations diverge in ways a shim cannot express -- 3.0 inverted the quaternion
        # convention globally, renamed or removed most of the accessors holosoma calls, and made
        # the physics engine selectable -- so each generation gets a complete implementation
        # rather than one tree with version branches. The relationship to `isaacsim` above
        # mirrors `mjwarp`'s to `mujoco`: a separate preset naming the same simulator identity.
        #
        # Deliberately a separate preset rather than replacing `isaacsim`: flipping the default
        # would silently change the physics engine under every existing config. That flip happens
        # at a release boundary, at which point this folder is renamed and the 5.1 one deleted.
        _target_="holosoma.simulator.isaacsim6.isaacsim.IsaacSim",
        _recursive_=False,
        config=SimulatorInitConfig(
            # "isaacsim", not "isaacsim6". `SimulatorType` stays at three members, so the ~31
            # `simulator_type ==` comparison sites and the silent `else` branches after them keep
            # working untouched. utils/simulator_config.py derives this name from the last segment
            # of `_target_` (`IsaacSim`.lower()) and rejects a mismatch, so the class in this lane
            # must stay named `IsaacSim`.
            name="isaacsim",
            sim=SimEngineConfig(
                fps=200,
                control_decimation=4,
                substeps=1,
                physx=PhysxConfig(
                    solver_type=1,
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    bounce_threshold_velocity=0.5,
                ),
                render_mode="human",
                render_interval=4,
            ),
            contact_sensor_history_length=3,
        ),
    ),
)


# Isaac Sim 6.1 under the non-default physics engines.
#
# A preset per engine, following `mujoco` / `mjwarp`: same `_target_`, same simulator identity,
# differing only in which engine `SimulationCfg.physics` resolves to. This is also what makes the
# engines testable -- the tests/simulators/*_assert.py harnesses take a `--simulator` preset name
# and plain argparse, so they cannot express a nested config override.
isaacsim6_newton = SIMULATOR_REGISTRY.add(
    "isaacsim6-newton",
    dataclasses.replace(
        isaacsim6,
        config=dataclasses.replace(
            isaacsim6.config,
            isaac_physics_backend=IsaacPhysicsBackend.NEWTON,
        ),
    ),
)


isaacsim6_ovphysx = SIMULATOR_REGISTRY.add(
    "isaacsim6-ovphysx",
    dataclasses.replace(
        isaacsim6,
        config=dataclasses.replace(
            isaacsim6.config,
            isaac_physics_backend=IsaacPhysicsBackend.OVPHYSX,
        ),
    ),
)

mujoco = SIMULATOR_REGISTRY.add(
    "mujoco",
    SimulatorConfig(
        _target_="holosoma.simulator.mujoco.mujoco.MuJoCo",
        _recursive_=False,
        config=SimulatorInitConfig(
            name="mujoco",
            sim=SimEngineConfig(
                fps=200,
                control_decimation=4,
                substeps=1,
                physx=PhysxConfig(
                    solver_type=1,
                    num_position_iterations=4,
                    num_velocity_iterations=0,
                    bounce_threshold_velocity=0.5,
                ),
                render_mode="fake",
                render_interval=1,
            ),
            mujoco_backend=MujocoBackend.CLASSIC,  # Explicit for clarity
        ),
    ),
)


mjwarp = SIMULATOR_REGISTRY.add(
    "mjwarp",
    SimulatorConfig(
        _target_="holosoma.simulator.mujoco.mujoco.MuJoCo",
        _recursive_=False,
        config=SimulatorInitConfig(
            name="mujoco",
            sim=SimEngineConfig(
                fps=200,
                control_decimation=4,
                substeps=1,
                physx=PhysxConfig(
                    solver_type=1,
                    num_position_iterations=4,
                    num_velocity_iterations=0,
                    bounce_threshold_velocity=0.5,
                ),
                render_mode="fake",
                render_interval=1,
            ),
            mujoco_backend=MujocoBackend.WARP,  # GPU-accelerated backend
        ),
    ),
)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, SIMULATOR_REGISTRY)
