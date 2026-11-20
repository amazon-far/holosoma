from holosoma.config_types.simulator import (
    MujocoBackend,
    PhysxConfig,
    SceneConfig,
    SimEngineConfig,
    SimulatorConfig,
    SimulatorInitConfig,
)

isaacgym = SimulatorConfig(
    _target_="holosoma.simulator.isaacgym.isaacgym.IsaacGym",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="isaacgym",
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
        ),
        # Must be >= control_decimation (4) so the contact-force history covers
        # a full control interval for the touchdown peak-force penalty.
        contact_sensor_history_length=4,
    ),
)


isaacsim = SimulatorConfig(
    _target_="holosoma.simulator.isaacsim.isaacsim.IsaacSim",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="isaacsim",
        scene=SceneConfig(
            replicate_physics=True,
        ),
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
        # Must be >= control_decimation (4) so the contact-force history covers
        # a full control interval for the touchdown peak-force penalty.
        contact_sensor_history_length=4,
    ),
)


mujoco = SimulatorConfig(
    _target_="holosoma.simulator.mujoco.mujoco.MuJoCo",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="mujoco",
        scene=SceneConfig(
            replicate_physics=True,
        ),
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
)


mjwarp = SimulatorConfig(
    _target_="holosoma.simulator.mujoco.mujoco.MuJoCo",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="mujoco",
        scene=SceneConfig(
            replicate_physics=True,
        ),
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
)


DEFAULTS = {
    "isaacgym": isaacgym,
    "isaacsim": isaacsim,
    "mujoco": mujoco,
    "mjwarp": mjwarp,
}
