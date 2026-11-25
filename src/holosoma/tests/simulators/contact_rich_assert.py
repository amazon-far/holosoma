"""A launched ball strikes G1, rolls down an incline, and reaches the terrain."""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Any

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.simulators._sim_harness import build_run_sim_config, launch_simulator, run_and_hard_exit  # noqa: E402

_DATA = Path(__file__).resolve().parent / "data"
_BALL_RADIUS = 0.08
_BALL_MASS = 1.0
_BOX_MASS = 2.0
_G1_MASS = 33.343142
_ROOT_HEIGHT = 0.76
_ANGLE = math.radians(15)
_RAMP_POSITION = [0.80, 0.0, 0.22]
_DURATION = 2.25


def _build(backend: str, num_envs: int, device: str | None, record_dir: str | None) -> Any:
    import torch

    from holosoma.config_types.scene import (
        IsaacSimPhysicsConfig,
        MujocoPhysicsConfig,
        PhysicsConfig,
        PhysXPhysicsConfig,
        RigidObjectConfig,
        SceneConfig,
    )
    from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig
    from holosoma.config_types.video import CartesianCameraConfig

    def actor(name: str) -> ContactTargetConfig:
        return ContactTargetConfig(target_kind="actor", target=name)

    robot = ContactTargetConfig()
    terrain = ContactTargetConfig(target_kind="terrain")
    hit = ContactTargetConfig(target_kind="robot_link", target="torso_link")
    untouched = ContactTargetConfig(target_kind="robot_link", target="right_wrist_yaw_link")
    pairs = ContactSensorConfig(sources=[actor("ball")], counterparts=[robot, actor("incline"), terrain])
    sensors = {
        "ball_pairs": pairs,
        "ball_total": dataclasses.replace(pairs, force_mode="total"),
        "ball_links": ContactSensorConfig(sources=[actor("ball")], counterparts=[hit, untouched]),
        "robot_ball": ContactSensorConfig(sources=[robot], counterparts=[actor("ball")]),
        "link_ball": ContactSensorConfig(sources=[hit], counterparts=[actor("ball")]),
        "incline_ball": ContactSensorConfig(sources=[actor("incline")], counterparts=[actor("ball")]),
        "stationary_ground": ContactSensorConfig(sources=[actor("stationary")], counterparts=[terrain]),
        "robot_ground": ContactSensorConfig(sources=[robot], counterparts=[terrain]),
    }

    def physics(mass: float, *, projectile: bool = False) -> PhysicsConfig:
        return PhysicsConfig(
            mass=mass,
            physx=PhysXPhysicsConfig(linear_damping=0.0, angular_damping=0.0),
            isaacsim=IsaacSimPhysicsConfig(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.25 if projectile else 0.0,
                restitution_combine_mode="max",
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            mujoco=MujocoPhysicsConfig(
                friction=[0.5, 0.001, 0.0001], solref=[0.004, 0.4 if projectile else 1.0], condim=3
            ),
        )

    scene = SceneConfig(
        rigid_objects={
            "ball": RigidObjectConfig(
                asset_root=str(_DATA),
                urdf_file="contact_rich_ball.urdf",
                usd_file="contact_rich_ball.usda",
                position=[0.72, 0.0, 1.24],
                linear_velocity=[-3.0, 0.0, 0.0],
                physics=physics(_BALL_MASS, projectile=True),
            ),
            "incline": RigidObjectConfig(
                asset_root=str(_DATA),
                urdf_file="contact_rich_ramp.urdf",
                usd_file="contact_rich_ramp.usda",
                position=_RAMP_POSITION,
                orientation=[math.cos(_ANGLE / 2), 0.0, math.sin(_ANGLE / 2), 0.0],
                fixed=True,
                physics=physics(1.0),
            ),
            "stationary": RigidObjectConfig(
                urdf_file="holosoma/data/scene_objects/boxes/small_box.urdf",
                usd_file="holosoma/data/scene_objects/boxes/small_box.usda",
                position=[2.0, 1.0, 0.05],
                physics=physics(_BOX_MASS),
            ),
        },
        env_spacing=20.0,
    )
    config = build_run_sim_config(
        backend,
        "empty",
        "g1-29dof",
        "terrain_locomotion_plane",
        sensors=sensors,
        record_dir=record_dir,
        video_camera=CartesianCameraConfig(offset=[3.0, -4.0, 1.25], target_offset=[1.0, 0.0, -0.06], smoothing=0.0),
        show_command_overlay=False,
    )
    config = dataclasses.replace(
        config,
        scene=scene,
        robot=dataclasses.replace(
            config.robot, init_state=dataclasses.replace(config.robot.init_state, pos=[0.0, 0.0, _ROOT_HEIGHT])
        ),
        device=device or ("cpu" if backend == "mujoco" else "cuda:0"),
        training=dataclasses.replace(config.training, num_envs=num_envs, headless=True),
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                debug_viz=False,
                sim=dataclasses.replace(config.simulator.config.sim, fps=500, control_decimation=10),
            ),
        ),
    )
    sim, device, _ = launch_simulator(config, device=config.device)
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    origins = torch.zeros(num_envs, 3, device=device)
    origins[:, 0] = torch.arange(num_envs, device=device) * 20
    initial = config.robot.init_state
    root = torch.tensor([*initial.pos, *initial.rot, *initial.lin_vel, *initial.ang_vel], device=device)
    sim.create_envs(num_envs, origins, root)
    sim.prepare_sim()
    # Place every robot at its own origin before the projectile's first physics step.
    ids = torch.arange(num_envs, device=device)
    robot_states = root.expand(num_envs, -1).clone()
    robot_states[:, :3] += sim.env_origins
    sim.set_actor_states(["robot"], ids, robot_states)
    angles = config.robot.init_state.default_joint_angles
    sim.set_dof_state_by_names(None, angles, dict.fromkeys(angles, 0.0))
    torch.testing.assert_close(sim.robot_root_states[:, :3], robot_states[:, :3], atol=1e-5, rtol=0)
    torch.testing.assert_close(sim.robot_root_states[:, 7:13], robot_states[:, 7:13], atol=1e-5, rtol=0)
    return sim


def _run(sim: Any, record: bool) -> dict[str, Any]:
    import torch

    from holosoma.simulator.base_simulator.hooks import Phase

    ids = torch.arange(sim.num_envs, device=sim.sim_device)
    targets = sim.dof_pos.clone()
    anchor = sim.env_origins.clone()
    anchor[:, 2] += _ROOT_HEIGHT
    samples: dict[str, list[Any]] = {name: [] for name in sim.sensor_config}
    states: list[Any] = []
    roots: list[Any] = []
    initial_ball = sim.get_actor_states(["ball"], ids).clone()

    def support_robot() -> None:
        root = sim.robot_root_states
        force = 3000 * (anchor - root[:, :3]) - 150 * root[:, 7:10]
        force[:, 2] += 0.85 * _G1_MASS * 9.81
        sign = torch.where(root[:, 6:7] < 0, -1.0, 1.0)
        torque = -500 * root[:, 3:6] * sign - 30 * root[:, 10:13]
        sim.apply_external_force("robot", forces=force[:, None], torques=torque[:, None], body_names=["pelvis"])
        sim.apply_torques_at_dof(100 * (targets - sim.dof_pos) - 5 * sim.dof_vel)

    def collect() -> None:
        # Copies stay on the simulation device until the run ends.
        for name, history in samples.items():
            history.append(sim.get_contact_data(name, "force_matrix").clone())
        states.append(sim.get_actor_states(["ball", "stationary"], ids).reshape(2, sim.num_envs, 13).clone())
        roots.append(sim.robot_root_states.clone())

    sim.hooks.add(Phase.PRE_STEP, support_robot, name="contact_rich.support")
    sim.hooks.add(Phase.POST_STEP, collect, name="contact_rich.collect")
    recorder = sim.video_recorder
    if record:
        assert recorder is not None
        recorder.setup_recording()
        recorder.on_episode_start(torch.tensor([recorder.config.record_env_id], device=sim.sim_device))
    decimation = sim.simulator_config.sim.control_decimation_steps
    frames = math.ceil(_DURATION / (sim.sim_dt * decimation))
    try:
        for _ in range(frames):
            sim.hooks.emit(Phase.FRAME_BEGIN)
            for _ in range(decimation):
                sim.hooks.emit(Phase.PRE_STEP)
                sim.simulate_at_each_physics_step()
                sim.hooks.emit(Phase.POST_STEP)
            sim.hooks.emit(Phase.FRAME_END)
    finally:
        if record and recorder is not None:
            recorder.on_episode_end(torch.tensor([recorder.config.record_env_id], device=sim.sim_device))
    return {
        "forces": {name: torch.stack(values).cpu() for name, values in samples.items()},
        "states": torch.stack(states).cpu(),
        "roots": torch.stack(roots).cpu(),
        "initial_ball": initial_ball.cpu(),
        "origins": sim.env_origins.cpu(),
        "dt": sim.sim_dt,
    }


def _assert_trace(trace: dict[str, Any]) -> list[dict[str, float]]:
    import torch

    force = trace["forces"]
    ball = trace["states"][:, 0]
    stationary = trace["states"][:, 1]
    dt = trace["dt"]
    time = (torch.arange(len(ball)) + 1) * dt
    normal = force["ball_pairs"][:, :, 0]
    total = force["ball_total"][:, :, 0]
    links = force["ball_links"][:, :, 0]
    torch.testing.assert_close(force["robot_ball"][:, :, 0, 0], -normal[:, :, 0], atol=0.02, rtol=0.002)
    torch.testing.assert_close(force["link_ball"][:, :, 0, 0], -links[:, :, 0], atol=0.02, rtol=0.002)
    torch.testing.assert_close(force["incline_ball"][:, :, 0, 0], -normal[:, :, 1], atol=0.02, rtol=0.002)
    assert links[:, :, 1].abs().max() < 1e-4, "The projectile touched the unintended right wrist."
    direction = torch.tensor([math.sin(_ANGLE), 0.0, math.cos(_ANGLE)])
    tangent = torch.tensor([math.cos(_ANGLE), 0.0, -math.sin(_ANGLE)])
    reports = []
    for env in range(ball.shape[1]):
        position = ball[:, env, :3] - trace["origins"][env]
        velocity = ball[:, env, 7:10]
        magnitude = torch.linalg.vector_norm(normal[:, env], dim=-1)
        assert normal[time < 0.10, env].abs().max() < 1e-4, "The airborne ball reported contact before the impact."
        hit = (position[:, 0] < 0.30) & (position[:, 2] > 0.70) & (magnitude[:, 0] > 5)
        assert hit.any(), f"Environment {env}: the ball missed the robot."
        assert normal[hit, env, 1:].abs().max() < 1e-4, "The robot impact leaked into another counterpart column."
        assert (torch.linalg.vector_norm(links[hit, env, 0], dim=-1) > 5).any(), (
            "The intended torso link did not receive the impact."
        )
        hit_time = time[hit][0].item()
        assert 0.10 < hit_time < 0.40, f"Unexpected robot impact time: {hit_time}"
        assert velocity[(time > hit_time) & (time < hit_time + 0.20), 0].max() > 0.1, (
            "The impact did not reverse the launched ball."
        )

        delta = position - torch.tensor(_RAMP_POSITION)
        local_x = delta[:, 0] * math.cos(_ANGLE) - delta[:, 2] * math.sin(_ANGLE)
        local_z = delta[:, 0] * math.sin(_ANGLE) + delta[:, 2] * math.cos(_ANGLE)
        on_incline = (
            (local_x.abs() < 0.525 - _BALL_RADIUS)
            & ((local_z - 0.05 - _BALL_RADIUS).abs() < 0.025)
            & (magnitude[:, 1] > 1)
        )
        assert on_incline.sum() >= 5, f"Environment {env}: the ball did not travel across the incline."
        incline_time = time[on_incline][0].item()
        assert incline_time > hit_time
        assert normal[on_incline, env][:, [0, 2]].abs().max() < 1e-4, "The incline contact leaked into another column."
        incline_force = normal[on_incline, env, 1]
        alignment = incline_force @ direction / torch.linalg.vector_norm(incline_force, dim=-1)
        assert alignment.min() > 0.99, f"Incline force does not follow its tilted normal: {alignment.min()}"
        tangential_speed = velocity[on_incline] @ tangent
        spin_speed = _BALL_RADIUS * ball[on_incline, env, 11]
        rolling = (tangential_speed > 0.2) & (
            (tangential_speed - spin_speed).abs() < 0.25 * tangential_speed.abs() + 0.2
        )
        assert rolling.any(), "The projectile did not roll down the incline."
        assert torch.linalg.vector_norm(total[on_incline, env, 1] - incline_force, dim=-1).max() > 0.1, (
            "Total force omitted tangential friction."
        )

        on_ground = (position[:, 0] > 1.35) & ((position[:, 2] - _BALL_RADIUS).abs() < 0.015) & (magnitude[:, 2] > 1)
        assert on_ground.any(), f"Environment {env}: the ball did not reach the terrain beyond the incline."
        ground_time = time[on_ground][0].item()
        assert ground_time > incline_time
        ground_force = normal[on_ground, env, 2]
        assert ground_force[:, 2].min() > 0
        assert ground_force[:, :2].abs().max() < 0.01
        ground_stable = on_ground & (time > ground_time + 0.2)
        assert ground_stable.sum() >= 20
        torch.testing.assert_close(
            normal[ground_stable, env, 2].mean(dim=0), torch.tensor([0.0, 0.0, _BALL_MASS * 9.81]), atol=0.5, rtol=0.05
        )

        # The ball has no applied forces. Its velocity change independently constrains total contact impulse.
        impulse = total[:, env].sum(dim=(0, 1)) * dt
        elapsed = len(ball) * dt
        measured = _BALL_MASS * (velocity[-1] - trace["initial_ball"][env, 7:10])
        measured[2] += _BALL_MASS * 9.81 * elapsed
        torch.testing.assert_close(impulse, measured, atol=0.15, rtol=0.10)
        calm = time > 0.15
        box_force = force["stationary_ground"][calm, env, 0, 0]
        torch.testing.assert_close(
            box_force.mean(dim=0), torch.tensor([0.0, 0.0, _BOX_MASS * 9.81]), atol=0.5, rtol=0.05
        )
        assert stationary[calm, env, 7:10].abs().max() < 0.02, "The second actor did not remain stationary."
        robot_ground = force["robot_ground"][calm, env, 0, 0, 2]
        assert robot_ground.min() > 0.5, "The gantry lifted the robot clear of the terrain."
        assert robot_ground.mean() < 0.5 * _G1_MASS * 9.81, "The gantry did not support most of the robot's weight."
        root_height = trace["roots"][calm, env, 2] - trace["origins"][env, 2]
        assert ((root_height > 0.70) & (root_height < 0.82)).all(), "The supported robot fell or lifted off."
        reports.append(
            {
                "robot": hit_time,
                "incline": incline_time,
                "terrain": ground_time,
                "robot_peak_N": magnitude[:, 0].max().item(),
                "incline_samples": float(on_incline.sum()),
            }
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", choices=["mujoco", "mjwarp", "isaacsim"], required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--record-dir", help="Save an optional video using the simulator's video recorder.")
    parser.add_argument("--result-file", type=Path, required=True)
    args = parser.parse_args()
    sim = _build(args.simulator, args.num_envs, args.device, args.record_dir)
    try:
        trace = _run(sim, args.record_dir is not None)
        try:
            reports = _assert_trace(trace)
        except AssertionError:
            import torch

            torch.save(trace, args.result_file.with_suffix(".trace.pt"))
            raise
        print(f"Contact-rich phases ({args.simulator}): {reports}")
        args.result_file.write_text("OK\n")
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    run_and_hard_exit(main)
