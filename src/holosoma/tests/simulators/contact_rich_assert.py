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
_SOURCES = ("ball", "robot", "torso", "wrist", "incline", "stationary")
_COUNTERPARTS = (*_SOURCES, "terrain")


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
    sources = [actor("ball"), robot, hit, untouched, actor("incline"), actor("stationary")]
    # Robot and link counterparts overlap, so they use separate matrices.
    sensors = {}
    for mode in ("normal", "total"):
        sensors[f"scene_{mode}"] = ContactSensorConfig(
            sources=sources,
            counterparts=[actor("ball"), robot, actor("incline"), actor("stationary"), terrain],
            force_mode=mode,
        )
        sensors[f"links_{mode}"] = ContactSensorConfig(sources=sources, counterparts=[hit, untouched], force_mode=mode)

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
            config.robot, init_state=dataclasses.replace(config.robot.init_state, position=[0.0, 0.0, _ROOT_HEIGHT])
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
    root = torch.tensor(initial.root_state_list(), device=device)
    sim.create_envs(num_envs, origins, root)
    sim.prepare_sim()
    # Place every robot at its own origin before the projectile's first physics step.
    ids = torch.arange(num_envs, device=device)
    robot_states = root.expand(num_envs, -1).clone()
    robot_states[:, :3] += sim.env_origins
    sim.set_actor_states(["robot"], ids, robot_states)
    angles = config.robot.default_dof_positions()
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
    torso_poses: list[Any] = []
    torso_id = sim.find_rigid_body_indice("torso_link")
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
        torso_poses.append(
            torch.cat(
                [sim.rigid_body_pos_w[:, torso_id], sim.rigid_body_quat_w[:, torso_id]],
                dim=-1,
            ).clone()
        )

    sim.hooks.add(Phase.PRE_STEP, support_robot, name="contact_rich.support")
    sim.hooks.add(Phase.POST_STEP, collect, name="contact_rich.collect", after=sim.sensor_contact_hook)
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
        "torso": torch.stack(torso_poses).cpu(),
        "initial_ball": initial_ball.cpu(),
        "origins": sim.env_origins.cpu(),
        "dt": sim.sim_dt,
    }


def _assert_stage(
    matrices: dict[str, Any],
    env: int,
    stage: str,
    mask: Any,
    contacts: set[tuple[int, int]],
) -> None:
    import torch

    assert mask.any(), f"Environment {env}: no geometry samples for {stage}."
    expected = torch.zeros(len(_SOURCES), len(_COUNTERPARTS), dtype=torch.bool)
    for source, counterpart in contacts:
        expected[source, counterpart] = True
    for mode, matrix in matrices.items():
        samples = matrix[mask, env]
        absent = samples.abs().amax(dim=(0, 3)).masked_fill(expected, 0)
        largest = int(absent.argmax())
        source, counterpart = divmod(largest, len(_COUNTERPARTS))
        assert absent.max() < 1e-4, (
            f"Environment {env}, {stage}, {mode}: unexpected force "
            f"{_SOURCES[source]} -> {_COUNTERPARTS[counterpart]}: {absent.max().item()} N."
        )
        if contacts:
            present = samples[:, expected].norm(dim=-1).amax(dim=0)
            assert (present > 1).all(), f"Environment {env}, {stage}, {mode}: missing expected contact."


def _assert_trace(trace: dict[str, Any]) -> list[dict[str, float]]:
    import torch

    from holosoma.utils.rotations import quat_rotate_inverse

    force = trace["forces"]
    ball = trace["states"][:, 0]
    stationary = trace["states"][:, 1]
    dt = trace["dt"]
    time = (torch.arange(len(ball)) + 1) * dt
    matrices = {
        mode: torch.cat(
            [force[f"scene_{mode}"][:, :, :, :2], force[f"links_{mode}"], force[f"scene_{mode}"][:, :, :, 2:]],
            dim=3,
        )
        for mode in ("normal", "total")
    }
    for matrix in matrices.values():
        torch.testing.assert_close(matrix[:, :, :, :6], -matrix[:, :, :, :6].transpose(2, 3), atol=0.02, rtol=0.002)
    normal = matrices["normal"][:, :, 0, [1, 4, 6]]
    total = matrices["total"][:, :, 0, [1, 4, 6]]
    direction = torch.tensor([math.sin(_ANGLE), 0.0, math.cos(_ANGLE)])
    tangent = torch.tensor([math.cos(_ANGLE), 0.0, -math.sin(_ANGLE)])
    permanent = {(1, 6), (5, 6)}
    impact_pairs = {(0, 1), (1, 0), (0, 2), (2, 0)}
    incline_pairs = {(0, 4), (4, 0)}
    reports = []
    for env in range(ball.shape[1]):
        position = ball[:, env, :3] - trace["origins"][env]
        velocity = ball[:, env, 7:10]
        torso = trace["torso"][:, env]
        local_ball = quat_rotate_inverse(torso[:, 3:7], ball[:, env, :3] - torso[:, :3], w_last=True)
        # The struck G1 torso capsule has a Y-axis segment at (x=.005, z=.22), radius .073.
        capsule_delta = local_ball - torch.tensor([0.005, 0.0, 0.22])
        capsule_delta[:, 1] = capsule_delta[:, 1].sign() * (capsule_delta[:, 1].abs() - 0.032).clamp(min=0)
        torso_gap = capsule_delta.norm(dim=-1) - 0.073 - _BALL_RADIUS
        hit = (torso_gap < 0.006) & (position[:, 2] > 0.70) & (time > 0.15) & (time < 0.35)
        assert hit.any(), f"Environment {env}: the ball did not reach the torso geometry."
        hit_time = time[hit][0].item()
        assert normal[hit, env, 0].norm(dim=-1).max() > 5, "The robot did not receive the geometric impact."
        assert matrices["normal"][hit, env, 0, 2].norm(dim=-1).max() > 5, (
            "The intended torso link did not receive the geometric impact."
        )
        assert 0.10 < hit_time < 0.40, f"Unexpected robot impact time: {hit_time}"
        assert velocity[(time > hit_time) & (time < hit_time + 0.20), 0].max() > 0.1, (
            "The impact did not reverse the launched ball."
        )

        delta = position - torch.tensor(_RAMP_POSITION)
        local_x = delta[:, 0] * math.cos(_ANGLE) - delta[:, 2] * math.sin(_ANGLE)
        local_z = delta[:, 0] * math.sin(_ANGLE) + delta[:, 2] * math.cos(_ANGLE)
        incline_gap = local_z - 0.05 - _BALL_RADIUS
        box_distance = (
            torch.stack(
                [local_x.abs() - 0.525, delta[:, 1].abs() - 0.325, local_z.abs() - 0.05],
                dim=-1,
            )
            .clamp(min=0)
            .norm(dim=-1)
            - _BALL_RADIUS
        )
        # Solved forces precede the recorded poses by at most one integration step.
        possible = torch.zeros(len(time), len(_SOURCES), len(_COUNTERPARTS), dtype=torch.bool)
        for source, counterpart in permanent:
            possible[:, source, counterpart] = True
        for source, counterpart in impact_pairs:
            possible[:, source, counterpart] = torso_gap < 0.015
        for source, counterpart in incline_pairs:
            possible[:, source, counterpart] = box_distance < 0.015
        possible[:, 0, 6] = position[:, 2] - _BALL_RADIUS < 0.015
        for mode, matrix in matrices.items():
            unexpected = matrix[:, env].abs().masked_fill(possible[..., None], 0)
            index = int(unexpected.argmax())
            sample, remainder = divmod(index, len(_SOURCES) * len(_COUNTERPARTS) * 3)
            source, remainder = divmod(remainder, len(_COUNTERPARTS) * 3)
            counterpart = remainder // 3
            assert unexpected.max() < 1e-4, (
                f"Environment {env}, {mode}, step {sample + 1}: force between separated bodies "
                f"{_SOURCES[source]} -> {_COUNTERPARTS[counterpart]}: {unexpected.max().item()} N."
            )
        near_incline = (
            (local_x.abs() < 0.525 + _BALL_RADIUS)
            & (delta[:, 1].abs() < 0.325 + _BALL_RADIUS)
            & (incline_gap.abs() < 0.003)
        )
        assert near_incline.any(), f"Environment {env}: the ball missed the incline geometry."
        first_incline = time[near_incline][0]
        tangential_speed = velocity @ tangent
        rolling_geometry = (
            (local_x.abs() < 0.525 - _BALL_RADIUS)
            & (delta[:, 1].abs() < 0.325 - _BALL_RADIUS)
            & (incline_gap.abs() < 0.0002)
            & ((velocity @ direction).abs() < 0.02)
            & (tangential_speed > 0.2)
        )
        # Require a stable geometry window to exclude the apex of a small bounce.
        on_incline = rolling_geometry.clone()
        for offset in range(1, 6):
            on_incline[offset:] &= rolling_geometry[:-offset]
            on_incline[:-offset] &= rolling_geometry[offset:]
        assert on_incline.sum() >= 5, f"Environment {env}: the ball did not travel across the incline."
        incline_time = time[on_incline][0].item()
        assert incline_time > hit_time
        incline_force = normal[on_incline, env, 1]
        assert incline_force.norm(dim=-1).min() > 1, "The rolling ball received no incline support."
        alignment = incline_force @ direction / torch.linalg.vector_norm(incline_force, dim=-1)
        assert alignment.min() > 0.99, f"Incline force does not follow its tilted normal: {alignment.min()}"
        spin_speed = _BALL_RADIUS * ball[on_incline, env, 11]
        rolling = (tangential_speed[on_incline] - spin_speed).abs() < 0.25 * tangential_speed[on_incline].abs() + 0.2
        assert rolling.any(), "The projectile did not roll down the incline."
        assert torch.linalg.vector_norm(total[on_incline, env, 1] - incline_force, dim=-1).max() > 0.1, (
            "Total force omitted tangential friction."
        )

        on_ground = (position[:, 0] > 1.35) & ((position[:, 2] - _BALL_RADIUS).abs() < 0.003)
        assert on_ground.any(), f"Environment {env}: the ball did not reach the terrain beyond the incline."
        ground_time = time[on_ground][0].item()
        assert ground_time > incline_time
        ground_geometry = (
            on_ground
            & (time > ground_time + 0.2)
            & ((position[:, 2] - _BALL_RADIUS).abs() < 0.0002)
            & (velocity[:, 2].abs() < 0.02)
        )
        ground_stable = ground_geometry.clone()
        for offset in range(1, 6):
            ground_stable[offset:] &= ground_geometry[:-offset]
            ground_stable[:-offset] &= ground_geometry[offset:]
        assert ground_stable.sum() >= 20
        ground_force = normal[ground_stable, env, 2]
        assert ground_force[:, 2].min() > 0
        assert ground_force[:, :2].abs().max() < 0.01
        torch.testing.assert_close(
            normal[ground_stable, env, 2].mean(dim=0), torch.tensor([0.0, 0.0, _BALL_MASS * 9.81]), atol=0.5, rtol=0.05
        )
        stages = {
            "launch": ((time < 0.18) & (torso_gap > 0.025), permanent),
            "robot impact": (hit, permanent | impact_pairs),
            "flight to incline": (
                (time > 0.21) & (time < first_incline) & (torso_gap > 0.03) & (incline_gap > 0.03),
                permanent,
            ),
            "incline rolling": (on_incline, permanent | incline_pairs),
            "flight to terrain": (
                (local_x > 0.525 + _BALL_RADIUS + 0.02)
                & (position[:, 2] > _BALL_RADIUS + 0.02)
                & (velocity[:, 2] < -0.02)
                & (time < ground_time),
                permanent,
            ),
            "terrain settling": (ground_stable, permanent | {(0, 6)}),
        }
        for stage, (mask, contacts) in stages.items():
            _assert_stage(matrices, env, stage, mask, contacts)

        # The ball has no applied forces. Its velocity change independently constrains total contact impulse.
        impulse = total[:, env].sum(dim=(0, 1)) * dt
        elapsed = len(ball) * dt
        measured = _BALL_MASS * (velocity[-1] - trace["initial_ball"][env, 7:10])
        measured[2] += _BALL_MASS * 9.81 * elapsed
        torch.testing.assert_close(impulse, measured, atol=0.15, rtol=0.10)
        calm = time > 0.15
        box_force = matrices["normal"][calm, env, 5, 6]
        torch.testing.assert_close(
            box_force.mean(dim=0), torch.tensor([0.0, 0.0, _BOX_MASS * 9.81]), atol=0.5, rtol=0.05
        )
        assert stationary[calm, env, 7:10].abs().max() < 0.02, "The second actor did not remain stationary."
        robot_ground = matrices["normal"][calm, env, 1, 6, 2]
        assert robot_ground.min() > 0.5, "The gantry lifted the robot clear of the terrain."
        assert robot_ground.mean() < 0.5 * _G1_MASS * 9.81, "The gantry did not support most of the robot's weight."
        root_height = trace["roots"][calm, env, 2] - trace["origins"][env, 2]
        assert ((root_height > 0.70) & (root_height < 0.82)).all(), "The supported robot fell or lifted off."
        reports.append(
            {
                "robot": hit_time,
                "incline": incline_time,
                "terrain": ground_time,
                "robot_peak_N": normal[:, env, 0].norm(dim=-1).max().item(),
                "incline_samples": float(on_incline.sum()),
                **{f"{stage}_samples": float(mask.sum()) for stage, (mask, _) in stages.items()},
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
