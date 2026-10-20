"""Cross-backend harness: the YAM arm follows its shipped joint trajectory and reaches the expected poses.

Builds the fixed-base ``yam`` robot with the ``yam-trajectory`` controller plugin, as in
``python -m holosoma.run_sim robot:yam plugin.motion:yam-trajectory --simulator.config.bridge.enabled=False``.
At the end of each hold segment it checks:

  JOINTS   dof_pos within DOF_ATOL of the waypoint targets (the PD has no gravity compensation, so the arm sags
           by gravity torque / kp: up to ~0.035 rad on joint3).
  EE       the ``gripper`` body's world pose is within EE_POS_ATOL / EE_ANG_ATOL of the forward kinematics of the
           targets (EXPECTED below, computed with MuJoCo from data/robots/yam/yam.xml; when mujoco is importable
           the table is recomputed and a drift fails the run).
  GRIPPER  the finger tips are OPEN_TIP_DISTANCE / CLOSED_TIP_DISTANCE apart.
  STILL    dof_vel is small at the end of a hold.
  BASE     robot_root_states stays at the spawn pose (fixed base).
  COLOURS  (IsaacSim) the spawned visual materials carry the URDF link colours (the importer leaves STL white).

Exits 0 on success, 1 with FAIL lines otherwise; writes "OK" to ``--result-file`` on success. ``--record DIR``
also saves an mp4 of the run through the backend's video recorder, from a fixed third-person camera, at 50 frames
per second of sim time (the control rate is set to 50 Hz for the run; the plugin PD runs every physics step).

Usage:
  python yam_trajectory_assert.py --simulator mujoco      # ClassicBackend, cpu
  python yam_trajectory_assert.py --simulator mjwarp      # WarpBackend, cuda
  python yam_trajectory_assert.py --simulator isaacgym
  python yam_trajectory_assert.py --simulator isaacsim
  python yam_trajectory_assert.py --simulator mujoco --record /tmp/yam_video
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from typing import Any

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.config_types.simulator import BridgeConfig
from holosoma.config_types.video import FixedCameraConfig
from holosoma.config_values.plugin import yam_trajectory
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.utils.path import resolve_asset_path
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit

ROBOT_KEY = "yam"
EE_BODY = "gripper"
TIPS = ("tip_left", "tip_right")
DOF_ATOL_ARM = 0.045  # rad; the gravity sag with kp 200 / 50 and no gravity compensation is 0.034 on every backend
DOF_ATOL_FINGER = 0.004  # m
EE_POS_ATOL = 0.025  # m (measured 0.019)
EE_ANG_ATOL = 0.10  # rad (measured 0.081)
VEL_ATOL = 0.25  # rad/s or m/s at the end of a hold
BASE_ATOL = 1e-5
FK_TABLE_ATOL = 5e-3  # EXPECTED is rounded to 4 decimals; a live MuJoCo forward-kinematics recompute must agree
OPEN_TIP_DISTANCE, CLOSED_TIP_DISTANCE = 0.0951, 0.0488  # finger joints at -0.003 / -0.04 m
TIP_DISTANCE_ATOL = 0.01
BACKEND_PARITY_DOF_ATOL = 1e-3

# End time of each hold segment of the shipped trajectory -> (arm pose name, gripper open?). Times and joint
# targets mirror ``yam_trajectory`` in config_values/plugin.py.
CHECKS = [
    (2.75, "reach", True),
    (4.0, "reach", False),
    (6.75, "swing", False),
    (8.0, "swing", True),
    (10.75, "home", True),
]
ARM_TARGETS = {
    "home": [0.0, 1.047, 1.047, 0.0, 0.0, 0.0],
    "reach": [0.0, 0.8, 0.6, 0.3, 0.0, 0.0],
    "swing": [0.8, 1.2, 0.6, 0.3, 0.0, 0.6],
}
FINGER_TARGET = {True: -0.003, False: -0.04}
# Settled joint states shared by MuJoCo classic/warp, IsaacGym and IsaacSim with the shipped dynamics. This is
# deliberately separate from the commanded targets above: gravity sag and the coupled fingers are physical, but
# those effects must remain aligned across backends. Each backend wrapper checks this same table within 1e-3.
SETTLED_DOF_REFERENCE = {
    ("reach", True): [0.0, 0.8048, 0.5660, 0.2688, 0.0, 0.0, -0.0030, -0.0030],
    ("reach", False): [0.0, 0.8048, 0.5660, 0.2688, 0.0, 0.0, -0.0400, -0.0400],
    ("swing", False): [0.8000, 1.2177, 0.5677, 0.2689, 0.0, 0.6001, -0.0382, -0.0418],
    ("swing", True): [0.8000, 1.2177, 0.5677, 0.2689, 0.0, 0.6001, -0.0012, -0.0048],
    ("home", True): [0.0, 1.0598, 1.0135, -0.0317, 0.0, 0.0, -0.0030, -0.0030],
}
# Forward kinematics of the arm targets: gripper body world position [m] and orientation [w, x, y, z], base at
# the origin (mujoco.mj_kinematics on data/robots/yam/yam.xml).
EXPECTED = {
    "home": ([0.2426, 0.0, 0.4021], [0.0, 0.7071, 0.0, -0.7071]),
    "reach": ([0.1972, 0.0, 0.324], [0.0, 0.7416, 0.0, -0.6709]),
    "swing": ([0.1847, 0.1902, 0.2384], [0.0804, 0.4539, 0.3823, -0.8008]),
}


# Third-person recording camera: in front-right of the base, looking at the arm's working volume.
CAMERA = FixedCameraConfig(position=[1.1, -1.1, 0.75], target=[0.12, 0.05, 0.28])
RECORD_FPS = 50


def _build(simulator: str, record_dir: str | None = None) -> tuple[Any, Any]:
    """Build the robot with a trajectory plugin and no SDK transport."""
    config = build_run_sim_config(simulator, "empty", ROBOT_KEY, "terrain_locomotion_plane")
    device = "cpu" if simulator == "mujoco" else "cuda:0"
    sim_init = dataclasses.replace(config.simulator.config, bridge=BridgeConfig(enabled=False))
    if record_dir is not None:
        # Frames are captured once per control step (FRAME_END), so the control rate is the video rate.
        sim_init = dataclasses.replace(
            sim_init, sim=dataclasses.replace(sim_init.sim, control_decimation=f"{RECORD_FPS}Hz")
        )
        video = dataclasses.replace(
            config.logger.video,
            enabled=True,
            interval=1,
            save_dir=record_dir,
            output_format="mp4",
            upload_to_wandb=False,
            show_command_overlay=False,
            width=1280,
            height=720,
            camera=CAMERA,
        )
        config = dataclasses.replace(config, logger=dataclasses.replace(config.logger, video=video))
    config = dataclasses.replace(
        config,
        simulator=dataclasses.replace(config.simulator, config=sim_init),
        device=device,
        plugin={"motion": yam_trajectory},
    )
    env, device, _app = setup_simulation_environment(config, device=device)
    import torch

    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    origins = torch.zeros(1, 3, device=device)
    base_init = torch.tensor(config.robot.init_state.root_state_list(), device=device, dtype=torch.float32)
    sim.create_envs(1, origins, base_init)
    sim.prepare_sim()
    sim.install_plugins()
    return sim, config


def _run_until(sim: Any, t_end: float) -> None:
    """Advance physics with the phase schedule ``DirectSimulation.run`` uses, until sim time reaches ``t_end``.

    FRAME_BEGIN opens on a control step's first substep and FRAME_END closes after its LAST substep (tensors
    refreshed first), so a recorded frame shows the state that control step wrote.
    """
    decimation = sim.simulator_config.sim.control_decimation_steps
    while sim.time() < t_end - 1e-9:
        step = round(sim.time() / sim.sim_dt)
        sim.refresh_sim_tensors()
        if step % decimation == 0:
            sim.hooks.emit(Phase.FRAME_BEGIN)
        sim.hooks.emit(Phase.PRE_STEP)
        sim.simulate_at_each_physics_step()
        sim.hooks.emit(Phase.POST_STEP)
        if step % decimation == decimation - 1:
            sim.refresh_sim_tensors()
            sim.hooks.emit(Phase.FRAME_END)
    sim.refresh_sim_tensors()


def _quat_angle(q_xyzw: Any, expected_wxyz: list[float]) -> float:
    """Rotation angle between a live xyzw quaternion and a (rounded, so re-normalised) wxyz reference."""
    norm = math.sqrt(sum(v * v for v in expected_wxyz))
    expected = [v / norm for v in expected_wxyz]
    dot = abs(float(q_xyzw[3]) * expected[0] + sum(float(q_xyzw[i]) * expected[i + 1] for i in range(3)))
    return 2.0 * math.acos(min(1.0, dot))


def _fmt(values: Any) -> list[float]:
    return [round(float(v), 4) for v in values]


def check_fk_table(config: Any) -> list[str]:
    """Recompute EXPECTED with MuJoCo when it is importable; a drift means the asset or the table changed."""
    try:
        import mujoco
    except ImportError:
        print("mujoco not importable: EXPECTED table not cross-checked")
        return []
    asset = config.robot.asset
    model = mujoco.MjModel.from_xml_path(resolve_asset_path(asset.xml_file, asset.asset_root))
    data = mujoco.MjData(model)
    fails = []
    for name, arm in ARM_TARGETS.items():
        data.qpos[:] = [*arm, 0.0, 0.0]
        mujoco.mj_kinematics(model, data)
        pos, quat = data.xpos[model.body(EE_BODY).id], data.xquat[model.body(EE_BODY).id]
        exp_pos, exp_quat = EXPECTED[name]
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
        if (
            max(abs(float(a) - b) for a, b in zip(pos, exp_pos)) > FK_TABLE_ATOL
            or _quat_angle(quat_xyzw, exp_quat) > FK_TABLE_ATOL
        ):
            fails.append(
                f"EXPECTED[{name!r}] {exp_pos}/{exp_quat} != MuJoCo FK {_fmt(pos)}/{_fmt(quat)}; update the table"
            )
    return fails


def check_isaacsim_visual_colors(config: Any, stage: Any = None) -> list[str]:
    """Check bound visual surface colours in env 0, including MDL and material subsets.

    This checks USD's resolved constant colour inputs, not rendered pixels. A connected texture with no
    inspectable RGB value is reported as unverifiable rather than checked against an unused fallback.
    """
    from pxr import Usd, UsdGeom, UsdShade

    from holosoma.simulator.isaacsim.urdf_visual_colors import (
        match_visual_colors_to_groups,
        urdf_visual_color_groups,
    )
    from holosoma.utils.path import resolve_asset_path

    expected = urdf_visual_color_groups(resolve_asset_path(config.robot.asset.urdf_file, config.robot.asset.asset_root))
    if stage is None:
        from isaaclab.sim import SimulationContext

        stage = SimulationContext.instance().stage
    groups: dict[str, dict[str, Any]] = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        if path.startswith("/World/envs/env_0/Robot/") and prim.GetParent().GetName() == "visuals":
            groups.setdefault(prim.GetParent().GetParent().GetName(), {})[prim.GetName()] = prim
    fails: list[str] = []
    checked = 0
    for link, per_visual in expected.items():
        actual_groups = groups.get(link, {})
        if not actual_groups and any(color is not None for color in per_visual):
            fails.append(f"IsaacSim: no visual geometry found for coloured URDF link '{link}'")
            continue
        matched = match_visual_colors_to_groups(per_visual, list(actual_groups))
        if not matched and any(color is not None for color in per_visual):
            fails.append(
                f"IsaacSim: cannot map {len(per_visual)} URDF visual(s) on link '{link}' to "
                f"{len(actual_groups)} imported visual group(s)"
            )
            continue
        for visual, expected_rgb in matched.items():
            consumers = []
            for prim in Usd.PrimRange(actual_groups[visual], Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
                faces = set()
                for subset in subsets:
                    indices = subset.GetIndicesAttr().Get()
                    if subset.GetElementTypeAttr().Get() == UsdGeom.Tokens.face and indices:
                        consumers.append(subset.GetPrim())
                        faces.update(indices)
                face_counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() if prim.IsA(UsdGeom.Mesh) else None
                # A fully partitioned mesh's parent binding shades no faces. A partial partition still uses it.
                if face_counts is None or not faces or faces != set(range(len(face_counts))):
                    consumers.append(prim)
            if not consumers:
                fails.append(f"IsaacSim: no visual geometry in '{link}/visuals/{visual}'")
            for prim in consumers:
                materials: dict[Any, tuple[Any, str]] = {}
                for purpose in UsdShade.MaterialBindingAPI.GetMaterialPurposes():
                    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(purpose)
                    if material:
                        materials.setdefault(material.GetPath(), (material, purpose))
                if not materials:
                    fails.append(f"IsaacSim: no bound material for visual geometry '{prim.GetPath()}'")
                for material, purpose in materials.values():
                    label = f"{prim.GetPath()} ({purpose or 'allPurpose'}, {material.GetPath()})"
                    colour_inputs = []
                    for output in material.GetSurfaceOutputs():
                        if not output.GetAttr().GetConnections():
                            continue
                        context = output.GetBaseName().rpartition(":")[0]
                        shader, _, _ = material.ComputeSurfaceSource(context)
                        if not shader:
                            fails.append(f"IsaacSim: unresolved {context or 'universal'} surface on '{label}'")
                            continue
                        for name in ("diffuse_color_constant", "diffuseColor"):
                            color_input = shader.GetInput(name)
                            if color_input:
                                colour_inputs.append(color_input)
                    if not colour_inputs:
                        fails.append(f"IsaacSim: no supported surface colour input on '{label}'")
                    for color_input in colour_inputs:
                        values = color_input.GetValueProducingAttributes()
                        value = values[0].Get() if len(values) == 1 else None
                        if value is None or not hasattr(value, "__len__") or len(value) != 3:
                            fails.append(f"IsaacSim: cannot resolve constant RGB for '{label}'")
                            continue
                        checked += 1
                        actual_rgb = tuple(float(v) for v in value)
                        if not all(math.isclose(a, b, abs_tol=1e-3) for a, b in zip(actual_rgb, expected_rgb)):
                            fails.append(
                                f"IsaacSim visual material on '{label}' is {actual_rgb}, "
                                f"URDF visual '{link}/visuals/{visual}' says {expected_rgb}"
                            )
    print(f"IsaacSim visual colours: checked {checked} bound surface colour input(s)")
    return fails


def run_checks(sim: Any, config: Any) -> list[str]:
    import torch

    fails: list[str] = []
    if config.simulator.config.name == "isaacsim":
        fails += check_isaacsim_visual_colors(config)
    ee = sim.find_rigid_body_indice(EE_BODY)
    tips = [sim.find_rigid_body_indice(name) for name in TIPS]
    spawn = torch.tensor(config.robot.init_state.root_state_list(), device=sim.dof_pos.device)
    for t_end, pose, is_open in CHECKS:
        _run_until(sim, t_end)
        label = f"t={t_end:g} {pose} {'open' if is_open else 'closed'}"
        target = torch.tensor(
            [*ARM_TARGETS[pose], FINGER_TARGET[is_open], FINGER_TARGET[is_open]], device=sim.dof_pos.device
        )
        q, qd = sim.dof_pos[0], sim.dof_vel[0]
        arm_err, finger_err = (q[:6] - target[:6]).abs().max().item(), (q[6:] - target[6:]).abs().max().item()
        if arm_err > DOF_ATOL_ARM or finger_err > DOF_ATOL_FINGER:
            fails.append(
                f"{label}: dof_pos {_fmt(q)} != targets {_fmt(target)} "
                f"(arm err {arm_err:.4f}, finger err {finger_err:.4f})"
            )
        parity_reference = torch.tensor(SETTLED_DOF_REFERENCE[(pose, is_open)], device=q.device)
        parity_err = (q - parity_reference).abs().max().item()
        if parity_err > BACKEND_PARITY_DOF_ATOL:
            fails.append(
                f"{label}: dof_pos {_fmt(q)} is {parity_err:.4g} from the shared backend reference "
                f"{_fmt(parity_reference)} (limit {BACKEND_PARITY_DOF_ATOL:g})"
            )
        if qd.abs().max().item() > VEL_ATOL:
            fails.append(f"{label}: dof_vel {_fmt(qd)} not settled")
        exp_pos, exp_quat = EXPECTED[pose]
        pos = sim.rigid_body_pos_w[0, ee]
        pos_err = (pos - torch.tensor(exp_pos, device=pos.device)).norm().item()
        ang_err = _quat_angle(sim._rigid_body_rot[0, ee], exp_quat)
        if pos_err > EE_POS_ATOL or ang_err > EE_ANG_ATOL:
            fails.append(
                f"{label}: {EE_BODY} pose {_fmt(pos)} / {_fmt(sim._rigid_body_rot[0, ee])} (xyzw) is {pos_err:.4f} m, "
                f"{ang_err:.4f} rad from the expected {exp_pos} / {exp_quat} (wxyz)"
            )
        tip_distance = (sim.rigid_body_pos_w[0, tips[0]] - sim.rigid_body_pos_w[0, tips[1]]).norm().item()
        expected_distance = OPEN_TIP_DISTANCE if is_open else CLOSED_TIP_DISTANCE
        if abs(tip_distance - expected_distance) > TIP_DISTANCE_ATOL:
            fails.append(f"{label}: finger tip distance {tip_distance:.4f} != {expected_distance}")
        root = torch.as_tensor(sim.robot_root_states[0, :7]).to(spawn.device)  # IsaacSim: a proxy over IsaacLab
        if (root - spawn[:7]).abs().max().item() > BASE_ATOL:
            fails.append(f"{label}: robot_root_states {_fmt(root)} left the spawn pose")
        print(
            f"[{label}] dof err arm {arm_err:.4f} finger {finger_err:.4f} parity {parity_err:.5f}; "
            f"{EE_BODY} err {pos_err:.4f} m "
            f"{ang_err:.4f} rad; tips {tip_distance:.4f} m; q {_fmt(q)}"
        )
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description="YAM joint-trajectory assertion harness.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    parser.add_argument("--record", default=None, metavar="DIR", help="save an mp4 of the run to DIR")
    args = parser.parse_args()

    sim, config = _build(args.simulator, args.record)
    if args.record is not None:
        sim.video_recorder.start_recording(episode_id=0)
    print(
        f"[{args.simulator}] robot={config.robot.asset.robot_type} sdk_type={config.robot.bridge.sdk_type} "
        f"dof={sim.dof_names} bodies={sim.body_names} sim_dt={sim.sim_dt}"
    )
    fails = check_fk_table(config) + run_checks(sim, config)
    if args.record is not None:
        sim.video_recorder.stop_recording()
        print(f"VIDEO saved under {args.record}")

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    for msg in fails:
        print(f"FAIL: {msg}")
    print("PASS" if not fails else f"{len(fails)} FAILURE(S)")
    return 0 if not fails else 1


if __name__ == "__main__":
    run_and_hard_exit(main)
