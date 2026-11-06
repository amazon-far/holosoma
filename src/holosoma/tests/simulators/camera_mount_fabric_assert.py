"""IsaacSim assertion harness: every mounted camera renders at its authored mount offset.

The renderer places cameras from Fabric. Cameras are mounted on the pelvis, both elbows and both
wrists, links that move during the physics steps Isaac Lab takes while it initializes sensors. After
startup and again after a few physics steps, each camera's Fabric world matrix must equal its mount
body's Fabric world matrix composed with the camera's USD-authored local transform. Exits 0 PASS,
1 FAIL.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step

_LINKS = ("pelvis", "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
_MAX_POS_ERR_M = 1e-3
_MAX_ROT_ERR_DEG = 0.2


def _cameras() -> dict[str, CameraSensorConfig]:
    # A distinct offset and rotation per mount, so a local re-derived against a moved body shows up.
    return {
        f"cam_{link}": CameraSensorConfig(
            mount=SensorMountConfig(
                target_kind="robot_link",
                target=link,
                position=[0.1, 0.05 * (i - 2), 0.02 * i],
                orientation=[math.cos(0.2 * i), 0.0, math.sin(0.2 * i), 0.0],
            ),
            width=32,
            height=32,
            vertical_fov=60.0,
            data_types=["rgb"],
        )
        for i, link in enumerate(_LINKS)
    }


def _mount_errors(sim) -> list[tuple[str, float, float]]:
    """``(prim, position error m, rotation error deg)`` for every camera prim of every env."""
    if sim.backend.modern:
        return _native_mount_errors(sim)

    import isaaclab.sim as sim_utils
    import numpy as np
    import usdrt
    from pxr import Usd, UsdGeom
    from usdrt import Rt

    fabric_stage = usdrt.Usd.Stage.Attach(sim_utils.get_current_stage_id())

    def fabric_world(path: str) -> np.ndarray:
        matrix = Rt.Xformable(fabric_stage.GetPrimAtPath(path)).GetFabricHierarchyWorldMatrixAttr().Get()
        return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])

    errors: list[tuple[str, float, float]] = []
    for camera in sim.tiled_cameras.values():
        for prim in sim_utils.find_matching_prims(camera.cfg.prim_path):
            path = str(prim.GetPath())
            local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
            local = np.array([[local[r][c] for c in range(4)] for r in range(4)])
            expected = local @ fabric_world(str(prim.GetParent().GetPath()))  # USD row-vector convention
            actual = fabric_world(path)
            pos_err = float(np.linalg.norm(actual[3, :3] - expected[3, :3]))
            cos = (np.trace(actual[:3, :3].T @ expected[:3, :3]) - 1.0) / 2.0
            errors.append((path, pos_err, math.degrees(math.acos(min(1.0, max(-1.0, cos))))))
    return errors


def _native_mount_errors(sim) -> list[tuple[str, float, float]]:
    """Compare native capture poses with live link poses, independently of Kit/Fabric."""
    import torch
    from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

    from holosoma.simulator.isaacsim.backends import as_tensor

    errors: list[tuple[str, float, float]] = []
    for name, camera in sim.tiled_cameras.items():
        camera.update(0.0, force_recompute=True)
        mount = sim.sensor_config[name].mount
        body_id = sim._robot.body_names.index(mount.target)
        body_pos = as_tensor(sim._robot.data.body_pos_w)[:, body_id]
        body_quat = as_tensor(sim._robot.data.body_quat_w)[:, body_id]
        offset_pos = torch.tensor(mount.position, device=body_pos.device).expand_as(body_pos)
        offset_quat = torch.tensor(sim.backend.config_quaternion(mount.orientation), device=body_quat.device).expand_as(
            body_quat
        )
        expected_pos = body_pos + quat_apply(body_quat, offset_pos)
        expected_quat = quat_mul(body_quat, offset_quat)
        position_error = torch.linalg.vector_norm(as_tensor(camera.data.pos_w) - expected_pos, dim=-1)
        angle_error = torch.rad2deg(quat_error_magnitude(as_tensor(camera.data.quat_w_opengl), expected_quat))
        errors.extend(
            (f"{name}/env{env_id}", float(position_error[env_id]), float(angle_error[env_id]))
            for env_id in range(sim.num_envs)
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    args = parser.parse_args()

    config = build_run_sim_config("isaacsim", "empty", args.robot, args.terrain, sensors=_cameras())
    config = dataclasses.replace(
        config, device="cuda:0", training=dataclasses.replace(config.training, num_envs=args.num_envs)
    )
    env, device, _app = setup_simulation_environment(config, device="cuda:0")
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(args.num_envs, torch.zeros(args.num_envs, 3, device=device), base_init)
    sim.prepare_sim()

    fails = []
    for when in ("after startup", "after 20 physics steps"):
        if when != "after startup":
            step(sim, 20)  # the un-actuated robot sags, so every mount body moves
        for path, pos_err, rot_err in _mount_errors(sim):
            print(f"[isaacsim] {when}: {path} pos {pos_err * 100:.3f} cm rot {rot_err:.3f} deg")
            if pos_err > _MAX_POS_ERR_M or rot_err > _MAX_ROT_ERR_DEG:
                fails.append(
                    f"{when}: {path} renders {pos_err * 100:.2f} cm / {rot_err:.2f} deg off its authored mount"
                )
    if not fails and len(_mount_errors(sim)) != len(_LINKS) * args.num_envs:
        fails.append(f"expected {len(_LINKS) * args.num_envs} camera prims, found {len(_mount_errors(sim))}")

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for f in fails:
            print(f"[isaacsim] FAIL: {f}")
        return 1
    print(f"[isaacsim] PASS: {len(_LINKS)} cameras at their mount offsets across {args.num_envs} env(s)")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
