"""Assert one actuator group over all DOFs matches one single-joint group per DOF.

Boots a headless app first because ``isaaclab.actuators`` imports ``carb``.
"""

from __future__ import annotations

import argparse
import sys
import traceback

# tests/simulators/ has an ``isaacsim/`` subpackage that would shadow the real IsaacSim
# ``isaacsim`` package when run as a script; drop sys.path[0] (mirrors the other harnesses).
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.utils.safe_torch_import import torch

NUM_ENVS = 4
DEVICE = "cpu"

# Distinct value per joint per parameter, so a mis-paired joint fails instead of coinciding.
JOINT_NAMES = [f"joint_{i:02d}_link" for i in range(29)]
EFFORT_LIMITS = [100.0 + i for i in range(29)]
VELOCITY_LIMITS = [20.0 + 0.5 * i for i in range(29)]
ARMATURES = [0.01 * (i + 1) for i in range(29)]
FRICTIONS = [0.001 * (i + 1) for i in range(29)]

PER_JOINT_PARAMS = ("effort_limit", "velocity_limit", "armature", "friction", "stiffness", "damping")

# Larger than any effort limit above, so the clip binds.
EFFORT_TARGET_SCALE = 500.0


def _usd_defaults(num_joints: int) -> dict:
    """The USD-derived values Articulation passes to each group."""
    zeroed = ("stiffness", "damping", "armature", "friction", "dynamic_friction", "viscous_friction")
    defaults = {name: torch.zeros(NUM_ENVS, num_joints, device=DEVICE) for name in zeroed}
    for name in ("effort_limit", "velocity_limit"):
        defaults[name] = torch.full((NUM_ENVS, num_joints), 1.0e9, device=DEVICE)
    return defaults


def _build_per_dof() -> list:
    from isaaclab.actuators import IdealPDActuatorCfg

    actuators = []
    for i, name in enumerate(JOINT_NAMES):
        cfg = IdealPDActuatorCfg(
            joint_names_expr=[name],
            effort_limit=EFFORT_LIMITS[i],
            velocity_limit=VELOCITY_LIMITS[i],
            stiffness=0.0,
            damping=0.0,
            armature=ARMATURES[i],
            friction=FRICTIONS[i],
        )
        actuators.append(
            cfg.class_type(
                cfg=cfg,
                joint_names=[name],
                joint_ids=torch.tensor([i], device=DEVICE),
                num_envs=NUM_ENVS,
                device=DEVICE,
                **_usd_defaults(1),
            )
        )
    return actuators


def _build_single_group():
    from isaaclab.actuators import IdealPDActuatorCfg

    cfg = IdealPDActuatorCfg(
        joint_names_expr=list(JOINT_NAMES),
        effort_limit=dict(zip(JOINT_NAMES, EFFORT_LIMITS)),
        velocity_limit=dict(zip(JOINT_NAMES, VELOCITY_LIMITS)),
        stiffness=0.0,
        damping=0.0,
        armature=dict(zip(JOINT_NAMES, ARMATURES)),
        friction=dict(zip(JOINT_NAMES, FRICTIONS)),
    )
    # Articulation collapses joint_ids to slice(None) when a group covers every joint.
    return cfg.class_type(
        cfg=cfg,
        joint_names=list(JOINT_NAMES),
        joint_ids=slice(None),
        num_envs=NUM_ENVS,
        device=DEVICE,
        **_usd_defaults(len(JOINT_NAMES)),
    )


def _check_fast_path(grouped) -> None:
    assert grouped.joint_indices == slice(None), (
        f"expected slice(None) for an all-joint group, got {grouped.joint_indices!r}"
    )
    print("OK: all-joint group resolves to slice(None)")


def _check_resolved_parameters(per_dof: list, grouped) -> None:
    for param in PER_JOINT_PARAMS:
        for i, actuator in enumerate(per_dof):
            per_dof_value = getattr(actuator, param)[:, 0]
            grouped_value = getattr(grouped, param)[:, i]
            assert torch.equal(per_dof_value, grouped_value), (
                f"{param} for {JOINT_NAMES[i]}: per-DOF grouping resolved "
                f"{per_dof_value[0].item()}, single group resolved {grouped_value[0].item()}"
            )
    print(f"OK: {len(PER_JOINT_PARAMS)} per-joint parameters identical across {len(JOINT_NAMES)} joints")


def _apply(actuators: list, effort_target, joint_pos, joint_vel):
    """Run each group's actuator model the way write_data_to_sim() does."""
    from isaaclab.utils.types import ArticulationActions

    applied = torch.zeros_like(effort_target)
    for actuator in actuators:
        ids = actuator.joint_indices
        action = ArticulationActions(
            joint_positions=torch.zeros_like(effort_target[:, ids]),
            joint_velocities=torch.zeros_like(effort_target[:, ids]),
            joint_efforts=effort_target[:, ids],
            joint_indices=ids,
        )
        action = actuator.compute(action, joint_pos=joint_pos[:, ids], joint_vel=joint_vel[:, ids])
        applied[:, ids] = action.joint_efforts
    return applied


def _check_applied_torque(per_dof: list, grouped) -> None:
    torch.manual_seed(0)
    num_joints = len(JOINT_NAMES)
    effort_target = torch.randn(NUM_ENVS, num_joints, device=DEVICE) * EFFORT_TARGET_SCALE
    joint_pos = torch.randn(NUM_ENVS, num_joints, device=DEVICE)
    joint_vel = torch.randn(NUM_ENVS, num_joints, device=DEVICE)

    per_dof_torque = _apply(per_dof, effort_target.clone(), joint_pos, joint_vel)
    grouped_torque = _apply([grouped], effort_target.clone(), joint_pos, joint_vel)

    assert torch.equal(per_dof_torque, grouped_torque), (
        f"applied torque differs, max abs diff {(per_dof_torque - grouped_torque).abs().max().item()}"
    )
    assert (per_dof_torque.abs() < effort_target.abs()).any(), "effort limit never bound"
    print("OK: applied torque bitwise identical, with the effort limit clipping")


def main() -> int:
    parser = argparse.ArgumentParser(description="IsaacSim actuator-grouping assertion harness.")
    parser.add_argument("--result-file", default=None, help="write 'OK' here after all checks pass")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    try:
        per_dof = _build_per_dof()
        grouped = _build_single_group()

        _check_fast_path(grouped)
        _check_resolved_parameters(per_dof, grouped)
        _check_applied_torque(per_dof, grouped)
    except BaseException:
        print("ACTUATOR GROUPING ASSERT FAILED:\n" + traceback.format_exc(), flush=True)
        simulation_app.close()
        return 1

    print("ACTUATOR GROUPING OK", flush=True)
    if args.result_file:
        with open(args.result_file, "w") as f:
            f.write("OK\n")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
