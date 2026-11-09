"""Read authored USD and native physics models independently of adapter accessors."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.scene import IsaacSimPhysicsConfig, PhysicsConfig
from holosoma.config_types.simulator import IsaacLab3Config, IsaacLab3NewtonConfig
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._native_runtime_assertions import assert_native_runtime
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", choices=["isaacsim3"], required=True)
    parser.add_argument("--physics", choices=["newton", "physx"], required=True)
    parser.add_argument("--renderer", choices=["newton_warp", "ovrtx"], required=True)
    parser.add_argument("--collision-source", choices=["mujoco", "newton"], default="mujoco")
    parser.add_argument("--legacy-newton", action="store_true")
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    config = build_run_sim_config(
        "isaacsim3",
        "panel-target",
        "g1-29dof",
        "terrain_locomotion_plane",
        sensors=_camera_presets.front_cam_depth,
    )
    physics: dict[str, Any] = {}
    if args.physics == "newton":
        physics = {"solver_cfg": {"iterations": 17, "ls_iterations": 9, "cone": "elliptic", "impratio": 3.0}}
        if args.collision_source == "newton":
            physics["solver_cfg"]["use_mujoco_contacts"] = False
            physics["collision_cfg"] = {
                "_type_": "isaaclab_newton.physics:NewtonCollisionPipelineCfg",
                "broad_phase": "sap",
            }
    native = IsaacLab3Config(
        physics_backend=args.physics,
        renderer=args.renderer,
        physics=physics,
        renderer_options={"enable_shadows": False},
    )
    if args.legacy_newton:
        assert args.physics == "newton"
        physics["solver_cfg"].pop("iterations")
        native = dataclasses.replace(
            native,
            physics=physics,
            newton=IsaacLab3NewtonConfig(solver_iterations=17, use_cuda_graph=False),
        )
    settings: dict[str, Any] = {
        "mass_props": [
            {
                "match": "/.*pelvis",
                "fragments": [{"_type_": "isaaclab.sim.schemas:MassCfg", "mass": 42.0}],
            }
        ],
        "physics_material": [
            {
                "match": "/.*left_ankle_roll_link(/.*)?",
                "fragments": [
                    {
                        "_type_": "isaaclab.sim.spawners.materials:UsdPhysicsRigidBodyMaterialCfg",
                        "static_friction": 0.45,
                        "dynamic_friction": 0.45,
                    },
                ],
            },
            {
                "match": "/.*left_ankle_roll_link(/.*)?",
                "fragments": [
                    {
                        "_type_": "isaaclab_newton.sim.spawners.materials:NewtonMaterialCfg",
                        "torsional_friction": 0.07,
                        "rolling_friction": 0.03,
                    },
                ],
            },
        ],
    }
    if args.physics == "newton":
        settings["collision_props"] = [
            {"_type_": "isaaclab_newton.sim.schemas:MujocoCollisionCfg", "condim": 6},
        ]
        settings["rigid_props"] = [
            {
                "match": "/.*",
                "fragments": [{"_type_": "isaaclab_newton.sim.schemas:MujocoRigidBodyCfg", "gravcomp": 0.1}],
            },
            {
                "match": "/.*pelvis",
                "fragments": [{"_type_": "isaaclab_newton.sim.schemas:MujocoRigidBodyCfg", "gravcomp": 0.25}],
            },
        ]
    config = dataclasses.replace(
        config,
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                sim=dataclasses.replace(config.simulator.config.sim, isaaclab3=native),
            ),
        ),
        robot=dataclasses.replace(
            config.robot,
            asset=dataclasses.replace(
                config.robot.asset,
                link_physics=PhysicsConfig(
                    isaacsim=IsaacSimPhysicsConfig(static_friction=0.7, dynamic_friction=0.7),
                    isaaclab3=settings,
                ),
            ),
        ),
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=2),
    )
    env, device, app = setup_simulation_environment(config, device="cuda:0")
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()

    import numpy as np
    import torch
    import warp as wp
    from isaaclab.sim.schemas import MassCfg
    from isaaclab.utils.string import string_to_callable
    from pxr import Usd, UsdPhysics, UsdShade

    from holosoma.simulator.isaacsim.backends.native_properties import author_properties

    fixture = Usd.Stage.CreateInMemory()
    fixture.DefinePrim("/Library", "Xform")
    fixture.DefinePrim("/Library/Geometry", "Xform")
    fixture.DefinePrim("/Library/Geometry/mesh", "Mesh")
    fixture.DefinePrim("/Asset", "Xform")
    body = fixture.DefinePrim("/Asset/body", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    visual = fixture.DefinePrim("/Asset/visual", "Xform")
    visual.GetReferences().AddInternalReference("/Library/Geometry")
    visual.SetInstanceable(True)
    author_properties("/Asset", {"mass_props": [("/body", [MassCfg(mass=2.0)])]}, flags={}, stage=fixture)
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 2.0
    assert visual.IsInstanceable(), "A mass override deinstanced unrelated visuals"
    other_body = fixture.DefinePrim("/Asset/other_body", "Xform")
    outside = fixture.DefinePrim("/other_body", "Xform")
    sibling = fixture.DefinePrim("/AssetOther/body", "Xform")
    for prim, mass in ((other_body, 3.0), (outside, 77.0), (sibling, 88.0)):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    author_properties(
        "/Asset",
        {"mass_props": [("/body|/other_body", [MassCfg(mass=4.0)])]},
        flags={},
        stage=fixture,
    )
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 4.0
    assert UsdPhysics.MassAPI(other_body).GetMassAttr().Get() == 4.0
    author_properties("/Asset", {"mass_props": [("(.*)", [MassCfg(mass=5.0)])]}, flags={}, stage=fixture)
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 5.0
    assert UsdPhysics.MassAPI(other_body).GetMassAttr().Get() == 5.0
    assert UsdPhysics.MassAPI(outside).GetMassAttr().Get() == 77.0
    assert UsdPhysics.MassAPI(sibling).GetMassAttr().Get() == 88.0
    assert visual.IsInstanceable(), "A broad mass rule deinstanced unrelated visuals"

    init = config.robot.init_state
    initial = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=device,
        dtype=torch.float32,
    )
    sim.create_envs(2, torch.zeros(2, 3, device=device), initial)
    sim.prepare_sim()
    manager = string_to_callable(sim.sim.cfg.physics.class_type)
    stage = sim.sim.stage
    for row in range(2):
        root = stage.GetPrimAtPath(f"/World/envs/env_{row}/Robot")
        bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        pelvis = next(prim for prim in bodies if prim.GetName() == "pelvis")
        assert UsdPhysics.MassAPI(pelvis).GetMassAttr().Get() == 42.0
        if args.physics == "newton":
            assert abs(pelvis.GetAttribute("mjc:gravcomp").Get() - 0.25) < 1e-6
            other = next(prim for prim in bodies if prim.GetName() != "pelvis")
            assert abs(other.GetAttribute("mjc:gravcomp").Get() - 0.1) < 1e-6
        for side, expected in [("left", 0.45), ("right", 0.7)]:
            colliders = [
                prim
                for prim in Usd.PrimRange(root)
                if f"{side}_ankle_roll_link/" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI)
            ]
            assert colliders, side
            for collider in colliders:
                bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose="physics")
                assert bound, collider.GetPath()
                value = UsdPhysics.MaterialAPI(bound.GetPrim()).GetDynamicFrictionAttr().Get()
                assert abs(value - expected) < 1e-6, (collider.GetPath(), value, expected)

    robot = sim._robot
    if args.physics == "newton":
        from newton import ShapeFlags

        model = manager.get_model()
        masses = model.body_mass.numpy()
        labels = model.body_label
        pelvis_ids = [index for index, label in enumerate(labels) if label.endswith("/pelvis")]
        assert len(pelvis_ids) == 2, labels
        np.testing.assert_allclose(masses[pelvis_ids], 42.0)
        shapes = model.shape_label
        flags = model.shape_flags.numpy()
        for side, expected in [("left", 0.45), ("right", 0.7)]:
            ids = [
                index
                for index, label in enumerate(shapes)
                if f"{side}_ankle_roll_link/" in label and flags[index] & int(ShapeFlags.COLLIDE_SHAPES)
            ]
            assert ids, shapes
            np.testing.assert_allclose(model.shape_material_mu.numpy()[ids], expected, atol=1e-6)
            if side == "left":
                np.testing.assert_allclose(model.shape_material_mu_torsional.numpy()[ids], 0.07, atol=1e-6)
                np.testing.assert_allclose(model.shape_material_mu_rolling.numpy()[ids], 0.03, atol=1e-6)
        # The compiled solver is the oracle, not the Holosoma config or accessors.
        solver = manager._solver
        assert solver.mj_model.opt.iterations == 17
        assert solver.mj_model.opt.ls_iterations == 9
        assert solver.mj_model.opt.impratio == 3.0
        assert (solver.mj_model.geom_condim == 6).any()
        assert abs(solver.mj_model.body_gravcomp.max() - 0.25) < 1e-6
        if args.legacy_newton:
            assert sim.sim.cfg.physics.use_cuda_graph is False
    else:
        native_mass = wp.to_torch(robot.root_view.get_attribute("articulation_body_mass"))
        index = robot.backend_body_names.index("pelvis")
        torch.testing.assert_close(native_mass[:, index], torch.full((2,), 42.0, device=native_mass.device))

    camera = sim.tiled_cameras["front_cam"]
    assert camera.cfg.renderer_cfg.enable_shadows is False
    before = sim.robot_root_states.clone()
    step(sim, 20, render=True)
    assert torch.isfinite(sim.robot_root_states).all()
    assert not torch.equal(before, sim.robot_root_states)
    assert camera.data.output["rgb"].torch.numel() > 0
    assert_native_runtime(physics=args.physics, uses_kit=False)
    Path(args.result_file).write_text("OK\n")
    Path(args.result_file).with_suffix(".json").write_text(
        json.dumps(
            {
                "physics": args.physics,
                "renderer": args.renderer,
                "collision_source": args.collision_source,
                "mass": 42.0,
                "left_friction": 0.45,
                "right_friction": 0.7,
            },
            indent=2,
        )
        + "\n"
    )
    app.close()
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
