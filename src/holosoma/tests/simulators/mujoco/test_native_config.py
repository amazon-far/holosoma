"""Real model/step evidence for native configuration; no application or stepping mocks."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields, replace
from inspect import signature
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
import yaml
from pydantic import TypeAdapter

mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_types.full_sim import FullSimConfig  # noqa: E402
from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig  # noqa: E402
from holosoma.config_types.run_sim import RunSimConfig  # noqa: E402
from holosoma.config_types.scene import MujocoPhysicsConfig, PhysicsConfig  # noqa: E402
from holosoma.config_values.robot import g1_29dof  # noqa: E402
from holosoma.config_values.simulator import mjwarp  # noqa: E402
from holosoma.config_values.simulator import mujoco as preset  # noqa: E402
from holosoma.simulator.mujoco.backends.native_config import (  # noqa: E402
    allocation_kwargs,
    apply_warp_model,
    validate_warp_options,
    warp_option_overrides,
)
from holosoma.simulator.mujoco.native_config import effective_options  # noqa: E402
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager  # noqa: E402
from holosoma.utils.config_registry import parse_config  # noqa: E402
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim, build_warp_sim  # noqa: E402

pytestmark = pytest.mark.mujoco_classic

XML = """<mujoco><option integrator="Euler" tolerance="0.003"/>
<worldbody><geom name="floor" type="plane" size="2 2 .1"/>
<body name="ball" pos="0 0 .099"><freejoint/>
<geom name="ball_geom" size=".1" mass="1" friction=".3 .01 .002"/></body>
</worldbody></mujoco>"""


def compiled(native, backend=preset, *, xml=XML):
    config = replace(backend.config, sim=replace(backend.config.sim, fps=100, mujoco=native))
    manager = MujocoSceneManager(config)
    manager.world_spec = mujoco.MjSpec.from_string(xml)
    return manager.compile()


def warp_backend_config(native):
    run = RunSimConfig()
    return FullSimConfig(
        simulator=replace(mjwarp.config, sim=replace(mjwarp.config.sim, fps=100, mujoco=native)),
        robot=run.robot,
        training=replace(run.training, num_envs=2),
        logger=run.logger,
        plugin={},
    )


@pytest.mark.parametrize("yaml_roundtrip", [False, True])
def test_unset_preserves_asset_options_and_contacts_but_fps_owns_dt(yaml_roundtrip):
    baseline = mujoco.MjModel.from_xml_string(XML)
    native = MujocoConfig()
    if yaml_roundtrip:
        # Full saved configs include default fields: reloading must keep them inactive.
        adapter = TypeAdapter(MujocoConfig)
        native = adapter.validate_python(yaml.safe_load(yaml.safe_dump(adapter.dump_python(native, mode="json"))))
    model = compiled(native)
    assert model.opt.integrator == baseline.opt.integrator
    assert model.opt.tolerance == baseline.opt.tolerance
    np.testing.assert_array_equal(model.geom_friction, baseline.geom_friction)
    assert model.opt.timestep == 0.01


@pytest.mark.parametrize("integrator", ["euler", "rk4", "implicit", "implicitfast"])
def test_all_integrators_are_applied_and_stepped(integrator):
    xml = XML if integrator == "rk4" else XML.replace('integrator="Euler"', 'integrator="RK4"')
    model = compiled(
        MujocoConfig(integrator=integrator, extra={"gravity": [0, 0, -2]}, disableflags=("contact",)), xml=xml
    )
    assert model.opt.integrator == getattr(mujoco.mjtIntegrator, "mjINT_" + integrator.upper())
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.time == pytest.approx(0.1)
    assert data.qvel[2] == pytest.approx(-0.2)
    expected_z = 0.099 - (0.01 if integrator == "rk4" else 0.011)
    assert data.qpos[2] == pytest.approx(expected_z)


@pytest.mark.parametrize("solver", ["pgs", "cg", "newton"])
def test_named_contact_solver_settings_override_assets_and_step(solver):
    native = MujocoConfig(
        solver=solver,
        cone="elliptic",
        jacobian="sparse",
        impratio=10,
        iterations=17,
        tolerance=1e-7,
        ls_iterations=9,
        ls_tolerance=0.005,
        noslip_iterations=1,
    )
    asset_solver = "CG" if solver == "newton" else "Newton"
    model = compiled(native, xml=XML.replace("<option ", f'<option solver="{asset_solver}" '))
    assert model.opt.solver == getattr(mujoco.mjtSolver, "mjSOL_" + solver.upper())
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE
    for name in ("impratio", "iterations", "tolerance", "ls_iterations", "ls_tolerance", "noslip_iterations"):
        assert getattr(model.opt, name) == pytest.approx(getattr(native, name)), name
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert data.ncon > 0
    assert data.qpos[2] > 0.09
    assert data.time == pytest.approx(1)


def test_contact_flags_change_real_dynamics():
    enabled = compiled(MujocoConfig())
    disabled = compiled(MujocoConfig(disableflags=("contact",)))
    outcomes = []
    for model in [enabled, disabled]:
        data = mujoco.MjData(model)
        for _ in range(100):
            mujoco.mj_step(model, data)
        outcomes.append(data.qpos[2])
    assert outcomes[0] > 0.09
    assert outcomes[1] < -4


def test_explicit_empty_flags_clear_asset_mask_and_invalid_names_fail():
    xml = XML.replace(
        '<option integrator="Euler" tolerance="0.003"/>',
        '<option><flag contact="disable" gravity="disable"/></option>',
    )
    config = replace(preset.config, sim=replace(preset.config.sim, mujoco=MujocoConfig(disableflags=())))
    manager = MujocoSceneManager(config)
    manager.world_spec = mujoco.MjSpec.from_string(xml)
    assert manager.world_spec.option.disableflags != 0
    model = manager.compile()
    assert model.opt.disableflags == 0
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.ncon > 0
    with pytest.raises(ValueError, match="unknown flag"):
        compiled(MujocoConfig(disableflags=("contact_typo",)))


def test_geom_masks_solver_mixing_and_signed_parameters_reach_model():
    manager = MujocoSceneManager(preset.config)
    manager.world_spec = mujoco.MjSpec.from_string(XML)
    body = manager.world_spec.bodies[1]
    physics = MujocoPhysicsConfig(
        friction=[0.9, 0.02, 0.003],
        solref=[-1000, -10],
        solimp=[0.8, 0.92, 0.002, 0.4, 3],
        condim=6,
        contype=2,
        conaffinity=4,
        priority=3,
        solmix=0,
        margin=0.002,
        gap=0.001,
        extra={"fluid_ellipsoid": 0},
    )
    manager._apply_physics_to_body(body, PhysicsConfig(mujoco=physics), "ball")
    model = manager.compile()
    np.testing.assert_allclose(model.geom_friction[1], [0.9, 0.02, 0.003])
    np.testing.assert_allclose(model.geom_solref[1], [-1000, -10])
    np.testing.assert_allclose(model.geom_solimp[1], [0.8, 0.92, 0.002, 0.4, 3])
    assert (model.geom_contype[1], model.geom_conaffinity[1], model.geom_condim[1]) == (2, 4, 6)
    assert model.geom_priority[1] == 3
    assert model.geom_solmix[1] == 0
    assert model.geom_margin[1] == pytest.approx(0.002)
    assert model.geom_gap[1] == pytest.approx(0.001)


def test_signed_native_collision_masks_priority_and_mixing_are_not_restricted_to_positive():
    manager = MujocoSceneManager(preset.config)
    manager.world_spec = mujoco.MjSpec.from_string(XML)
    physics = MujocoPhysicsConfig(contype=-1, conaffinity=-2147483648, priority=-2, solmix=-1)
    manager._apply_physics_to_body(manager.world_spec.bodies[1], PhysicsConfig(mujoco=physics), "ball")
    model = manager.compile()
    assert (model.geom_contype[1], model.geom_conaffinity[1], model.geom_priority[1]) == (-1, -2147483648, -2)
    assert model.geom_solmix[1] == -1
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.ncon > 0


ROBOT_XML = """<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/>
<geom name="visual" size=".03" mass="1" contype="0" conaffinity="0"/>
<body name="left" pos="-.05 0 0"><joint name="left_joint"/><geom name="left_collision" size=".08"/></body>
<body name="right" pos=".05 0 0"><joint name="right_joint"/><geom name="right_collision" size=".08"/></body>
</body></worldbody></mujoco>"""


def robot_scene(tmp_path, native, self_collisions, *, xml=ROBOT_XML):
    asset_path = tmp_path / "robot.xml"
    asset_path.write_text(xml)
    robot = replace(
        g1_29dof,
        asset=replace(
            g1_29dof.asset,
            asset_root=str(tmp_path),
            xml_file=str(asset_path),
            enable_self_collisions=self_collisions,
            link_physics=PhysicsConfig(mujoco=native) if native is not None else None,
        ),
    )
    manager = MujocoSceneManager(preset.config)
    manager.add_robot(terrain_state=SimpleNamespace(geom=True), robot_config=robot)
    return manager


@pytest.mark.parametrize(
    ("masks", "self_collisions", "expected", "contacts"),
    [
        ({}, False, (1, 2), 0),
        ({"conaffinity": 3}, True, (1, 3), 1),
        ({"contype": 4, "conaffinity": 2}, False, (4, 2), 0),
        ({"contype": 4, "conaffinity": 6}, True, (4, 6), 1),
        ({"contype": 0, "conaffinity": 0}, False, (0, 0), 0),
    ],
)
def test_robot_native_masks_preserve_visuals_and_select_real_self_contacts(
    tmp_path, masks, self_collisions, expected, contacts
):
    model = robot_scene(tmp_path, MujocoPhysicsConfig(**masks), self_collisions).compile()
    np.testing.assert_array_equal(model.geom_contype, [0, expected[0], expected[0]])
    np.testing.assert_array_equal(model.geom_conaffinity, [0, expected[1], expected[1]])
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.ncon == contacts


@pytest.mark.parametrize("masks", [{"conaffinity": 3}, {"contype": 2}, {"contype": 4, "conaffinity": 6}])
def test_robot_native_masks_cannot_enable_disabled_self_collisions(tmp_path, masks):
    with pytest.raises(ValueError, match=r"robot\.asset\.link_physics\.mujoco.*enable_self_collisions=False"):
        robot_scene(tmp_path, MujocoPhysicsConfig(**masks), False).compile()


def test_authored_custom_robot_masks_are_preserved_when_unset(tmp_path):
    xml = ROBOT_XML.replace('name="left_collision"', 'name="left_collision" contype="4" conaffinity="2"')
    xml = xml.replace('name="right_collision"', 'name="right_collision" contype="8" conaffinity="2"')
    model = robot_scene(tmp_path, None, False, xml=xml).compile()
    np.testing.assert_array_equal(model.geom_contype, [0, 4, 8])
    np.testing.assert_array_equal(model.geom_conaffinity, [0, 2, 2])
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.ncon == 0


def test_robot_native_masks_preserve_retained_asset_floor_contacts(tmp_path):
    xml = ROBOT_XML.replace(
        "<worldbody>", '<worldbody><geom name="floor" type="plane" size="2 2 .1" contype="2" conaffinity="1"/>'
    ).replace('pos="0 0 1"', 'pos="0 0 .079"')
    model = robot_scene(tmp_path, MujocoPhysicsConfig(contype=4, conaffinity=2), False, xml=xml).compile()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_floor")
    assert model.geom_bodyid[floor] == 0
    assert (model.geom_contype[floor], model.geom_conaffinity[floor]) == (2, 1)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.ncon == 2
    assert all(floor in data.contact[index].geom for index in range(data.ncon))


def test_robot_native_masks_keep_separate_scene_actors_outside_self_collision_check(tmp_path):
    manager = robot_scene(
        tmp_path,
        MujocoPhysicsConfig(contype=4, conaffinity=2),
        False,
        xml=ROBOT_XML.replace('pos="0 0 1"', 'pos="0 0 .079"'),
    )
    body = manager.world_spec.worldbody.add_body(name="fixture")
    body.add_geom(name="fixture_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], contype=2, conaffinity=1)
    model = manager.compile()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixture_floor")
    body_id = model.geom_bodyid[floor]
    assert body_id > mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_right")
    assert model.body_parentid[body_id] == 0
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.ncon == 2
    assert all(floor in data.contact[index].geom for index in range(data.ncon))


FILTERED_ROBOT_XML = """<mujoco><worldbody><body name="root"><freejoint/><geom size=".1" mass="1"/>
<body name="child"><joint name="hinge"/><geom size=".1" mass="1"/></body>
</body></worldbody></mujoco>"""


@pytest.mark.parametrize(
    "xml",
    [
        FILTERED_ROBOT_XML,
        FILTERED_ROBOT_XML.replace('<joint name="hinge"/>', ""),
        ROBOT_XML.replace("</mujoco>", '<contact><exclude body1="left" body2="right"/></contact></mujoco>'),
    ],
    ids=["parent", "weld", "exclude"],
)
def test_robot_native_masks_respect_native_body_collision_filters(tmp_path, xml):
    model = robot_scene(tmp_path, MujocoPhysicsConfig(conaffinity=1), False, xml=xml).compile()
    assert np.any(model.geom_contype & model.geom_conaffinity)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.ncon == 0


def test_disabling_parent_filter_requires_self_collision_permission(tmp_path):
    for self_collisions in [False, True]:
        manager = robot_scene(tmp_path, MujocoPhysicsConfig(conaffinity=1), self_collisions, xml=FILTERED_ROBOT_XML)
        manager.simulator_config = replace(
            preset.config, sim=replace(preset.config.sim, mujoco=MujocoConfig(disableflags=("filterparent",)))
        )
        if not self_collisions:
            with pytest.raises(ValueError, match="enable_self_collisions=False"):
                manager.compile()
        else:
            model = manager.compile()
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
            assert data.ncon == 1


@pytest.mark.parametrize(
    "extra",
    [{"type": "box"}, {"pos": [1, 2, 3]}, {"size": [1, 1, 1]}, {"meshname": "mesh"}, {"mass": 7}, {"density": 900}],
)
def test_geom_extras_respect_asset_and_shared_physics_ownership(extra):
    manager = MujocoSceneManager(preset.config)
    manager.world_spec = mujoco.MjSpec.from_string(XML)
    with pytest.raises(ValueError, match="asset authoring"):
        manager._apply_physics_to_body(
            manager.world_spec.bodies[1], PhysicsConfig(mujoco=MujocoPhysicsConfig(extra=extra)), "ball"
        )


def test_extra_actual_api_validation_and_zero():
    model = compiled(MujocoConfig(iterations=0, impratio=0, extra={"ccd_iterations": 0, "wind": [0, 0, 0]}))
    assert model.opt.iterations == model.opt.ccd_iterations == 0
    assert model.opt.impratio == 0
    for extra, reason in [
        ({"ccdd_iterations": 3}, "unknown"),
        ({"wind": [1, 2]}, "shape"),
        ({"gravity": [1, 2]}, "shape"),
        ({"gravity": [1, 2, 3, 4]}, "shape"),
        ({"ccd_iterations": 1.5}, "integer"),
    ]:
        with pytest.raises(ValueError, match=reason):
            compiled(MujocoConfig(extra=extra))


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in dir(mujoco.MjOption)
        if isinstance(getattr(mujoco.MjOption, name), property)
        and name not in MujocoConfig.__dataclass_fields__
        and name != "timestep"
    ],
)
def test_every_remaining_public_option_has_a_working_generic_path(name):
    baseline = getattr(mujoco.MjOption(), name)
    if isinstance(baseline, np.ndarray):
        override = baseline * 0.5 if baseline.any() else baseline + 0.1
    elif isinstance(baseline, int):
        override = baseline + 1
    else:
        override = baseline * 0.5 if baseline else 0.1
    model = compiled(MujocoConfig(extra={name: override}))
    np.testing.assert_allclose(getattr(model.opt, name), override)
    assert not np.array_equal(getattr(model.opt, name), baseline)


def test_same_shared_settings_reach_both_compiled_backends():
    pytest.importorskip("mujoco_warp")
    native = MujocoConfig(
        iterations=0,
        impratio=8,
        jacobian="dense",
        disableflags=("gravity",),
    )
    classic = compiled(native)
    warp = compiled(native, backend=mjwarp)
    for model in (classic, warp):
        assert (model.opt.iterations, model.opt.impratio) == (0, 8)
        assert model.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE
        assert model.opt.disableflags == int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    assert warp_option_overrides(native.warp) == {}


@pytest.mark.parametrize(
    "settings",
    [
        {"graph_conditional": False},
        {"nconmax_per_env": 96},
        {"naconmax": 0},
        {"extras": {"nccdmax": 0}},
        {"extras": {"run_collision_detection": False}},
        {"njmax_per_env": 800},
    ],
)
def test_classic_keeps_warp_only_inputs_inactive_and_preserves_shared_dynamics(settings):
    shared = {"integrator": "rk4", "extra": {"gravity": [0, 0, -2]}, "disableflags": ("contact",)}
    native = MujocoConfig(**shared, warp=MujocoWarpConfig(**settings))
    model = compiled(native)
    baseline = compiled(MujocoConfig(**shared))
    for name in dir(mujoco.MjOption):
        if isinstance(getattr(mujoco.MjOption, name), property):
            np.testing.assert_array_equal(getattr(model.opt, name), getattr(baseline.opt, name))
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.qvel[2] == pytest.approx(-0.2)
    assert data.qpos[2] == pytest.approx(0.099 - 0.01)


def test_saved_warp_preset_can_override_only_backend_then_initialize_classic():
    warp_config = parse_config(
        RunSimConfig,
        args=[
            "simulator:mjwarp",
            "--simulator.config.sim.mujoco.integrator",
            "rk4",
            "--simulator.config.sim.mujoco.cone",
            "elliptic",
            "--simulator.config.sim.mujoco.impratio",
            "10",
            "--simulator.config.sim.mujoco.warp.nconmax-per-env",
            "128",
            "--simulator.config.sim.mujoco.warp.naconmax",
            "300",
            "--simulator.config.sim.mujoco.warp.graph-conditional",
            "False",
            "--simulator.config.sim.mujoco.extra",
            '{"ccd_iterations": 12}',
            "--simulator.config.sim.mujoco.warp.extras",
            '{"nccdmax": 64}',
        ],
    )
    saved = TypeAdapter(RunSimConfig).validate_python(
        yaml.safe_load(yaml.safe_dump(TypeAdapter(RunSimConfig).dump_python(warp_config, mode="json")))
    )
    classic = parse_config(RunSimConfig, default=saved, args=["--simulator.config.mujoco-backend", "CLASSIC"])
    assert classic.simulator.config.sim.mujoco == saved.simulator.config.sim.mujoco
    sim = build_classic_sim(simulator=classic.simulator)
    try:
        assert sim.plugin_config == sim.installed_plugins == {}
        assert sim.root_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        assert sim.root_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        assert sim.root_model.opt.impratio == 10
        assert sim.root_model.opt.ccd_iterations == 12
        assert sim.simulator_config.sim.mujoco == saved.simulator.config.sim.mujoco
        sim.backend.step()
        assert sim.root_data.time == pytest.approx(sim.sim_dt)
        mujoco.mj_resetData(sim.root_model, sim.root_data)
        sim.backend.step()
        assert sim.root_data.time == pytest.approx(sim.sim_dt)
    finally:
        sim.close()


def test_scene_compilation_and_missing_warp_startup_without_optional_dependencies():
    # Only optional imports are blocked. Model construction, application and stepping are real.
    script = f"""
import importlib.abc
import sys
class NoWarp(importlib.abc.MetaPathFinder):
    blocked = ("warp", "mujoco_warp")
    def find_spec(self, fullname, path, target=None):
        if fullname.partition(".")[0] in self.blocked:
            raise ModuleNotFoundError(fullname)
finder = NoWarp()
sys.meta_path.insert(0, finder)
from dataclasses import replace
import mujoco
from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig
from holosoma.config_values.simulator import mujoco as preset
from holosoma.config_types.simulator import MujocoBackend
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager
native = MujocoConfig(integrator="rk4", extra={{"gravity": [0, 0, -2]}}, disableflags=("contact",),
                      warp=MujocoWarpConfig(graph_conditional=False, naconmax=300,
                            extras={{"nccdmax": 64, "run_collision_detection": False}}))
for backend in MujocoBackend:
    config = replace(preset.config, mujoco_backend=backend, sim=replace(preset.config.sim, fps=100, mujoco=native))
    manager = MujocoSceneManager(config)
    manager.world_spec = mujoco.MjSpec.from_string({XML!r})
    model = manager.compile()
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert abs(data.time - 0.1) < 1e-12
    assert abs(data.qvel[2] + 0.2) < 1e-12
    assert abs(data.qpos[2] - (0.099 - 0.01)) < 1e-12
assert "warp" not in sys.modules and "mujoco_warp" not in sys.modules
# Core terrain code uses the shared warp-lang dependency; mujoco_warp remains optional.
finder.blocked = ("mujoco_warp",)
from tests.simulators.mujoco._build import build_classic_sim
try:
    build_classic_sim(simulator=replace(preset, config=config))
except RuntimeError as error:
    assert "dependencies not available" in str(error)
    assert "bash scripts/setup_mujoco.sh" in str(error)
else:
    raise AssertionError("Warp startup must report its missing dependency")
assert "mujoco_warp" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_checkpoint_native_physics_reaches_standalone_core_and_stepping(tmp_path):
    native = MujocoConfig(
        integrator="rk4",
        cone="elliptic",
        impratio=10,
        disableflags=("contact",),
        extra={"gravity": [0, 0, -2]},
        warp=MujocoWarpConfig(nconmax_per_env=128, graph_conditional=False),
    )
    simulator = replace(mjwarp, config=replace(mjwarp.config, sim=replace(mjwarp.config.sim, mujoco=native)))
    training = replace(ExperimentConfig(), simulator=simulator, plugin={})
    checkpoint = tmp_path / "model.pt"
    torch.save({"experiment_config": training.to_serializable_dict()}, checkpoint)
    loaded, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    standalone = parse_config(
        RunSimConfig,
        default=RunSimConfig(simulator=loaded.simulator, robot=loaded.robot, plugin={}),
        args=["--simulator.config.mujoco-backend", "CLASSIC"],
    )
    sim = build_classic_sim(simulator=standalone.simulator)
    try:
        assert sim.plugin_config == sim.installed_plugins == {}
        assert sim.root_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        assert sim.root_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        assert sim.root_model.opt.impratio == 10
        sim.root_data.qpos[2] = 5
        sim.root_data.qvel[:] = 0
        for _ in range(20):
            sim.backend.step()
        elapsed = 20 * sim.sim_dt
        assert sim.root_data.qvel[2] == pytest.approx(-2 * elapsed, abs=1e-6)
        assert sim.root_data.qpos[2] == pytest.approx(5 - elapsed**2, abs=1e-6)
        assert sim.root_data.time == pytest.approx(elapsed)
        assert sim.simulator_config.sim.mujoco.warp == native.warp
    finally:
        sim.close()


@pytest.mark.parametrize("reported_version", ["3.9.0", "3.11.0"])
def test_native_options_use_public_api_without_runtime_version_pins(reported_version, monkeypatch):
    # Changing only metadata tests the absence of a runtime version gate, not other binaries' dynamics.
    monkeypatch.setattr(mujoco, "__version__", reported_version)
    model = compiled(MujocoConfig(integrator="rk4", extra={"gravity": [0, 0, -2]}, disableflags=("contact",)))
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert data.time == pytest.approx(0.1)
    assert data.qvel[2] == pytest.approx(-0.2)
    assert data.qpos[2] == pytest.approx(0.099 - 0.01)


def test_warp_allocation_uses_real_api_and_preserves_default_units():
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    model = compiled(MujocoConfig())
    baseline = allocation_kwargs(model, MujocoWarpConfig(), 3)
    assert baseline == {"nworld": 3, "nconmax": 96, "njmax": max(576, model.nv * 4)}
    with wp.ScopedDevice("cpu"):
        default_data = mjw.make_data(model, **baseline)
    assert default_data.contact.geom.shape == (96 * 3,)
    assert default_data.njmax == 576
    model = compiled(MujocoConfig(jacobian="sparse"))
    cfg = MujocoWarpConfig(
        nconmax_per_env=5,
        njmax_per_env=40,
        naconmax=11,
        extras={"nccdmax": 2, "naccdmax": 4, "njmax_nnz": 120, "nvmax": 3},
    )
    with wp.ScopedDevice("cpu"):
        data = mjw.make_data(model, **allocation_kwargs(model, cfg, 3))
    assert data.contact.geom.shape == (11,)
    assert data.njmax == 40
    assert data.qpos.shape[0] == 3
    assert data.naccdmax == 4
    assert data.nvmax == 3
    assert data.efc.J.shape == (3, 1, 120)
    with wp.ScopedDevice("cpu"):
        zero_ccd = mjw.make_data(model, **allocation_kwargs(model, MujocoWarpConfig(extras={"nccdmax": 0}), 3))
    assert zero_ccd.naccdmax == 0
    for settings, reason in [
        ({"extras": {"unknown": 3}}, "make_data"),
        ({"extras": {"nccdmax": -1}}, "nonnegative"),
        ({"extras": {"nvmax": model.nv + 1}}, "exceeds model.nv"),
        ({"nconmax_per_env": 5, "extras": {"nccdmax": 6}}, "CCD capacity"),
        ({"naconmax": 11, "extras": {"naccdmax": 12}}, "CCD capacity"),
        ({"extras": {"contact_sensor_maxmatch": 1.5}}, "integer"),
        ({"extras": {"run_collision_detection": 2}}, "boolean"),
        ({"extras": {"broadphase": "typo"}}, "BroadphaseType"),
        ({"extras": {"broadphase_filter": 128}}, "BroadphaseFilter"),
        ({"extras": {"gravity": (0, 0, -2)}}, "sim.mujoco.extra.gravity"),
        ({"extras": {"impratio_invsqrt": 2}}, "derived"),
    ]:
        config = MujocoWarpConfig(**settings)
        validate, args = (
            (allocation_kwargs, (model, config, 3))
            if config.extras.keys() & signature(mjw.make_data).parameters.keys()
            else (warp_option_overrides, (config,))
        )
        with pytest.raises(ValueError, match=r"simulator\.config\.sim\.mujoco\.warp") as error:
            validate(*args)
        assert reason in str(error.value)


def test_remaining_warp_only_options_reach_real_cpu_converted_model():
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    model = compiled(MujocoConfig())
    option_types = get_type_hints(mjw.Option)
    with wp.ScopedDevice("cpu"):
        for field in fields(mjw.Option):
            name = field.name
            if isinstance(getattr(mujoco.MjOption, name, None), property) or name in {
                "impratio_invsqrt",  # Derived from the named shared impratio setting.
                "graph_conditional",  # Named Warp field, covered separately.
            }:
                continue
            uploaded = mjw.put_model(model)
            current = getattr(uploaded.opt, name)
            native_type = option_types[name]
            if hasattr(native_type, "__members__"):
                expected = next(member for member in native_type if int(member) != int(current))
                override = expected.name.lower()
            else:
                override = expected = not current if native_type is bool else current + 1
            native = MujocoConfig(warp=MujocoWarpConfig(extras={name: override}))
            apply_warp_model(uploaded, native, warp_option_overrides(native.warp))
            assert getattr(uploaded.opt, name) == expected, name
            assert getattr(uploaded.opt, name) != current, name


def test_warp_device_array_conversion_on_cpu_does_not_claim_gpu_execution():
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    native = MujocoConfig(
        extra={"gravity": [0, 0, -2]},
        tolerance=1e-8,
        impratio=10,
        disableflags=(),
        warp=MujocoWarpConfig(graph_conditional=False, extras={"broadphase_filter": "sphere|sphere|plane"}),
    )
    model = compiled(native, backend=mjwarp)
    with wp.ScopedDevice("cpu"):
        uploaded = mjw.put_model(model)
        assert uploaded.opt.tolerance.numpy()[0] == pytest.approx(1e-6)
        apply_warp_model(uploaded, native, warp_option_overrides(native.warp))
    np.testing.assert_allclose(uploaded.opt.tolerance.numpy(), [1e-8])
    np.testing.assert_allclose(uploaded.opt.gravity.numpy(), [[0, 0, -2]])
    np.testing.assert_allclose(uploaded.opt.impratio_invsqrt.numpy(), [1 / np.sqrt(10)])
    assert uploaded.opt.disableflags == int(mjw.DisableBit.ISLAND)
    assert uploaded.opt.graph_conditional is False
    assert uploaded.opt.broadphase_filter == int(mjw.BroadphaseFilter.SPHERE | mjw.BroadphaseFilter.PLANE)
    with wp.ScopedDevice("cpu"):
        native = replace(native, impratio=0)
        uploaded = mjw.put_model(compiled(native, backend=mjwarp))
        apply_warp_model(uploaded, native, warp_option_overrides(native.warp))
    np.testing.assert_allclose(uploaded.opt.impratio_invsqrt.numpy(), [1 / np.sqrt(mujoco.mjMINVAL)])


@pytest.mark.parametrize("authored", [False, True])
def test_warp_rejects_sleep_before_device_allocation_but_scene_compilation_is_backend_neutral(authored):
    pytest.importorskip("mujoco_warp")
    from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend

    native = MujocoConfig(solver="newton", enableflags=None if authored else ("sleep",))
    xml = XML.replace("<worldbody>", '<option><flag sleep="enable"/></option><worldbody>') if authored else XML
    model = compiled(native, backend=mjwarp, xml=xml)
    assert model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP
    config = warp_backend_config(native)
    assert config.plugin == {}
    with pytest.raises(ValueError, match=r"simulator\.config\.sim\.mujoco\.enableflags.*sleep.*state writes"):
        WarpBackend(model, mujoco.MjData(model), config, "cpu")


def test_warp_rejects_cpu_with_device_argument_in_error():
    pytest.importorskip("mujoco_warp")
    from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend

    native = MujocoConfig()
    model = compiled(native, backend=mjwarp)
    config = warp_backend_config(native)
    assert config.plugin == {}
    with pytest.raises(ValueError, match=r"device='cpu'.*CUDA"):
        WarpBackend(model, mujoco.MjData(model), config, "cpu")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=[
                pytest.mark.mujoco_warp,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="Captured contact stepping requires CUDA"),
            ],
        ),
    ],
)
def test_warp_contact_extra_changes_real_kernel_trajectories(device):
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    outcomes = {}
    for collision in (True, False):
        native = MujocoConfig(
            integrator="euler",
            cone="elliptic",
            impratio=10,
            extra={"gravity": [0, 0, -2]},
            warp=MujocoWarpConfig(
                nconmax_per_env=8,
                njmax_per_env=40,
                graph_conditional=False,
                extras={"run_collision_detection": collision, "contact_sensor_maxmatch": 0},
            ),
        )
        config = warp_backend_config(native)
        model = compiled(native, backend=mjwarp)
        initial = mujoco.MjData(model)
        backend = None
        with wp.ScopedDevice(device):
            try:
                if device == "cpu":
                    # Execute real upstream kernels on CPU; CUDA backend/capture is tested in the other case.
                    uploaded = mjw.put_model(model)
                    apply_warp_model(uploaded, native, warp_option_overrides(native.warp))
                    data = mjw.make_data(model, **allocation_kwargs(model, native.warp, 2))
                    data.qpos.assign(np.tile(initial.qpos, (2, 1)).astype(np.float32))
                    for _ in range(100):
                        mjw.step(uploaded, data)
                else:
                    from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend

                    assert config.plugin == {}
                    backend = WarpBackend(model, initial, config, device)
                    backend.initialize_state(model, initial)
                    assert backend.step_graph is not None
                    for _ in range(100):
                        backend.step()
                    data = backend.mjw_data
                np.testing.assert_allclose(data.time.numpy(), [1, 1], atol=1e-6, rtol=0)
                assert (data.nacon.numpy()[0] > 0) == collision
                outcomes[collision] = data.qpos.numpy()[:, 2]
                if not collision:
                    np.testing.assert_allclose(data.qvel.numpy()[:, 2], [-2, -2], atol=1e-5, rtol=0)
            finally:
                if backend is not None:
                    backend.close()
    assert np.all(outcomes[True] > 0.09)
    # Semi-implicit Euler free fall for 100 steps at dt=0.01 under gravity=-2.
    np.testing.assert_allclose(outcomes[False], [0.099 - 1.01] * 2, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    ("native", "error", "reason"),
    [
        (MujocoConfig(solver="pgs"), NotImplementedError, "mjSOL_PGS"),
        (MujocoConfig(noslip_iterations=1), NotImplementedError, "noslip solver"),
        (MujocoConfig(enableflags=("override",)), NotImplementedError, "mjENBL_OVERRIDE"),
        (MujocoConfig(extra={"o_margin": 0.01}), ValueError, "not implemented"),
        (MujocoConfig(warp=MujocoWarpConfig(extras={"impratio_invsqrt": 2})), ValueError, "derived"),
        (
            MujocoConfig(warp=MujocoWarpConfig(extras={"contact_sensor_maxmatch": -1})),
            ValueError,
            r"simulator\.config\.sim\.mujoco\.warp\.extras\.contact_sensor_maxmatch.*nonnegative",
        ),
    ],
)
def test_warp_unsupported_combinations_are_rejected_by_native_conversion(native, error, reason):
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")

    def convert():
        model = compiled(native, backend=mjwarp)
        validate_warp_options(model, effective_options(warp_backend_config(native).simulator))
        warp_option_overrides(native.warp)
        mjw.put_model(model)

    with wp.ScopedDevice("cpu"), pytest.raises(error, match=reason):
        convert()


def test_nativeccd_flag_is_accepted_by_mjwarp():
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    native = MujocoConfig(disableflags=("NATIVECCD",))
    with wp.ScopedDevice("cpu"):
        uploaded = mjw.put_model(compiled(native, backend=mjwarp))
    assert uploaded.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD


def test_core_classic_empty_plugins_initialization_reset_and_reconstruction():
    native = MujocoConfig(
        integrator="implicitfast", cone="elliptic", impratio=10, noslip_iterations=1, extra={"ccd_iterations": 12}
    )
    for _ in range(2):
        sim = build_classic_sim(native=native)
        try:
            assert sim.plugin_config == sim.installed_plugins == {}
            assert sim.root_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            assert sim.root_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
            assert sim.root_model.opt.impratio == 10
            assert sim.root_model.opt.noslip_iterations == 1
            sim.backend.step()
            assert sim.root_data.time == pytest.approx(sim.sim_dt)
            mujoco.mj_resetData(sim.root_model, sim.root_data)
            sim.backend.step()
            assert sim.root_data.time == pytest.approx(sim.sim_dt)
            assert sim.root_model.opt.ccd_iterations == 12
        finally:
            sim.close()


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MJWarp upload/allocation/captured stepping requires CUDA")
def test_core_warp_actual_upload_allocation_capture_and_reset():
    native = MujocoConfig(
        integrator="rk4",
        cone="elliptic",
        impratio=10,
        extra={"gravity": [0, 0, -2]},
        tolerance=1e-8,
        warp=MujocoWarpConfig(nconmax_per_env=128, njmax_per_env=800, naconmax=300, graph_conditional=False),
    )
    sim = build_warp_sim(num_envs=2, native=native)
    try:
        assert sim.plugin_config == sim.installed_plugins == {}
        backend = sim.backend
        assert backend.mjw_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        np.testing.assert_allclose(backend.mjw_model.opt.gravity.numpy(), [[0, 0, -2]])
        np.testing.assert_allclose(backend.mjw_model.opt.tolerance.numpy(), [1e-8])
        np.testing.assert_allclose(backend.mjw_model.opt.impratio_invsqrt.numpy(), [1 / np.sqrt(10)])
        assert backend.mjw_model.opt.graph_conditional is False
        assert backend.mjw_data.contact.geom.shape[0] == 300
        assert backend.mjw_data.njmax == 800
        assert backend.step_graph is not None
        # Lift the robot above contacts and give every world a known zero initial velocity.
        state = backend.qpos_t.clone()
        state[:, 2] = 5
        backend.qpos_t.copy_(state)
        backend.qvel_t.zero_()
        for _ in range(20):
            backend.step()
        elapsed = 20 * sim.sim_dt
        assert backend.qvel_t[:, 2].cpu().tolist() == pytest.approx([-2 * elapsed] * 2, rel=0.01)
        # RK4's analytic free-fall position differs from Euler by 20*dt**2 here.
        # Check the trajectory so an uploaded enum alone cannot establish kernel application.
        np.testing.assert_allclose(backend.qpos_t[:, 2].cpu(), [5 - elapsed**2] * 2, atol=3e-6, rtol=0)
        backend.initialize_state(sim.root_model, sim.root_data)
        assert backend.mjw_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        backend.step()
        assert backend.get_render_data().time == pytest.approx(sim.sim_dt)
    finally:
        sim.close()
