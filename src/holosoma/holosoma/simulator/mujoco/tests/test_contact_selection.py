"""Live MuJoCo tests for pair identity, terrain selection, friction, and body fusion."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from holosoma.config_types.full_sim import FullSimConfig  # noqa: E402
from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig  # noqa: E402
from holosoma.config_values.simulator import mujoco as mujoco_config  # noqa: E402
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase  # noqa: E402
from holosoma.simulator.mujoco.backends import ClassicBackend  # noqa: E402
from holosoma.simulator.mujoco.contact_setup import (  # noqa: E402
    ContactPairForces,
    prepare_contact_targets,
    register_contacts,
    resolve_target_geoms,
    selection_masks,
)
from holosoma.simulator.mujoco.mujoco import MuJoCo  # noqa: E402
from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager  # noqa: E402
from holosoma.simulator.mujoco.sensor_setup import render_sensors  # noqa: E402
from holosoma.simulator.shared.sensor_manager import SensorManager  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402

pytestmark = pytest.mark.mujoco_classic

_PAIR_SCENE = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 .1" pos="0 0 -1"/>
    <body name="robot_root" pos="-.08 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="box_root" pos=".08 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="robot_decoy_root" pos="3 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>
"""


def _scene(xml: str = _PAIR_SCENE) -> MujocoSceneManager:
    scene = MujocoSceneManager(mujoco_config.config)
    scene.world_spec = mujoco.MjSpec.from_string(xml)
    scene.robot_spec_meta = ActorSpecMeta(prefix="robot_", root_body="root", body_names=["root"])
    scene.rigid_object_root_bodies = {"box": "box_root", "robot_decoy": "robot_decoy_root"}
    scene.terrain_geom_name = "ground"
    return scene


def _actor(name: str) -> ContactTargetConfig:
    return ContactTargetConfig(target_kind="actor", target=name)


def _pairs(model: Any, data: Any) -> ContactPairForces:
    config = cast("FullSimConfig", SimpleNamespace(training=SimpleNamespace(num_envs=1)))
    return ClassicBackend(model, data, config, "cpu").contact_pairs()


def test_filtered_pair_forces_are_equal_opposite_and_exclude_terrain_and_prefix_decoys() -> None:
    scene = _scene()
    config = ContactSensorConfig(
        sources=[ContactTargetConfig(), _actor("box"), _actor("robot_decoy")],
        counterparts=[_actor("box"), ContactTargetConfig(target_kind="terrain")],
    )
    targets = prepare_contact_targets(scene, {"pairs": config})
    model = scene.world_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sources, counterparts = selection_masks(model, targets, config)
    matrix = _pairs(model, data).matrix(sources, counterparts, force_mode="normal").numpy()

    assert data.ncon == 1
    assert matrix.shape == (1, 3, 2, 3)
    assert matrix[0, 0, 0, 0] < -1
    np.testing.assert_array_equal(matrix[0, :, 1], 0)
    np.testing.assert_array_equal(matrix[0, 1:], 0)

    reverse = ContactSensorConfig(sources=[_actor("box")], counterparts=[ContactTargetConfig()])
    sources, counterparts = selection_masks(model, targets, reverse)
    reaction = _pairs(model, data).matrix(sources, counterparts, force_mode="normal").numpy()
    np.testing.assert_allclose(reaction[0, 0, 0], -matrix[0, 0, 0], atol=1e-6)


def test_whole_source_group_cancels_internal_contact() -> None:
    scene = _scene()
    config = ContactSensorConfig(sources=[ContactTargetConfig(), _actor("box")])
    targets = prepare_contact_targets(scene, {"pairs": config})
    model = scene.world_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sources, counterparts = selection_masks(model, targets, config)
    pairs = _pairs(model, data)
    individual = pairs.matrix(sources, counterparts, force_mode="normal")
    combined = pairs.matrix(sources.sum(axis=0, keepdims=True), counterparts, force_mode="normal")
    assert abs(individual[0, 0, 0, 0]) > 1
    np.testing.assert_allclose(combined.numpy(), 0, atol=1e-6)


def test_actor_sources_include_descendant_geometry_and_keep_configuration_order() -> None:
    scene = _scene(
        """
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="ground" type="plane" size="5 5 .1"/>
            <body name="robot_root" pos="0 0 5"><geom type="sphere" size=".1"/></body>
            <body name="box_root" pos="2 0 .1">
              <freejoint/><geom type="box" size=".1 .1 .1" mass="2"/>
              <body name="payload" pos=".3 0 0"><geom type="box" size=".1 .1 .1" mass="3"/></body>
            </body>
            <body name="box_decoy_root" pos="3 0 .1">
              <freejoint/><geom type="box" size=".1 .1 .1" mass="7"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    scene.rigid_object_root_bodies = {"box": "box_root", "box_decoy": "box_decoy_root"}
    config = ContactSensorConfig(
        sources=[_actor("box_decoy"), _actor("box")],
        counterparts=[ContactTargetConfig(target_kind="terrain")],
    )
    targets = prepare_contact_targets(scene, {"weights": config})
    model = scene.world_spec.compile()
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    sources, counterparts = selection_masks(model, targets, config)
    actual = _pairs(model, data).matrix(sources, counterparts, force_mode="normal").numpy()

    # Independent mass/gravity values require both root and descendant contacts.
    np.testing.assert_allclose(actual[0, :, 0], [[0, 0, 7 * 9.81], [0, 0, 5 * 9.81]], atol=0.03)
    assert set(model.geom_bodyid[resolve_target_geoms(model, targets, _actor("box"))]) == {
        model.body("box_root").id,
        model.body("payload").id,
    }


def test_robot_link_pair_measures_internal_contact_while_the_whole_robot_cancels_it() -> None:
    scene = _scene(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="robot_root">
              <body name="robot_left" pos="-.08 0 0">
                <joint type="slide" axis="1 0 0"/><geom type="sphere" size=".1"/>
              </body>
              <body name="robot_right" pos=".08 0 0">
                <joint type="slide" axis="1 0 0"/><geom type="sphere" size=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    assert scene.robot_spec_meta is not None
    scene.robot_spec_meta.body_names = ["left", "right"]
    left = ContactTargetConfig(target_kind="robot_link", target="left")
    right = ContactTargetConfig(target_kind="robot_link", target="right")
    config = ContactSensorConfig(sources=[right, left], counterparts=[left, right])
    whole_robot = ContactSensorConfig()
    targets = prepare_contact_targets(scene, {"links": config, "whole": whole_robot})
    model = scene.world_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sources, counterparts = selection_masks(model, targets, config)
    pairs = _pairs(model, data)
    matrix = pairs.matrix(sources, counterparts, force_mode="normal").numpy()

    assert data.ncon == 1
    assert matrix[0, 0, 0, 0] > 1
    np.testing.assert_allclose(matrix[0, 1, 1], -matrix[0, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(matrix[0, [0, 1], [1, 0]], 0)
    sources, counterparts = selection_masks(model, targets, whole_robot)
    np.testing.assert_allclose(pairs.matrix(sources, counterparts, force_mode="normal"), 0, atol=1e-6)


def test_static_actor_geometry_survives_fusion_without_selecting_the_ground() -> None:
    scene = _scene()
    scene.world_spec.delete(scene.world_spec.body("box_root").joints[0])
    scene.world_spec.compiler.fusestatic = True
    config = ContactSensorConfig(sources=[ContactTargetConfig()], counterparts=[_actor("box")])
    targets = prepare_contact_targets(scene, {"pairs": config})
    model = scene.world_spec.compile()
    ids = resolve_target_geoms(model, targets, _actor("box"))
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert ground not in ids
    assert len(ids) == 1
    assert model.geom_bodyid[ids[0]] == 0
    sources, counterparts = selection_masks(model, targets, config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert _pairs(model, data).matrix(sources, counterparts, force_mode="normal")[0, 0, 0, 0] < -1


def test_fused_robot_link_follows_its_native_owner_and_rejects_overlapping_counterparts() -> None:
    scene = _scene()
    child = scene.world_spec.body("robot_root").add_body(name="robot_fixed", pos=[0, 0, 0.2])
    child.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0])
    assert scene.robot_spec_meta is not None
    scene.robot_spec_meta.body_names.append("fixed")
    scene.world_spec.compiler.fusestatic = True
    link = ContactTargetConfig(target_kind="robot_link", target="fixed")
    config = ContactSensorConfig(counterparts=[ContactTargetConfig(), link])
    targets = prepare_contact_targets(scene, {"overlap": config})
    model = scene.world_spec.compile()
    link_geoms = resolve_target_geoms(model, targets, link)
    assert set(model.geom_bodyid[link_geoms]) == {model.body("robot_root").id}
    with pytest.raises(ValueError, match="overlap"):
        selection_masks(model, targets, config)


def test_normal_and_total_modes_separate_friction_on_a_tilted_plane() -> None:
    scene = _scene(
        """
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="ground" type="plane" size="5 5 .1" euler="0 20 0" friction="5 5 5"/>
            <body name="robot_root" pos="0 0 .4">
              <freejoint/><geom type="box" size=".1 .1 .1" mass="2" friction="5 5 5"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    config = ContactSensorConfig(counterparts=[ContactTargetConfig(target_kind="terrain")])
    targets = prepare_contact_targets(scene, {"terrain": config})
    model = scene.world_spec.compile()
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    sources, counterparts = selection_masks(model, targets, config)
    pairs = _pairs(model, data)
    normal = pairs.matrix(sources, counterparts, force_mode="normal")[0, 0, 0].numpy()
    total = pairs.matrix(sources, counterparts, force_mode="total")[0, 0, 0].numpy()

    np.testing.assert_allclose(total, [0, 0, 19.62], atol=0.03)
    direction = np.array([np.sin(np.deg2rad(20)), 0, np.cos(np.deg2rad(20))])
    np.testing.assert_allclose(normal / np.linalg.norm(normal), direction, atol=1e-5)
    assert np.linalg.norm(total - normal) > 5
    # A separated body must produce zeros even after a nonzero measurement.
    data.qpos[2] = 5
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(_pairs(model, data).matrix(sources, counterparts, force_mode="normal"), 0)


def test_missing_actor_and_terrain_fail_during_scene_setup() -> None:
    scene = _scene()
    with pytest.raises(ValueError, match="not a registered scene object"):
        prepare_contact_targets(scene, {"missing": ContactSensorConfig(sources=[_actor("missing")])})
    scene.terrain_geom_name = None
    with pytest.raises(ValueError, match="requires a terrain collider"):
        prepare_contact_targets(
            scene, {"missing": ContactSensorConfig(counterparts=[ContactTargetConfig(target_kind="terrain")])}
        )


def test_sensors_and_legacy_api_share_one_native_extraction_per_substep(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = _scene()
    normal = ContactSensorConfig(counterparts=[_actor("box")])
    sensors = {
        "normal": dataclasses.replace(normal, record_substeps=True),
        "total": dataclasses.replace(normal, force_mode="total"),
        "slow": dataclasses.replace(normal, update_decimation=2),
    }
    targets = prepare_contact_targets(scene, sensors)
    model = scene.world_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = cast("FullSimConfig", SimpleNamespace(training=SimpleNamespace(num_envs=1)))
    hooks = HookRegistry()
    sim = cast(
        "MuJoCo",
        SimpleNamespace(
            root_model=model,
            backend=ClassicBackend(model, data, config, "cpu"),
            body_names=["root"],
            body_ids=[model.body("robot_root").id],
            _body_ids_t=torch.tensor([model.body("robot_root").id]),
            sensor_config=sensors,
            sensor_manager=SensorManager("cpu", 50.0, physics_hz=200.0, hooks=hooks),
            contact_target_map=targets,
            contact_pairs=None,
        ),
    )
    register_contacts(sim)
    native = mujoco.mj_contactForce
    calls = []

    def counted(model: Any, data: Any, index: int, result: Any) -> None:
        calls.append(index)
        native(model, data, index, result)

    monkeypatch.setattr(mujoco, "mj_contactForce", counted)
    sim.sensor_manager.contact_step_completed()
    hooks.emit(Phase.POST_STEP)
    render_sensors(sim)
    assert calls == list(range(data.ncon))
    assert sim.sensor_manager.get_contact("normal").buffers["forces"][0, 0, 0] < -1
    assert sim.sensor_manager.get_contact("total").buffers["forces"][0, 0, 0] < -1
    legacy = sim.sensor_manager.get_contact("contact_sensor")
    assert legacy.legacy_recorder is not None
    torch.testing.assert_close(legacy.read_legacy_forces(), legacy.legacy_recorder.latest_forces)
    assert calls == list(range(data.ncon))
    data.qpos[0] = -4
    mujoco.mj_forward(model, data)
    sim.contact_pairs = None
    sim.sensor_manager.contact_step_completed()
    render_sensors(sim)
    np.testing.assert_array_equal(sim.sensor_manager.get_contact("normal").buffers["forces"], 0)
