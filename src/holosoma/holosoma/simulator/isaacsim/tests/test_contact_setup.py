"""Isaac contact grouping tests with the native SDK boundary replaced."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig, IsaacSimContactConfig
from holosoma.simulator.shared.sensor_manager import SensorManager

pytestmark = pytest.mark.no_sim

_ENV = "/World/envs/env_0"
_BODY_A = f"{_ENV}/Robot/a"
_BODY_B = f"{_ENV}/Robot/b"
_TARGET_A = f"{_ENV}/bundle/a"
_TARGET_B = f"{_ENV}/bundle/b"
_GROUND = "/World/ground/collider"


class _NativeContactSensor:
    def __init__(self, cfg: SimpleNamespace) -> None:
        self.cfg = cfg
        self.num_bodies = 1
        self.contact_physx_view = SimpleNamespace(
            sensor_count=2,
            filter_count=len(cfg.filter_prim_paths_expr),
            sensor_paths=[cfg.prim_path.replace("env_.*", f"env_{env}") for env in range(2)],
        )
        shape = (2, 1, len(cfg.filter_prim_paths_expr), 3)
        self._data = SimpleNamespace(
            net_forces_w=torch.zeros(2, 1, 3),
            force_matrix_w=torch.zeros(shape) if cfg.filter_prim_paths_expr else None,
            friction_forces_w=torch.zeros(shape) if cfg.track_friction_forces else None,
        )
        self.data_reads = 0

    @property
    def data(self) -> SimpleNamespace:
        self.data_reads += 1
        return self._data


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> Any:
    native = ModuleType("isaaclab.sensors")
    native.__dict__.update(
        ContactSensor=_NativeContactSensor,
        ContactSensorCfg=lambda **kwargs: SimpleNamespace(track_friction_forces=False, **kwargs),
    )
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", native)
    pxr = ModuleType("pxr")
    pxr.__dict__.update(Usd=SimpleNamespace(), UsdPhysics=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    prim_utils = ModuleType("holosoma.simulator.isaacsim.prim_utils")
    prim_utils.__dict__.update(
        get_current_stage=lambda: None,
        is_rigid_body_enabled=lambda _prim: True,
        resolve_robot_link_rigid_body_prim_expression=lambda name: f"/World/envs/env_.*/Robot/{name}",
        set_instanceable=lambda *_args: None,
    )
    monkeypatch.setitem(sys.modules, prim_utils.__name__, prim_utils)

    name = "_isaac_contact_setup_under_test"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "contact_setup.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    def source_bodies(_sim: object, target: ContactTargetConfig) -> list[str]:
        if target.target_kind == "robot":
            return [_BODY_A, _BODY_B]
        if target.target_kind == "robot_link":
            return [_BODY_B] if target.target in ("b", "fixed_on_b") else [_BODY_A]
        return [_TARGET_B]

    def counterparts(_sim: object, target: ContactTargetConfig) -> list[str]:
        if target.target_kind == "terrain":
            return [_GROUND]
        return [_TARGET_A, _TARGET_B] if target.target == "bundle" else [_TARGET_B]

    monkeypatch.setattr(module, "_source_bodies", source_bodies)
    monkeypatch.setattr(module, "_counterpart_paths", counterparts)
    return module


def _actor(name: str) -> ContactTargetConfig:
    return ContactTargetConfig(target_kind="actor", target=name)


def _sim() -> SimpleNamespace:
    terrain = ContactTargetConfig(target_kind="terrain")
    existing = object()
    robot_sensor = _NativeContactSensor(
        SimpleNamespace(prim_path=f"{_ENV}/Robot/.*", filter_prim_paths_expr=[], track_friction_forces=False)
    )
    robot_sensor.num_bodies = 2
    robot_sensor.contact_physx_view.sensor_count = 4
    robot_sensor.contact_physx_view.sensor_paths = [
        path.replace("env_0", f"env_{env}") for env in range(2) for path in (_BODY_B, _BODY_A)
    ]
    robot_sensor._data.net_forces_w = torch.tensor([[[10.0, 20, 30], [1.0, 2, 3]], [[30.0, 60, 90], [2.0, 4, 6]]])
    return SimpleNamespace(
        contact_readers={},
        contact_views={},
        sensor_config={
            "grouped": ContactSensorConfig(
                sources=[ContactTargetConfig(), ContactTargetConfig(target_kind="robot_link", target="a")],
                counterparts=[_actor("bundle"), terrain],
                isaacsim=IsaacSimContactConfig(max_contact_data_count_per_prim=16),
            ),
            "reversed_total": ContactSensorConfig(
                sources=[ContactTargetConfig(target_kind="robot_link", target="a")],
                counterparts=[terrain, _actor("single")],
                force_mode="total",
                isaacsim=IsaacSimContactConfig(max_contact_data_count_per_prim=128),
            ),
            "__holosoma_contact_0": ContactSensorConfig(
                sources=[ContactTargetConfig(target_kind="robot_link", target="a")]
            ),
        },
        scene=SimpleNamespace(
            sensors={"contact_sensor": robot_sensor, "__holosoma_contact_1": existing},
            rigid_objects={"bundle": object(), "single": object()},
        ),
        training_config=SimpleNamespace(num_envs=2),
        num_envs=2,
        sim_device="cpu",
        debug_viz_enabled=True,
        # Canonical outputs omit b and use a different order from the native view.
        body_names=["a"],
        contact_sensor=robot_sensor,
        _contact_to_robot_body_ids=torch.tensor([1]),
        sensor_manager=SensorManager("cpu", 50.0, physics_hz=200.0, num_envs=2),
    )


def test_unique_source_views_share_union_filters_capacity_and_friction(adapter: Any) -> None:
    sim = _sim()
    existing = sim.scene.sensors["__holosoma_contact_1"]
    adapter.build_contacts(sim)

    assert set(sim.contact_views) == {_BODY_A, _BODY_B}
    assert len(sim.scene.sensors) == 4
    assert sim.scene.sensors["__holosoma_contact_1"] is existing
    assert "__holosoma_contact_0" not in sim.scene.sensors
    assert sum(isinstance(sensor, _NativeContactSensor) for sensor in sim.scene.sensors.values()) == 3
    for reader in sim.contact_readers.values():
        if not reader.config.counterparts:
            continue
        for bodies in reader.source_bodies:
            for body in bodies:
                assert reader.views[body] is sim.contact_views[body]
    a, b = sim.contact_views[_BODY_A].cfg, sim.contact_views[_BODY_B].cfg
    assert a.filter_prim_paths_expr == [
        "/World/envs/env_.*/bundle/a",
        "/World/envs/env_.*/bundle/b",
        _GROUND,
    ]
    assert a.track_friction_forces
    assert a.max_contact_data_count_per_prim == 128
    assert not b.track_friction_forces
    assert b.max_contact_data_count_per_prim == 16
    assert a.debug_vis and b.debug_vis
    assert a.update_period == b.update_period == 0.0


def test_grouped_normal_total_and_unfiltered_readers_use_their_own_columns(adapter: Any) -> None:
    sim = _sim()
    adapter.build_contacts(sim)
    adapter.register_contacts(sim)
    a, b = sim.contact_views[_BODY_A].data, sim.contact_views[_BODY_B].data
    a.force_matrix_w[:] = torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 3]])
    b.force_matrix_w[:] = torch.tensor([[10.0, 0, 0], [0, 20, 0], [0, 0, 30]])
    a.friction_forces_w[:] = torch.tensor([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
    a.net_forces_w[:] = torch.tensor([100.0, 200, 300])
    # Distinct environments make an accidental reduction over the environment axis visible.
    a.force_matrix_w[1] *= 2
    b.force_matrix_w[1] *= 2
    a.friction_forces_w[1] *= 2
    a.net_forces_w[1] *= 2

    grouped = torch.tensor([[[11.0, 22, 0], [0, 0, 33]], [[1.0, 2, 0], [0, 0, 3]]])
    torch.testing.assert_close(sim.contact_readers["grouped"](), torch.stack([grouped, grouped * 2]))
    total = torch.tensor([[[0.0, 0, 3.3], [0, 2.2, 0]]])
    torch.testing.assert_close(sim.contact_readers["reversed_total"](), torch.stack([total, total * 2]))
    net = torch.tensor([[[1.0, 2, 3]]])
    torch.testing.assert_close(sim.contact_readers["__holosoma_contact_0"](), torch.stack([net, net * 2]))


def test_overlapping_counterpart_groups_fail_before_native_views_are_created(adapter: Any) -> None:
    sim = _sim()
    sim.sensor_config = {
        "overlap": ContactSensorConfig(counterparts=[_actor("bundle"), _actor("single")]),
    }
    with pytest.raises(ValueError, match="counterpart groups overlap"):
        adapter.build_contacts(sim)
    assert len(sim.scene.sensors) == 2


def test_unfiltered_robot_sources_reuse_one_read_and_include_omitted_and_fixed_bodies(adapter: Any) -> None:
    sim = _sim()
    sim.sensor_config = {
        "whole": ContactSensorConfig(),
        "links": ContactSensorConfig(
            sources=[ContactTargetConfig(target_kind="robot_link", target=name) for name in ("b", "a", "fixed_on_b")]
        ),
        "slow": ContactSensorConfig(update_decimation=2),
    }
    adapter.build_contacts(sim)
    adapter.register_contacts(sim)
    assert sim.contact_views == {}
    assert len(sim.scene.sensors) == 2
    native = sim.contact_sensor._data.net_forces_w.clone()
    sim.sensor_manager.contact_step_completed()
    assert sim.contact_sensor.data_reads == 1
    torch.testing.assert_close(sim.sensor_manager.get_contact("contact_sensor").buffers["forces"], native[:, 1:2])
    whole = sim.sensor_manager.get_contact("whole").buffers["forces"]
    torch.testing.assert_close(whole, native.sum(dim=1, keepdim=True))
    torch.testing.assert_close(sim.sensor_manager.get_contact("links").buffers["forces"], native[:, [0, 1, 0]])
    assert sim.sensor_manager.get_contact("slow").buffers["forces"].count_nonzero() == 0

    sim.contact_sensor._data.net_forces_w.mul_(2)
    sim.sensor_manager.contact_step_completed()
    assert sim.contact_sensor.data_reads == 2
    torch.testing.assert_close(whole, native.sum(dim=1, keepdim=True))
    torch.testing.assert_close(sim.sensor_manager.get_contact("slow").buffers["forces"], 2 * whole)

    sim.sensor_manager.reset_contacts(torch.tensor([0]))
    assert sim.contact_sensor.data_reads == 2
    assert sim.sensor_manager.get_contact("slow").buffers["forces"][0].count_nonzero() == 0
    sim.contact_sensor._data.net_forces_w = 3 * native
    sim.sensor_manager.contact_step_completed()
    assert sim.contact_sensor.data_reads == 3
    torch.testing.assert_close(sim.sensor_manager.get_contact("whole").buffers["forces"], 3 * whole)
    assert sim.sensor_manager.get_contact("slow").buffers["forces"][0].count_nonzero() == 0
    sim.sensor_manager.contact_step_completed()
    torch.testing.assert_close(sim.sensor_manager.get_contact("slow").buffers["forces"], 3 * whole)


def test_mixed_unfiltered_actor_and_robot_sources_keep_filtered_views_separate(adapter: Any) -> None:
    sim = _sim()
    sim.sensor_config["mixed"] = ContactSensorConfig(
        sources=[_actor("single"), ContactTargetConfig(target_kind="robot_link", target="a"), ContactTargetConfig()]
    )
    adapter.build_contacts(sim)
    adapter.register_contacts(sim)
    assert set(sim.contact_views) == {_BODY_A, _BODY_B, _TARGET_B}
    a, b, actor = (sim.contact_views[body]._data for body in (_BODY_A, _BODY_B, _TARGET_B))
    a.force_matrix_w.fill_(1)
    b.force_matrix_w.fill_(10)
    a.friction_forces_w.fill_(0.5)
    a.net_forces_w.fill_(1000)
    actor.net_forces_w[:] = torch.tensor([[[100.0, 200, 300]], [[400.0, 500, 600]]])
    native = sim.contact_sensor._data.net_forces_w
    sim.sensor_manager.contact_step_completed()
    assert sim.contact_sensor.data_reads == 1
    expected = torch.stack([actor.net_forces_w[:, 0], native[:, 1], native.sum(dim=1)], dim=1)
    torch.testing.assert_close(sim.sensor_manager.get_contact("mixed").buffers["forces"], expected)
    torch.testing.assert_close(
        sim.sensor_manager.get_contact("reversed_total").buffers["force_matrix"],
        torch.full((2, 1, 2, 3), 1.5),
    )
    torch.testing.assert_close(
        sim.sensor_manager.get_contact("grouped").buffers["force_matrix"][:, 0],
        torch.tensor([[[22.0] * 3, [11.0] * 3]]).expand(2, -1, -1),
    )


def test_unfiltered_source_missing_from_native_robot_view_fails_registration(adapter: Any) -> None:
    sim = _sim()
    sim.sensor_config = {"whole": ContactSensorConfig()}
    sim.contact_sensor.contact_physx_view.sensor_paths = [_BODY_A]
    adapter.build_contacts(sim)
    with pytest.raises(ValueError, match="absent from the robot contact view"):
        adapter.register_contacts(sim)


@pytest.mark.parametrize("field,value", [("filter_count", 2), ("sensor_count", 1), ("num_bodies", 2)])
def test_registration_rejects_native_layout_that_breaks_group_mapping(adapter: Any, field: str, value: int) -> None:
    sim = _sim()
    adapter.build_contacts(sim)
    sensor = sim.contact_views[_BODY_A]
    target = sensor if field == "num_bodies" else sensor.contact_physx_view
    setattr(target, field, value)
    with pytest.raises(RuntimeError, match="unexpected native layout"):
        adapter.register_contacts(sim)
