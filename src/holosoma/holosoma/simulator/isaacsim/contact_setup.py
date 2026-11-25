# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Translate Holosoma contact groups to native single-body contact views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from pxr import Usd, UsdPhysics

from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig
from holosoma.simulator.isaacsim.prim_utils import (
    get_current_stage,
    is_rigid_body_enabled,
    resolve_robot_link_rigid_body_prim_expression,
    set_instanceable,
)
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim

_ENV_PATH = "/World/envs/env_0"
_ENV_EXPRESSION = "/World/envs/env_.*"


def _root(sim: IsaacSim, target: ContactTargetConfig) -> str:
    if target.target_kind == "robot":
        return f"{_ENV_PATH}/Robot"
    if target.target_kind == "robot_link":
        return resolve_robot_link_rigid_body_prim_expression(target.target).replace(_ENV_EXPRESSION, _ENV_PATH)
    if target.target_kind == "terrain":
        return "/World/ground"
    if target.target not in sim.scene.rigid_objects:
        raise ValueError(
            f"Contact actor '{target.target}' is not registered. Known actors: {list(sim.scene.rigid_objects)}."
        )
    return f"{_ENV_PATH}/{target.target}"


def _prims(sim: IsaacSim, target: ContactTargetConfig) -> list[Usd.Prim]:
    path = _root(sim, target)
    prim = get_current_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"Contact selector '{target.label}' has no prim '{path}' in the scene.")
    return list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))


def _source_bodies(sim: IsaacSim, target: ContactTargetConfig) -> list[str]:
    prims = _prims(sim, target)
    if target.target_kind == "robot_link":
        bodies = [_root(sim, target)]
    else:
        bodies = [str(prim.GetPath()) for prim in prims if is_rigid_body_enabled(prim)]
    if not bodies:
        raise ValueError(f"Contact source '{target.label}' has no enabled rigid bodies.")
    return bodies


def _counterpart_paths(sim: IsaacSim, target: ContactTargetConfig) -> list[str]:
    # GPU contact filters use rigid bodies. The terrain has static collider paths.
    if target.target_kind != "terrain":
        return _source_bodies(sim, target)
    colliders = []
    for prim in _prims(sim, target):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        if enabled and enabled.HasAuthoredValue() and not enabled.Get():
            continue
        colliders.append(str(prim.GetPath()))
    if not colliders:
        raise ValueError(f"Contact counterpart '{target.label}' has no enabled collision geometry.")
    return colliders


@dataclass
class IsaacContactReader:
    """Aggregate native normal and friction forces into the configured group axes."""

    views: list[list[list[ContactSensor]]]
    config: ContactSensorConfig
    num_envs: int
    device: str

    def __call__(self) -> torch.Tensor:
        matrix = torch.zeros(
            self.num_envs, len(self.views), len(self.views[0]), 3, dtype=torch.float32, device=self.device
        )
        for source, groups in enumerate(self.views):
            for counterpart, sensors in enumerate(groups):
                for sensor in sensors:
                    data = sensor.data
                    if not self.config.counterparts:
                        matrix[:, source, counterpart] += data.net_forces_w.sum(dim=1)
                    else:
                        if data.force_matrix_w is None:
                            raise RuntimeError("The native contact sensor returned no filtered forces.")
                        matrix[:, source, counterpart] += data.force_matrix_w.sum(dim=(1, 2))
                        if self.config.force_mode == "total":
                            if data.friction_forces_w is None:
                                raise RuntimeError("The native contact sensor returned no friction forces.")
                            matrix[:, source, counterpart] += data.friction_forces_w.sum(dim=(1, 2))
        return matrix


def build_contacts(sim: IsaacSim) -> None:
    """Build one native view per source body and counterpart group before cloning."""
    sim.contact_readers = {}
    for name, config in sim.sensor_config.items():
        if not isinstance(config, ContactSensorConfig):
            continue
        for target in [*config.sources, *config.counterparts]:
            if target.target_kind == "actor":
                set_instanceable(get_current_stage(), _root(sim, target), False)
        counterparts = [_counterpart_paths(sim, target) for target in config.counterparts]
        selected = [path for group in counterparts for path in group]
        if len(set(selected)) != len(selected):
            raise ValueError(f"Contact sensor '{name}' counterpart groups overlap after native body resolution.")
        groups: list[list[list[ContactSensor]]] = []
        for source_index, source in enumerate(config.sources):
            source_views: list[list[ContactSensor]] = []
            bodies = _source_bodies(sim, source)
            for counterpart_index, colliders in enumerate(counterparts or [[]]):
                views = []
                for body_index, body in enumerate(bodies):
                    cfg = ContactSensorCfg(
                        prim_path=body.replace(_ENV_PATH, _ENV_EXPRESSION),
                        update_period=0.0,
                        filter_prim_paths_expr=[path.replace(_ENV_PATH, _ENV_EXPRESSION) for path in colliders],
                        track_friction_forces=config.force_mode == "total",
                        max_contact_data_count_per_prim=config.isaacsim.max_contact_data_count_per_prim,
                        track_air_time=False,
                        debug_vis=config.isaacsim.debug_vis,
                    )
                    sensor = ContactSensor(cfg)
                    # Single-body views support native one-to-many filtering, including grouped sources.
                    key = f"__holosoma_contact_{name}_{source_index}_{counterpart_index}_{body_index}"
                    sim.scene.sensors[key] = sensor
                    views.append(sensor)
                source_views.append(views)
            groups.append(source_views)
        sim.contact_readers[name] = IsaacContactReader(groups, config, sim.training_config.num_envs, sim.sim_device)


def register_contacts(sim: IsaacSim) -> None:
    """Expose the existing robot sensor and configured contact views through the shared manager."""
    default = ContactSensorConfig(
        sources=[ContactTargetConfig(target_kind="robot_link", target=name) for name in sim.body_names]
    )
    sim.sensor_manager.register_contact(
        "contact_sensor",
        default,
        lambda: sim._robot_contact_forces().unsqueeze(2),
        legacy_capture=sim._robot_contact_forces,
    )
    for name, reader in sim.contact_readers.items():
        sim.sensor_manager.register_contact(name, reader.config, reader)
