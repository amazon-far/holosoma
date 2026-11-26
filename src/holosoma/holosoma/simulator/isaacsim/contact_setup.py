# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Translate Holosoma contact groups to native single-body contact views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from pxr import Usd, UsdPhysics

from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig
from holosoma.simulator.isaacsim.prim_utils import (
    get_current_stage,
    is_rigid_body_enabled,
    resolve_robot_link_rigid_body_prim_expression,
    set_instanceable,
)
from holosoma.simulator.shared.sensor_manager import robot_body_contact_config
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

    views: dict[str, ContactSensor]
    source_bodies: list[list[str]]
    counterpart_columns: dict[str, list[list[int]]]
    config: ContactSensorConfig
    num_envs: int
    device: str

    def __call__(self) -> torch.Tensor:
        matrix = torch.zeros(
            self.num_envs,
            len(self.source_bodies),
            len(self.config.counterparts) or 1,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        for source, bodies in enumerate(self.source_bodies):
            for body in bodies:
                data = self.views[body].data
                if not self.config.counterparts:
                    matrix[:, source, 0] += data.net_forces_w[:, 0]
                    continue
                if data.force_matrix_w is None:
                    raise RuntimeError("The native contact sensor returned no filtered forces.")
                forces = data.force_matrix_w[:, 0]
                if self.config.force_mode == "total":
                    if data.friction_forces_w is None:
                        raise RuntimeError("The native contact sensor returned no friction forces.")
                    forces = forces + data.friction_forces_w[:, 0]
                for counterpart, columns in enumerate(self.counterpart_columns[body]):
                    matrix[:, source, counterpart] += forces[:, columns].sum(dim=1)
        return matrix


def build_contacts(sim: IsaacSim) -> None:
    """Share one native single-body view across all named contact measurements."""
    view_configs: dict[str, ContactSensorCfg] = {}
    selections: dict[str, tuple[list[list[str]], list[list[str]]]] = {}
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
        sources = [_source_bodies(sim, source) for source in config.sources]
        selections[name] = sources, counterparts
        filters = [path.replace(_ENV_PATH, _ENV_EXPRESSION) for path in selected]
        for body in dict.fromkeys(path for group in sources for path in group):
            if body not in view_configs:
                view_configs[body] = ContactSensorCfg(
                    prim_path=body.replace(_ENV_PATH, _ENV_EXPRESSION),
                    update_period=0.0,
                    filter_prim_paths_expr=[],
                    track_air_time=False,
                    debug_vis=sim.debug_viz_enabled,
                    max_contact_data_count_per_prim=config.isaacsim.max_contact_data_count_per_prim,
                )
            cfg = view_configs[body]
            cfg.filter_prim_paths_expr.extend(path for path in filters if path not in cfg.filter_prim_paths_expr)
            cfg.track_friction_forces |= config.force_mode == "total"
            cfg.max_contact_data_count_per_prim = max(
                cfg.max_contact_data_count_per_prim, config.isaacsim.max_contact_data_count_per_prim
            )

    index = 0
    for body, cfg in view_configs.items():
        sensor = ContactSensor(cfg)
        key = f"__holosoma_contact_{index}"
        while key in sim.scene.sensors or key in sim.sensor_config:
            index += 1
            key = f"__holosoma_contact_{index}"
        sim.scene.sensors[key] = sensor
        sim.contact_views[body] = sensor
        index += 1

    for name, (sources, counterparts) in selections.items():
        columns = {
            body: [
                [
                    view_configs[body].filter_prim_paths_expr.index(path.replace(_ENV_PATH, _ENV_EXPRESSION))
                    for path in group
                ]
                for group in counterparts
            ]
            for body in dict.fromkeys(path for group in sources for path in group)
        }
        sim.contact_readers[name] = IsaacContactReader(
            sim.contact_views,
            sources,
            columns,
            cast(ContactSensorConfig, sim.sensor_config[name]),
            sim.training_config.num_envs,
            sim.sim_device,
        )


def register_contacts(sim: IsaacSim) -> None:
    """Expose the existing robot sensor and configured contact views through the shared manager."""
    for body, sensor in sim.contact_views.items():
        view = sensor.contact_physx_view
        if (
            sensor.num_bodies != 1
            or view.sensor_count != sim.num_envs
            or view.filter_count != len(sensor.cfg.filter_prim_paths_expr)
        ):
            raise RuntimeError(
                f"Contact view '{body}' resolved an unexpected native layout: "
                f"{view.sensor_count} sensors, {sensor.num_bodies} bodies per environment, "
                f"{view.filter_count} filters for {len(sensor.cfg.filter_prim_paths_expr)} patterns."
            )
    default = robot_body_contact_config(sim.body_names)
    sim.sensor_manager.register_contact(
        "contact_sensor",
        default,
        lambda: sim._robot_contact_forces().unsqueeze(2),
    )
    for name, reader in sim.contact_readers.items():
        sim.sensor_manager.register_contact(name, reader.config, reader)
