# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Resolve Holosoma contact selectors and read solved MuJoCo contact forces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import numpy.typing as npt

from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig, SensorConfig
from holosoma.simulator.mujoco.backends import ClassicBackend, WarpBackend
from holosoma.simulator.shared.sensor_manager import robot_body_contact_config
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.mujoco import MuJoCo
    from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager


@dataclass
class ContactTargetMap:
    """Authored actor geometry and link markers that survive native body fusion."""

    geom_names: dict[str, list[str]] = field(default_factory=dict)
    link_sites: dict[str, str] = field(default_factory=dict)


def _subtree_geoms(body: Any) -> list[Any]:
    geoms = list(body.geoms)
    for child in body.bodies:
        geoms.extend(_subtree_geoms(child))
    return geoms


def prepare_contact_targets(scene: MujocoSceneManager, sensors: Mapping[str, SensorConfig]) -> ContactTargetMap:
    """Capture exact actor membership before compilation can merge static bodies."""
    result = ContactTargetMap()
    selectors = {
        target.label: target
        for config in sensors.values()
        if isinstance(config, ContactSensorConfig)
        for target in [*config.sources, *config.counterparts]
    }
    if not selectors:
        return result

    # Names keep a fused actor's geoms addressable without selecting the whole world body.
    named = {geom.name for geom in scene.world_spec.geoms if geom.name}
    next_id = 0
    for geom in scene.world_spec.geoms:
        if not geom.name:
            while f"holosoma_contact_geom_{next_id}" in named:
                next_id += 1
            geom.name = f"holosoma_contact_geom_{next_id}"
            named.add(geom.name)
            next_id += 1

    for label, selector in selectors.items():
        if selector.target_kind == "terrain":
            if scene.terrain_geom_name is None:
                raise ValueError("Contact counterpart 'terrain' requires a terrain collider.")
            result.geom_names[label] = [scene.terrain_geom_name]
            continue
        if selector.target_kind == "robot":
            meta = scene.robot_spec_meta
            if meta is None:
                raise ValueError("Contact selector 'robot' requires a robot.")
            body_name = f"{meta.prefix}{meta.root_body}"
        else:
            body_name = scene.resolve_sensor_body_name(selector.target_kind, selector.target)
        body = scene.world_spec.body(body_name)
        if body is None:
            raise ValueError(f"Contact selector '{label}' has no body '{body_name}' in the scene.")
        if selector.target_kind == "robot_link":
            site = f"holosoma_contact_link_{len(result.link_sites)}"
            if scene.world_spec.site(site) is not None:
                raise ValueError(f"Contact marker name '{site}' conflicts with an authored site.")
            body.add_site(name=site, size=[0.001, 0.001, 0.001], rgba=[0, 0, 0, 0])
            result.link_sites[label] = site
        else:
            geoms = _subtree_geoms(body)
            result.geom_names[label] = [geom.name for geom in geoms]
    return result


def resolve_target_geoms(
    model: mujoco.MjModel, targets: ContactTargetMap, selector: ContactTargetConfig
) -> npt.NDArray[np.int32]:
    """Resolve an exact actor group or a link's native rigid-body owner."""
    label = selector.label
    if label in targets.link_sites:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, targets.link_sites[label])
        if site_id < 0:
            raise RuntimeError(f"Contact selector '{label}' lost its compiled body marker.")
        body_id = int(model.site_bodyid[site_id])
        if body_id == 0:
            raise ValueError(f"Contact link '{selector.target}' fused into the world and has no selective rigid body.")
        ids = np.flatnonzero(model.geom_bodyid == body_id).astype(np.int32)
    else:
        ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in targets.geom_names[label]],
            dtype=np.int32,
        )
        if np.any(ids < 0):
            raise RuntimeError(f"Contact selector '{label}' lost geometry during compilation.")
    return ids


def selection_masks(
    model: mujoco.MjModel, targets: ContactTargetMap, config: ContactSensorConfig
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Build source and counterpart membership, rejecting overlapping counterpart groups."""
    source_masks = np.zeros((len(config.sources), model.ngeom), dtype=np.int32)
    for index, source in enumerate(config.sources):
        source_masks[index, resolve_target_geoms(model, targets, source)] = 1
    counterpart_masks = np.ones((1, model.ngeom), dtype=np.int32)
    if config.counterparts:
        counterpart_masks = np.zeros((len(config.counterparts), model.ngeom), dtype=np.int32)
        for index, counterpart in enumerate(config.counterparts):
            counterpart_masks[index, resolve_target_geoms(model, targets, counterpart)] = 1
        if np.any(counterpart_masks.sum(axis=0) > 1):
            raise ValueError("Contact counterpart groups overlap after native body resolution.")
    return source_masks, counterpart_masks


def _reader(
    sim: MuJoCo,
    sources: npt.NDArray[np.int32],
    counterparts: npt.NDArray[np.int32],
    config: ContactSensorConfig,
) -> Callable[[], torch.Tensor]:
    backend = sim.backend
    if isinstance(backend, WarpBackend):
        from holosoma.simulator.mujoco.contact_warp import WarpContactReader

        return WarpContactReader(backend, sources, counterparts, config.force_mode)
    assert isinstance(backend, ClassicBackend)
    return lambda: backend.contact_pairs().matrix(sources, counterparts, force_mode=config.force_mode)


def register_contacts(sim: MuJoCo) -> None:
    """Register the default per-robot-body sensor and every configured contact sensor."""
    assert sim.root_model is not None
    model = sim.root_model
    sources = np.zeros((len(sim.body_names), model.ngeom), dtype=np.int32)
    for index, body_id in enumerate(sim.body_ids):
        sources[index, model.geom_bodyid == body_id] = 1
    default = robot_body_contact_config(sim.body_names)
    sim.sensor_manager.register_contact(
        "contact_sensor",
        default,
        _reader(sim, sources, np.ones((1, model.ngeom), dtype=np.int32), default),
    )
    for name, config in sim.sensor_config.items():
        if isinstance(config, ContactSensorConfig):
            sources, counterparts = selection_masks(model, sim.contact_target_map, config)
            sim.sensor_manager.register_contact(name, config, _reader(sim, sources, counterparts, config))
