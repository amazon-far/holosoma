"""TerrainLocomotion resolves its feet-height bodies in setup() via env.robot_names.resolve_body_indices."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from loguru import logger

from holosoma.config_types.terrain import MeshType, TerrainTermCfg
from holosoma.utils.name_tags import expand_name_spec
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorConfig, SimulatorType

pytestmark = pytest.mark.no_sim

# holosoma.utils.draw (imported by the terrain term) binds its backend at import time. Bind it to MuJoCo for this
# module's import whatever the process-global type is (its adapter imports mujoco, so an env without it skips),
# then restore the previous type and the previously bound modules so no other test sees this module's state.
pytest.importorskip("mujoco")
_previous_simulator_type = SimulatorConfig._simulator_type
_REBOUND_PREFIXES = ("holosoma.utils.draw", "holosoma.utils.adapters", "holosoma.managers.terrain.terms.locomotion")
_previous_modules = {name: sys.modules.pop(name) for name in list(sys.modules) if name.startswith(_REBOUND_PREFIXES)}
SimulatorConfig._simulator_type = SimulatorType.MUJOCO
try:
    from holosoma.managers.terrain.terms import locomotion as terrain_locomotion
    from holosoma.managers.terrain.terms.locomotion import TerrainLocomotion
finally:
    SimulatorConfig._simulator_type = _previous_simulator_type
    for name in [n for n in sys.modules if n.startswith(_REBOUND_PREFIXES)]:
        sys.modules.pop(name, None)
    sys.modules.update(_previous_modules)

NUM_ENVS = 3
# The stub resolver's per-body indices; a resolved list follows the resolved NAME order, not this dict's.
BODY_INDEX = {"left_foot_contact_point": 9, "right_foot_contact_point": 4, "front_contact_point": 7}


class _StubTerrain:
    def __init__(self, cfg: TerrainTermCfg, num_envs: int) -> None:
        self.mesh = SimpleNamespace(vertices=np.zeros((3, 3)), faces=np.zeros((1, 3), dtype=np.int64))
        self._num_envs = num_envs
        self._env_origins = np.zeros((1, 1, 3))

    def sample_env_origins(self) -> npt.NDArray[np.float64]:
        return np.zeros((self._num_envs, 3))


class _EnvProxy:
    """Mirror of run_sim's EnvProxy: TerrainManager is built with only num_envs/device and setup() never runs."""

    def __init__(self) -> None:
        self.num_envs = NUM_ENVS
        self.device = "cpu"


class _StubEnv(_EnvProxy):
    def __init__(self, body_tags: dict[str, list[str]]) -> None:
        super().__init__()
        self.robot_config = SimpleNamespace(body_tags=body_tags)
        self.resolve_calls: list[tuple[str, ...]] = []

    def resolve_body_indices(self, spec: Sequence[str]) -> torch.Tensor:
        self.resolve_calls.append(tuple(spec))
        names = expand_name_spec(spec, self.robot_config.body_tags, None, kind="body")
        return torch.tensor([BODY_INDEX[name] for name in names], dtype=torch.long)

    @property
    def robot_names(self) -> _StubEnv:
        return self


@pytest.fixture(autouse=True)
def _stub_terrain_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terrain_locomotion, "Terrain", _StubTerrain)
    monkeypatch.setattr(terrain_locomotion.warp_utils, "convert_to_wp_mesh", lambda *_args, **_kwargs: object())


@pytest.fixture
def info_logs() -> Iterator[list[str]]:
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(m.record["message"]), level="INFO")
    yield msgs
    logger.remove(sink_id)


def _cfg(feet_height_bodies: list[str] | None = None) -> TerrainTermCfg:
    return TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        mesh_type=MeshType.PLANE,
        feet_height_bodies=feet_height_bodies,
    )


def _setup(env: Any, cfg: TerrainTermCfg) -> TerrainLocomotion:
    term = TerrainLocomotion(cfg, env)
    term.setup()
    return term


def test_init_with_env_proxy_does_not_touch_robot_config() -> None:
    term = TerrainLocomotion(_cfg(), _EnvProxy())

    assert term.env_origins.shape == (NUM_ENVS, 3)


def test_auto_off_when_robot_declares_no_feet_height_tag(info_logs: list[str]) -> None:
    env = _StubEnv(body_tags={"#base": ["baseLink"]})

    term = _setup(env, _cfg())

    assert env.resolve_calls == []
    assert term._compute_feet_heights is False
    assert term.feet_heights.shape == (NUM_ENVS, 0)
    assert term._feet_height_indices.shape == (0,)
    assert term._feet_height_indices.dtype == torch.long
    assert "feet height raycasts off: no '#feet_height' body tag; body tags: ['#base']" in info_logs


def test_auto_off_on_a_misspelled_tag_names_the_defined_tags(info_logs: list[str]) -> None:
    """A typo cannot be told from an absent tag, so the log lists what the robot does declare."""
    env = _StubEnv(body_tags={"#base": ["baseLink"], "#feet_heigth": ["left_foot_contact_point"]})

    term = _setup(env, _cfg())

    assert term._compute_feet_heights is False
    assert "feet height raycasts off: no '#feet_height' body tag; body tags: ['#base', '#feet_heigth']" in info_logs


def test_auto_on_when_robot_declares_feet_height_tag(info_logs: list[str]) -> None:
    env = _StubEnv(body_tags={"#feet_height": ["left_foot_contact_point", "right_foot_contact_point"]})

    term = _setup(env, _cfg())

    assert env.resolve_calls == [("#feet_height",)]
    assert term._compute_feet_heights is True
    assert term.feet_heights.shape == (NUM_ENVS, 2)
    # Indices in the tag's member order (left, right), taken from the resolver untouched.
    assert term._feet_height_indices.tolist() == [9, 4]
    assert term._feet_height_indices.dtype == torch.long
    assert term._ray_directions_feet.shape == (NUM_ENVS, 2, 3)
    assert torch.equal(term._ray_directions_feet[..., 2], torch.full((NUM_ENVS, 2), -1.0))
    assert "feet height raycasts on: robot declares the '#feet_height' body tag" in info_logs


def test_empty_list_turns_feet_heights_off_despite_tag(info_logs: list[str]) -> None:
    env = _StubEnv(body_tags={"#feet_height": ["left_foot_contact_point", "right_foot_contact_point"]})

    term = _setup(env, _cfg(feet_height_bodies=[]))

    assert env.resolve_calls == []
    assert term._compute_feet_heights is False
    assert term.feet_heights.shape == (NUM_ENVS, 0)
    assert "feet height raycasts off: feet_height_bodies=[]" in info_logs


def test_explicit_list_is_resolved_as_given(info_logs: list[str]) -> None:
    env = _StubEnv(body_tags={})

    term = _setup(env, _cfg(feet_height_bodies=["front_contact_point"]))

    assert env.resolve_calls == [("front_contact_point",)]
    assert term._compute_feet_heights is True
    assert term.feet_heights.shape == (NUM_ENVS, 1)
    assert term._feet_height_indices.tolist() == [BODY_INDEX["front_contact_point"]]
    assert "feet height raycasts on: feet_height_bodies=['front_contact_point']" in info_logs


def test_explicit_unknown_tag_fails_loud() -> None:
    env = _StubEnv(body_tags={})

    with pytest.raises(ValueError, match="Unknown body tag"):
        _setup(env, _cfg(feet_height_bodies=["#feet_height"]))


def test_update_heights_skips_feet_raycast_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched at the module boundary (``warp_utils.ray_cast``): one base cast, no feet cast."""
    env: Any = _StubEnv(body_tags={})
    term = _setup(env, _cfg())
    root_z = torch.tensor([0.5, 0.8, 1.1])
    env.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(NUM_ENVS, 1)
    env.simulator = SimpleNamespace(
        robot_root_states=torch.zeros(NUM_ENVS, 13), _rigid_body_pos=torch.zeros(NUM_ENVS, 12, 3)
    )
    env.simulator.robot_root_states[:, 2] = root_z
    casts: list[tuple[int, ...]] = []

    def _ray_cast(starts: torch.Tensor, _directions: torch.Tensor, _mesh: object) -> torch.Tensor:
        casts.append(tuple(starts.shape))
        hits = starts.clone()
        hits[..., 2] = 0.0  # flat ground at z = 0
        return hits

    monkeypatch.setattr(terrain_locomotion.warp_utils, "ray_cast", _ray_cast)

    term.update_heights()

    assert casts == [(NUM_ENVS, term._num_base_height_points, 3)]
    assert torch.allclose(term.base_heights, root_z)
    assert term.feet_heights.shape == (NUM_ENVS, 0)


def test_update_heights_raycasts_from_the_resolved_feet_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    """The feet cast starts at ``_rigid_body_pos[:, resolved indices]`` (tag order) + ATTACHMENT_POS; each column
    of ``feet_heights`` is that body's height above its own hit."""
    env: Any = _StubEnv(body_tags={"#feet_height": ["left_foot_contact_point", "right_foot_contact_point"]})
    term = _setup(env, _cfg())
    env.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(NUM_ENVS, 1)
    env.simulator = SimpleNamespace(
        robot_root_states=torch.zeros(NUM_ENVS, 13), _rigid_body_pos=torch.zeros(NUM_ENVS, 12, 3)
    )
    env.simulator.robot_root_states[:, 2] = 0.7
    # Distinct z per body and x per env, so a read from the wrong body, in the wrong order or for the wrong env
    # shows in the result: body i at z = i / 10, env i at x = i.
    body_z = torch.arange(12, dtype=torch.float) / 10
    env_x = torch.arange(NUM_ENVS, dtype=torch.float).unsqueeze(1)
    env.simulator._rigid_body_pos[..., 2] = body_z
    env.simulator._rigid_body_pos[..., 0] = env_x
    casts: list[torch.Tensor] = []

    def _ray_cast(starts: torch.Tensor, directions: torch.Tensor, _mesh: object) -> torch.Tensor:
        casts.append(starts.clone())
        assert torch.equal(directions[..., 2], torch.full(starts.shape[:2], -1.0))
        hits = starts.clone()
        hits[..., 2] = 0.1 * starts[..., 0]  # the ground rises with x: env i stands on z = 0.1 * i
        return hits

    monkeypatch.setattr(terrain_locomotion.warp_utils, "ray_cast", _ray_cast)

    term.update_heights()

    assert len(casts) == 2  # one base cast, one feet cast
    feet_starts = casts[1]
    assert feet_starts.shape == (NUM_ENVS, 2, 3)
    expected_feet = torch.tensor([body_z[9], body_z[4]])  # left (body 9), right (body 4)
    assert torch.allclose(
        feet_starts[..., 2], (expected_feet + terrain_locomotion.ATTACHMENT_POS[2]).expand(NUM_ENVS, 2)
    )
    assert torch.equal(feet_starts[..., 0], env_x.expand(NUM_ENVS, 2))
    ground = 0.1 * env_x
    assert torch.allclose(term.feet_heights, expected_feet - ground)
    assert torch.allclose(term._ray_hits_world_feet[..., 2], ground.expand(NUM_ENVS, 2))
