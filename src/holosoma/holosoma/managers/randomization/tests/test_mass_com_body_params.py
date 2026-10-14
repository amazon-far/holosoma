"""Body-spec params of ``randomize_mass_startup`` / ``randomize_base_com_startup`` (no sim).

The two terms take their target bodies as params (``link_bodies``, ``base_body``, ``body``: exact names
or ``#tag`` references) resolved through ``env.robot_names.resolve_body_names``. A stub env running the real
``expand_name_spec`` plus a stand-in simulator whose ``backend.model`` is a bare ``mujoco.MjModel``
drive the argument validation and the MuJoCo write path end to end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Sequence

import pytest

if TYPE_CHECKING:
    import mujoco
else:
    # no_sim lane runs in an env without the mujoco package (isaacsim conda); skip when unavailable.
    mujoco = pytest.importorskip("mujoco")

from holosoma.managers.randomization.terms import locomotion as L
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.utils.name_tags import expand_name_spec
from holosoma.utils.sampler import STAGE_SETUP, DistributionLike, TermSampler
from holosoma.utils.simulator_config import SimulatorType

pytestmark = pytest.mark.no_sim

_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="10"/>
      <body name="link_a" pos="0 0 -0.3">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.05" mass="2"/>
      </body>
      <body name="link_b" pos="0 0 0.3">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.05" mass="3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_BODY_NAMES = ["base", "link_a", "link_b"]
_DEFAULT_TAGS = {"#base": ["base"], "#legs": ["link_a", "link_b"]}
_ZERO_COM = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}


class _Sim(BaseSimulator):
    """The MuJoCo ClassicBackend attributes ``randomize_field`` reads."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.backend = SimpleNamespace(model=model)
        self.sim_device = "cpu"
        self.num_envs = 1
        self.robot_config = SimpleNamespace(body_names=list(_BODY_NAMES), dof_names=[])  # type: ignore[assignment]
        self.body_ids = [model.body(name).id for name in self.body_names]

    def get_simulator_type(self) -> SimulatorType:
        return SimulatorType.MUJOCO


class _Env:
    """Stub env: ``resolve_body_names`` runs the real tag expansion against ``body_names``."""

    def __init__(self, body_tags: dict[str, list[str]], simulator: _Sim) -> None:
        self.robot_config = SimpleNamespace(body_tags=body_tags)
        self.body_names = list(_BODY_NAMES)
        self.simulator = simulator
        self.num_envs = 1
        self.device = "cpu"

    def resolve_body_names(self, spec: Sequence[str]) -> list[str]:
        return expand_name_spec(spec, self.robot_config.body_tags, self.body_names, kind="body")

    @property
    def robot_names(self) -> _Env:
        return self


def _make(body_tags: dict[str, list[str]] | None = None) -> tuple[_Env, mujoco.MjModel]:
    model = mujoco.MjModel.from_xml_string(_XML)
    return _Env(dict(_DEFAULT_TAGS) if body_tags is None else body_tags, _Sim(model)), model


def _sampler() -> TermSampler:
    return TermSampler.bind(0, "test_mass_com_body_params", STAGE_SETUP)


def _bid(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _masses(model: mujoco.MjModel) -> dict[str, float]:
    return {name: float(model.body_mass[_bid(model, name)]) for name in _BODY_NAMES}


# --- argument validation ---------------------------------------------------------------------------


def test_link_mass_requires_link_bodies() -> None:
    env, _ = _make()
    with pytest.raises(ValueError, match="enable_link_mass=True requires link_bodies"):
        L.randomize_mass_startup(env, sampler=_sampler(), enable_link_mass=True, enable_base_mass=False)


def test_base_body_must_resolve_to_one_body() -> None:
    env, _ = _make({"#base": ["base", "link_a"]})
    with pytest.raises(ValueError, match="base mass DR targets one body"):
        L.randomize_mass_startup(env, sampler=_sampler(), enable_link_mass=False, enable_base_mass=True)
    with pytest.raises(ValueError, match="base CoM DR targets one body"):
        L.randomize_base_com_startup(env, sampler=_sampler(), base_com_range=_ZERO_COM)


def test_unknown_link_body_raises_from_resolver() -> None:
    env, _ = _make()
    with pytest.raises(ValueError, match="Unknown body name 'nope'"):
        L.randomize_mass_startup(
            env, sampler=_sampler(), enable_link_mass=True, link_bodies=["nope"], enable_base_mass=False
        )


def test_unknown_base_tag_raises_from_resolver() -> None:
    env, _ = _make({"legs": ["link_a", "link_b"]})  # no "base" tag
    with pytest.raises(ValueError, match="Unknown body tag '#base'"):
        L.randomize_base_com_startup(env, sampler=_sampler(), base_com_range=_ZERO_COM)


def test_base_body_not_resolved_when_base_mass_disabled() -> None:
    env, model = _make({"legs": ["link_a", "link_b"]})  # no "base" tag: must not be needed
    L.randomize_mass_startup(
        env,
        sampler=_sampler(),
        enable_link_mass=True,
        link_bodies=["link_a"],
        link_mass_range=(1.0, 1.0),
        enable_base_mass=False,
    )
    assert _masses(model) == {"base": 10.0, "link_a": 2.0, "link_b": 3.0}


def test_disabled_term_skips_resolution() -> None:
    env, _ = _make({})  # no tags at all; enabled=False must return before resolving
    L.randomize_mass_startup(env, sampler=_sampler(), enabled=False)
    L.randomize_base_com_startup(env, sampler=_sampler(), base_com_range=_ZERO_COM, enabled=False)


# --- MuJoCo write path -----------------------------------------------------------------------------


def test_default_base_tag_adds_base_mass_only() -> None:
    env, model = _make()
    inertia_before = model.body_inertia[_bid(model, "base")].copy()
    L.randomize_mass_startup(
        env, sampler=_sampler(), enable_link_mass=False, enable_base_mass=True, added_mass_range=(2.0, 2.0)
    )
    assert _masses(model) == pytest.approx({"base": 12.0, "link_a": 2.0, "link_b": 3.0})
    assert model.body_inertia[_bid(model, "base")] == pytest.approx(inertia_before * 1.2)


def test_base_body_param_overrides_default() -> None:
    env, model = _make()
    L.randomize_mass_startup(
        env,
        sampler=_sampler(),
        enable_link_mass=False,
        enable_base_mass=True,
        base_body="link_b",
        added_mass_range=(2.0, 2.0),
    )
    assert _masses(model) == pytest.approx({"base": 10.0, "link_a": 2.0, "link_b": 5.0})


def test_link_bodies_tag_expands_and_scales() -> None:
    env, model = _make()
    inertia_before = model.body_inertia[_bid(model, "link_a")].copy()
    L.randomize_mass_startup(
        env,
        sampler=_sampler(),
        enable_link_mass=True,
        link_bodies=["#legs"],
        link_mass_range=(2.0, 2.0),
        enable_base_mass=False,
    )
    assert _masses(model) == pytest.approx({"base": 10.0, "link_a": 4.0, "link_b": 6.0})
    assert model.body_inertia[_bid(model, "link_a")] == pytest.approx(inertia_before * 2.0)


def test_link_bodies_without_inertia_recompute_leaves_inertia() -> None:
    env, model = _make()
    inertia_before = model.body_inertia[_bid(model, "link_b")].copy()
    L.randomize_mass_startup(
        env,
        sampler=_sampler(),
        enable_link_mass=True,
        link_bodies=["link_b"],
        link_mass_range=(2.0, 2.0),
        enable_base_mass=False,
        recompute_inertia=False,
    )
    assert _masses(model) == pytest.approx({"base": 10.0, "link_a": 2.0, "link_b": 6.0})
    assert model.body_inertia[_bid(model, "link_b")] == pytest.approx(inertia_before)


def test_default_base_tag_offsets_base_com() -> None:
    env, model = _make()
    ipos_before = model.body_ipos.copy()
    L.randomize_base_com_startup(
        env, sampler=_sampler(), base_com_range={"x": [0.1, 0.1], "y": [0.0, 0.0], "z": [0.0, 0.0]}
    )
    expected = ipos_before.copy()
    expected[_bid(model, "base"), 0] += 0.1
    assert model.body_ipos == pytest.approx(expected)


def test_body_param_overrides_default_com_target() -> None:
    env, model = _make()
    ipos_before = model.body_ipos.copy()
    L.randomize_base_com_startup(
        env, sampler=_sampler(), base_com_range={"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.2, 0.2]}, body="link_a"
    )
    expected = ipos_before.copy()
    expected[_bid(model, "link_a"), 2] += 0.2
    assert model.body_ipos == pytest.approx(expected)


@pytest.mark.parametrize(
    "invalid_addition",
    [(1.0, -1.0), {"kind": "log_uniform", "low": -1.0, "high": 1.0}],
)
def test_mass_distributions_are_all_validated_before_any_write(invalid_addition: DistributionLike) -> None:
    env, model = _make()
    masses_before, inertia_before = model.body_mass.copy(), model.body_inertia.copy()
    with pytest.raises(ValueError, match="must be >= low|strictly positive"):
        L.randomize_mass_startup(
            env,
            sampler=_sampler(),
            link_bodies=["#legs"],
            link_mass_range=(1.5, 1.5),
            added_mass_range=invalid_addition,
        )
    assert model.body_mass == pytest.approx(masses_before)
    assert model.body_inertia == pytest.approx(inertia_before)
