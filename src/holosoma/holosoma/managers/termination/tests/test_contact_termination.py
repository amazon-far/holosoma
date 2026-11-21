"""contact_forces_exceeded names its bodies (exact names / #tags) and resolves them through the env."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.managers.termination.terms.locomotion import contact_forces_exceeded
from holosoma.utils.name_tags import expand_name_spec
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 4
NUM_BODIES = 6
BODY_TAGS = {"#hips": ["left_hip", "right_hip"]}
BODY_INDEX = {"pelvis": 0, "left_hip": 2, "right_hip": 3, "torso": 5}


class _StubEnv:
    def __init__(self) -> None:
        self.num_envs = NUM_ENVS
        self.device = "cpu"
        self.simulator = SimpleNamespace(contact_forces=torch.zeros(NUM_ENVS, NUM_BODIES, 3))
        self.resolve_calls: list[tuple[str, ...]] = []

    def resolve_body_indices(self, spec: Sequence[str]) -> torch.Tensor:
        self.resolve_calls.append(tuple(spec))
        names = expand_name_spec(spec, BODY_TAGS, list(BODY_INDEX), kind="body")
        return torch.tensor([BODY_INDEX[name] for name in names], dtype=torch.long)

    @property
    def robot_names(self) -> _StubEnv:
        return self


def _env() -> Any:
    return _StubEnv()


def test_terminates_only_on_named_bodies() -> None:
    env = _env()
    forces = env.simulator.contact_forces
    forces[0, BODY_INDEX["right_hip"]] = torch.tensor([0.0, 0.0, 3.0])  # named via #hips -> terminate
    forces[1, BODY_INDEX["torso"]] = torch.tensor([0.0, 0.0, 3.0])  # not named -> keep
    forces[2, BODY_INDEX["pelvis"]] = torch.tensor([0.6, 0.0, 0.6])  # named, |f| ~ 0.85 < threshold -> keep

    out = contact_forces_exceeded(env, bodies=["pelvis", "#hips"], force_threshold=1.0)

    assert env.resolve_calls == [("pelvis", "#hips")]
    assert out.dtype == torch.bool
    assert out.tolist() == [True, False, False, False]


def test_force_threshold_is_honoured() -> None:
    env = _env()
    env.simulator.contact_forces[2, BODY_INDEX["pelvis"]] = torch.tensor([0.6, 0.0, 0.8])  # |f| == 1.0 exactly
    env.simulator.contact_forces[3, BODY_INDEX["pelvis"], 2] = 2.0

    # Strictly above the threshold terminates; a force norm equal to it does not.
    assert contact_forces_exceeded(env, bodies=["pelvis"], force_threshold=1.0).tolist() == [False, False, False, True]
    assert contact_forces_exceeded(env, bodies=["pelvis"], force_threshold=0.5).tolist() == [False, False, True, True]
    assert contact_forces_exceeded(env, bodies=["pelvis"], force_threshold=2.0).tolist() == [False] * NUM_ENVS
    assert contact_forces_exceeded(env, bodies=["pelvis"], force_threshold=2.5).tolist() == [False] * NUM_ENVS


def test_bare_string_bodies_param_raises_when_the_term_resolves_it() -> None:
    """``params={"bodies": "pelvis"}``, the likeliest preset typo, fails at the resolver, not by iterating chars."""
    env = _env()
    with pytest.raises(
        TypeError, match=r"body spec must be a list of names, got the string 'pelvis'; wrap it in a list"
    ):
        contact_forces_exceeded(env, bodies="pelvis")


def test_bodies_has_no_default() -> None:
    """Presets must name their bodies: no env attribute fallback, no implicit default."""
    params = inspect.signature(contact_forces_exceeded).parameters
    assert params["bodies"].default is inspect.Parameter.empty
    assert "contact_indices_attr" not in params


@pytest.mark.parametrize(
    ("preset", "bodies"),
    [
        (g1_29dof_termination, ["#root", "#shoulder_links", "#hip_links"]),
        (t1_29dof_termination, ["#root", "Waist", "#hip_links"]),
    ],
)
def test_presets_pass_bodies_not_an_env_attribute(preset: Any, bodies: list[str]) -> None:
    params = preset.terms["contact"].params

    assert params["bodies"] == bodies
    assert "contact_indices_attr" not in params
    # The preset's params must bind to the term's signature (the manager calls func(env, **params)).
    inspect.signature(contact_forces_exceeded).bind(object(), **params)
