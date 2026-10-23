"""Isaac Sim 6.2-only configuration must not be silently ignored by another backend.

``isaac_physics_backend`` / ``isaac_visualizers`` / ``isaac_newton`` live on the shared
``SimulatorInitConfig``, so every backend accepts them and only ``simulator/isaacsim6`` reads
them. Without a check, `simulator:isaacsim --simulator.config.isaac-physics-backend=newton` runs
Isaac Sim 5.1 on PhysX and reports success -- the field docstrings promise the opposite, and a run
that quietly uses a different physics engine than configured is the worst kind of wrong, because
nothing in the output says so.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest

import holosoma.config_values.simulator as simulator_values
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import IsaacNewtonConfig, IsaacPhysicsBackend
from holosoma.utils.sim_utils import reject_unsupported_isaac_fields

pytestmark = pytest.mark.no_sim


def _config(preset: str, **overrides: Any) -> Any:
    """A minimal stand-in for the slice of ``RunSimConfig`` the check reads."""
    simulator = simulator_values.SIMULATOR_REGISTRY[preset]
    if overrides:
        simulator = dataclasses.replace(simulator, config=dataclasses.replace(simulator.config, **overrides))
    return dataclasses.replace(simulator_values.SIMULATOR_REGISTRY[preset], config=simulator.config)


class _Holder:
    """Just ``.simulator``, which is all :func:`reject_unsupported_isaac_fields` touches."""

    def __init__(self, simulator: Any) -> None:
        self.simulator = simulator


def _holder(preset: str, **overrides: Any) -> RunSimConfig:
    """A minimal stand-in, cast once at this boundary.

    The function under test is annotated for the real config union and reads only
    ``.simulator`` (defensively, via ``getattr``), so a stand-in is the honest fixture: building a
    real ``RunSimConfig`` would drag in a tyro parse per case to exercise one attribute. The cast
    is confined here rather than repeated at every call site.
    """
    return cast("RunSimConfig", _Holder(_config(preset, **overrides)))


@pytest.mark.parametrize("preset", ["isaacsim", "mujoco", "mjwarp"])
def test_defaults_are_accepted_for_every_backend(preset: str) -> None:
    """The check must not fire on configs that set nothing -- that is every existing config."""
    reject_unsupported_isaac_fields(_holder(preset))


@pytest.mark.parametrize("preset", ["isaacsim6", "isaacsim6-newton", "isaacsim6-ovphysx"])
def test_the_6_1_presets_are_always_accepted(preset: str) -> None:
    """The 6.1 lane is the backend that reads these fields, including its engine variants."""
    reject_unsupported_isaac_fields(_holder(preset))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("isaac_physics_backend", IsaacPhysicsBackend.NEWTON),
        ("isaac_visualizers", ("kit",)),
        ("isaac_newton", IsaacNewtonConfig(num_substeps=4)),
    ],
)
def test_each_6_1_field_is_rejected_on_the_5_1_backend(field: str, value: Any) -> None:
    """Isaac Sim 5.1 has one engine and different launcher flags, so none of these can apply."""
    with pytest.raises(ValueError, match="Isaac Sim 6.2 settings"):
        reject_unsupported_isaac_fields(_holder("isaacsim", **{field: value}))


def test_rejection_is_not_isaac_specific() -> None:
    """A non-Isaac backend cannot honour them either, so it must not accept them quietly."""
    with pytest.raises(ValueError, match="Isaac Sim 6.2 settings"):
        reject_unsupported_isaac_fields(_holder("mujoco", isaac_physics_backend=IsaacPhysicsBackend.NEWTON))


def test_the_message_names_the_field_and_the_alternative() -> None:
    """ "Unsupported" alone leaves the reader stuck; the fix is a different preset."""
    with pytest.raises(ValueError, match="Isaac Sim 6.2 settings") as excinfo:
        reject_unsupported_isaac_fields(_holder("isaacsim", isaac_visualizers=("rerun",)))
    message = str(excinfo.value)
    assert "isaac_visualizers" in message
    assert "simulator:isaacsim6" in message
