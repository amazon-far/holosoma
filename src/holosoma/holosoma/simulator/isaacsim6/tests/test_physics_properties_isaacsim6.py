"""The bulk-physics-property adapter's layout translation, without a simulator.

What these pin, and why it is worth pinning on the CPU lane
-----------------------------------------------------------
``physics_properties`` exists because Newton's ``ArticulationView`` provides none of the seven
PhysX tensor-view members the randomization terms use, and its arrays are shaped differently.
Every one of those shape differences produced a real failure during the port, and none of them
were guessable from the API:

* Newton adds a per-articulation axis, which broke indexing inside the sampler.
* ``body_inertia`` is ``mat33``, so the equivalent flat array is 5-D and exceeds warp's
  4-dimension limit.
* PhysX returns these arrays on the **host** while Newton returns them on the **device**.
* PhysX's own layout is not uniform: it keeps a length-1 body axis for ``get_masses`` and
  ``get_material_properties`` on a rigid object but **drops** it for ``get_inertias``.

A fake view is enough to cover all of that, because the translation is pure shape arithmetic. The
alternative is a GPU harness run per case, which is why this was previously untested.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from holosoma.simulator.isaacsim import physics_properties as pp

pytestmark = pytest.mark.no_sim

_NUM_ENVS = 4
_NUM_BODIES = 3
_NUM_SHAPES = 2


class _PhysxView:
    """A stand-in for the PhysX tensor view, which defines the whole ``get_*`` family."""

    def __init__(self) -> None:
        self.max_shapes = _NUM_SHAPES
        self.written: dict[str, Any] = {}

    def get_masses(self) -> torch.Tensor:
        return torch.zeros(_NUM_ENVS, _NUM_BODIES)

    def get_inertias(self) -> torch.Tensor:
        return torch.zeros(_NUM_ENVS, _NUM_BODIES, 9)

    def get_coms(self) -> torch.Tensor:
        return torch.zeros(_NUM_ENVS, _NUM_BODIES, 7)

    def get_material_properties(self) -> torch.Tensor:
        return torch.zeros(_NUM_ENVS, _NUM_SHAPES, 3)

    def set_masses(self, values: Any, indices: Any) -> None:
        self.written["masses"] = values


class _Articulation:
    """Satisfies ``isinstance(asset, BaseArticulation)`` so the body axis is kept."""

    def __init__(self, view: Any) -> None:
        self.root_view = view
        self.device = "cpu"


def test_physx_view_is_detected_by_its_accessor_family() -> None:
    """Engine dispatch keys off the view, not configuration.

    The randomization terms are Isaac Lab manager terms -- they never see holosoma's simulator
    config -- so threading an engine enum through every one of them was not an option. ``_is_physx_view``
    probes ``get_masses`` as the stand-in for the whole family.
    """
    assert pp._is_physx_view(_PhysxView())
    assert not pp._is_physx_view(object())


def test_physx_reads_pass_through_untouched() -> None:
    """The verified engine must not be reshaped on the way through.

    PhysX is the path the harness suite measured, so the adapter has to be transparent for it --
    any translation here would move numerics on the engine that was already signed off.
    """
    asset = _Articulation(_PhysxView())
    assert pp.get_masses(asset).shape == (_NUM_ENVS, _NUM_BODIES)
    assert pp.get_inertias(asset).shape == (_NUM_ENVS, _NUM_BODIES, 9)
    assert pp.get_coms(asset).shape == (_NUM_ENVS, _NUM_BODIES, 7)
    assert pp.get_material_properties(asset).shape == (_NUM_ENVS, _NUM_SHAPES, 3)
    assert pp.max_shapes(asset) == _NUM_SHAPES


def test_squeeze_articulations_drops_the_newton_only_axis() -> None:
    """Newton reports ``(num_envs, num_articulations, ...)``; holosoma's layout has no such axis."""
    values = torch.zeros(_NUM_ENVS, 1, _NUM_BODIES)
    assert pp._squeeze_articulations(values, "body_mass").shape == (_NUM_ENVS, _NUM_BODIES)


def test_squeeze_articulations_refuses_to_collapse_several_articulations() -> None:
    """Silently taking the first articulation would randomize the wrong bodies.

    Each view addresses one asset, so a second articulation means the view matched more than it
    should have. Raising names that; squeezing anyway would write plausible numbers to the wrong
    rigid bodies, which no assertion downstream would catch.
    """
    values = torch.zeros(_NUM_ENVS, 2, _NUM_BODIES)
    with pytest.raises(RuntimeError, match="articulations per environment"):
        pp._squeeze_articulations(values, "body_mass")


@pytest.mark.parametrize(
    ("is_articulation", "expected"),
    [(True, (_NUM_ENVS, 1, 9)), (False, (_NUM_ENVS, 9))],
    ids=["articulation", "rigid_object"],
)
def test_body_axis_follows_physx_non_uniform_layout(
    monkeypatch: pytest.MonkeyPatch, is_articulation: bool, expected: tuple[int, ...]
) -> None:
    """Reproduces PhysX's non-uniform layout, which is what the terms are written against.

    ``randomize_rigid_body_mass`` does ``inertias[env_ids] *= ratios`` on the rigid-object branch.
    With the extra axis present that broadcasts ``(4, 1, 9)`` against ``(4, 1)`` into ``(4, 4, 9)``
    and raises, which is how this surfaced.

    The articulation test patches ``_is_articulation`` rather than building a ``BaseArticulation``
    subclass. That is not only cheaper -- importing ``BaseArticulation`` reaches
    ``omni.physics.tensors``, which exists only inside a running Kit runtime, so a fixture that
    needed it would make this file uncollectable in the 5.1 environment and abort the whole
    ``no_sim`` run. The branch under test is the shape decision, not the isinstance.
    """
    monkeypatch.setattr(pp, "_is_articulation", lambda _asset: is_articulation)
    values = torch.zeros(_NUM_ENVS, 1, 9)
    assert pp._matches_physx_body_axis(values, _Articulation(_PhysxView())).shape == expected


@pytest.mark.parametrize("is_articulation", [True, False], ids=["articulation", "rigid_object"])
def test_body_axis_restore_is_the_exact_inverse(monkeypatch: pytest.MonkeyPatch, is_articulation: bool) -> None:
    """The write direction has to undo the read direction, or writes land on the wrong shape."""
    monkeypatch.setattr(pp, "_is_articulation", lambda _asset: is_articulation)
    asset = _Articulation(_PhysxView())
    original = torch.zeros(_NUM_ENVS, 1, 9)
    reduced = pp._matches_physx_body_axis(original, asset)
    assert pp._restore_physx_body_axis(reduced, asset).shape == original.shape


def test_torch_device_of_renders_a_torch_acceptable_string() -> None:
    """``warp.array.device`` is a ``warp.Device``; torch needs its string form.

    Reads are normalised to the host to match PhysX, so the write path has to move values back
    onto whichever device Newton keeps the attribute on -- this is how it learns which.
    """

    class _Device:
        def __str__(self) -> str:
            return "cuda:0"

    assert pp.torch_device_of(type("_A", (), {"device": _Device()})()) == "cuda:0"
    assert torch.zeros(1).to("cpu").device.type == "cpu"


def test_split_friction_is_reported_per_engine() -> None:
    """PhysX carries static and dynamic friction; Newton has one ``shape_material_mu``.

    The randomization term asks this so it can refuse a static-only request rather than write a
    column Newton does not have -- which would discard the drawn values in silence.
    """
    assert pp.has_split_friction(_Articulation(_PhysxView()))
    assert not pp.has_split_friction(_Articulation(object()))
