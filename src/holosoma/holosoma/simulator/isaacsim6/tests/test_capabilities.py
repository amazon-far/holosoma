"""Pure configuration validation and consistency checks for the GA capability report."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import holosoma.config_values.simulator as simulator_values
from holosoma.config_types.simulator import IsaacNewtonConfig, IsaacPhysicsBackend, SimulatorInitConfig
from holosoma.simulator.isaacsim6.capabilities import (
    BACKEND_STATUS,
    CAPABILITIES,
    HEADLESS,
    KNOWN_VISUALIZERS,
    RENDERER_AGNOSTIC,
    Verification,
    validate,
)

pytestmark = pytest.mark.no_sim


def _config(**overrides: Any) -> SimulatorInitConfig:
    """The shipped ``isaacsim6`` preset's init config, with overrides applied."""
    base: SimulatorInitConfig = simulator_values.SIMULATOR_REGISTRY["isaacsim6"].config
    return dataclasses.replace(base, **overrides)


def test_the_shipped_preset_validates() -> None:
    """The default 6.2 preset must pass its own gate, or nobody can run the backend."""
    validate(_config(), {})


def test_unknown_visualizer_is_rejected() -> None:
    """An unrecognised name is ignored by Isaac Lab's launcher, so the run would come up headless.

    That is the failure worth catching: a user asking for a window or camera frames gets neither,
    with no error anywhere.
    """
    with pytest.raises(ValueError, match="Unknown isaac_visualizers"):
        validate(_config(isaac_visualizers=("opengl",)), {})


@pytest.mark.parametrize("name", KNOWN_VISUALIZERS)
def test_each_known_visualizer_is_accepted(name: str) -> None:
    validate(_config(isaac_visualizers=(name,)), {})


def test_newton_tuning_on_a_physx_run_is_rejected() -> None:
    """Engine-specific tuning aimed at the wrong engine would be silently dropped.

    PhysX and Newton share no tuning fields, so there is no partial application to fall back on --
    the values would simply have no effect while the config claims otherwise.
    """
    with pytest.raises(ValueError, match="isaac_newton tuning"):
        validate(
            _config(
                isaac_physics_backend=IsaacPhysicsBackend.PHYSX,
                isaac_newton=IsaacNewtonConfig(num_substeps=4),
            ),
            {},
        )


@pytest.mark.parametrize("backend", [b.backend for b in BACKEND_STATUS if not b.usable], ids=lambda b: b.value)
def test_backends_measured_as_unusable_are_rejected(backend: IsaacPhysicsBackend) -> None:
    """Selecting a backend holosoma cannot run on must fail here, not deep in the engine.

    OvPhysX is the remaining case: it is Isaac Lab's Kit-free PhysX runtime, and forcing it while
    Kit is running **segfaults** inside the ovphysx wheel with no Python exception at all. So
    there is not even an error to read, which is why the gate has to speak first.
    """
    with pytest.raises(NotImplementedError, match=f"{backend.value!r} physics backend"):
        validate(_config(isaac_physics_backend=backend), {})


@pytest.mark.parametrize("backend", [b.backend for b in BACKEND_STATUS if b.usable], ids=lambda b: b.value)
def test_supported_backends_are_accepted(backend: IsaacPhysicsBackend) -> None:
    """A backend the harness suite passes on must not be rejected.

    The counterpart of the test above, and the reason it exists: Newton *was* rejected here while
    the real blockers were two holosoma-side gaps (model population and bulk-property access).
    Once those were fixed the gate still said no, so this pins the two together -- a backend
    cannot be measured as working and refused at startup.
    """
    validate(_config(isaac_physics_backend=backend), {})


def test_the_rejection_explains_the_measured_cause() -> None:
    """The message must carry the detail, so a user is not left guessing what to do."""
    with pytest.raises(NotImplementedError, match="Kit-free"):
        validate(_config(isaac_physics_backend=IsaacPhysicsBackend.OVPHYSX), {})


def test_every_unusable_backend_has_a_detail() -> None:
    """A backend marked unusable with no detail renders as an unexplained "no" in the docs."""
    for status in BACKEND_STATUS:
        if not status.usable:
            assert status.detail.strip(), f"{status.backend.value} is unusable but has no detail"
        assert status.summary.strip(), f"{status.backend.value} has no summary"


def test_every_capability_entry_has_a_note() -> None:
    """The matrix doubles as the source for ``docs/isaac-sim-backends.md``.

    An entry with no note renders as an unexplained gap, which is how documentation starts lying
    about why something is unsupported.
    """
    for capability in CAPABILITIES:
        assert capability.note.strip(), f"{capability.name} has no note"


def test_unsupported_capabilities_name_the_alternative() -> None:
    """A capability with no supported backend must tell the user where to go instead.

    "Not supported" alone leaves someone stuck; these are exactly the cases where the 5.1 backend
    still works, so the message has to say so.
    """
    for capability in CAPABILITIES:
        if not capability.ported:
            assert "isaacsim" in capability.note, (
                f"{capability.name} is unsupported but does not point at an alternative backend"
            )


def test_renderer_verification_names_only_real_visualizers() -> None:
    """The rendering axis must not claim verification under a name Isaac Lab does not accept.

    ``Renderers verified`` is read as "measured under these selections", so an entry that is not
    a real visualizer name would advertise a measurement that could not have happened.
    """
    allowed = {*KNOWN_VISUALIZERS, HEADLESS}
    for capability in CAPABILITIES:
        unknown = set(capability.renderers) - allowed
        assert not unknown, f"{capability.name} claims unknown renderers {sorted(unknown)}"


def test_unported_capabilities_claim_no_renderer_verification() -> None:
    """An unported capability cannot have been verified under any renderer."""
    for capability in CAPABILITIES:
        if not capability.ported:
            assert capability.renderers == RENDERER_AGNOSTIC, (
                f"{capability.name} is unported but lists verified renderers"
            )


def test_unported_capabilities_are_unported_on_every_backend() -> None:
    """An unported capability must not read as merely ``blocked`` anywhere.

    ``blocked`` means "the backend failed before this code ran, so nothing is known". For a
    capability holosoma has not implemented, something *is* known -- it is absent regardless of
    backend -- so reporting it as blocked would hide real porting work behind a backend excuse.
    """
    for capability in CAPABILITIES:
        if capability.ported:
            continue
        for status in BACKEND_STATUS:
            assert capability.on(status.backend) is Verification.UNPORTED, (
                f"{capability.name} is unported but reports "
                f"{capability.on(status.backend).value!r} on {status.backend.value}"
            )


def test_unreachable_backends_report_blocked_not_failed() -> None:
    """A backend that cannot construct must block, never claim a per-capability result.

    This is the distinction the matrix exists to preserve. Newton and OvPhysX each fail once, at
    simulator construction, for a single shared reason. Rendering that as a per-capability "no"
    would present one backend failure as a dozen independent capability findings -- which is
    precisely the overstatement an earlier version of this table made.
    """
    for status in BACKEND_STATUS:
        if status.reachable:
            continue
        for capability in CAPABILITIES:
            if not capability.ported:
                continue
            assert capability.on(status.backend) is Verification.BLOCKED


def test_unreachable_backends_are_also_unusable() -> None:
    """``validate()`` must reject any backend that cannot construct.

    Otherwise the run proceeds to the backend's own failure, which for OvPhysX is a segfault with
    no Python exception -- nothing names the backend as unsupported.
    """
    for status in BACKEND_STATUS:
        if not status.reachable:
            assert not status.usable, f"{status.backend.value} cannot construct but is not rejected"


def test_unusable_backends_explain_the_measurement() -> None:
    """A rejected backend's detail must carry the measured cause, not a bare refusal.

    The message is what a user gets instead of a working run, so it has to say what was observed
    and whether anything can be done about it.
    """
    for status in BACKEND_STATUS:
        if not status.usable:
            assert len(status.detail.strip()) > 100, f"{status.backend.value} detail is too thin to act on"


def test_configured_lidar_is_accepted() -> None:
    from tests.simulators import _lidar_presets

    validate(_config(), {"scan": _lidar_presets.isaacsim_ground_visual_lidar})


def test_actor_mounted_camera_is_accepted() -> None:
    """Every existing camera mount is available through the shared sensor builder."""
    from tests.simulators import _camera_presets

    actor_cams = {
        name: cfg
        for name, cfg in _camera_presets.SENSOR_PRESETS["actor-cam"].items()
        if getattr(getattr(cfg, "mount", None), "target_kind", None) == "actor"
    }
    assert actor_cams, "the actor-cam preset no longer contains an actor-mounted camera"
    validate(_config(), actor_cams)


def test_robot_mounted_camera_is_accepted() -> None:
    """The supported mount kinds must not be caught by the actor-mount rejection."""
    from tests.simulators import _camera_presets

    validate(_config(), _camera_presets.SENSOR_PRESETS["front-cam"])
