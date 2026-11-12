from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.config_types.video import VideoConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import (
    HookCloseError,
    HookOrderError,
    HookRef,
    HookRegistry,
    HookRegistryError,
    Phase,
    if_present,
)

pytestmark = pytest.mark.no_sim


def test_emit_uses_registration_order() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.FRAME_END, lambda: calls.append("a"), name="a")
    hooks.add(Phase.FRAME_END, lambda: calls.append("b"), name="b")

    hooks.emit(Phase.FRAME_END)

    assert calls == ["a", "b"]


def test_an_unnamed_hook_is_labelled_by_its_qualname() -> None:
    hooks = HookRegistry()

    class Producer:
        def tick(self) -> None: ...

    handle = hooks.add(Phase.FRAME_END, Producer().tick)

    assert handle.name == "test_an_unnamed_hook_is_labelled_by_its_qualname.<locals>.Producer.tick"


def test_handle_disable_enable_remove() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    handle = hooks.add(Phase.FRAME_END, lambda: calls.append("a"), name="a")

    handle.disable()
    hooks.emit(Phase.FRAME_END)
    handle.enable()
    hooks.emit(Phase.FRAME_END)
    handle.remove()
    hooks.emit(Phase.FRAME_END)
    handle.remove()

    assert calls == ["a"]


def test_mutating_current_phase_during_emit_raises() -> None:
    hooks = HookRegistry()
    handle = hooks.add(Phase.FRAME_END, lambda: None, name="a")
    hooks.add(Phase.FRAME_END, handle.remove, name="remove")

    with pytest.raises(HookRegistryError, match="Cannot mutate hooks"):
        hooks.emit(Phase.FRAME_END)


def test_close_runs_reverse_order_once() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.CLOSE, lambda: calls.append("first"), name="first")
    hooks.add(Phase.CLOSE, lambda: calls.append("second"), name="second")

    hooks.emit(Phase.CLOSE)
    hooks.emit(Phase.CLOSE)

    assert calls == ["second", "first"]


def test_close_failure_keeps_earlier_provider_open() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    hooks.add(Phase.CLOSE, lambda: calls.append("provider"), name="provider")

    def fail_consumer() -> None:
        calls.append("consumer")
        raise RuntimeError("boom")

    hooks.add(Phase.CLOSE, fail_consumer, name="consumer")
    with pytest.raises(HookCloseError) as exc_info:
        hooks.emit(Phase.CLOSE)

    hooks.emit(Phase.CLOSE)

    assert calls == ["consumer"]
    assert len(exc_info.value.failures) == 1
    name, error = exc_info.value.failures[0]
    assert name == "consumer"
    assert isinstance(error, RuntimeError)
    assert str(error) == "boom"


def test_constructor_defers_plugins_until_explicit_installation() -> None:
    created: list[object] = []

    class _Plugin:
        def __init__(self, cfg: object, sim: BaseSimulator) -> None:
            assert cfg is config
            assert sim is simulator
            created.append(self)

    class _Config:
        @staticmethod
        def get_cls() -> type[_Plugin]:
            return _Plugin

    config = _Config()
    simulator_config = SimpleNamespace(
        debug_viz=False,
        sim=SimpleNamespace(kinematic_playback=False, fps=200.0, control_decimation_steps=4),
    )
    tyro_config = SimpleNamespace(
        training=SimpleNamespace(num_envs=1),
        simulator=simulator_config,
        scene=SimpleNamespace(),
        sensors={},
        robot=SimpleNamespace(),
        logger=SimpleNamespace(video=VideoConfig(enabled=False), headless_recording=False),
        plugin={"probe": config},
        experiment_dir=None,
    )

    simulator = BaseSimulator(tyro_config, terrain_manager=SimpleNamespace(), device="cpu")  # type: ignore[arg-type]
    assert created == []
    assert simulator.installed_plugins == {}

    simulator.install_plugins()
    simulator.install_plugins()

    assert simulator.installed_plugins == {"probe": created[0]}
    assert len(created) == 1


def test_close_phase_rejects_payload() -> None:
    hooks = HookRegistry()

    with pytest.raises(HookRegistryError, match="expects no arguments"):
        hooks.emit(Phase.CLOSE, object())


def test_phase_payloads_have_fixed_signatures() -> None:
    hooks = HookRegistry()
    calls: list[int] = []

    def record_episode_start(env_id: int) -> None:
        calls.append(env_id)

    hooks.add(Phase.EPISODE_START, record_episode_start, name="episode_start")

    hooks.emit(Phase.EPISODE_START, 7)

    with pytest.raises(HookRegistryError, match="expects env_id"):
        hooks.emit(Phase.EPISODE_START)
    with pytest.raises(HookRegistryError, match="expects no arguments"):
        hooks.emit(Phase.FRAME_END, 1)
    assert calls == [7]


def test_every_int_decimates_emissions() -> None:
    calls: list[int] = []
    hooks = HookRegistry()
    hooks.add(Phase.POST_STEP, lambda: calls.append(1), name="third", every=3)

    for _ in range(7):
        hooks.emit(Phase.POST_STEP)

    # fires on the 3rd, 6th emission (counter resets after each fire)
    assert len(calls) == 2


def test_every_default_fires_every_emission() -> None:
    calls: list[int] = []
    hooks = HookRegistry()
    hooks.add(Phase.POST_STEP, lambda: calls.append(1), name="always")

    for _ in range(5):
        hooks.emit(Phase.POST_STEP)

    assert len(calls) == 5


def test_every_frequency_string_resolves_against_base_rate() -> None:
    calls: list[int] = []
    # base 200Hz physics tick; "50Hz" -> decimation 4.
    hooks = HookRegistry(base_rates={Phase.POST_STEP: 200.0})
    hooks.add(Phase.POST_STEP, lambda: calls.append(1), name="rate", every="50Hz")

    for _ in range(8):
        hooks.emit(Phase.POST_STEP)

    assert len(calls) == 2  # fired on the 4th and 8th


def test_every_frequency_string_needs_periodic_phase() -> None:
    hooks = HookRegistry()
    with pytest.raises(HookRegistryError, match="not periodic"):
        hooks.add(Phase.EPISODE_START, lambda _env_id: None, name="bad", every="10Hz")


def test_every_frequency_string_needs_base_rate() -> None:
    hooks = HookRegistry()  # no base rates registered
    with pytest.raises(HookRegistryError, match="no base rate"):
        hooks.add(Phase.POST_STEP, lambda: None, name="bad", every="50Hz")


def test_every_int_on_event_phase_is_allowed() -> None:
    calls: list[int] = []
    hooks = HookRegistry()
    hooks.add(Phase.EPISODE_START, lambda env_id: calls.append(env_id), name="every_other", every=2)

    for i in range(4):
        hooks.emit(Phase.EPISODE_START, i)

    assert calls == [1, 3]


def test_close_rejects_decimation() -> None:
    hooks = HookRegistry()
    with pytest.raises(HookRegistryError, match="CLOSE fires once"):
        hooks.add(Phase.CLOSE, lambda: None, name="bad", every=2)


def test_invalid_every_raises() -> None:
    hooks = HookRegistry()
    with pytest.raises(ValueError, match="must be >= 1"):
        hooks.add(Phase.POST_STEP, lambda: None, name="bad", every=0)


def test_set_every_live_changes_cadence_and_resets_counter() -> None:
    calls: list[int] = []
    hooks = HookRegistry()
    handle = hooks.add(Phase.POST_STEP, lambda: calls.append(1), name="rate")

    hooks.emit(Phase.POST_STEP)  # every=1 -> fires
    assert len(calls) == 1

    handle.set_every(3)
    assert handle.every == 3
    for _ in range(5):
        hooks.emit(Phase.POST_STEP)
    # counter reset by set_every; fires on the 3rd of the 5 emissions
    assert len(calls) == 2


def test_set_every_accepts_frequency_string() -> None:
    calls: list[int] = []
    hooks = HookRegistry(base_rates={Phase.POST_STEP: 100.0})
    handle = hooks.add(Phase.POST_STEP, lambda: calls.append(1), name="rate")

    handle.set_every("25Hz")  # 100 / 25 -> 4
    assert handle.every == 4
    for _ in range(8):
        hooks.emit(Phase.POST_STEP)
    assert len(calls) == 2


def test_set_every_during_own_phase_emit_raises() -> None:
    hooks = HookRegistry()
    handle = hooks.add(Phase.POST_STEP, lambda: None, name="a")
    hooks.add(Phase.POST_STEP, lambda: handle.set_every(2), name="mutator")

    with pytest.raises(HookRegistryError, match="Cannot mutate hooks"):
        hooks.emit(Phase.POST_STEP)


def test_set_every_on_removed_hook_raises() -> None:
    hooks = HookRegistry()
    handle = hooks.add(Phase.POST_STEP, lambda: None, name="a")
    handle.remove()
    with pytest.raises(HookRegistryError, match="removed hook"):
        handle.set_every(2)


# ----- callback-arity guard (add() rejects a signature that can't take the phase payload) -----


def test_add_rejects_required_arg_on_zero_payload_phase() -> None:
    hooks = HookRegistry()

    def needs_env(env_id: int) -> None: ...

    # POST_STEP emits no args, so a required positional param can never be filled.
    # The ``type: ignore[call-overload]`` is load-bearing: mypy also rejects this mismatch via
    # add()'s typed overloads, so if the overloads regress this ignore goes unused and CI notices.
    with pytest.raises(HookRegistryError, match="requires 1 positional arg"):
        hooks.add(Phase.POST_STEP, needs_env, name="bad")  # type: ignore[call-overload]


def test_add_rejects_zero_arg_callback_on_episode_phase() -> None:
    hooks = HookRegistry()

    def takes_nothing() -> None: ...

    # EPISODE_START emits env_id, but the callback accepts no positional args.
    # ``type: ignore[call-overload]`` load-bearing (see test above): the overloads reject this too.
    with pytest.raises(HookRegistryError, match="accepts at most 0"):
        hooks.add(Phase.EPISODE_START, takes_nothing, name="bad")  # type: ignore[call-overload]


def test_add_accepts_defaulted_arg_on_zero_payload_phase() -> None:
    # The video recorder's ``capture_frame(env_id: int = 0)`` shape: the default covers the
    # zero-arg FRAME/STEP emit, so it is a valid registration (not a mismatch).
    hooks = HookRegistry()
    calls: list[int] = []

    def capture_frame(env_id: int = 0) -> None:
        calls.append(env_id)

    hooks.add(Phase.POST_STEP, capture_frame, name="video.capture_frame")
    hooks.emit(Phase.POST_STEP)
    assert calls == [0]  # default supplied, no crash


def test_add_accepts_varargs_on_any_phase() -> None:
    hooks = HookRegistry()

    def anything(*args: object) -> None: ...

    hooks.add(Phase.POST_STEP, anything, name="a")
    hooks.add(Phase.EPISODE_END, anything, name="b")  # *args absorbs env_id too


def test_add_arity_guard_ignores_self_on_bound_method() -> None:
    # inspect.signature on a bound method excludes ``self``, so a one-arg method matches an
    # episode phase and a defaulted-arg method matches a zero-payload phase.
    hooks = HookRegistry()

    class Participant:
        def on_episode_start(self, env_id: int) -> None: ...
        def capture_frame(self, env_id: int = 0) -> None: ...

    p = Participant()
    hooks.add(Phase.EPISODE_START, p.on_episode_start, name="ep")
    hooks.add(Phase.FRAME_END, p.capture_frame, name="frame")


def test_add_arity_guard_tolerates_uninspectable_callback() -> None:
    # Some C-level callables reject inspect.signature; the guard must not block them.
    hooks = HookRegistry()
    hooks.add(Phase.EPISODE_END, print, name="builtin")  # print(*args) — accepts env_id


def recorder(calls: list[str], name: str) -> Callable[[], None]:
    return lambda: calls.append(name)


def test_after_runs_hook_behind_its_dependency() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    render = hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="x.render")
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="x.consumer", after=[render])

    hooks.emit(Phase.FRAME_END)

    assert calls == ["render", "consumer"]


def test_a_single_dependency_needs_no_list() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    render = hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="x.render")
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="x.consumer", after=render)

    hooks.emit(Phase.FRAME_END)

    assert calls == ["render", "consumer"]


def test_before_runs_hook_ahead_of_its_dependency() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    late = hooks.add(Phase.FRAME_END, recorder(calls, "late"), name="x.late")
    hooks.add(Phase.FRAME_END, recorder(calls, "early"), name="x.early", before=[late])

    hooks.emit(Phase.FRAME_END)

    assert calls == ["early", "late"]


def test_unconstrained_hooks_keep_their_relative_order() -> None:
    # 'other' and 'gate' stay in that order; the blocked hook loses its slot rather than holding it.
    calls: list[str] = []
    hooks = HookRegistry()

    blocked = hooks.add(Phase.FRAME_END, recorder(calls, "blocked"), name="x.blocked")
    hooks.add(Phase.FRAME_END, recorder(calls, "other"), name="x.other")
    hooks.add(Phase.FRAME_END, recorder(calls, "gate"), name="x.gate", before=[blocked])

    hooks.emit(Phase.FRAME_END)

    assert calls == ["other", "gate", "blocked"]


def test_order_is_recomputed_after_a_later_registration() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    consumer = hooks.add(Phase.FRAME_END, recorder(calls, "x.consumer"), name="x.consumer")

    hooks.emit(Phase.FRAME_END)
    hooks.add(Phase.FRAME_END, recorder(calls, "x.late"), name="x.late", before=[consumer])
    hooks.emit(Phase.FRAME_END)

    assert calls == ["x.consumer", "x.late", "x.consumer"]


def test_cycle_raises_on_emit() -> None:
    hooks = HookRegistry()
    first = hooks.add(Phase.FRAME_END, lambda: None, name="x.a")
    hooks.add(Phase.FRAME_END, lambda: None, name="x.b", after=[first], before=[first])

    with pytest.raises(HookOrderError, match="cycle on phase 'frame_end'"):
        hooks.emit(Phase.FRAME_END)


def test_disabling_a_link_drops_the_order_it_carried() -> None:
    # Only b's edges put a ahead of c, so disabling b takes that order with it.
    calls: list[str] = []
    hooks = HookRegistry()
    c = hooks.add(Phase.FRAME_END, recorder(calls, "c"), name="x.c")
    a = hooks.add(Phase.FRAME_END, recorder(calls, "a"), name="x.a")
    b = hooks.add(Phase.FRAME_END, recorder(calls, "b"), name="x.b", after=[a], before=[c])

    hooks.emit(Phase.FRAME_END)
    calls.clear()
    b.disable()
    hooks.emit(Phase.FRAME_END)

    assert calls == ["c", "a"]


def test_a_cycle_broken_by_a_disabled_hook_still_resolves() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    a = hooks.add(Phase.FRAME_END, recorder(calls, "a"), name="x.a")
    b = hooks.add(Phase.FRAME_END, recorder(calls, "b"), name="x.b", after=[a])
    c = hooks.add(Phase.FRAME_END, recorder(calls, "c"), name="x.c", after=[b])
    hooks.add(Phase.FRAME_END, recorder(calls, "d"), name="x.d", after=[c], before=[a])

    b.disable()
    hooks.emit(Phase.FRAME_END)

    assert calls == ["c", "d", "a"]


def test_disabled_dependency_is_inert_and_returns_on_enable() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    render = hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="x.render")
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="x.consumer", after=[render])

    render.disable()
    hooks.emit(Phase.FRAME_END)
    render.enable()
    hooks.emit(Phase.FRAME_END)

    assert calls == ["consumer", "render", "consumer"]


def test_removing_a_required_dependency_fails_the_phase() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    render = hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="x.render")
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="x.consumer", after=[render])

    render.remove()

    with pytest.raises(HookOrderError, match="requires 'x.render'"):
        hooks.emit(Phase.FRAME_END)


def test_close_declares_teardown_order_directly() -> None:
    # A node's cleanup says it runs before the context it publishes through, and that is the order.
    calls: list[str] = []
    hooks = HookRegistry()

    runtime = hooks.add(Phase.CLOSE, recorder(calls, "runtime"), name="x.runtime")
    hooks.add(Phase.CLOSE, recorder(calls, "node"), name="x.node", before=[runtime])

    hooks.emit(Phase.CLOSE)

    assert calls == ["node", "runtime"]


def test_close_cycle_raises_like_any_other_phase() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    first = hooks.add(Phase.CLOSE, recorder(calls, "a"), name="x.a")
    hooks.add(Phase.CLOSE, recorder(calls, "b"), name="x.b", after=[first], before=[first])

    with pytest.raises(HookOrderError, match="cycle on phase 'close'"):
        hooks.emit(Phase.CLOSE)

    assert calls == []


def test_handle_orders_against_one_instance_not_its_namesake() -> None:
    # A.foo(seed=1) -> B.foo -> A.foo(seed=2): same callback twice, ordered by handle.
    calls: list[str] = []
    hooks = HookRegistry()

    def seeded(seed: int) -> Callable[[], None]:
        return lambda: calls.append(f"a{seed}")

    first = hooks.add(Phase.FRAME_END, seeded(1), name="a.foo")
    second = hooks.add(Phase.FRAME_END, seeded(2), name="a.foo")
    hooks.add(Phase.FRAME_END, recorder(calls, "b"), name="b.foo", after=[first], before=[second])

    hooks.emit(Phase.FRAME_END)

    assert calls == ["a1", "b", "a2"]


@pytest.mark.parametrize(
    ("dependency", "message"),
    [
        ("other_phase", "Constraints apply within one phase"),
        ("other_registry", "from a different registry"),
        ("name", "names are labels, not dependency targets"),
        ("object", "pass the HookHandle add\\(\\) returned"),
    ],
)
def test_a_dependency_that_cannot_be_resolved_is_rejected(dependency: str, message: str) -> None:
    hooks = HookRegistry()
    target = hooks.add(Phase.FRAME_END, lambda: None, name="x.target")
    bad: Any = {
        "other_phase": hooks.add(Phase.FRAME_BEGIN, lambda: None, name="x.elsewhere"),
        "other_registry": HookRegistry().add(Phase.FRAME_END, lambda: None, name="x.elsewhere"),
        "name": target.name,
        "object": object(),
    }[dependency]

    with pytest.raises(HookRegistryError, match=message):
        hooks.add(Phase.FRAME_END, lambda: None, name="x.dependent", after=[bad])


def test_handle_to_a_removed_hook_fails_unless_optional() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    render = hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="render.one")
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="consumer.one", after=[render])
    optional = hooks.add(Phase.FRAME_END, recorder(calls, "optional"), name="optional.one")
    hooks.add(Phase.FRAME_END, recorder(calls, "lenient"), name="lenient.one", after=[if_present(optional)])

    render.remove()
    optional.remove()

    with pytest.raises(HookOrderError, match="requires 'render.one'"):
        hooks.emit(Phase.FRAME_END)


def test_names_reports_resolved_order_per_phase() -> None:
    hooks = HookRegistry()
    render = hooks.add(Phase.FRAME_END, lambda: None, name="x.render")
    hooks.add(Phase.FRAME_END, lambda: None, name="x.consumer", after=[render])
    runtime = hooks.add(Phase.CLOSE, lambda: None, name="x.runtime")
    hooks.add(Phase.CLOSE, lambda: None, name="x.node", before=[runtime])

    assert hooks.names(Phase.FRAME_END) == ("x.render", "x.consumer")
    assert hooks.names(Phase.CLOSE) == ("x.node", "x.runtime")
    assert hooks.names(Phase.PRE_STEP) == ()


def test_validate_reports_a_bad_graph_before_anything_emits() -> None:
    hooks = HookRegistry()
    with hooks.installing("inspect"):
        hooks.add(
            Phase.EPISODE_END,
            lambda _env_id: None,
            name="x.late",
            after=[HookRef("absent", _CameraHook.PUBLISH)],
        )

    with pytest.raises(HookOrderError, match="published no such hook"):
        hooks.validate()


class _CameraHook(Enum):
    PUBLISH = auto()
    CLOSE = auto()


class _OtherHook(Enum):
    PUBLISH = auto()


def test_hookref_resolves_a_publication_registered_after_the_reference() -> None:
    # The consumer registers first: a reference does not require its provider to exist yet.
    calls: list[str] = []
    hooks = HookRegistry()

    with hooks.installing("inspect"):
        hooks.add(
            Phase.FRAME_END,
            recorder(calls, "consumer"),
            name="inspector.read",
            after=[HookRef("head", _CameraHook.PUBLISH)],
        )
    with hooks.installing("head"):
        hooks.add(Phase.FRAME_END, recorder(calls, "head"), name="camera.publish", key=_CameraHook.PUBLISH)

    hooks.emit(Phase.FRAME_END)

    assert calls == ["head", "consumer"]


def test_hookref_picks_the_named_instance_of_a_plugin_installed_twice() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    for slot in ("head", "wrist"):
        with hooks.installing(slot):
            hooks.add(Phase.FRAME_END, recorder(calls, slot), name="camera.publish", key=_CameraHook.PUBLISH)
    with hooks.installing("inspect"):
        hooks.add(
            Phase.FRAME_END,
            recorder(calls, "consumer"),
            name="inspector.read",
            before=[HookRef("wrist", _CameraHook.PUBLISH)],
        )

    hooks.emit(Phase.FRAME_END)

    assert calls == ["head", "consumer", "wrist"]


def test_same_member_of_a_different_enum_is_a_different_key() -> None:
    hooks = HookRegistry()
    with hooks.installing("head"):
        hooks.add(Phase.FRAME_END, lambda: None, name="camera.publish", key=_CameraHook.PUBLISH)
        hooks.add(Phase.FRAME_END, lambda: None, name="other.publish", key=_OtherHook.PUBLISH)
    with hooks.installing("inspect"):
        hooks.add(Phase.FRAME_END, lambda: None, name="inspector.read", after=[HookRef("head", _OtherHook.PUBLISH)])

    assert hooks.names(Phase.FRAME_END)[-1] == "inspector.read"


def test_a_mixin_enum_does_not_collide_with_another_contract() -> None:
    # A str-mixin enum compares by value, so these two members are equal outside the registry.
    class CameraHook(str, Enum):
        PUBLISH = "publish"

    class LidarHook(str, Enum):
        PUBLISH = "publish"

    hooks = HookRegistry()
    with hooks.installing("head"):
        hooks.add(Phase.FRAME_END, lambda: None, name="camera.publish", key=CameraHook.PUBLISH)
    with hooks.installing("inspect"):
        hooks.add(
            Phase.FRAME_END,
            lambda: None,
            name="inspector.read",
            after=[HookRef("head", LidarHook.PUBLISH)],
        )

    with pytest.raises(HookOrderError, match="published no such hook"):
        hooks.validate()


def test_one_instance_cannot_publish_a_key_twice() -> None:
    hooks = HookRegistry()

    with hooks.installing("head"):
        hooks.add(Phase.FRAME_END, lambda: None, name="camera.publish", key=_CameraHook.PUBLISH)

        with pytest.raises(HookRegistryError, match="already published"):
            hooks.add(Phase.CLOSE, lambda: None, name="camera.stop", key=_CameraHook.PUBLISH)


def test_publishing_a_key_outside_an_install_scope_is_rejected() -> None:
    hooks = HookRegistry()

    with pytest.raises(HookRegistryError, match="no owning plugin instance"):
        hooks.add(Phase.FRAME_END, lambda: None, name="camera.publish", key=_CameraHook.PUBLISH)


def test_unresolved_hookref_fails_unless_optional() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    with hooks.installing("inspect"):
        hooks.add(
            Phase.FRAME_END,
            recorder(calls, "consumer"),
            name="inspector.read",
            after=[HookRef("head", _CameraHook.PUBLISH)],
        )

    with pytest.raises(HookOrderError, match="published no such hook"):
        hooks.validate()

    hooks = HookRegistry()
    with hooks.installing("inspect"):
        hooks.add(
            Phase.FRAME_END,
            recorder(calls, "lenient"),
            name="inspector.read",
            after=[if_present(HookRef("head", _CameraHook.PUBLISH))],
        )
    hooks.validate()
    hooks.emit(Phase.FRAME_END)

    assert calls == ["lenient"]


def test_a_required_close_reference_must_resolve_when_declared() -> None:
    # A run that fails to finish starting still has to tear down, so CLOSE cannot carry an edge
    # whose provider may never appear.
    hooks = HookRegistry()

    with hooks.installing("inspect"):
        with pytest.raises(HookRegistryError, match="must resolve when declared"):
            hooks.add(
                Phase.CLOSE,
                lambda: None,
                name="inspector.stop",
                before=HookRef("head", _CameraHook.CLOSE),
            )

        # Optional is still allowed: it declares that absence is acceptable.
        hooks.add(
            Phase.CLOSE,
            lambda: None,
            name="inspector.stop",
            before=if_present(HookRef("head", _CameraHook.CLOSE)),
        )

    hooks.validate()


def test_a_close_reference_resolves_once_its_provider_has_published() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    with hooks.installing("head"):
        hooks.add(Phase.CLOSE, recorder(calls, "camera"), name="camera.stop", key=_CameraHook.CLOSE)
    with hooks.installing("inspect"):
        hooks.add(
            Phase.CLOSE,
            recorder(calls, "inspector"),
            name="inspector.stop",
            before=HookRef("head", _CameraHook.CLOSE),
        )

    hooks.emit(Phase.CLOSE)

    assert calls == ["inspector", "camera"]


def test_hookref_to_another_phase_is_rejected() -> None:
    hooks = HookRegistry()
    with hooks.installing("head"):
        hooks.add(Phase.CLOSE, lambda: None, name="camera.stop", key=_CameraHook.CLOSE)
    with hooks.installing("inspect"):
        hooks.add(Phase.FRAME_END, lambda: None, name="inspector.read", after=[HookRef("head", _CameraHook.CLOSE)])

    with pytest.raises(HookOrderError, match="published on phase 'close'"):
        hooks.emit(Phase.FRAME_END)


def test_publishing_a_key_re_resolves_phases_that_reference_it() -> None:
    # The reference resolves while the key is absent, so the phase it is on has to be re-resolved
    # when the key appears -- on any phase, since whether it matches is only knowable then.
    hooks = HookRegistry()
    with hooks.installing("inspect"):
        hooks.add(
            Phase.FRAME_END,
            lambda: None,
            name="inspector.read",
            after=if_present(HookRef("head", _CameraHook.CLOSE)),
        )
    hooks.validate()

    with hooks.installing("head"):
        hooks.add(Phase.CLOSE, lambda: None, name="camera.stop", key=_CameraHook.CLOSE)

    with pytest.raises(HookOrderError, match="published on phase 'close'"):
        hooks.validate()


def test_a_hook_cannot_order_against_itself() -> None:
    hooks = HookRegistry()
    with hooks.installing("head"):
        hooks.add(
            Phase.FRAME_END,
            lambda: None,
            name="camera.publish",
            key=_CameraHook.PUBLISH,
            after=HookRef("head", _CameraHook.PUBLISH),
        )

    with pytest.raises(HookOrderError, match="which is itself"):
        hooks.validate()
