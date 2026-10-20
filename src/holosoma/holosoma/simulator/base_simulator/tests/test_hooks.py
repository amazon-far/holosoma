from __future__ import annotations

import functools
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest
from loguru import logger

from holosoma.config_types.video import VideoConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import (
    HookCloseError,
    HookOrderError,
    HookRegistry,
    HookRegistryError,
    Phase,
)

pytestmark = pytest.mark.no_sim


def test_emit_uses_registration_order() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.FRAME_END, lambda: calls.append("a"), name="a")
    hooks.add(Phase.FRAME_END, lambda: calls.append("b"), name="b")

    hooks.emit(Phase.FRAME_END)

    assert calls == ["a", "b"]


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
        training=SimpleNamespace(),
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


@contextmanager
def captured_logs(level: str) -> Iterator[list[str]]:
    """Collect loguru messages at or above ``level`` for the duration of the block."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level=level)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def recorder(calls: list[str], name: str) -> Callable[[], None]:
    return lambda: calls.append(name)


def test_after_runs_hook_behind_its_dependency() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="consumer", after=["render"])
    hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="render")

    hooks.emit(Phase.FRAME_END)

    assert calls == ["render", "consumer"]


def test_before_runs_hook_ahead_of_its_dependency() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.FRAME_END, recorder(calls, "late"), name="late")
    hooks.add(Phase.FRAME_END, recorder(calls, "early"), name="early", before=["late"])

    hooks.emit(Phase.FRAME_END)

    assert calls == ["early", "late"]


def test_unconstrained_hooks_keep_their_relative_order() -> None:
    # 'b' and 'c' stay in that order, but both overtake the blocked 'a' rather than holding its slot.
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.FRAME_END, recorder(calls, "a"), name="a", after=["c"])
    hooks.add(Phase.FRAME_END, recorder(calls, "b"), name="b")
    hooks.add(Phase.FRAME_END, recorder(calls, "c"), name="c")

    hooks.emit(Phase.FRAME_END)

    assert calls == ["b", "c", "a"]


def test_constraint_matches_every_hook_sharing_the_name() -> None:
    # Two plugin slots of the same class register the same hook name; the constraint covers both.
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="consumer", after=["publish"])
    hooks.add(Phase.FRAME_END, recorder(calls, "publish_1"), name="publish")
    hooks.add(Phase.FRAME_END, recorder(calls, "publish_2"), name="publish")

    hooks.emit(Phase.FRAME_END)

    assert calls == ["publish_1", "publish_2", "consumer"]


def test_order_is_recomputed_after_a_later_registration() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="consumer", after=["render"])

    hooks.emit(Phase.FRAME_END)
    hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="render")
    hooks.emit(Phase.FRAME_END)

    assert calls == ["consumer", "render", "consumer"]


def test_cycle_raises_on_emit() -> None:
    hooks = HookRegistry()
    hooks.add(Phase.FRAME_END, lambda: None, name="a", after=["b"])
    hooks.add(Phase.FRAME_END, lambda: None, name="b", after=["a"])

    with pytest.raises(HookOrderError, match="cycle on phase 'frame_end' among: a, b"):
        hooks.emit(Phase.FRAME_END)


def test_missing_dependency_is_ignored_silently() -> None:
    # A constraint on something this run didn't configure is satisfied by its absence, not a problem.
    calls: list[str] = []
    hooks = HookRegistry()
    hooks.add(Phase.FRAME_END, recorder(calls, "a"), name="a", after=["nope"])
    hooks.add(Phase.FRAME_END, recorder(calls, "b"), name="b", after=["nope"])

    with captured_logs("WARNING") as messages:
        hooks.emit(Phase.FRAME_END)

    assert calls == ["a", "b"]
    assert messages == []


def test_dependency_on_another_phase_does_not_order_or_resolve() -> None:
    # Names are looked up per phase: a FRAME_BEGIN hook is not a FRAME_END dependency.
    calls: list[str] = []
    hooks = HookRegistry()
    hooks.add(Phase.FRAME_BEGIN, recorder(calls, "elsewhere"), name="elsewhere")
    hooks.add(Phase.FRAME_END, recorder(calls, "a"), name="a", after=["elsewhere"])

    with captured_logs("WARNING") as messages:
        hooks.emit(Phase.FRAME_END)

    assert calls == ["a"]
    assert messages == []


def test_disabled_dependency_is_inert_and_returns_on_enable() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    render = hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="render")
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="consumer", after=["render"])

    render.disable()
    hooks.emit(Phase.FRAME_END)
    render.enable()
    hooks.emit(Phase.FRAME_END)

    assert calls == ["consumer", "render", "consumer"]


def test_removed_dependency_stops_constraining() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    render = hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="render")
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="consumer", after=["render"])

    render.remove()
    with captured_logs("WARNING") as messages:
        hooks.emit(Phase.FRAME_END)

    assert calls == ["consumer"]
    assert messages == []


def test_close_runs_resolved_order_backwards() -> None:
    # The consumer follows the runtime while running, so it tears down before the runtime does.
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.CLOSE, recorder(calls, "consumer"), name="consumer", after=["runtime"])
    hooks.add(Phase.CLOSE, recorder(calls, "runtime"), name="runtime")

    hooks.emit(Phase.CLOSE)

    assert calls == ["consumer", "runtime"]


def test_close_cycle_falls_back_to_registration_order() -> None:
    # A cycle must not cost every hook its teardown.
    calls: list[str] = []
    hooks = HookRegistry()
    hooks.add(Phase.CLOSE, recorder(calls, "a"), name="a", after=["b"])
    hooks.add(Phase.CLOSE, recorder(calls, "b"), name="b", after=["a"])

    with captured_logs("ERROR") as messages:
        hooks.emit(Phase.CLOSE)

    assert calls == ["b", "a"]
    assert any("cycle" in message for message in messages)


class _Producer:
    """Two instances register the same callback; a subclass override registers a different one."""

    def tick(self) -> None: ...


class _Subclass(_Producer):
    def tick(self) -> None: ...


def test_default_name_is_module_and_qualname() -> None:
    hooks = HookRegistry()

    handle = hooks.add(Phase.FRAME_END, _Producer().tick, name=None)

    assert handle.name == f"{__name__}._Producer.tick"


def test_instances_share_a_default_name_but_an_override_gets_its_own() -> None:
    hooks = HookRegistry()

    first = hooks.add(Phase.FRAME_END, _Producer().tick)
    second = hooks.add(Phase.FRAME_END, _Producer().tick)
    override = hooks.add(Phase.FRAME_END, _Subclass().tick)

    assert first.name == second.name
    assert override.name == f"{__name__}._Subclass.tick"


def test_default_name_unwraps_partial_and_decorators() -> None:
    def target(env_id: int) -> None: ...

    @functools.wraps(target)
    def decorated(env_id: int) -> None: ...

    hooks = HookRegistry()

    partial_handle = hooks.add(Phase.EPISODE_START, functools.partial(target))
    decorated_handle = hooks.add(Phase.EPISODE_END, decorated)

    expected = f"{__name__}.test_default_name_unwraps_partial_and_decorators.<locals>.target"
    assert partial_handle.name.startswith(expected)
    assert decorated_handle.name.startswith(expected)


def test_lambdas_on_different_lines_get_distinct_default_names() -> None:
    hooks = HookRegistry()

    first = hooks.add(Phase.FRAME_END, lambda: None)
    second = hooks.add(Phase.FRAME_END, lambda: None)

    assert first.name != second.name


def test_one_name_reference_covers_every_hook_sharing_the_default() -> None:
    # The multi-slot case: one callback registered twice, one reference orders against both.
    calls: list[str] = []
    hooks = HookRegistry()

    class Egress:
        def __init__(self, label: str) -> None:
            self.label = label

        def publish(self) -> None:
            calls.append(self.label)

    head = hooks.add(Phase.FRAME_END, Egress("head").publish)
    waist = hooks.add(Phase.FRAME_END, Egress("waist").publish)
    assert head.name == waist.name  # one place in the source, so one name
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="consumer", after=[head.name])

    hooks.emit(Phase.FRAME_END)

    assert calls == ["head", "waist", "consumer"]


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


def test_handle_for_another_phase_is_rejected() -> None:
    hooks = HookRegistry()
    elsewhere = hooks.add(Phase.FRAME_BEGIN, lambda: None, name="elsewhere.tick")

    with pytest.raises(HookRegistryError, match="Constraints apply within one phase"):
        hooks.add(Phase.FRAME_END, lambda: None, name="here.tick", after=[elsewhere])


def test_dependency_that_is_neither_name_nor_handle_is_rejected() -> None:
    # The overloads reject this at type-check time; the runtime guard catches whoever ignores them,
    # since a callback or plugin object would match no hook and silence is for absent names only.
    hooks = HookRegistry()
    target = hooks.add(Phase.FRAME_END, lambda: None, name="target.tick")
    deps: list[Any] = [target.name, object()]

    with pytest.raises(HookRegistryError, match="pass a hook name or the HookHandle"):
        hooks.add(Phase.FRAME_END, lambda: None, name="dependent.tick", after=deps)


def test_handle_from_another_registry_is_rejected() -> None:
    hooks = HookRegistry()
    other = HookRegistry()
    elsewhere = other.add(Phase.FRAME_END, lambda: None, name="elsewhere.tick")

    with pytest.raises(HookRegistryError, match="from a different registry"):
        hooks.add(Phase.FRAME_END, lambda: None, name="here.tick", after=[elsewhere])


def test_handle_to_a_removed_hook_stops_constraining() -> None:
    calls: list[str] = []
    hooks = HookRegistry()
    render = hooks.add(Phase.FRAME_END, recorder(calls, "render"), name="render.one")
    hooks.add(Phase.FRAME_END, recorder(calls, "consumer"), name="consumer.one", after=[render])

    render.remove()
    hooks.emit(Phase.FRAME_END)

    assert calls == ["consumer"]


def test_names_reports_resolved_order_per_phase() -> None:
    hooks = HookRegistry()
    hooks.add(Phase.FRAME_END, lambda: None, name="consumer", after=["render"])
    hooks.add(Phase.FRAME_END, lambda: None, name="render")
    hooks.add(Phase.CLOSE, lambda: None, name="consumer", after=["runtime"])
    hooks.add(Phase.CLOSE, lambda: None, name="runtime")

    assert hooks.names(Phase.FRAME_END) == ("render", "consumer")
    assert hooks.names(Phase.CLOSE) == ("consumer", "runtime")  # teardown order
    assert hooks.names(Phase.PRE_STEP) == ()


def test_bare_string_dependency_is_rejected() -> None:
    # A str satisfies Sequence[str], so the types allow this and only the runtime guard catches it.
    hooks = HookRegistry()

    with pytest.raises(HookRegistryError, match="pass a sequence of hook names"):
        hooks.add(Phase.FRAME_END, lambda: None, name="a", after="render")
