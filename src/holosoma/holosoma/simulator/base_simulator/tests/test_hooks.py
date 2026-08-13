from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.hooks import HookCloseError, HookRegistry, HookRegistryError, Phase

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


def test_close_runs_reverse_order_once_and_reports_failures() -> None:
    calls: list[str] = []
    hooks = HookRegistry()

    hooks.add(Phase.CLOSE, lambda: calls.append("first"), name="first")

    def fail() -> None:
        calls.append("second")
        raise RuntimeError("boom")

    hooks.add(Phase.CLOSE, fail, name="second")

    with pytest.raises(HookCloseError) as exc_info:
        hooks.emit(Phase.CLOSE)

    hooks.emit(Phase.CLOSE)

    assert calls == ["second", "first"]
    assert exc_info.value.failures[0][0] == "second"


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
