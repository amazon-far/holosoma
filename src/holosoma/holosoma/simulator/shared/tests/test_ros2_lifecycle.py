"""ROS-free tests for deterministic private-executor teardown."""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest

from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.ros2_lifecycle import (
    ROS2Runtime,
    close_ros2_executor,
    get_ros2_runtime,
    spin_executor_until_stopped,
)

pytestmark = pytest.mark.no_sim


class _Node:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def destroy_node(self) -> None:
        self.events.append("destroy")
        if self.fail:
            raise RuntimeError("destroy failed")


class _Executor:
    def __init__(
        self,
        events: list[str],
        wakeup: threading.Event,
        *,
        fail: bool = False,
        stopped: bool = True,
    ) -> None:
        self.events = events
        self.wakeup = wakeup
        self.fail = fail
        self.stopped = stopped

    def spin_once(self, timeout_sec: float | None = None) -> None:
        self.wakeup.wait(timeout_sec)

    def shutdown(self, timeout_sec: float | None = None) -> bool:
        self.events.append("shutdown")
        self.wakeup.set()
        if self.fail:
            raise RuntimeError("shutdown failed")
        return self.stopped

    def remove_node(self, node: Any) -> None:
        self.events.append("remove")


def test_close_wakes_executor_joins_thread_then_destroys_node() -> None:
    events: list[str] = []
    stop = threading.Event()
    wakeup = threading.Event()
    executor = _Executor(events, wakeup)
    node = _Node(events)

    def _spin() -> None:
        spin_executor_until_stopped(executor, stop)
        events.append("thread-exit")

    thread = threading.Thread(target=_spin)
    thread.start()
    close_ros2_executor(
        executor=executor,
        node=node,
        spin_thread=thread,
        stop_event=stop,
        label="test",
    )

    assert events == ["shutdown", "thread-exit", "remove", "destroy"]
    assert not thread.is_alive()


def test_close_does_not_destroy_node_when_executor_stop_fails() -> None:
    events: list[str] = []
    stop = threading.Event()
    wakeup = threading.Event()
    executor = _Executor(events, wakeup, fail=True)
    node = _Node(events, fail=True)

    with pytest.raises(RuntimeError, match="executor shutdown failed"):
        close_ros2_executor(
            executor=executor,
            node=node,
            spin_thread=None,
            stop_event=stop,
            label="test",
        )

    assert events == ["shutdown"]


def test_close_does_not_destroy_node_when_executor_times_out() -> None:
    events: list[str] = []
    executor = _Executor(events, threading.Event(), stopped=False)
    node = _Node(events)

    with pytest.raises(RuntimeError, match="executor did not stop"):
        close_ros2_executor(
            executor=executor,
            node=node,
            spin_thread=None,
            stop_event=threading.Event(),
            label="test",
        )

    assert events == ["shutdown"]


def test_runtime_uses_private_context_without_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    init_calls: list[dict[str, Any]] = []

    class _Context:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def ok(self) -> bool:
            return self.shutdown_calls == 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    class _SignalHandlerOptions:
        NO = object()

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda **kwargs: init_calls.append(kwargs)  # type: ignore[attr-defined]
    context_module = types.ModuleType("rclpy.context")
    context_module.Context = _Context  # type: ignore[attr-defined]
    signals_module = types.ModuleType("rclpy.signals")
    signals_module.SignalHandlerOptions = _SignalHandlerOptions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.context", context_module)
    monkeypatch.setitem(sys.modules, "rclpy.signals", signals_module)

    runtime = ROS2Runtime()
    context = runtime.start()
    runtime.close()
    runtime.close()

    assert init_calls == [
        {
            "args": None,
            "context": context,
            "signal_handler_options": _SignalHandlerOptions.NO,
        }
    ]
    assert context.shutdown_calls == 1


def test_runtime_context_closes_after_plugin_hooks() -> None:
    events: list[str] = []

    class _Context:
        @staticmethod
        def ok() -> bool:
            return True

        @staticmethod
        def shutdown() -> None:
            events.append("context")

    simulator = types.SimpleNamespace(hooks=HookRegistry())
    runtime = get_ros2_runtime(simulator)
    runtime._context = _Context()
    simulator.hooks.add(Phase.CLOSE, lambda: events.append("plugin"), name="plugin.close")

    simulator.hooks.emit(Phase.CLOSE)

    assert events == ["plugin", "context"]
