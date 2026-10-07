"""Shared lifecycle helpers for ROS2 simulator plugins."""

from __future__ import annotations

import threading
from typing import Any, cast

from loguru import logger

from holosoma.simulator.base_simulator.hooks import Phase

_RUNTIME_ATTRIBUTE = "_holosoma_ros2_runtime"


class ROS2Runtime:
    """One private ROS context owned by a simulator."""

    def __init__(self) -> None:
        self._context: Any = None
        self._closed = False

    def start(self) -> Any:
        """Initialize the context without installing process signal handlers."""
        if self._closed:
            raise RuntimeError("ROS2 runtime is closed")
        if self._context is not None:
            return self._context

        import rclpy
        from rclpy.context import Context
        from rclpy.signals import SignalHandlerOptions

        context = Context()
        try:
            rclpy.init(
                args=None,
                context=context,
                signal_handler_options=SignalHandlerOptions.NO,
            )
        except BaseException:
            try:
                if context.ok():
                    context.shutdown()
            except Exception:
                logger.exception("ROS2 context rollback failed while preserving the startup error")
            raise
        self._context = context
        return context

    def close(self) -> None:
        """Shut down the context once, after plugin nodes have closed."""
        if self._closed:
            return
        self._closed = True
        context = self._context
        self._context = None
        if context is not None and context.ok():
            context.shutdown()


def get_ros2_runtime(simulator: Any) -> ROS2Runtime:
    """Return the simulator's ROS runtime and register its close hook once."""
    runtime = cast("ROS2Runtime | None", getattr(simulator, _RUNTIME_ATTRIBUTE, None))
    if runtime is not None:
        return runtime

    runtime = ROS2Runtime()
    setattr(simulator, _RUNTIME_ATTRIBUTE, runtime)
    simulator.hooks.add(Phase.CLOSE, runtime.close, name="ros2.runtime.close")
    return runtime


def spin_executor_until_stopped(executor: Any, stop_event: threading.Event, *, timeout_sec: float = 0.1) -> None:
    """Spin a private executor until its owner requests shutdown.

    Executor shutdown and a process-wide rclpy context shutdown both wake ``spin_once`` by
    raising. Those exceptions are expected only while this plugin is stopping; an unexpected
    executor failure while the plugin is active is still allowed to reach the thread exception
    handler.
    """
    try:
        from rclpy.executors import ExternalShutdownException, ShutdownException

        shutdown_exceptions: tuple[type[Exception], ...] = (ExternalShutdownException, ShutdownException)
    except (ImportError, AttributeError):
        shutdown_exceptions = ()

    while not stop_event.is_set():
        try:
            executor.spin_once(timeout_sec=timeout_sec)
        except shutdown_exceptions:  # noqa: PERF203 - spin-loop shutdown is exception-signaled by rclpy.
            return
        except Exception:
            if stop_event.is_set():
                return
            raise


def close_ros2_executor(
    *,
    executor: Any,
    node: Any,
    spin_thread: threading.Thread | None,
    stop_event: threading.Event,
    label: str,
    timeout_sec: float = 2.0,
) -> None:
    """Stop one plugin's executor, join its thread, then destroy its node.

    The simulator-scoped ROS context is closed by a separate, earlier-registered CLOSE hook.
    """
    stop_event.set()
    failures: list[str] = []
    executor_stopped = executor is None

    if executor is not None:
        try:
            stopped = executor.shutdown(timeout_sec=timeout_sec)
            if stopped is False:
                failures.append("executor did not stop before the timeout")
            else:
                executor_stopped = True
        except Exception as exc:
            failures.append(f"executor shutdown failed: {exc!r}")

    thread_stopped = spin_thread is None or not spin_thread.is_alive()
    if spin_thread is threading.current_thread():
        failures.append("cannot close ROS resources from the executor thread")
    elif spin_thread is not None and spin_thread.is_alive():
        try:
            spin_thread.join(timeout=timeout_sec)
            if spin_thread.is_alive():
                failures.append("spin thread did not stop before the timeout")
            else:
                thread_stopped = True
        except Exception as exc:
            failures.append(f"spin thread join failed: {exc!r}")

    if executor_stopped and thread_stopped:
        node_detached = executor is None or node is None
        if executor is not None and node is not None:
            try:
                executor.remove_node(node)
            except Exception as exc:
                failures.append(f"executor node removal failed: {exc!r}")
            else:
                node_detached = True

        if node is not None and node_detached:
            try:
                node.destroy_node()
            except Exception as exc:
                failures.append(f"node destruction failed: {exc!r}")

    if failures:
        raise RuntimeError(f"{label} ROS2 shutdown failed: {'; '.join(failures)}")
