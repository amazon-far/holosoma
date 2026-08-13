"""Generic multiprocess proxy for the G1 high-level clients.

Runs an arbitrary object in a spawned child process and forwards method calls
over queues. Same idea as ``holosoma_telemetry.unitree.unitree_interface_mp``,
but factored so any client can be wrapped — each proxy owns its own child
process, so the arm SDK's CycloneDDS and the loco SDK's CycloneDDS live in
*separate* address spaces (and both are isolated from the parent's rclpy).

We use the ``spawn`` start method so the child is a fresh interpreter that
never inherited the parent's loaded DDS. The factory is passed as a dotted
import path + kwargs (both picklable) and constructed inside the child — the
parent never imports ``unitree_sdk2py``.

    arm = MPClientProxy("...arm_client.arm_client:G1j29ArmController", bridge_host="...")
    arm.ctrl_dual_arm_initialization_pose()      # proxied into the child
    arm.close()
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import signal
from functools import partial
from typing import Any, Generic, TypeVar

# The wrapped client's type. Purely for static typing — factories ``cast`` the
# proxy to the concrete client so callers see its real, typed method surface.
T = TypeVar("T")

_STOP = None


# ── child process ──────────────────────────────────────────────────────


def _worker(factory_path: str, kwargs: dict, req_q: mp.Queue, res_q: mp.Queue):
    # Let the parent own Ctrl+C (mirrors unitree_interface_mp).
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        module_path, _, attr = factory_path.partition(":")
        factory = getattr(importlib.import_module(module_path), attr)
        obj = factory(**kwargs)
    except Exception as exc:  # surface construction failure instead of hanging
        res_q.put(("err", exc))
        return
    res_q.put(("ready", None))

    while True:
        msg = req_q.get()
        if msg is _STOP:
            break
        method, args, kwargs = msg
        try:
            res_q.put(("ok", getattr(obj, method)(*args, **kwargs)))
        except Exception as exc:
            res_q.put(("err", exc))


# ── parent-side proxy ──────────────────────────────────────────────────


class MPClientProxy(Generic[T]):
    """Owns one object in a child process; forwards method calls over queues.

    ``factory_path`` is ``"package.module:Callable"`` (class or factory fn),
    constructed in the child with ``**kwargs``. Methods are reachable either as
    attributes (``proxy.foo(1)``) or via :meth:`call` (``proxy.call("foo", 1)``).
    Calls are synchronous: each returns the method's result (or re-raises).

    The ``T`` type parameter is the wrapped client type. It's only a static-typing
    hint — a constructor can ``cast(T, proxy)`` so callers get the real client's
    typed method surface (autocomplete + checking) while runtime stays the proxy.

    NOTE: only method *calls* are proxied, not attribute *reads* — ``proxy.foo``
    always yields a callable, so ``proxy.some_field`` returns a function, not
    the field's value. Expose any needed state via a getter method instead.
    """

    def __init__(self, factory_path: str, **kwargs: Any):
        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._proc = ctx.Process(target=_worker, args=(factory_path, kwargs, self._req_q, self._res_q), daemon=True)
        self._proc.start()

        # Block until the child has constructed the object (or failed).
        tag, payload = self._res_q.get()
        if tag == "err":
            raise payload

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._req_q.put((method, args, kwargs))
        tag, payload = self._res_q.get()
        if tag == "err":
            raise payload
        return payload

    def __getattr__(self, name: str):
        # Only reached for names not found normally (so _req_q etc. are safe).
        if name.startswith("_"):
            raise AttributeError(name)
        return partial(self.call, name)

    def close(self) -> None:
        try:
            self._req_q.put(_STOP)
            self._proc.join(timeout=5)
        finally:
            if self._proc.is_alive():
                self._proc.kill()

    def __del__(self):
        if hasattr(self, "_proc") and self._proc.is_alive():
            self.close()
