"""Subprocess scenarios for failures around native simulation shutdown."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, cast

import pytest

from holosoma.utils import sim_utils
from holosoma.utils.simulator_config import SimulatorType


def configure_logging(monkeypatch: pytest.MonkeyPatch, sink_kind: str, diagnostic: str) -> int | None:
    """Configure a child sink or inject a fault; return any added fault sink."""
    logger = sim_utils.logger
    if sink_kind == "queued":
        sink_started = Event()
        release_sink = Event()

        def queued_sink(message: str) -> None:
            sink_started.set()
            assert release_sink.wait(10), "Queued sink was never released"
            sys.stderr.write(message)
            sys.stderr.flush()

        logger.remove()
        logger.add(queued_sink, level="WARNING", enqueue=True, catch=False)

        def release_before_drain(drain: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                # Hold the real queue worker until draining begins; a missing
                # drain must not pass merely because the worker ran first.
                assert sink_started.wait(10), "Queued sink never received the failure"
                release_sink.set()
                return drain(*args, **kwargs)

            return wrapped

        monkeypatch.setattr(logger, "complete", release_before_drain(logger.complete))
        monkeypatch.setattr(logger, "remove", release_before_drain(logger.remove))

    def diagnostic_failure(*_args: Any, **_kwargs: Any) -> None:
        print(f"DIAGNOSTIC_FAILURE {diagnostic}", file=sys.__stderr__, flush=True)
        raise OSError("Diagnostic delivery failed")

    if diagnostic in {"sink_error", "sink_all"}:
        return logger.add(diagnostic_failure, level="ERROR" if diagnostic == "sink_error" else "DEBUG", catch=False)
    if diagnostic == "drain":
        monkeypatch.setattr(logger, "complete", diagnostic_failure)
    elif diagnostic == "flush":
        # Keep writes working so this isolates the explicit stderr flush.
        monkeypatch.setattr(
            sys, "stderr", SimpleNamespace(write=sys.stderr.write, flush=diagnostic_failure, isatty=lambda: False)
        )
    return None


class UsdContext:
    def close_stage(self) -> bool:
        return False


class NativeApp:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, quit_kind: str) -> None:
        self.quit_kind = quit_kind
        self.quit_requests: list[int] = []
        self.usd_context = UsdContext()
        self.sim_context = SimpleNamespace(_disable_app_control_on_stop_handle=False)

        # Substitute only the native provider boundary; run the real workarounds.
        omni = ModuleType("omni")
        usd = ModuleType("omni.usd")
        omni.__dict__["usd"] = usd
        usd.__dict__["get_context"] = lambda: self.usd_context
        isaaclab = ModuleType("isaaclab")
        sim = ModuleType("isaaclab.sim")
        isaaclab.__dict__["sim"] = sim
        sim.__dict__["SimulationContext"] = SimpleNamespace(instance=lambda: self.sim_context)
        for name, module in (("omni", omni), ("omni.usd", usd), ("isaaclab", isaaclab), ("isaaclab.sim", sim)):
            monkeypatch.setitem(sys.modules, name, module)

    @property
    def app(self) -> NativeApp:
        if self.quit_kind == "missing_app":
            raise AttributeError("Kit app is unavailable")
        return self

    def post_uncancellable_quit(self, return_code: int) -> None:
        if self.quit_kind == "binding_error":
            raise TypeError("Unsupported Kit binding")
        self.quit_requests.append(return_code)

    def close(self, **_kwargs: Any) -> None:
        assert self.usd_context.close_stage(), "Stage shutdown workaround was skipped"
        assert self.sim_context._disable_app_control_on_stop_handle, "Stop workaround was skipped"
        print("WORKAROUNDS_OK", flush=True)
        print("APP_CLOSE", flush=True)
        os._exit(self.quit_requests[-1] if self.quit_requests else 0)


class Environment:
    def __init__(self, ready_file: Path | None) -> None:
        self.sim: Any = object()
        self.ready_file = ready_file

    def close(self) -> None:
        print("ENV_CLOSE", flush=True)
        if self.ready_file is not None:
            self.ready_file.touch()
            Event().wait()


def run_entrypoint(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, fail: Callable[[], None], env: Environment, app: NativeApp
) -> None:
    config = cast(
        "sim_utils.RunSimConfig",
        SimpleNamespace(
            simulator=SimpleNamespace(config=SimpleNamespace(sim=SimpleNamespace(fps=100, control_decimation_steps=1))),
            time_scale=0,
            viewer_dt=0.02,
            training=SimpleNamespace(headless=True),
        ),
    )
    if entrypoint == "setup":
        monkeypatch.setattr(sim_utils, "setup_simulator_imports", lambda _config: None)
        monkeypatch.setattr(sim_utils, "setup_isaaclab_launcher", lambda _config, _device: app)
        monkeypatch.setattr(sim_utils, "_create_simulation_environment", lambda _config, _device: fail())
        sim_utils.setup_simulation_environment(config, device="cpu")
    elif entrypoint == "app":
        try:
            fail()
        finally:
            sim_utils.close_simulation_app(app)
    elif entrypoint in {"direct_initialize", "direct_body", "direct_run"}:
        monkeypatch.setattr(
            sim_utils.DirectSimulation,
            "initialize",
            lambda _self: fail() if entrypoint == "direct_initialize" else None,
        )
        env.sim = SimpleNamespace(
            shutdown_requested=False,
            refresh_sim_tensors=lambda: None,
            hooks=SimpleNamespace(emit=lambda *_args: None),
            simulate_at_each_physics_step=fail,
        )
        with sim_utils.DirectSimulation(config, env, "cpu", app) as simulation:
            if entrypoint == "direct_run":
                simulation.run()
            elif entrypoint == "direct_initialize":
                raise AssertionError("Initialization unexpectedly succeeded")
            else:
                fail()
    else:
        with sim_utils.simulation_resource_session(env, app):
            fail()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint")
    parser.add_argument("error")
    parser.add_argument("--sink", default="direct")
    parser.add_argument("--quit", default="normal")
    parser.add_argument("--diagnostic", default="normal")
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()

    failure = {
        "runtime": RuntimeError("Missing LiDAR prim: /World/missing_chair"),
        "interrupt": KeyboardInterrupt(),
        "exit_code": SystemExit(7),
        "exit_message": SystemExit("Invalid simulator configuration"),
        "exit_success": SystemExit(),
    }.get(args.error)

    def fail() -> None:
        if args.error == "terminate":
            os.kill(os.getpid(), signal.SIGTERM)
        elif failure is not None:
            raise failure

    monkeypatch = pytest.MonkeyPatch()
    if args.diagnostic in {"sink_error", "sink_all"}:
        # A healthy default sink would hide a missing fallback.
        sim_utils.logger.remove()
    configure_logging(monkeypatch, args.sink, args.diagnostic)
    app = NativeApp(monkeypatch, args.quit)
    env = Environment(args.ready_file)
    monkeypatch.setattr(sim_utils, "get_simulator_type", lambda: SimulatorType.ISAACSIM)
    with sim_utils.exit_on_sigterm():
        run_entrypoint(monkeypatch, args.entrypoint, fail, env, app)

    raise AssertionError("Native provider shutdown must terminate the process")


if __name__ == "__main__":
    main()
