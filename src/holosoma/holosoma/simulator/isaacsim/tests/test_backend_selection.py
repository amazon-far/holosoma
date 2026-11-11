"""Select the native API family without importing the optional simulator runtime."""

from __future__ import annotations

import importlib.metadata
import gc
import sys
import weakref
import re
from argparse import Namespace
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import make_dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest

from holosoma.simulator.isaacsim import backends


@pytest.mark.no_sim
@pytest.mark.parametrize(
    ("metadata_version", "physx_package", "expected_family", "expected_backend"),
    [("0.54.2", False, 2, "isaacsim5"), ("28.0.0", True, 3, "isaacsim6")],
)
def test_source_extension_versions_validate_the_explicit_api_family(
    monkeypatch: pytest.MonkeyPatch,
    metadata_version: str,
    physx_package: bool,
    expected_family: int,
    expected_backend: str,
) -> None:
    """Source-installed metadata numbers must not be interpreted as release tags."""
    # A conflicting environment value must not affect declarative selection.
    conflicting_backend = "isaacsim6" if expected_backend == "isaacsim5" else "isaacsim5"
    monkeypatch.setenv("HOLOSOMA_ISAACSIM_BACKEND", conflicting_backend)
    monkeypatch.setattr(backends.importlib.metadata, "version", lambda _name: metadata_version)
    monkeypatch.setattr(
        backends.importlib.util,
        "find_spec",
        lambda _name: ModuleSpec("isaaclab_physx", loader=None) if physx_package else None,
    )
    selected = backends.get_backend(expected_backend)
    assert selected.isaaclab_major == expected_family
    assert selected.is_isaacsim6 == (expected_family == 3)
    assert type(backends.get_backend(expected_backend)) is type(selected)
    with pytest.raises(RuntimeError, match="requires Isaac Lab"):
        backends.get_backend(conflicting_backend)


@pytest.mark.no_sim
def test_optional_runtime_absent_leaves_explicit_selection_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("isaaclab")

    monkeypatch.setattr(backends.importlib.metadata, "version", missing)
    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setenv("HOLOSOMA_ISAACSIM_BACKEND", "unsupported-obsolete-value")
    assert backends.get_backend("isaacsim5").isaaclab_major == 2
    assert backends.get_backend("isaacsim6").isaaclab_major == 3
    with pytest.raises(TypeError, match="backend"):
        backends.get_backend()  # type: ignore[call-arg]


@pytest.mark.no_sim
def test_unknown_selection_fails_with_the_supported_choices() -> None:
    with pytest.raises(ValueError, match="'isaacsim5' or 'isaacsim6'"):
        backends.get_backend("physx")


@pytest.mark.no_sim
@pytest.mark.parametrize("family", [2, 3])
def test_backend_selection_accepts_the_config_enum(monkeypatch: pytest.MonkeyPatch, family: int) -> None:
    from holosoma.config_types.simulator import IsaacSimBackend

    monkeypatch.setattr(backends, "_installed_isaaclab_major", lambda: None)
    selection = IsaacSimBackend.ISAACSIM6 if family == 3 else IsaacSimBackend.ISAACSIM5
    assert type(backends.get_backend(selection)) is type(backends.get_backend(selection.value))
    assert backends.get_backend(selection).isaaclab_major == family


@pytest.mark.no_sim
@pytest.mark.parametrize("family", [2, 3])
def test_tensor_boundary_keeps_the_native_torch_view(family: int) -> None:
    """An adapter must return the native tensor itself, preserving writes and storage."""
    from holosoma.simulator.isaacsim.backends.isaacsim5 import IsaacSim5Backend
    from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend
    from holosoma.utils.safe_torch_import import torch

    native_torch = torch.tensor([1.0, 2.0, 3.0])
    backend = IsaacSim6Backend() if family == 3 else IsaacSim5Backend()
    native = SimpleNamespace(torch=native_torch) if family == 3 else native_torch
    converted = backend.tensor(native)
    assert converted is native_torch
    converted[1] = 17.0
    assert native_torch.tolist() == [1.0, 17.0, 3.0]


@pytest.mark.no_sim
@pytest.mark.parametrize(("family", "expected_namespace"), [(2, "/World/envs/env_.*"), (3, "/World/envs/env_[^/]+")])
def test_environment_namespace_keeps_native_clone_roots_in_one_path_segment(
    family: int, expected_namespace: str
) -> None:
    from holosoma.simulator.isaacsim.backends.isaacsim5 import IsaacSim5Backend
    from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend

    backend = IsaacSim6Backend() if family == 3 else IsaacSim5Backend()
    assert backend.env_regex_ns == expected_namespace
    assert re.fullmatch(backend.env_regex_ns, "/World/envs/env_0")
    assert re.fullmatch(f"{backend.env_regex_ns}/Robot", "/World/envs/env_12/Robot")
    assert re.fullmatch(f"{backend.env_regex_ns}/Robot/.*", "/World/envs/env_0/Robot/Geometry/pelvis")
    if family == 3:
        assert not re.fullmatch(backend.env_regex_ns, "/World/envs/env_0/freebox")
        assert not re.fullmatch(f"{backend.env_regex_ns}/Robot", "/World/envs/env_0/freebox/Geometry/Robot")
        assert not re.fullmatch(
            f"{backend.env_regex_ns}/freebox", "/World/envs/env_0/freebox/Geometry/envs/env_0/freebox"
        )


@pytest.mark.no_sim
def test_native_physics_clock_uses_steps_when_the_manager_clock_is_stopped() -> None:
    """Lab 3 PhysxManager's inherited clock stays zero while physical steps advance."""
    from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend

    step_count = 0
    sim = SimpleNamespace(
        get_physics_step_count=lambda: step_count,
        get_physics_dt=lambda: 0.005,
        physics_manager=SimpleNamespace(get_simulation_time=lambda: 0.0),
    )
    backend = IsaacSim6Backend()
    assert backend.physics_time(sim) == 0.0
    for step_count, expected_seconds in ((1, 0.005), (20, 0.1), (200, 1.0)):
        assert backend.physics_time(sim) == pytest.approx(expected_seconds)
        assert backend.physics_time(sim) == pytest.approx(expected_seconds), "reading time must not advance it"


@pytest.mark.no_sim
@pytest.mark.parametrize(("namespace_args", "exit_code"), [(False, 0), (True, 23)])
def test_native_launch_context_outlives_the_returned_app(
    monkeypatch: pytest.MonkeyPatch, namespace_args: bool, exit_code: int
) -> None:
    """Dropping a wrapper must not finalize the real external launch context."""
    from holosoma.simulator.isaacsim.backends import isaacsim6

    events: list[tuple[str, str | int | None]] = []
    context_refs: list[weakref.ReferenceType[object]] = []
    native_app = SimpleNamespace(is_running=lambda: True)

    @contextmanager
    def runtime() -> Iterator[None]:
        events.append(("enter", None))
        try:
            yield
        finally:
            exception = sys.exc_info()[1]
            events.append(("exit", exception.code if isinstance(exception, SystemExit) else None))

    def launch_simulation(_cfg: object, _args: dict[str, object]) -> object:
        context = runtime()
        context_refs.append(weakref.ref(context))
        return context

    modules = {
        name: ModuleType(name)
        for name in (
            "isaaclab",
            "isaaclab.app",
            "isaaclab_physx",
            "isaaclab_physx.physics",
            "isaaclab_physx.renderers",
            "isaaclab_newton.renderers",
            "omni",
            "omni.kit",
            "omni.kit.app",
        )
    }
    setattr(modules["isaaclab"], "app", modules["isaaclab.app"])
    setattr(modules["isaaclab.app"], "launch_simulation", launch_simulation)
    setattr(modules["isaaclab_physx"], "physics", modules["isaaclab_physx.physics"])
    setattr(modules["isaaclab_physx.physics"], "PhysxCfg", make_dataclass("PhysxCfg", []))
    setattr(modules["isaaclab_physx.renderers"], "IsaacRtxRendererCfg", make_dataclass("IsaacRtxRendererCfg", []))
    setattr(modules["isaaclab_newton.renderers"], "NewtonWarpRendererCfg", make_dataclass("NewtonWarpRendererCfg", []))
    setattr(modules["omni"], "kit", modules["omni.kit"])
    setattr(modules["omni.kit"], "app", modules["omni.kit.app"])
    setattr(modules["omni.kit.app"], "get_app", lambda: native_app)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(isaacsim6, "_RUNTIME", ExitStack())

    telemetry_setting = "--/exts/omni.kit.telemetry/skipDeferredStartup"
    requested_args = {
        "headless": False,
        "visualizer": None,
        "kit_args": f"{telemetry_setting}=false --/custom/sentinel=23" if namespace_args else "--/custom/sentinel=23",
    }
    args = Namespace(**requested_args) if namespace_args else requested_args.copy()
    app = isaacsim6.IsaacSim6Backend().launch_app(args)
    assert app.is_running()
    forwarded = vars(args) if isinstance(args, Namespace) else args
    assert forwarded["visualizer"] == "kit"
    assert forwarded["kit_args"] == (
        requested_args["kit_args"] if namespace_args else f"--/custom/sentinel=23 {telemetry_setting}=true"
    )
    app_ref = weakref.ref(app)
    del app
    gc.collect()

    assert app_ref() is None
    assert context_refs[0]() is not None
    assert events == [("enter", None)]

    # The adapter's explicit close entry point unwinds the retained context.
    closer = isaacsim6.IsaacSim6Runtime(native_app)
    closer.close(exit_code=exit_code)
    assert events == [("enter", None), ("exit", exit_code)]
    closer.close(exit_code=91)
    assert events == [("enter", None), ("exit", exit_code)]


@pytest.mark.no_sim
@pytest.mark.parametrize("launcher_failure", [False, True])
def test_native_teardown_failure_still_exits_launch_context(
    monkeypatch: pytest.MonkeyPatch, launcher_failure: bool
) -> None:
    """A failed native release must not retain launchers or report a successful exit."""
    from holosoma.simulator.isaacsim.backends import isaacsim6

    events: list[str] = []

    @contextmanager
    def runtime() -> Iterator[None]:
        try:
            yield
        finally:
            events.append(str(sys.exc_info()[1]))
            if launcher_failure:
                raise ValueError("launcher cleanup failed")

    def clear_instance() -> None:
        events.append("clear")
        raise RuntimeError("native cleanup failed")

    sim = ModuleType("isaaclab.sim")
    setattr(sim, "SimulationContext", SimpleNamespace(clear_instance=clear_instance))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    stack = ExitStack()
    stack.enter_context(runtime())
    monkeypatch.setattr(isaacsim6, "_RUNTIME", stack)
    app = isaacsim6.IsaacSim6Runtime()
    with pytest.raises(ValueError if launcher_failure else RuntimeError) as error:
        app.close()
    assert str(error.value) == ("launcher cleanup failed" if launcher_failure else "native cleanup failed")
    if launcher_failure:
        assert str(error.value.__context__) == "native cleanup failed"
    assert events == ["clear", "native cleanup failed"]
    app.close(exit_code=91)
    assert events == ["clear", "native cleanup failed"]
