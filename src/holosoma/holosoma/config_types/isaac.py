"""Isaac runtime selection without importing a simulator into config/CLI processes."""

from __future__ import annotations

import ast
import json
from dataclasses import field
from typing import Annotated, Any, Literal

from pydantic import model_validator
from pydantic.dataclasses import dataclass
from tyro.constructors import PrimitiveConstructorSpec

IsaacRenderer = Literal["isaacsim_rtx", "ovrtx", "newton_renderer", "none"]
IsaacVisualizer = Literal["kit", "newton_gl", "newton_rtx", "viser"]
IsaacSolver = Literal["mjwarp", "kamino_padmm", "kamino_dvi", "xpbd", "featherstone", "vbd"]


def _options(args: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(args[0])
    except ValueError:
        value = ast.literal_eval(args[0])
    if not isinstance(value, dict):
        raise ValueError("Native options must be a JSON object or Python dictionary.")
    return value


NativeOptions = Annotated[
    dict[str, Any],
    PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=_options,
        is_instance=lambda value: isinstance(value, dict),
        str_from_instance=lambda value: [json.dumps(value)],
    ),
]


@dataclass(frozen=True)
class IsaacCameraRendererConfig:
    """Per-camera override; None inherits the simulator's renderer."""

    backend: IsaacRenderer | None = None
    options: NativeOptions = field(default_factory=dict)


@dataclass(frozen=True)
class IsaacSimRenderConfig:
    """Isaac runtime and native options. Existing 5.1 launcher fields are unchanged."""

    rendering_mode: Literal["performance", "balanced", "quality"] | None = None
    kit_args: str = ""
    runtime: Literal["5.1", "6.x"] = "5.1"
    physics: Literal["isaacsim_physx", "newton", "ovphysx"] = "isaacsim_physx"
    solver: IsaacSolver | None = None
    renderer: IsaacRenderer = "isaacsim_rtx"
    visualizers: list[IsaacVisualizer] | None = None
    physics_options: NativeOptions = field(default_factory=dict)
    solver_options: NativeOptions = field(default_factory=dict)
    renderer_options: NativeOptions = field(default_factory=dict)
    visualizer_options: NativeOptions = field(default_factory=dict)

    @model_validator(mode="after")
    def validate_runtime(self) -> IsaacSimRenderConfig:
        if self.renderer == "none" and self.renderer_options:
            raise ValueError("renderer_options requires a selected renderer.")
        if self.runtime == "5.1":
            if self.physics != "isaacsim_physx" or self.solver is not None:
                raise ValueError("Isaac Sim 5.1 supports Isaac Sim PhysX only; select runtime='6.x'.")
            if self.renderer not in ("isaacsim_rtx", "none"):
                raise ValueError("Isaac Sim 5.1 supports the Isaac RTX renderer only.")
            if any((self.physics_options, self.solver_options, self.renderer_options, self.visualizer_options)):
                raise ValueError("Native backend options require runtime='6.x'; 5.1 retains its existing settings.")
            if self.visualizers is not None and any(v != "kit" for v in self.visualizers):
                raise ValueError("Isaac Sim 5.1 supports the Kit viewer only.")
        if self.physics != "newton" and (self.solver is not None or self.solver_options):
            raise ValueError("solver and solver_options apply only to Newton physics.")
        unknown = set(self.visualizer_options) - set(self.visualizers or [])
        if unknown:
            raise ValueError(f"visualizer_options requires explicit visualizer selection: {sorted(unknown)}")
        return self

    def app_launcher_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if self.rendering_mode is not None:
            args["rendering_mode"] = self.rendering_mode
        if self.kit_args:
            args["kit_args"] = self.kit_args
        return args

    def selected_visualizers(self, headless: bool) -> list[IsaacVisualizer]:
        if headless:
            if self.visualizers:
                raise ValueError("Headless execution requires visualizers=[]; disable headless to open a viewer.")
            return []
        if self.visualizers is not None:
            return list(dict.fromkeys(self.visualizers))
        return ["kit"] if self.physics == "isaacsim_physx" else ["newton_gl"]
