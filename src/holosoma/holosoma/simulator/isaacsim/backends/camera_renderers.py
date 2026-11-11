"""Camera submission adapters for the pinned Isaac Lab 3 renderers.

Imports remain inside factories: selecting a standalone backend must not import
Kit, PhysX, or an unused renderer. The SDK still owns cameras, native resources,
material writes, render buffers, cloning and optional OVStage ordinals.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


class IsaacRtxCapture:
    """Own one product's completion subscription and pause it between captures."""

    def __init__(self, texture: Any, *, name: str) -> None:
        from carb.eventdispatcher import get_eventdispatcher
        import omni.hydratexture

        self.texture = texture
        self.completed = 0
        texture.set_async(False)
        texture.set_updates_enabled(False)
        self._subscription = get_eventdispatcher().observe_event(
            observer_name=f"holosoma_camera_{name}",
            event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
            on_event=self._complete,
            filter=texture.get_event_key(),
        )

    def _complete(self, event: Any) -> None:
        # Pausing before completion cancels pending work. Pausing here prevents
        # an extra draw while Kit finishes the annotator readback.
        self.texture.set_updates_enabled(False)
        self.completed += 1

    def close(self) -> None:
        self._subscription = None


def capture_isaac_rtx_products(captures: list[IsaacRtxCapture]) -> None:
    """Complete the requested products without stepping physics or drawing idle ones."""
    if not captures:
        return
    import time
    import omni.kit.app
    from carb.settings import get_settings

    before = [capture.completed for capture in captures]
    settings = get_settings()
    previous = settings.get("/app/player/playSimulations")
    try:
        settings.set("/app/player/playSimulations", False)
        for capture in captures:
            capture.texture.set_updates_enabled(True)
        deadline = time.monotonic() + 30.0
        # A fixed number of app updates can read an older image. Wait for the
        # actual drawable events, which also pause each completed product.
        while any(capture.completed == count for capture, count in zip(captures, before, strict=True)):
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for Isaac RTX camera products.")
            omni.kit.app.get_app().update()
    finally:
        for capture in captures:
            capture.texture.set_updates_enabled(False)
        settings.set("/app/player/playSimulations", previous)


def create_scheduled_ovrtx_renderer(cfg: Any) -> Any:
    """Resolve the renderer implementation after the selected runtime has launched."""
    return scheduled_ovrtx_renderer_class()(cfg)


def create_scheduled_isaac_rtx_renderer(cfg: Any) -> Any:
    """Delay all Kit imports until the selected runtime has launched."""
    return scheduled_isaac_rtx_renderer_class()(cfg)


@lru_cache(maxsize=1)
def scheduled_isaac_rtx_renderer_class() -> type[Any]:
    """Request one completed draw for each due camera, leaving idle products paused."""
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    class ScheduledIsaacRtxRenderer(IsaacRtxRenderer):
        def __init__(self, cfg: Any) -> None:
            super().__init__(cfg)
            self._products: dict[str, Any] = {}
            self._captures: dict[str, IsaacRtxCapture] = {}
            self._configured_cameras: set[str] = set()

        def configure_cadence(self, path: str, *, control_hz: float, decimation: int) -> None:
            # The shared scheduler owns the rate. Product activation follows
            # its due batch, avoiding a second timer with a different phase.
            self._configured_cameras.add(path)

        def create_render_data(self, spec: Any) -> Any:
            data = super().create_render_data(spec)
            path = data.render_product.path
            self._products[path] = data
            self._captures[path] = IsaacRtxCapture(data.render_product.hydra_texture, name=path)
            return data

        def render_batch(self, render_data: Any) -> None:
            if not render_data:
                return
            from isaaclab.sim import SimulationContext

            sim = SimulationContext.instance()
            # RenderContext already prepared all due poses. Publish physics
            # and deformable state before the first draw, as SDK ensure_update
            # does, without its global per-step/app-update deduplication.
            self._fabric.update_transforms(sim.get_scene_data_provider())
            self._fabric.update_geometries(sim.get_scene_data_provider(), sim.render_generation)
            products = render_data if self._configured_cameras else list(self._products.values())
            capture_isaac_rtx_products([self._captures[data.render_product.path] for data in products])
            for data in render_data:
                self._read_annotator_output(data)

        def cleanup(self, render_data: Any) -> None:
            if render_data is None or render_data.render_product is None:
                return
            path = render_data.render_product.path
            self._captures.pop(path).close()
            self._products.pop(path, None)
            super().cleanup(render_data)

        def close(self) -> None:
            # Teardown can follow STOP callbacks that already cleaned cameras.
            # Release remaining event subscriptions before Kit destroys products.
            for data in list(self._products.values()):
                self.cleanup(data)
            self._configured_cameras.clear()
            super().close()

    return ScheduledIsaacRtxRenderer


@lru_cache(maxsize=1)
def scheduled_ovrtx_renderer_class() -> type[Any]:
    """Resolve stable product ownership only after OVRTX has launched."""
    from .ovrtx_cameras import scheduled_renderer_class

    return scheduled_renderer_class()
