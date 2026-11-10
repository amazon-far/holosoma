"""Camera submission adapters for the pinned Isaac Lab 3 renderers.

Imports remain inside factories: selecting a standalone backend must not import
Kit, PhysX, or an unused renderer. The SDK still owns cameras, native resources,
material writes, render buffers, cloning and optional OVStage ordinals.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


class _CaptureSchedule:
    """Align native capture windows with the existing shared control-step cadence."""

    def __init__(self) -> None:
        self.decimations: dict[str, int] = {}
        self.control_hz = 1.0
        self.step = 0
        self._last_time = 0.0
        self._started = False

    def configure(self, path: str, *, control_hz: float, decimation: int) -> None:
        if self._started:
            raise RuntimeError("Camera cadence must be configured before rendering.")
        if self.decimations and self.control_hz != control_hz:
            raise ValueError("Cameras sharing a renderer must use the same control frequency.")
        self.control_hz = control_hz
        self.decimations[path] = decimation

    @property
    def epoch(self) -> float:
        return max(self.decimations.values(), default=0) / self.control_hz

    def delta_time(self) -> float:
        # A positive first window gives every sensor one initial frame. Native
        # startTime offsets put the ends of their first exposures at this epoch.
        # Use absolute cycle time to avoid accumulated floating-point drift.
        time = self.epoch + self.step / self.control_hz + 1e-7 / self.control_hz
        delta = time - self._last_time
        if delta < 0:
            raise RuntimeError("Camera capture steps must be monotonic.")
        self._last_time = time
        self._started = True
        return delta


class _ScheduledOvrtxEngine:
    """Forward native operations while supplying the current capture interval."""

    def __init__(self, engine: Any, schedule: _CaptureSchedule) -> None:
        self._engine = engine
        self._schedule = schedule
        self.scheduled_request = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def _delta_time(self, default: float) -> float:
        if not self._schedule.decimations:
            return default  # A movie-only renderer retains the SDK's behavior.
        if self.scheduled_request:
            return self._schedule.delta_time()
        # Auto-triggered movie captures render at the current scene state,
        # without consuming a future scheduled perception-camera exposure.
        return 0.0

    def step(self, *, render_products: set[str], delta_time: float, **kwargs: Any) -> Any:
        return self._engine.step(render_products=render_products, delta_time=self._delta_time(delta_time), **kwargs)

    def step_async(self, *, render_products: set[str], delta_time: float, **kwargs: Any) -> Any:
        return self._engine.step_async(
            render_products=render_products, delta_time=self._delta_time(delta_time), **kwargs
        )


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
            self._completed: dict[str, int] = {}
            self._subscriptions: dict[str, Any] = {}
            self._configured_cameras: set[str] = set()

        def configure_cadence(self, path: str, *, control_hz: float, decimation: int) -> None:
            # The shared scheduler owns the rate. Product activation follows
            # its due batch, avoiding a second timer with a different phase.
            self._configured_cameras.add(path)

        def create_render_data(self, spec: Any) -> Any:
            from carb.eventdispatcher import get_eventdispatcher
            import omni.hydratexture

            data = super().create_render_data(spec)
            path = data.render_product.path
            texture = data.render_product.hydra_texture
            texture.set_async(False)
            texture.set_updates_enabled(False)
            self._products[path] = data
            self._completed[path] = 0

            def complete(event: Any) -> None:
                # Disabling before completion cancels pending work. Disabling
                # here prevents an extra draw while Kit finishes the readback.
                texture.set_updates_enabled(False)
                self._completed[path] += 1

            self._subscriptions[path] = get_eventdispatcher().observe_event(
                observer_name=f"holosoma_camera_{path}",
                event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
                on_event=complete,
                filter=texture.get_event_key(),
            )
            return data

        def render_batch(self, render_data: Any) -> None:
            if not render_data:
                return
            import time
            import omni.kit.app
            from carb.settings import get_settings
            from isaaclab.sim import SimulationContext

            sim = SimulationContext.instance()
            # RenderContext already prepared all due poses. Publish physics
            # and deformable state before the first draw, as SDK ensure_update
            # does, without its global per-step/app-update deduplication.
            self._fabric.update_transforms(sim.get_scene_data_provider())
            self._fabric.update_geometries(sim.get_scene_data_provider(), sim.render_generation)
            products = render_data if self._configured_cameras else list(self._products.values())
            before = {data.render_product.path: self._completed[data.render_product.path] for data in products}
            settings = get_settings()
            previous = settings.get("/app/player/playSimulations")
            try:
                settings.set("/app/player/playSimulations", False)
                for data in products:
                    data.render_product.hydra_texture.set_updates_enabled(True)
                deadline = time.monotonic() + 30.0
                # App updates submit work and deliver completed drawable and
                # annotator events. Wait for the actual requested products;
                # a fixed number of updates can read an older image.
                while any(self._completed[path] == count for path, count in before.items()):
                    omni.kit.app.get_app().update()
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"Timed out waiting for Isaac RTX camera products: {list(before)}")
                for data in render_data:
                    self._read_annotator_output(data)
            finally:
                for data in products:
                    data.render_product.hydra_texture.set_updates_enabled(False)
                settings.set("/app/player/playSimulations", previous)

        def cleanup(self, render_data: Any) -> None:
            if render_data is None or render_data.render_product is None:
                return
            path = render_data.render_product.path
            self._subscriptions.pop(path, None)
            self._completed.pop(path, None)
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
    """Keep OVRTX's product set stable while its native scheduler skips idle cameras."""
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer

    class ScheduledOVRTXRenderer(OVRTXRenderer):
        def __init__(self, cfg: Any) -> None:
            super().__init__(cfg)
            self._capture_schedule = _CaptureSchedule()
            self._scheduled_products: set[str] = set()

        def configure_cadence(self, path: str, *, control_hz: float, decimation: int) -> None:
            self._capture_schedule.configure(path, control_hz=control_hz, decimation=decimation)

        def set_capture_step(self, step: int) -> None:
            self._capture_schedule.step = step

        def prepare_stage(self, stage: Any, num_envs: int) -> None:
            from isaaclab.sim import find_matching_prims
            from pxr import Gf, Sdf

            schedule = self._capture_schedule
            for path, decimation in schedule.decimations.items():
                for prim in find_matching_prims(path):
                    # Standalone OpenUSD does not register OmniSensorAPI.
                    # Preserve existing schemas; OVRTX resolves this API when
                    # it imports the exported stage with its native plugins.
                    schemas = list(prim.GetAppliedSchemas())
                    if "OmniSensorAPI" not in schemas:
                        schemas.append("OmniSensorAPI")
                    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=schemas))
                    prim.CreateAttribute("omni:sensor:frameRate", Sdf.ValueTypeNames.Double2).Set(
                        Gf.Vec2d(schedule.control_hz, decimation)
                    )
                    prim.CreateAttribute("omni:sensor:startTime", Sdf.ValueTypeNames.Double2).Set(
                        Gf.Vec2d(max(schedule.decimations.values()) - decimation, schedule.control_hz)
                    )
            super().prepare_stage(stage, num_envs)

        def create_render_data(self, spec: Any) -> Any:
            data = super().create_render_data(spec)
            if spec.cfg.prim_path in self._capture_schedule.decimations:
                self._scheduled_products.add(data.render_product_path)
            if not isinstance(self.backend.renderer, _ScheduledOvrtxEngine):
                self.backend.renderer = _ScheduledOvrtxEngine(self.backend.renderer, self._capture_schedule)
            return data

        def render_batch(self, render_data: Any) -> None:
            if not render_data:
                return
            self.backend.renderer.scheduled_request = any(
                data.render_product_path in self._scheduled_products for data in render_data
            )
            # The SDK submits a stable set of all products and handles both
            # native paths. frameRate/startTime decide which products draw.
            super().render_batch(render_data)

        def cleanup(self, render_data: Any) -> None:
            if render_data is not None:
                self._scheduled_products.discard(render_data.render_product_path)
            super().cleanup(render_data)

    return ScheduledOVRTXRenderer
