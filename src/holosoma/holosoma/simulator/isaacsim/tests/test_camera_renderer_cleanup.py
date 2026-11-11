"""Exercise adapter failures at the native resource boundary, without launching Kit."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from holosoma.simulator.isaacsim.backends.camera_renderers import (
    capture_isaac_rtx_products,
    scheduled_isaac_rtx_renderer_class,
)

pytestmark = [
    pytest.mark.no_sim,
    pytest.mark.skipif(sys.version_info < (3, 11), reason="Native Lab 3 requires Python 3.12."),
]


@pytest.fixture
def native_boundary(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace only SDK/Kit resources; run the production ownership and capture code."""
    state = SimpleNamespace(
        products=[],
        callbacks={},
        cleanup_attempts=[],
        cleanup_failures={},
        subscription_failure=False,
        app_failure=False,
        play_simulations=True,
        close_calls=0,
        publications=[],
    )

    class Texture:
        def __init__(self, path: str) -> None:
            self.path = path
            self.enabled = False
            self.disable_failure = False
            self.disable_attempts = 0

        def set_async(self, value: bool) -> None:
            pass

        def set_updates_enabled(self, value: bool) -> None:
            if not value:
                self.disable_attempts += 1
                if self.disable_failure:
                    raise RuntimeError(f"disable failed: {self.path}")
            self.enabled = value

        def get_event_key(self) -> str:
            return self.path

    class NativeRenderer:
        def __init__(self, cfg: Any) -> None:
            pass

        def create_render_data(self, spec: Any) -> Any:
            texture = Texture(spec)
            product = SimpleNamespace(path=spec, hydra_texture=texture, destroyed=False)
            data = SimpleNamespace(render_product=product)
            state.products.append(product)
            return data

        def cleanup(self, data: Any) -> None:
            # Pinned SDK detach/destroy failures occur before clearing render_product.
            path = data.render_product.path
            state.cleanup_attempts.append(path)
            if state.cleanup_failures.get(path, 0):
                state.cleanup_failures[path] -= 1
                raise RuntimeError(f"release failed: {path}")
            data.render_product.destroyed = True
            data.render_product = None

        def close(self) -> None:
            state.close_calls += 1

        def update_transforms(self) -> None:
            state.publications.append("transforms")

        def update_geometries(self) -> None:
            state.publications.append("geometries")

        def _read_annotator_output(self, data: Any) -> None:
            state.publications.append(f"read:{data.render_product.path}")

    class Subscription:
        def __init__(self, path: str) -> None:
            self.path = path

        def __del__(self) -> None:
            state.callbacks.pop(self.path, None)

    def observe_event(**kwargs: Any) -> Any:
        if state.subscription_failure:
            raise RuntimeError("subscription failed")
        state.callbacks[kwargs["filter"]] = kwargs["on_event"]
        return Subscription(kwargs["filter"])

    def update() -> None:
        if state.app_failure:
            raise RuntimeError("app update failed")
        for product in state.products:
            if product.hydra_texture.enabled:
                state.publications.append(f"draw:{product.path}")
                state.callbacks[product.path](None)

    settings = SimpleNamespace(
        get=lambda key: state.play_simulations,
        set=lambda key, value: setattr(state, "play_simulations", value),
    )
    modules = {
        name: ModuleType(name)
        for name in (
            "carb",
            "carb.eventdispatcher",
            "carb.settings",
            "omni",
            "omni.hydratexture",
            "omni.kit",
            "omni.kit.app",
            "isaaclab_physx.renderers.isaac_rtx_renderer",
        )
    }
    modules["carb.eventdispatcher"].get_eventdispatcher = lambda: SimpleNamespace(observe_event=observe_event)  # type: ignore[attr-defined]
    modules["carb.settings"].get_settings = lambda: settings  # type: ignore[attr-defined]
    modules["omni.hydratexture"].GLOBAL_EVENT_DRAWABLE_CHANGED = "drawable_changed"  # type: ignore[attr-defined]
    modules["omni"].hydratexture = modules["omni.hydratexture"]  # type: ignore[attr-defined]
    modules["omni"].kit = modules["omni.kit"]  # type: ignore[attr-defined]
    modules["omni.kit"].app = modules["omni.kit.app"]  # type: ignore[attr-defined]
    modules["omni.kit.app"].get_app = lambda: SimpleNamespace(update=update)  # type: ignore[attr-defined]
    modules["isaaclab_physx.renderers.isaac_rtx_renderer"].IsaacRtxRenderer = NativeRenderer  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    scheduled_isaac_rtx_renderer_class.cache_clear()
    state.renderer = scheduled_isaac_rtx_renderer_class()(None)
    yield state
    scheduled_isaac_rtx_renderer_class.cache_clear()


@pytest.mark.parametrize("rollback_failures", [0, 1])
def test_failed_subscription_releases_product_and_retries_failed_rollback(
    native_boundary: Any, rollback_failures: int
) -> None:
    state = native_boundary
    state.subscription_failure = True
    state.cleanup_failures["camera"] = rollback_failures
    with pytest.raises(RuntimeError, match="subscription failed"):
        state.renderer.create_render_data("camera")
    state.renderer.close()
    assert state.products[0].destroyed
    assert state.cleanup_attempts == ["camera"] * (1 + rollback_failures)
    assert not state.callbacks


def test_close_releases_other_products_and_keeps_failed_release_for_retry(native_boundary: Any) -> None:
    state = native_boundary
    state.renderer.create_render_data("first")
    state.renderer.create_render_data("second")
    state.cleanup_failures["first"] = 1
    with pytest.raises(Exception, match="Failed closing Isaac RTX"):
        state.renderer.close()
    assert state.cleanup_attempts == ["first", "second"]
    assert [product.destroyed for product in state.products] == [False, True]
    assert state.close_calls == 1
    assert not state.callbacks
    state.renderer.close()
    assert all(product.destroyed for product in state.products)
    assert state.cleanup_attempts == ["first", "second", "first"]


def test_explicit_cleanup_failure_remains_owned_until_close(native_boundary: Any) -> None:
    state = native_boundary
    data = state.renderer.create_render_data("camera")
    state.cleanup_failures["camera"] = 1
    with pytest.raises(RuntimeError, match="release failed"):
        state.renderer.cleanup(data)
    state.renderer.close()
    assert state.products[0].destroyed
    assert state.cleanup_attempts == ["camera", "camera"]
    assert not state.callbacks


@pytest.mark.parametrize("failed_subscription", [False, True])
def test_failed_release_does_not_block_surviving_camera(native_boundary: Any, failed_subscription: bool) -> None:
    state = native_boundary
    state.cleanup_failures["failed"] = 1
    state.subscription_failure = failed_subscription
    if failed_subscription:
        with pytest.raises(RuntimeError, match="subscription failed"):
            state.renderer.create_render_data("failed")
    else:
        data = state.renderer.create_render_data("failed")
        with pytest.raises(RuntimeError, match="release failed"):
            state.renderer.cleanup(data)
    state.subscription_failure = False
    survivor = state.renderer.create_render_data("survivor")
    state.renderer.render_batch([survivor])
    assert state.publications[-1] == "read:survivor"
    assert set(state.callbacks) == {"survivor"}
    state.renderer.close()
    assert all(product.destroyed for product in state.products)
    assert not state.callbacks


@pytest.mark.parametrize("playing", [False, True])
def test_render_batch_publishes_scene_and_restores_physics_setting(native_boundary: Any, playing: bool) -> None:
    state = native_boundary
    state.play_simulations = playing
    data = state.renderer.create_render_data("camera")
    state.renderer.render_batch([data])
    assert state.publications == ["transforms", "geometries", "draw:camera", "read:camera"]
    assert state.play_simulations is playing
    assert not state.products[0].hydra_texture.enabled
    state.renderer.close()
    assert not state.callbacks


def test_capture_failure_attempts_every_disable_and_restores_physics_setting(native_boundary: Any) -> None:
    state = native_boundary
    state.renderer.create_render_data("first")
    state.renderer.create_render_data("second")
    captures = list(state.renderer._captures.values())
    state.app_failure = True
    state.products[1].hydra_texture.disable_failure = True
    with pytest.raises(RuntimeError, match="disable failed: second") as error:
        capture_isaac_rtx_products(captures)
    assert str(error.value.__context__) == "app update failed"
    assert state.play_simulations is True
    assert not state.products[0].hydra_texture.enabled
    assert all(product.hydra_texture.disable_attempts == 2 for product in state.products)
    state.renderer.close()
    assert not state.callbacks
