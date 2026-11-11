"""Keep fixed OVRTX multiview batches for independently scheduled cameras.

The pinned native renderer can substitute camera poses when multiview batch
membership changes. Mixed rates and perception/movie coexistence use one SDK
renderer and native backend per product. Uniform and single-camera setups keep
the normal shared batch. The SDK owns cloning, bindings, materials and buffers.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast


def scheduled_renderer_class() -> type[Any]:
    """Build the SDK adapter lazily, keeping unused native runtimes unloaded."""
    # Match Holosoma's existing lazy factory; importing this file starts no runtime.
    # Lab 3 uses Python 3.12; Holosoma's shared type-check target remains 3.10.
    from builtins import ExceptionGroup  # type: ignore[attr-defined]

    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
    from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXRendererCfg

    @configclass
    class _ProductRendererCfg(OVRTXRendererCfg):
        # Internal registry identity, never a Holosoma user configuration field.
        owner_key: int = 0
        product_index: int = 0

    class _MaterialWriters:
        """Fan shared material buffers out to every owner, including later cameras.

        Each SDK writer owns its native binding; close writers before releasing
        camera data or backends. Publishing a later owner preserves current colors.
        """

        def __init__(self, renderers: Any, batches: Any) -> None:
            self.batches = batches
            self.writers: dict[int, Any] = {}
            self.closed = False
            try:
                for renderer in renderers:
                    self.add(renderer, publish_current=False)
            except Exception as primary:
                try:
                    self.close()
                except Exception as cleanup_error:
                    cast(Any, primary).add_note(f"Material-writer cleanup failed: {cleanup_error}")
                raise

        def add(self, renderer: Any, *, publish_current: bool = True) -> None:
            if self.closed:
                raise RuntimeError("Cannot attach a renderer to a closed material writer.")
            factory = (
                super(ScheduledOVRTXRenderer, renderer).visual_material_writer
                if isinstance(renderer, ScheduledOVRTXRenderer)
                else renderer.visual_material_writer
            )
            if factory is None:
                raise RuntimeError("OVRTX renderer does not provide visual material writes.")
            writer = factory(self.batches)
            self.writers[id(renderer)] = writer
            try:
                if publish_current:
                    # A later camera's stage snapshot must observe current randomized buffers.
                    writer()
            except Exception as primary:
                self.writers.pop(id(renderer), None)
                try:
                    writer.close()
                except Exception as cleanup_error:
                    cast(Any, primary).add_note(f"Material-writer cleanup failed: {cleanup_error}")
                raise

        def __call__(self, material_offsets: Any = None, env_ids: Any = None) -> None:
            for writer in self.writers.values():
                writer(material_offsets, env_ids)

        def close(self) -> None:
            if self.closed:
                return
            self.closed = True
            errors = []
            for writer in self.writers.values():
                try:
                    writer.close()
                except Exception as error:
                    errors.append(error)
            self.writers.clear()
            if errors:
                raise ExceptionGroup("Failed closing isolated OVRTX material writers", errors)

    class ScheduledOVRTXRenderer(OVRTXRenderer):
        """Keep fixed products per owner when camera schedules can diverge.

        Declare cadence and independent cameras before stage preparation. The
        prepared SDK scene is reused without authoring the shared stage again;
        uniform rates without independent cameras retain the ordinary shared
        renderer and multiview batch.
        """

        def __init__(self, cfg: Any) -> None:
            super().__init__(cfg)
            # The parent owns the first product and uses its ordinary registry engine.
            self._owners: list[Any] = [self]
            self._data_owners: dict[int, Any] = {}
            self._all_data: dict[int, Any] = {}
            self._private_backends: list[Any] = []
            self._decimations: dict[str, int] = {}
            self._independent_cameras: set[str] = set()
            self._isolated: bool | None = None
            self._prepared_scene: Any = None
            self._material_writers: Any = None
            self._closed = False

        def configure_cadence(self, path: str, *, control_hz: float, decimation: int) -> None:
            del control_hz
            self._decimations[path] = decimation

        def register_independent_camera(self, path: str) -> None:
            self._independent_cameras.add(path)

        def _requires_isolation(self) -> bool:
            return len(set(self._decimations.values())) > 1 or bool(self._decimations and self._independent_cameras)

        def prepare_stage(self, stage: Any, num_envs: int) -> None:
            super().prepare_stage(stage, num_envs)
            # SDK prepare_stage authors scene partitions and cannot be repeated
            # on the same host stage. Retain its immutable export and descriptors;
            # children ingest the same prepared snapshot into independent engines.
            if self._requires_isolation():
                self._prepared_scene = (
                    self._clone_plan,
                    self._exported_usd_string,
                    dict(self._object_scales_by_path),
                )

        def _new_owner(self) -> Any:
            sim = SimulationContext.instance()
            # Let the SDK create the correct backend through its normal
            # constructor. Registry identity differs; rendering settings do not.
            child = OVRTXRenderer(
                _ProductRendererCfg(
                    **{field.name: getattr(self.cfg, field.name) for field in fields(self.cfg) if field.init},
                    owner_key=id(self),
                    product_index=len(self._owners),
                )
            )
            backend = child.backend
            try:
                if self._prepared_scene is None:
                    raise RuntimeError("OVRTX stage must be prepared before camera data.")
                clone_plan, exported_usd, object_scales = self._prepared_scene
                child._clone_plan = clone_plan
                child._exported_usd_string = exported_usd
                child._object_scales_by_path = dict(object_scales)
            except Exception as primary:
                for release in (child.close, lambda: sim.close_backend(backend)):
                    try:
                        release()
                    except Exception as cleanup_error:
                        cast(Any, primary).add_note(f"Camera-owner cleanup failed: {cleanup_error}")
                raise
            self._private_backends.append(backend)
            return child

        def create_render_data(self, spec: Any) -> Any:
            if self._closed:
                raise RuntimeError("Cannot create camera data after OVRTX close.")
            if self._isolated is None:
                self._isolated = self._requires_isolation()
            # Late layout changes need to be declared before SDK binding creation.
            if (
                self._all_data
                and not self._isolated
                and self._decimations
                and (len(set(self._decimations.values())) > 1 or spec.cfg.prim_path in self._independent_cameras)
            ):
                raise RuntimeError("Configure independent OVRTX cameras before initialization.")
            owner = self._new_owner() if self._isolated and self._all_data else self
            data = None
            try:
                data = OVRTXRenderer.create_render_data(owner, spec)
                if owner is not self and self._material_writers is not None:
                    self._material_writers.add(owner)
            except Exception as primary:
                if owner is not self:
                    backend = owner.backend
                    for release in (
                        lambda: OVRTXRenderer.cleanup(owner, data),
                        owner.close,
                        lambda: SimulationContext.instance().close_backend(backend),
                    ):
                        try:
                            release()
                        except Exception as cleanup_error:
                            cast(Any, primary).add_note(f"Camera-owner cleanup failed: {cleanup_error}")
                    self._private_backends.remove(backend)
                raise
            if owner is not self:
                self._owners.append(owner)
            self._data_owners[id(data)] = owner
            self._all_data[id(data)] = data
            return data

        def _owner(self, data: Any) -> Any:
            return self._data_owners[id(data)]

        def _groups(self, cameras: Any) -> Any:
            groups: dict[Any, list[Any]] = {}
            for data in cameras:
                owner = self._owner(data)
                groups.setdefault(owner, []).append(data)
            return groups.items()

        def set_outputs(self, data: Any, output_data: Any) -> Any:
            return OVRTXRenderer.set_outputs(self._owner(data), data, output_data)

        def prepare_capture(self, data: Any, camera_data: Any, frame: Any) -> Any:
            return OVRTXRenderer.prepare_capture(self._owner(data), data, camera_data, frame)

        def read_output(self, data: Any, camera_data: Any) -> Any:
            return OVRTXRenderer.read_output(self._owner(data), data, camera_data)

        def update_camera(self, data: Any, positions: Any, orientations: Any, intrinsics: Any) -> Any:
            return OVRTXRenderer.update_camera(self._owner(data), data, positions, orientations, intrinsics)

        def update_camera_intrinsics(self, data: Any, intrinsics: Any, parameters: Any) -> Any:
            return OVRTXRenderer.update_camera_intrinsics(self._owner(data), data, intrinsics, parameters)

        def update_transforms(self) -> None:
            if not self._isolated:
                OVRTXRenderer.update_transforms(self)

        def update_geometries(self) -> None:
            if not self._isolated:
                OVRTXRenderer.update_geometries(self)

        def render(self, data: Any) -> Any:
            return self.render_batch((data,))

        def render_batch(self, cameras: Any) -> None:
            for owner, cameras in self._groups(cameras):
                if self._isolated:
                    # Publish only due scenes. Idle owners consume current SDP
                    # buffers on their next capture, without background uploads.
                    OVRTXRenderer.update_transforms(owner)
                    OVRTXRenderer.update_geometries(owner)
                # Each isolated owner always submits its fixed one-product batch.
                # SDK extraction finishes before another submission on that engine.
                OVRTXRenderer.render_batch(owner, cameras)

        def drain_pending_renders(self, cameras: Any = None) -> list[Any]:
            errors = []
            cameras = tuple(self._all_data.values()) if cameras is None else cameras
            for owner, group in self._groups(cameras):
                errors.extend(OVRTXRenderer.drain_pending_renders(owner, group))
            return errors

        def reset(self, data: Any, env_ids: Any = None) -> Any:
            return OVRTXRenderer.reset(self._owner(data), data, env_ids)

        def cleanup(self, data: Any) -> None:
            if data is None or id(data) not in self._data_owners:
                return
            owner = self._owner(data)
            try:
                OVRTXRenderer.cleanup(owner, data)
            finally:
                # The SDK releases buffers/bindings even if draining reports a failure.
                self._data_owners.pop(id(data), None)
                self._all_data.pop(id(data), None)

        def _make_material_writers(self, batches: Any) -> Any:
            self._material_writers = _MaterialWriters(self._owners, batches)
            return self._material_writers

        @property
        def visual_material_writer(self) -> Any:
            return self._make_material_writers

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            errors = []
            if self._material_writers is not None:
                try:
                    self._material_writers.close()
                except Exception as error:
                    errors.append(error)
            for data in tuple(self._all_data.values()):
                try:
                    self.cleanup(data)
                except Exception as error:
                    errors.append(error)
            # Destroy per-renderer bindings before releasing any native backend.
            for owner in reversed(self._owners):
                try:
                    OVRTXRenderer.close(owner)
                except Exception as error:
                    errors.append(error)
            for backend in reversed(self._private_backends):
                try:
                    SimulationContext.instance().close_backend(backend)
                except Exception as error:
                    errors.append(error)
            self._owners = [self]
            self._private_backends.clear()
            self._prepared_scene = None
            # The first/native backend remains in the normal simulation registry.
            if errors:
                raise ExceptionGroup("Failed closing isolated OVRTX resources", errors)

    return ScheduledOVRTXRenderer
