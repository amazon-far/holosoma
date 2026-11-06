"""Isaac Lab 3 adapter. Native factories select PhysX, Newton, or OvPhysX."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from contextlib import AbstractContextManager
from importlib import import_module, metadata
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence

from holosoma.config_types.isaac import IsaacSimRenderConfig

from . import _active, as_tensor
from .lab2 import Lab2Backend, ViewportConfig
from .native_config import isaac_rtx_rendering_mode, physics_config, renderer_config, visualizer_configs

if TYPE_CHECKING:
    import torch

    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.config_types.run_sim import RunSimConfig
    from holosoma.config_types.sensor import CameraSensorConfig
    from holosoma.config_types.simulator import SimulatorInitConfig
    from holosoma.config_types.video import VideoConfig
    from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

    from .properties import NativeProperties


class Lab3Backend(Lab2Backend):
    modern = True
    env_regex = "/World/envs/env_[^/]+"

    def __init__(self, config: IsaacSimRenderConfig) -> None:
        self.config = config
        self.single_friction = config.physics == "newton"
        self.native_config: Any = None
        self.session: NativeSession | None = None

    def has_gui(self, sim: Any) -> bool:
        return bool(sim.has_gui)

    def needs_rendering(self, simulator: Any) -> bool:
        return bool(simulator.tiled_cameras or simulator.sim.is_rendering)

    def physics_time(self, sim: Any) -> float:
        # Lab 3's PhysX manager manually steps with a constant timeline time;
        # neither its legacy clock nor PhysicsManager._sim_time advances.
        # The context owns the monotonic integration count for all three engines.
        return float(sim.get_physics_step_count() * sim.get_physics_dt())

    def prepare_recording_camera(self, simulator: Any) -> None:
        if not simulator.video_config.enabled:
            return
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab.sim import PinholeCameraCfg

        video = simulator.video_config
        focal_length = 24.0
        vertical_aperture = 2.0 * focal_length * math.tan(math.radians(video.vertical_fov) / 2.0)
        camera = Camera(
            CameraCfg(
                prim_path=f"{self.env_regex}/HolosomaVideoCamera",
                width=video.width,
                height=video.height,
                data_types=["rgb"],
                update_period=0.0,
                renderer_cfg=renderer_config(self.config),
                spawn=PinholeCameraCfg(
                    focal_length=focal_length,
                    vertical_aperture=vertical_aperture,
                    horizontal_aperture=vertical_aperture * video.width / video.height,
                    clipping_range=(0.1, 1000.0),
                ),
            )
        )
        simulator.recording_camera = camera
        simulator.scene.sensors["__holosoma_video"] = camera

    def config_quaternion(self, wxyz: Sequence[float]) -> tuple[float, float, float, float]:
        w, x, y, z = wxyz
        return (x, y, z, w)

    def quaternion_to_wxyz(self, quaternion: Sequence[float]) -> tuple[float, float, float, float]:
        x, y, z, w = quaternion
        return w, x, y, z

    def acquire_debug_draw(self) -> Any:
        from isaaclab.utils.version import has_kit

        return super().acquire_debug_draw() if has_kit() else None

    def read_state(self, value: Any) -> torch.Tensor:
        return as_tensor(value)

    def urdf_spawn_config(self, config: Any, *, preserve_fixed_bodies: Sequence[str] = ()) -> Any:
        from .urdf import urdf_spawn_config

        return urdf_spawn_config(config, preserve_fixed_bodies=preserve_fixed_bodies)

    def convert_urdf(self, config: Any) -> str:
        from .urdf import convert_urdf

        return convert_urdf(config)

    def write_state(self, value: torch.Tensor) -> torch.Tensor:
        return value

    @property
    def stage_utils(self) -> ModuleType:
        return import_module("isaaclab.sim.utils.stage")

    @property
    def prim_utils(self) -> ModuleType:
        return import_module("isaaclab.sim.utils")

    def simulation_config(self, config: SimulatorInitConfig, device: str) -> Any:
        from isaaclab.sim import RigidBodyMaterialCfg, SimulationCfg

        if self.native_config is None:
            self.native_config = SimulationCfg(
                dt=1.0 / config.sim.fps,
                render_interval=config.sim.render_interval_steps,
                device=device,
                physics=physics_config(config),
                use_newton_actuators=config.isaacsim.physics == "newton",
                physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            )
        return self.native_config

    def clone(self, scene: Any, global_paths: list[str]) -> None:
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.cloner import clone_plan_from_env_0, replicate
        from isaaclab.cloner.usd import UsdReplicateContext
        from isaaclab.sensors import RayCasterCfg

        if self.config.physics == "ovphysx" and scene.rigid_objects:
            from isaaclab_ov.physics import OvPhysxManager

            # Live body-damping edits reach parsed prims, not native clones.
            # Materialize these worlds so every free object's properties remain
            # mutable through the shared randomization API.
            OvPhysxManager.require_full_stage()
        assets = [a.cfg for a in (*scene.articulations.values(), *scene.rigid_objects.values())]
        sensors = [s.cfg for s in scene.sensors.values()]
        if any(getattr(getattr(cfg, "renderer_cfg", None), "renderer_type", None) == "ovrtx" for cfg in sensors):
            from pxr import UsdGeom

            # OVRTX's asset-level clone operation requires existing parent prims
            # when it writes each environment's transform. Native physics clones
            # alone need not create these USD roots.
            for index in range(scene.num_envs):
                UsdGeom.Xform.Define(scene.stage, f"/World/envs/env_{index}")
        configs = [*assets, *sensors, *(AssetBaseCfg(prim_path=p, spawn=None) for p in global_paths)]
        prefix = scene.env_regex_ns + "/"
        owned = [
            *assets,
            *(s for s in sensors if not isinstance(s, RayCasterCfg) and getattr(s, "spawn", None) is not None),
        ]
        declared = {c.prim_path[len(prefix) :].split("/", 1)[0] for c in owned if c.prim_path.startswith(prefix)}
        for child in scene.stage.GetPrimAtPath("/World/envs/env_0").GetChildren():
            if child.GetName() not in declared:
                configs.append(
                    AssetBaseCfg(
                        prim_path=f"{scene.env_regex_ns}/{child.GetName()}",
                        spawn=None,
                        cloning_contexts=(UsdReplicateContext,),
                    )
                )
        plan = clone_plan_from_env_0(
            scene.cfg.clone_cfg,
            configs,
            scene.num_envs,
            scene.cfg.env_spacing,
            positions=as_tensor(scene.env_origins).cpu().numpy(),
        )
        replicate(plan, replicate_physics=scene.cfg.replicate_physics)
        if self.config.physics != "newton":
            # Native PhysX replication disables environment IDs; full-stage
            # OVPhysX also needs explicit groups. Isolate overlapping worlds.
            scene.filter_collisions(global_prim_paths=global_paths)
        if self.config.physics == "newton":
            self._normalize_rigid_object_labels(scene)

    def _normalize_rigid_object_labels(self, scene: Any) -> None:
        """Align Lab 3.0's RigidObject view with Newton 1.6.1 URDF labels."""
        if not scene.rigid_objects:
            return
        from collections import defaultdict
        from isaaclab.sim import SimulationContext
        from isaaclab_newton.physics import NewtonBuilderCfg

        builder = SimulationContext.instance().get_or_create_backend(
            NewtonBuilderCfg(physics_cfg=self.native_config.physics)
        )
        patterns = [re.compile(asset.cfg.prim_path) for asset in scene.rigid_objects.values()]
        bodies = defaultdict(set)
        for joint, articulation in enumerate(builder.joint_articulation):
            bodies[articulation].add(builder.joint_child[joint])
        for index, label in enumerate(builder.articulation_label):
            if any(pattern.fullmatch(label) for pattern in patterns):
                if len(bodies[index]) != 1:
                    raise ValueError(f"Rigid scene object {label!r} contains multiple articulated bodies.")
                # URDF's wrapper names its one-body articulation after the asset;
                # Lab's RigidObject builds its view from the nested rigid prim.
                # Only the lookup label changes, not topology or physical data.
                builder.articulation_label[index] = builder.body_label[next(iter(bodies[index]))]

    def properties(self, asset: Any) -> NativeProperties:
        from .properties import NativeProperties

        return NativeProperties(asset, self.config.physics)

    def camera_options(self, camera: CameraSensorConfig) -> dict[str, Any]:
        renderer = camera.isaacsim.renderer.backend or self.config.renderer
        if renderer == "newton_renderer" and camera.isaacsim.projection_type != "pinhole":
            raise ValueError(
                "Newton Warp does not support Holosoma's F-theta camera calibration. "
                "Select isaacsim_rtx or ovrtx for this camera, or use projection_type='pinhole'."
            )
        # Standalone renderers consume the pose buffer; unlike RTX, they cannot
        # follow a moving USD parent while that buffer remains at its initial pose.
        return {
            "renderer_cfg": renderer_config(self.config, camera.isaacsim.renderer),
            "update_latest_camera_pose": True,
        }

    def sync_cameras(self, sim: Any) -> None:
        # Lab 3 initializes native renderer bindings from the resolved frame views.
        # The Lab 2 Fabric repair is deliberately confined to that adapter.
        pass

    def create_viewport(self, sim: Any, config: ViewportConfig) -> Any:
        if not sim.sim.cfg.visualizer_cfgs:
            return None
        from ..isaaclab_viewpoint_camera_controller import ViewportCameraController

        return ViewportCameraController(sim, config, native_callback=True)

    def create_recorder(self, config: VideoConfig, sim: Any) -> VideoRecorderInterface:
        from .recording import NativeVideoRecorder

        return NativeVideoRecorder(config, sim)

    def finish_state_write(self, sim: Any) -> None:
        sim.forward()


class NativeSession:
    """Own the native launch context for the entire simulator lifetime."""

    def __init__(self, context: AbstractContextManager[Any], backend: Lab3Backend) -> None:
        self.context = context
        self.backend = backend
        self.closed = False
        self.token = _active.set(backend)
        try:
            context.__enter__()
        except BaseException:
            _active.reset(self.token)
            raise
        # Keep the context alive even when a caller discards the returned app handle.
        backend.session = self

    def is_running(self) -> bool:
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        return not self.closed and (sim is None or bool(sim.is_running()))

    def close(self, **kwargs: Any) -> None:
        self.close_with_failure(sys.exc_info()[1])

    def close_with_failure(self, failure: BaseException | None) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.context.__exit__(
                type(failure) if failure is not None else None,
                failure,
                failure.__traceback__ if failure is not None else None,
            )
        finally:
            self.backend.session = None
            _active.reset(self.token)


def launch(config: ExperimentConfig | RunSimConfig, device: str | None) -> NativeSession:
    from isaaclab.app import add_launcher_args, launch_simulation, validate_launch
    from isaaclab.sensors import CameraCfg
    from holosoma.config_types.sensor import CameraSensorConfig

    version = metadata.version("isaaclab")
    if not version.startswith("28."):
        raise RuntimeError(f"runtime='6.x' requires the pinned Isaac Lab 3.0 environment; found isaaclab {version}.")
    selection = config.simulator.config.isaacsim
    if selection.renderer != "none":
        # Reject misspelled or invalid options even when this task has no cameras.
        renderer_config(selection)
    backend = Lab3Backend(selection)
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if distributed else device or "cuda:0"
    args.distributed = distributed
    args.visualizer = selection.selected_visualizers(config.training.headless)
    for name, value in selection.app_launcher_args().items():
        setattr(args, name, value)
    # Kit-specific launch options are an explicit requirement, never a silent no-op.
    args.require_kit = bool(selection.kit_args or selection.rendering_mode)
    native = backend.simulation_config(config.simulator.config, args.device)
    native.visualizer_cfgs = visualizer_configs(selection, headless=config.training.headless)
    cameras = [
        CameraCfg(prim_path=f"/World/launch_camera_{i}", **backend.camera_options(c))
        for i, c in enumerate(config.sensor.values())
        if isinstance(c, CameraSensorConfig)
    ]
    if config.logger.video.enabled or config.logger.headless_recording:
        cameras.append(CameraCfg(prim_path="/World/launch_video", renderer_cfg=renderer_config(selection)))
    # Lab's launch scan walks object fields, but skips camera lists. Expose each
    # camera so RTX startup and mixed-renderer compatibility see its requirements.
    tree = SimpleNamespace(sim=native, **{f"camera_{i}": camera for i, camera in enumerate(cameras)})
    _, kit_sources = validate_launch(tree, args)
    if not kit_sources:
        # The standalone URDF importer authors both PhysX and MuJoCo
        # attributes. Register their data-only schemas before fresh conversion.
        import ovphysx
        from pxr import Plug

        import_module("mujoco_usd_converter")  # registers the mjcPhysics USD schema
        Plug.Registry().RegisterPlugins([str(path) for path in ovphysx.codeless_schema_paths()])
    session = NativeSession(launch_simulation(tree, args), backend)
    if selection.rendering_mode is not None:
        # Apply before scene/camera creation, including viewport-only runs.
        # Renderer configs carry the same preset plus explicit camera overrides.
        try:
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

            settings = (
                renderer_config(selection).global_settings
                if selection.renderer == "isaacsim_rtx"
                else isaac_rtx_rendering_mode(selection.rendering_mode)
            )
            apply_isaac_rtx_global_settings(settings)
        except BaseException as exc:
            session.close_with_failure(exc)
            raise
    sys.argv = [sys.argv[0], *unknown]
    return session
