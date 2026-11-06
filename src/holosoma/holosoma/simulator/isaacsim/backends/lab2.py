"""Isaac Sim 5.1 behavior, also the default for shared helper callers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import torch

    from holosoma.config_types.robot import RobotConfig
    from holosoma.config_types.sensor import CameraSensorConfig
    from holosoma.config_types.simulator import SimulatorInitConfig
    from holosoma.config_types.video import VideoConfig
    from holosoma.simulator.shared.video_recorder import VideoRecorderInterface


@dataclass
class ViewportConfig:
    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type: str = "world"
    env_index: int = 0
    asset_name: str | None = None
    body_name: str | None = None


class Lab2Backend:
    modern = False
    single_friction = False
    env_regex = "/World/envs/env_.*"

    def has_gui(self, sim: Any) -> bool:
        return bool(sim.has_gui())

    def needs_rendering(self, simulator: Any) -> bool:
        return bool(simulator.sim.has_gui() or simulator.sim.has_rtx_sensors())

    def physics_time(self, sim: Any) -> float:
        return float(sim.current_time)

    def prepare_recording_camera(self, simulator: Any) -> None:
        pass

    def config_quaternion(self, wxyz: Sequence[float]) -> tuple[float, float, float, float]:
        w, x, y, z = wxyz
        return w, x, y, z

    def quaternion_to_wxyz(self, quaternion: Sequence[float]) -> tuple[float, float, float, float]:
        w, x, y, z = quaternion
        return w, x, y, z

    def acquire_debug_draw(self) -> Any:
        from isaacsim.util.debug_draw import _debug_draw

        return _debug_draw.acquire_debug_draw_interface()

    def validate_friction(self, static: float, dynamic: float) -> None:
        if self.single_friction and static != dynamic:
            raise ValueError(
                "Newton has one friction coefficient. Set static_friction and dynamic_friction "
                "to the same value, or select PhysX."
            )

    def merge_fixed_joints(self, robot: RobotConfig, urdf_path: str) -> bool:
        return robot.asset.collapse_fixed_joints

    def urdf_spawn_config(self, config: Any, *, preserve_fixed_bodies: Sequence[str] = ()) -> Any:
        return config

    def convert_urdf(self, config: Any) -> str:
        from isaaclab.sim.converters import UrdfConverter

        path: str = UrdfConverter(config).usd_path
        return path

    def read_state(self, value: torch.Tensor) -> torch.Tensor:
        from ..state_utils import fullstate_wxyz_to_xyzw

        return fullstate_wxyz_to_xyzw(value)

    def write_state(self, value: torch.Tensor) -> torch.Tensor:
        from ..state_utils import fullstate_xyzw_to_wxyz

        return fullstate_xyzw_to_wxyz(value)

    @property
    def stage_utils(self) -> ModuleType:
        return import_module("isaacsim.core.utils.stage")

    @property
    def prim_utils(self) -> ModuleType:
        return import_module("isaacsim.core.utils.prims")

    def simulation_config(self, config: SimulatorInitConfig, device: str) -> Any:
        from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg, SimulationCfg

        physics = config.sim.physx
        return SimulationCfg(
            dt=1.0 / config.sim.fps,
            render_interval=config.sim.render_interval_steps,
            device=device,
            physx=PhysxCfg(
                bounce_threshold_velocity=physics.bounce_threshold_velocity,
                solver_type=physics.solver_type,
                max_position_iteration_count=physics.num_position_iterations,
                max_velocity_iteration_count=physics.num_velocity_iterations,
                gpu_max_rigid_patch_count=10 * 2**15,
            ),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )

    def clone(self, scene: Any, global_paths: list[str]) -> None:
        scene.clone_environments(copy_from_source=False)
        scene.filter_collisions(global_prim_paths=global_paths)

    def properties(self, asset: Any) -> Any:
        return asset.root_physx_view

    def camera_options(self, camera: CameraSensorConfig) -> dict[str, Any]:
        if camera.isaacsim.renderer.backend is not None or camera.isaacsim.renderer.options:
            raise ValueError("Per-camera native renderer configuration requires Isaac Lab 3.")
        return {}

    def sync_cameras(self, sim: Any) -> None:
        from ..sensor_setup import sync_camera_fabric_transforms

        sync_camera_fabric_transforms(sim)

    def create_viewport(self, sim: Any, config: ViewportConfig) -> Any:
        if sim.sim.render_mode < sim.sim.RenderMode.PARTIAL_RENDERING:
            return None
        from ..isaaclab_viewpoint_camera_controller import ViewportCameraController

        return ViewportCameraController(sim, config)

    def create_recorder(self, config: VideoConfig, sim: Any) -> VideoRecorderInterface:
        from ..video_recorder import IsaacSimVideoRecorder

        return IsaacSimVideoRecorder(config, sim)

    def rigid_object(self, config: Any) -> Any:
        from isaaclab.assets import RigidObject

        return RigidObject(config)

    def finish_state_write(self, sim: Any) -> None:
        pass
