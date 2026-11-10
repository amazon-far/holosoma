"""Small API boundary shared by the Isaac Lab 2 and 3 backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from holosoma.config_types.simulator import SimEngineConfig


class IsaacLabBackend(ABC):
    """Native operations that changed between Isaac Lab releases.

    The frontend owns scene authoring, state registration, scheduling, and
    Holosoma conventions. The backend only translates the native API boundary.
    """

    major: int
    env_regex_ns: str

    @property
    def is_isaaclab3(self) -> bool:
        """Whether this adapter implements the Lab 3 API."""
        return self.major == 3

    @property
    def uses_kit(self) -> bool:
        """Whether scene and viewer operations run inside Isaac Sim Kit."""
        return True

    def configure_physics_scene(self, sim: Any, config: Any) -> None:
        """Apply engine scene settings before assets are authored."""

    @abstractmethod
    def mutable_ray_caster_class(self) -> type[Any]:
        """Return this API family's ray caster with fixed-shape scan updates."""

    def world_lidar_mount(self, name: str, position: Any, orientation: Any) -> tuple[str, Any]:
        """Author a fixed sensor frame under each environment."""
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import MultiMeshRayCasterCfg

        path = f"{self.env_regex_ns}/holosoma_lidar_{name}"
        sim_utils.create_prim(
            path.replace(self.env_regex_ns, "/World/envs/env_0", 1),
            prim_type="Xform",
            translation=tuple(position),
            orientation=self.config_quat_to_native(orientation),
        )
        return path, MultiMeshRayCasterCfg.OffsetCfg()

    @property
    def material_channels(self) -> tuple[str, ...]:
        return ("static_friction", "dynamic_friction", "restitution")

    def camera_depth(self, value: Any) -> Tensor:
        return self.tensor(value)

    @abstractmethod
    def tensor(self, value: Any) -> Tensor:
        """Return a zero-copy Torch view of a native data field."""

    @abstractmethod
    def native_tensor(self, value: Any) -> Tensor:
        """Return a Torch view of a raw physics view's buffer."""

    @abstractmethod
    def config_quat_to_native(self, quat: Sequence[float]) -> tuple[float, float, float, float]:
        """Translate WXYZ scene or sensor configuration to the native convention."""

    @abstractmethod
    def config_quat_from_native(self, quat: Sequence[float]) -> tuple[float, float, float, float]:
        """Translate a native scene quaternion to the scalar-first USD convention."""

    @abstractmethod
    def quat_from_native(self, quat: Tensor) -> Tensor:
        """Translate native runtime quaternions to XYZW."""

    @abstractmethod
    def quat_to_native(self, quat: Tensor) -> Tensor:
        """Translate XYZW runtime quaternions to the native convention."""

    @property
    @abstractmethod
    def stage_utils(self) -> Any:
        """Native stage utility module."""

    @abstractmethod
    def add_launcher_args(self, parser: Any) -> None:
        """Add the runtime's supported launcher arguments."""

    @abstractmethod
    def launch_app(self, args: dict[str, Any] | Namespace, *, sim_config: Any = None) -> Any:
        """Launch and retain the native application's lifetime."""

    @abstractmethod
    def root_view(self, asset: Any) -> Any:
        """Return the native physics view for an articulation or rigid body."""

    @abstractmethod
    def physics_property(self, asset: Any, name: str) -> Tensor:
        """Read physics properties in public body order."""

    @abstractmethod
    def write_physics_property(self, asset: Any, name: str, value: Tensor, env_ids: Any) -> None:
        """Write selected environments and invalidate native property caches."""

    @abstractmethod
    def root_state(self, asset: Any) -> Tensor:
        """Read XYZW root link pose and root COM velocity."""

    @abstractmethod
    def default_root_pose(self, asset: Any) -> Tensor:
        """Read the default XYZW root link pose."""

    @abstractmethod
    def body_quat(self, asset: Any) -> Tensor:
        """Read XYZW body link orientations."""

    @abstractmethod
    def write_root_pose(self, asset: Any, pose: Tensor, env_ids: Any) -> None:
        """Write selected XYZW root link poses."""

    @abstractmethod
    def write_root_velocity(self, asset: Any, velocity: Tensor, env_ids: Any) -> None:
        """Write selected root COM velocities."""

    @abstractmethod
    def write_joint_position(self, asset: Any, value: Tensor, joint_ids: Any, env_ids: Any) -> None:
        """Write selected joint positions."""

    @abstractmethod
    def write_joint_velocity(self, asset: Any, value: Tensor, joint_ids: Any, env_ids: Any) -> None:
        """Write selected joint velocities."""

    @abstractmethod
    def set_joint_effort(self, asset: Any, value: Tensor, joint_ids: Any) -> None:
        """Set effort targets without clearing other joints."""

    @abstractmethod
    def applied_effort(self, asset: Any) -> Tensor:
        """Read applied actuator efforts."""

    @abstractmethod
    def contact_forces(self, sensor: Any, config: SimEngineConfig) -> Tensor:
        """Read normal or total contact forces according to configuration."""

    @abstractmethod
    def simulation_config(self, config: SimEngineConfig, device: str) -> Any:
        """Build the native simulation configuration."""

    @abstractmethod
    def clone_environments(self, scene: Any) -> None:
        """Replicate the authored source environment."""

    @abstractmethod
    def has_gui(self, sim: Any) -> bool:
        """Whether the native application has an interactive viewer."""

    @abstractmethod
    def is_rendering(self, sim: Any) -> bool:
        """Whether the native renderer needs scheduled simulation renders."""

    @abstractmethod
    def physics_time(self, sim: Any) -> float:
        """Return the physics clock in seconds."""

    @abstractmethod
    def forward_kinematics(self, sim: Any, asset: Any) -> None:
        """Refresh native kinematics after state writes without integrating."""

    @abstractmethod
    def write_external_wrench(self, asset: Any, wrench: Tensor, body_ids: Any, env_ids: Any) -> None:
        """Submit world-frame forces and torques for the selected bodies."""

    @abstractmethod
    def create_camera_cfg(self, *, offset_position: Any, offset_orientation: Any, **kwargs: Any) -> Any:
        """Build the runtime's optimized camera configuration."""

    @abstractmethod
    def create_camera(self, cfg: Any) -> Any:
        """Create the runtime's optimized camera."""

    def configure_camera_cadence(self, camera: Any, *, control_hz: float, decimation: int) -> None:
        """Translate the shared camera cadence when a native renderer needs it."""

    def refresh_cameras(self, cameras: list[Any], *, capture_step: int = 0) -> None:
        """Prepare due cameras for readout; Lab 2 reads its existing render pass."""

    def body_positions(self, asset: Any) -> Tensor:
        return self.tensor(asset.data.body_link_pos_w)

    def body_linear_velocity(self, asset: Any) -> Tensor:
        return self.tensor(asset.data.body_link_lin_vel_w)

    def body_angular_velocity(self, asset: Any) -> Tensor:
        return self.tensor(asset.data.body_com_ang_vel_w)

    def write_root_state(self, asset: Any, states: Tensor, env_ids: Any) -> None:
        """Write XYZW root link poses and COM velocities for the selected rows."""
        self.write_root_pose(asset, states[:, :7], env_ids)
        self.write_root_velocity(asset, states[:, 7:], env_ids)

    def has_viewport(self, sim: Any) -> bool:
        return self.has_gui(sim)

    def render(self, sim: Any) -> None:
        sim.render()

    def prepare_scene(self, scene: Any) -> None:
        """Prepare cloning before assets are authored."""

    def stage_context(self, sim: Any) -> AbstractContextManager[None]:
        """Select the simulation stage while authoring or initializing assets."""
        return nullcontext()

    def register_scene(self, sim: Any, scene: Any) -> None:
        """Publish a constructed scene to native renderer and visualizer consumers."""

    def write_scene_data(self, sim: Any, scene: Any) -> None:
        """Flush staged scene writes to the native simulation."""
        scene.write_data_to_sim()

    def restore_initial_articulation_state(self, asset: Any, env_origins: Tensor) -> None:
        """Restore configured state when native physics initialization advances it."""

    def create_viewport_controller(self, frontend: Any, viewer_cfg: Any) -> Any:
        """Create the frontend's camera controller when an interactive view exists."""
        if not self.has_viewport(frontend.sim):
            return None
        from holosoma.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController

        return ViewportCameraController(frontend, viewer_cfg)
