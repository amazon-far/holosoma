"""Isaac Sim 5.1 native API through Isaac Lab 2.3."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from .base import IsaacSimBackendAdapter


class IsaacSim5Backend(IsaacSimBackendAdapter):
    isaaclab_major = 2
    env_regex_ns = "/World/envs/env_.*"

    @classmethod
    def mutable_ray_caster_class(cls) -> type[Any]:
        from .ray_casters import legacy_mutable_ray_caster_class

        return legacy_mutable_ray_caster_class()

    @staticmethod
    def tensor(value: Any) -> Any:
        return value

    @staticmethod
    def config_quat_to_native(quat: Any) -> tuple[float, float, float, float]:
        """Config sensor/scene quaternions are WXYZ."""
        return tuple(quat)

    @staticmethod
    def config_quat_from_native(quat: Any) -> tuple[float, float, float, float]:
        """Lab 2 scene quaternions already match USD's WXYZ convention."""
        return tuple(quat)

    @staticmethod
    def quat_from_native(quat: Any) -> Any:
        from holosoma.simulator.isaacsim.state_utils import quat_wxyz_to_xyzw

        return quat_wxyz_to_xyzw(quat)

    @staticmethod
    def quat_to_native(quat: Any) -> Any:
        from holosoma.simulator.isaacsim.state_utils import quat_xyzw_to_wxyz

        return quat_xyzw_to_wxyz(quat)

    @property
    def stage_utils(self) -> Any:
        import isaacsim.core.utils.stage as stage_utils

        return stage_utils

    @staticmethod
    def add_launcher_args(parser: Any) -> None:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)

    @staticmethod
    def launch_app(args: dict[str, Any] | Namespace, *, sim_config: Any = None) -> Any:
        from isaaclab.app import AppLauncher

        return AppLauncher(args).app

    @staticmethod
    def root_view(asset: Any) -> Any:
        return asset.root_physx_view

    @staticmethod
    def native_tensor(value: Any) -> Any:
        """PhysX views return Torch tensors in Lab 2."""
        return value

    def physics_property(self, asset: Any, name: str) -> Any:
        return getattr(self.root_view(asset), f"get_{name}")()

    def write_physics_property(self, asset: Any, name: str, value: Any, env_ids: Any) -> None:
        getattr(self.root_view(asset), f"set_{name}")(value, env_ids)

    @staticmethod
    def root_state(asset: Any) -> Any:
        from holosoma.simulator.isaacsim.state_utils import fullstate_wxyz_to_xyzw

        return fullstate_wxyz_to_xyzw(asset.data.root_state_w)

    @staticmethod
    def default_root_pose(asset: Any) -> Any:
        return asset.data.default_root_state[..., [0, 1, 2, 4, 5, 6, 3]]

    @staticmethod
    def body_quat(asset: Any) -> Any:
        return asset.data.body_quat_w[..., [1, 2, 3, 0]]

    @staticmethod
    def write_root_pose(asset: Any, pose: Any, env_ids: Any) -> None:
        import torch

        native = torch.cat((pose[:, :3], pose[:, [6, 3, 4, 5]]), dim=-1)
        asset.write_root_pose_to_sim(native, env_ids)

    @staticmethod
    def write_root_velocity(asset: Any, velocity: Any, env_ids: Any) -> None:
        asset.write_root_velocity_to_sim(velocity, env_ids)

    @staticmethod
    def write_joint_position(asset: Any, value: Any, joint_ids: Any, env_ids: Any) -> None:
        asset.write_joint_position_to_sim(value, joint_ids=joint_ids, env_ids=env_ids)

    @staticmethod
    def write_joint_velocity(asset: Any, value: Any, joint_ids: Any, env_ids: Any) -> None:
        asset.write_joint_velocity_to_sim(value, joint_ids=joint_ids, env_ids=env_ids)

    @staticmethod
    def set_joint_effort(asset: Any, value: Any, joint_ids: Any) -> None:
        asset.set_joint_effort_target(value, joint_ids=joint_ids)

    @staticmethod
    def applied_effort(asset: Any) -> Any:
        return asset.data.applied_torque

    @staticmethod
    def contact_forces(sensor: Any, config: Any) -> Any:
        return sensor.data.net_forces_w

    @staticmethod
    def simulation_config(config: Any, device: str) -> Any:
        import isaaclab.sim as sim_utils

        return sim_utils.SimulationCfg(
            dt=1.0 / config.fps,
            render_interval=config.control_decimation_steps,
            device=device,
            physx=sim_utils.PhysxCfg(
                bounce_threshold_velocity=config.physx.bounce_threshold_velocity,
                solver_type=config.physx.solver_type,
                max_position_iteration_count=config.physx.num_position_iterations,
                max_velocity_iteration_count=config.physx.num_velocity_iterations,
                gpu_max_rigid_patch_count=10 * 2**15,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )

    @staticmethod
    def clone_environments(scene: Any) -> None:
        scene.clone_environments(copy_from_source=False)

    @staticmethod
    def has_gui(sim: Any) -> bool:
        return bool(sim.has_gui())

    @staticmethod
    def is_rendering(sim: Any) -> bool:
        return bool(sim.has_gui() or sim.has_rtx_sensors())

    @staticmethod
    def has_viewport(sim: Any) -> bool:
        return bool(sim.render_mode >= sim.RenderMode.PARTIAL_RENDERING)

    @staticmethod
    def physics_time(sim: Any) -> float:
        return float(sim.current_time)

    @staticmethod
    def forward_kinematics(sim: Any, asset: Any) -> None:
        sim.forward()
        # Lab 2 invalidates these cached velocities only on joint-position writes.
        asset.data._body_link_vel_w.timestamp = -1.0
        asset.data._body_com_vel_w.timestamp = -1.0

    @staticmethod
    def write_external_wrench(asset: Any, wrench: Any, body_ids: Any, env_ids: Any) -> None:
        from isaaclab.utils.math import quat_apply_inverse

        quat = asset.data.body_quat_w[:, body_ids, :]
        asset.set_external_force_and_torque(
            forces=quat_apply_inverse(quat, wrench[..., :3]),
            torques=quat_apply_inverse(quat, wrench[..., 3:]),
            env_ids=env_ids,
            body_ids=body_ids,
        )

    def create_camera_cfg(self, *, offset_position: Any, offset_orientation: Any, **kwargs: Any) -> Any:
        from isaaclab.sensors import TiledCameraCfg

        return TiledCameraCfg(
            offset=TiledCameraCfg.OffsetCfg(
                pos=offset_position, rot=self.config_quat_to_native(offset_orientation), convention="opengl"
            ),
            **kwargs,
        )

    @staticmethod
    def create_camera(cfg: Any) -> Any:
        from isaaclab.sensors import TiledCamera

        return TiledCamera(cfg)

    def refresh_cameras(self, cameras: list[Any]) -> None:
        """Extract requested products from the completed Kit render without advancing time."""
        for camera in cameras:
            camera.update(dt=0.0, force_recompute=True)
