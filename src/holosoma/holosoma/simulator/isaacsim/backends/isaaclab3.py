"""Isaac Lab 3.0 / Isaac Sim 6.2 native API.

See Isaac Lab's migrating.rst, how-to/cloning.rst and how-to/proxy_array.rst.
Lab 3 already uses Holosoma's XYZW convention and cached zero-copy Torch views.
"""

from __future__ import annotations

from contextlib import ExitStack
from argparse import Namespace
from typing import Any

from .base import IsaacLabBackend


_RUNTIME = ExitStack()
"""Keep launched runtimes alive for the process, even when callers discard the app."""

# Lab 3 removed --rendering_mode. Preserve explicit Holosoma presets using the
# RTX settings from Lab 2's apps/rendering_modes/{performance,balanced,quality}.kit.
# These are native Kit settings, also exposed by IsaacRtxRendererGlobalSettingsCfg.
_RTX_PRESETS = {
    "/rtx/translucency/enabled": (False, False, True),
    "/rtx/reflections/enabled": (False, False, True),
    "/rtx/reflections/denoiser/enabled": (False, True, True),
    "/rtx/directLighting/sampledLighting/enabled": (False, True, True),
    "/rtx/sceneDb/ambientLightIntensity": (1.0, 1.0, 1.0),
    "/rtx/shadows/enabled": (True, True, True),
    "/rtx/indirectDiffuse/enabled": (False, False, True),
    "/rtx/indirectDiffuse/denoiser/enabled": (False, True, True),
    "/rtx/ambientOcclusion/enabled": (False, False, True),
    "/rtx/ambientOcclusion/denoiserMode": (1, 1, 0),
    "/rtx/raytracing/subpixel/mode": (0, 0, 1),
    "/rtx/raytracing/cached/enabled": (False, True, True),
    "/rtx-transient/dlssg/enabled": (False, False, False),
    "/rtx-transient/dldenoiser/enabled": (False, True, True),
    "/rtx/post/dlss/execMode": (0, 1, 2),
    "/rtx/pathtracing/maxSamplesPerLaunch": (1000000, 1000000, 1000000),
    "/rtx/viewTile/limit": (1000000, 1000000, 1000000),
}


class _LaunchedApp:
    """Keep the canonical launch context alive for Holosoma's app lifecycle."""

    isaaclab_major = 3

    def __init__(self, app: Any):
        self._app = app

    def close(self, *, exit_code: int = 0, **kwargs: Any) -> None:
        _RUNTIME.__exit__(SystemExit, SystemExit(exit_code), None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)


class IsaacLab3Backend(IsaacLabBackend):
    major = 3
    is_isaaclab3 = True
    env_regex_ns = "/World/envs/env_[^/]+"

    @staticmethod
    def tensor(value: Any) -> Any:
        return value.torch

    @staticmethod
    def config_quat_to_native(quat: Any) -> tuple[float, float, float, float]:
        """Config sensor/scene quaternions are WXYZ; Lab 3 takes XYZW."""
        return (quat[1], quat[2], quat[3], quat[0])

    @staticmethod
    def quat_from_native(quat: Any) -> Any:
        return quat

    @staticmethod
    def quat_to_native(quat: Any) -> Any:
        return quat

    @property
    def stage_utils(self) -> Any:
        from isaaclab.sim.utils import stage as stage_utils

        return stage_utils

    @staticmethod
    def add_launcher_args(parser: Any) -> None:
        from isaaclab.app import add_launcher_args

        add_launcher_args(parser)

    @staticmethod
    def launch_app(args: dict[str, Any] | Namespace) -> Any:
        from isaaclab.app import launch_simulation
        from isaaclab_physx.physics import PhysxCfg

        args = vars(args) if isinstance(args, Namespace) else args
        # Holosoma's viewer contract predates Lab 3's independent --viz switch.
        # Select the native Kit visualizer for interactive launches.
        if not args.get("headless", False) and args.get("visualizer") is None:
            args["visualizer"] = "kit"
        # Sim 6.2's deferred telemetry CPU-info startup can race native
        # interface initialization. Use Kit's supported serial startup mode.
        telemetry_setting = "/exts/omni.kit.telemetry/skipDeferredStartup"
        kit_args = args.get("kit_args", "")
        rendering_mode = args.pop("rendering_mode", None)
        if rendering_mode is not None:
            mode_index = ("performance", "balanced", "quality").index(rendering_mode)
            preset_args = [f"--{path}={str(values[mode_index]).lower()}" for path, values in _RTX_PRESETS.items()]
            if rendering_mode == "performance":
                preset_args.extend(
                    (
                        "--/rtx/directLighting/sampledLighting/denoisingTechnique=0",
                        "--/rtx/domeLight/upperLowerStrategy=3",
                    )
                )
            # Later caller arguments take precedence over preset defaults.
            kit_args = " ".join((*preset_args, kit_args)).strip()
        if telemetry_setting not in kit_args:
            kit_args = f"{kit_args} --{telemetry_setting}=true".strip()
        args["kit_args"] = kit_args
        _RUNTIME.enter_context(launch_simulation(PhysxCfg(), args))
        import omni.kit.app

        return _LaunchedApp(omni.kit.app.get_app())

    @staticmethod
    def root_view(asset: Any) -> Any:
        return asset.root_view

    @staticmethod
    def native_tensor(value: Any) -> Any:
        """PhysX native views return Warp arrays, distinct from data proxies."""
        import warp as wp

        return wp.to_torch(value)

    def physics_property(self, asset: Any, name: str) -> Any:
        data_names = {"masses": "body_mass", "coms": "body_com_pose_b", "inertias": "body_inertia"}
        if name in data_names:
            # The public data views carry user body ordering. The native view
            # is ordered by PhysX and should only be used for shape materials.
            value = getattr(asset.data, data_names[name]).torch.cpu().clone()
            if name == "inertias" and asset.num_bodies == 1 and not hasattr(asset, "num_joints"):
                return value[:, 0]
            return value
        return self.native_tensor(getattr(self.root_view(asset), f"get_{name}")())

    def write_physics_property(self, asset: Any, name: str, value: Any, env_ids: Any) -> None:
        import torch
        import warp as wp

        if name in ("masses", "coms", "inertias"):
            if name == "inertias" and value.ndim == 2:
                value = value.unsqueeze(1)
            # Native setters launch Warp kernels on the asset's device before
            # staging indices for PhysX. CPU Torch selectors cannot cross that
            # kernel boundary, unlike the raw material setter below.
            native_ids = wp.from_torch(env_ids.to(device=asset.device, dtype=torch.int64).contiguous(), dtype=wp.int64)
            getattr(asset, f"set_{name}_index")(**{name: value.to(asset.device)}, env_ids=native_ids, full_data=True)
        else:
            native_value = wp.from_torch(value.contiguous(), dtype=wp.float32)
            native_ids = wp.from_torch(env_ids.to(device="cpu", dtype=torch.int32).contiguous())
            getattr(self.root_view(asset), f"set_{name}")(native_value, native_ids)

    @staticmethod
    def restore_initial_articulation_state(asset: Any, env_origins: Any) -> None:
        """Restore configured state after PhysX startup integration."""
        root_pose = asset.data.default_root_pose.torch.clone()
        root_pose[:, :3] += env_origins
        asset.write_root_pose_to_sim_index(root_pose=root_pose)
        asset.write_root_velocity_to_sim_index(root_velocity=asset.data.default_root_vel.torch.clone())
        asset.write_joint_position_to_sim_index(position=asset.data.default_joint_pos.torch.clone())
        asset.write_joint_velocity_to_sim_index(velocity=asset.data.default_joint_vel.torch.clone())
        asset.reset()

    @staticmethod
    def root_state(asset: Any) -> Any:
        import torch

        return torch.cat((asset.data.root_link_pose_w.torch, asset.data.root_com_vel_w.torch), dim=-1)

    @staticmethod
    def default_root_pose(asset: Any) -> Any:
        return asset.data.default_root_pose.torch

    @staticmethod
    def body_quat(asset: Any) -> Any:
        return asset.data.body_link_quat_w.torch

    @staticmethod
    def write_root_pose(asset: Any, pose: Any, env_ids: Any) -> None:
        asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)

    @staticmethod
    def write_root_velocity(asset: Any, velocity: Any, env_ids: Any) -> None:
        asset.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)

    @staticmethod
    def write_joint_position(asset: Any, value: Any, joint_ids: Any, env_ids: Any) -> None:
        asset.write_joint_position_to_sim_index(position=value, joint_ids=joint_ids, env_ids=env_ids)

    @staticmethod
    def write_joint_velocity(asset: Any, value: Any, joint_ids: Any, env_ids: Any) -> None:
        asset.write_joint_velocity_to_sim_index(velocity=value, joint_ids=joint_ids, env_ids=env_ids)

    @staticmethod
    def set_joint_effort(asset: Any, value: Any, joint_ids: Any) -> None:
        asset.actuators.target_command.set_effort_index(value=value, joint_ids=joint_ids)

    @staticmethod
    def applied_effort(asset: Any) -> Any:
        return asset.actuators.applied_effort.torch

    @staticmethod
    def contact_forces(sensor: Any, config: Any) -> Any:
        # Lab 2's net_forces_w meant normal force. Sim 6.2 adds friction to
        # net_forces_w; keep the prior reward/termination contract by default.
        if config.isaaclab3.contact_force_mode == "total":
            return sensor.data.net_forces_w.torch
        return sensor.data.net_normal_forces_w.torch

    @staticmethod
    def simulation_config(config: Any, device: str) -> Any:
        from isaaclab.sim import SimulationCfg
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

        native = config.isaaclab3
        return SimulationCfg(
            dt=1.0 / config.fps,
            render_interval=config.render_interval_steps,
            device=device,
            use_fabric=native.use_fabric,
            create_stage_in_memory=native.create_stage_in_memory,
            use_newton_actuators=native.use_newton_actuators,
            physics=PhysxCfg(
                bounce_threshold_velocity=config.physx.bounce_threshold_velocity,
                solver_type=config.physx.solver_type,
                max_position_iteration_count=config.physx.num_position_iterations,
                max_velocity_iteration_count=config.physx.num_velocity_iterations,
                gpu_max_rigid_patch_count=10 * 2**15,
            ),
            physics_material=PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )

    @staticmethod
    def stage_context(sim: Any) -> Any:
        from isaaclab.sim import use_stage

        return use_stage(sim.stage)

    @staticmethod
    def register_scene(sim: Any, scene: Any) -> None:
        sim.register_interactive_scene(scene)

    @staticmethod
    def prepare_scene(scene: Any) -> None:
        from isaaclab import cloner
        from isaaclab.assets import AssetBaseCfg
        from isaaclab_physx.cloner import PhysxReplicateContext

        # Holosoma owns a complete env subtree, including expanded scene-file
        # bodies, material bindings and lights. Declare that ownership before
        # authoring assets, so one native plan clones the whole subtree.
        root = AssetBaseCfg(
            prim_path=scene.env_regex_ns,
            cloning_contexts=(cloner.UsdReplicateContext, PhysxReplicateContext),
        )
        cloner.clone_plan_from_env_0(scene.cfg.clone_cfg, (root,), scene.num_envs, scene.cfg.env_spacing)

    @staticmethod
    def clone_environments(scene: Any) -> None:
        from isaaclab import cloner

        cloner.replicate(scene.clone_plan, replicate_physics=scene.cfg.replicate_physics)

    @staticmethod
    def has_gui(sim: Any) -> bool:
        return bool(sim.has_gui)

    @staticmethod
    def is_rendering(sim: Any) -> bool:
        return bool(sim.is_rendering)

    @staticmethod
    def physics_time(sim: Any) -> float:
        # PhysxManager leaves the shared manager clock at zero in Lab 3.
        # The public counter advances only on SimulationContext.step, so
        # rendering does not change Holosoma's fixed-timestep physics clock.
        return float(sim.get_physics_step_count() * sim.get_physics_dt())

    @staticmethod
    def forward_kinematics(sim: Any, asset: Any) -> None:
        sim.forward()

    @staticmethod
    def write_external_wrench(asset: Any, wrench: Any, body_ids: Any, env_ids: Any) -> None:
        # Instantaneous composers are cleared by write_data_to_sim each step.
        # World-frame input avoids a quaternion rotation/allocation here, and
        # preserves forces other writers put in the permanent composer.
        asset.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=wrench[..., :3],
            torques=wrench[..., 3:],
            env_ids=env_ids,
            body_ids=body_ids,
            is_global=True,
        )

    def create_camera_cfg(self, *, offset_position: Any, offset_orientation: Any, **kwargs: Any) -> Any:
        from isaaclab.sensors import CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        return CameraCfg(
            offset=CameraCfg.OffsetCfg(
                pos=offset_position, rot=self.config_quat_to_native(offset_orientation), convention="opengl"
            ),
            renderer_cfg=IsaacRtxRendererCfg(),
            **kwargs,
        )

    @staticmethod
    def create_camera(cfg: Any) -> Any:
        from isaaclab.sensors import Camera

        return Camera(cfg)
