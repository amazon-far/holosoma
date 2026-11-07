"""Isaac Lab 3.0 / Isaac Sim 6.2 native API.

See Isaac Lab's migrating.rst, how-to/cloning.rst and how-to/proxy_array.rst.
Lab 3 already uses Holosoma's XYZW convention and cached zero-copy Torch views.
"""

from __future__ import annotations

from contextlib import ExitStack
from argparse import Namespace
from typing import Any

from holosoma.config_types.simulator import IsaacLab3Config

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


class IsaacLab3Runtime:
    """Retain either runtime through Lab 3's supported launch context.

    ``launch_simulation`` owns the runtime and preserves failure status during
    shutdown. Closing its yielded physics config or Kit app directly bypasses
    that contract. The process ExitStack retains contexts when assertion
    harnesses keep only the constructed simulator.
    """

    def __init__(self, app: Any = None):
        self._app = app
        self._closed = False

    def close(self, *, exit_code: int = 0, **kwargs: Any) -> None:
        if self._closed:
            return
        self._closed = True
        if self._app is None:
            from isaaclab.sim import SimulationContext

            SimulationContext.clear_instance()
        _RUNTIME.__exit__(SystemExit, SystemExit(exit_code), None)

    def is_running(self) -> bool:
        if self._closed:
            return False
        if self._app is not None:
            return bool(self._app.is_running())
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        return sim is None or bool(sim.is_running())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)


class IsaacLab3Backend(IsaacLabBackend):
    major = 3
    env_regex_ns = "/World/envs/env_[^/]+"

    def __init__(self, native: IsaacLab3Config | None = None, *, rendering: bool = True):
        self.options = native if native is not None else IsaacLab3Config()
        self._rendering = rendering

    @property
    def uses_kit(self) -> bool:
        # A renderer preference does not require its runtime when no camera,
        # movie or viewer uses it. Warp and OVRTX are both native renderers.
        return self._rendering and self.options.renderer == "isaac_rtx"

    def mutable_ray_caster_class(self) -> type[Any]:
        from .ray_casters import (
            newton_mutable_ray_caster_class,
            ovphysx_mutable_ray_caster_class,
            physx_mutable_ray_caster_class,
        )

        if self.options.physics_backend == "newton":
            factory = newton_mutable_ray_caster_class
        elif self.uses_kit:
            factory = physx_mutable_ray_caster_class
        else:
            factory = ovphysx_mutable_ray_caster_class
        return factory()

    def world_lidar_mount(self, name: str, position: Any, orientation: Any) -> tuple[str, Any]:
        if self.options.physics_backend != "newton":
            return super().world_lidar_mount(name, position, orientation)
        from isaaclab.sensors import MultiMeshRayCasterCfg

        # Newton's supported env-root path registers a bodyless site per world.
        # An arbitrary child Xform is interpreted as a physics body by its
        # legacy mesh caster. Put the fixed mount in the ray offset instead.
        return self.env_regex_ns, MultiMeshRayCasterCfg.OffsetCfg(
            pos=tuple(position), rot=self.config_quat_to_native(orientation)
        )

    @staticmethod
    def tensor(value: Any) -> Any:
        return value.torch

    @staticmethod
    def config_quat_to_native(quat: Any) -> tuple[float, float, float, float]:
        """Config sensor/scene quaternions are WXYZ; Lab 3 takes XYZW."""
        import numpy as np
        from isaaclab.utils.math import convert_quat

        # NumPy scalars make downstream torch.tensor infer float64, while
        # native ray patterns use float32. Config values must be Python floats.
        return tuple(convert_quat(np.asarray(quat), to="xyzw").tolist())

    @staticmethod
    def config_quat_from_native(quat: Any) -> tuple[float, float, float, float]:
        """Convert Lab 3's XYZW scene quaternions to USD's WXYZ convention."""
        import numpy as np
        from isaaclab.utils.math import convert_quat

        return tuple(convert_quat(np.asarray(quat), to="wxyz").tolist())

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

    def add_launcher_args(self, parser: Any) -> None:
        if not self.uses_kit:
            from .kitless_importers import select_standalone_importers

            # The SDK's argument definitions import KitLauncher, which imports
            # the isaacsim root even when no Kit runtime will be launched.
            select_standalone_importers()
        from isaaclab.app import add_launcher_args

        add_launcher_args(parser)

    def launch_app(self, args: dict[str, Any] | Namespace, *, sim_config: Any = None) -> Any:
        if not self.uses_kit:
            from .kitless_importers import select_standalone_importers

            # Also cover callers that do not register CLI arguments first.
            select_standalone_importers()
        from isaaclab.app import launch_simulation

        args = vars(args) if isinstance(args, Namespace) else args
        args["require_kit"] = self.uses_kit
        # The declared Holosoma config owns physics selection, rather than an
        # additional SDK CLI override changing only the launcher's prototype.
        requested_physics = args.pop("physics", None)
        expected_physics = "newton_mjwarp" if self.options.physics_backend == "newton" else "isaacsim_physx"
        if self.options.physics_backend == "physx" and not self.uses_kit:
            expected_physics = "ovphysx"
        if requested_physics not in (None, expected_physics):
            raise ValueError(
                f"SDK --physics {requested_physics!r} conflicts with configured {expected_physics!r}; "
                "select --simulator.config.sim.isaaclab3.physics-backend instead."
            )
        # Holosoma's viewer contract predates Lab 3's independent --viz switch.
        # Native visualizers preserve the standalone runtime for Warp/OVRTX.
        if not args.get("headless", False) and args.get("visualizer") is None:
            args["visualizer"] = (
                "kit" if self.uses_kit else "newton_rtx" if self.options.renderer == "ovrtx" else "newton_gl"
            )
        # Only cameras/movies need a camera renderer. A physics-only run must
        # not initialize OVRTX (or any other unused renderer) at launch.
        renderer = self.renderer_config() if args.get("enable_cameras") or args.get("video") else None
        if not self.uses_kit:
            if args.get("rendering_mode") is not None or args.get("kit_args"):
                raise ValueError(
                    "This run is Kit-less; rendering_mode and kit_args configure the Kit renderer. "
                    "Leave them unset, or enable an Isaac RTX camera, movie or viewer."
                )
            # Movie capture uses native Camera sensors, avoiding the SDK's
            # video-only hook which would automatically start Kit.
            args["video"] = False
            launch_cfg = Namespace(physics=self.physics_config(sim_config), renderer=renderer)
            from isaaclab.app import validate_launch

            from .kitless_importers import register_standalone_schemas

            _scan, kit_sources = validate_launch(launch_cfg, args)
            if kit_sources:
                raise ValueError(
                    f"This run is Kit-less; remove Kit launch requirements: {', '.join(kit_sources)}. "
                    "Use renderer='isaac_rtx' with cameras, movies or a viewer when Kit is needed."
                )
            register_standalone_schemas()
            _RUNTIME.enter_context(launch_simulation(launch_cfg, args))
            return IsaacLab3Runtime()
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
        launch_cfg = Namespace(physics=self.physics_config(sim_config), renderer=renderer)
        _RUNTIME.enter_context(launch_simulation(launch_cfg, args))
        import omni.kit.app

        return IsaacLab3Runtime(omni.kit.app.get_app())

    def physics_config(self, config: Any = None) -> Any:
        """Build the explicitly selected physics engine's supported config."""
        if self.options.physics_backend == "newton":
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

            options = self.options.newton
            solver_kwargs = {}
            for source, target in (("nconmax", "nconmax"), ("njmax", "njmax"), ("solver_iterations", "iterations")):
                value = getattr(options, source)
                if value is not None:
                    solver_kwargs[target] = value
            return NewtonCfg(
                num_substeps=config.substeps if config is not None else 1,
                use_cuda_graph=options.use_cuda_graph,
                solver_cfg=MJWarpSolverCfg(**solver_kwargs),
                default_shape_cfg=NewtonShapeCfg(margin=options.default_shape_margin),
            )
        if not self.uses_kit:
            from isaaclab_ov.physics import OvPhysxCfg

            return OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
        from isaaclab_physx.physics import PhysxCfg

        if config is None:
            return PhysxCfg()
        return PhysxCfg(
            bounce_threshold_velocity=config.physx.bounce_threshold_velocity,
            solver_type=config.physx.solver_type,
            max_position_iteration_count=config.physx.num_position_iterations,
            max_velocity_iteration_count=config.physx.num_velocity_iterations,
            gpu_max_rigid_patch_count=10 * 2**15,
        )

    def configure_physics_scene(self, sim: Any, config: Any) -> None:
        if self.uses_kit or self.options.physics_backend != "physx":
            return
        from pxr import Sdf

        # OVPhysX consumes PhysX scene policy from USD; its config owns the
        # detached runtime and buffer capacities, rather than these attributes.
        prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        fields = (
            ("bounceThreshold", Sdf.ValueTypeNames.Float, config.physx.bounce_threshold_velocity),
            ("solverType", Sdf.ValueTypeNames.Token, "TGS" if config.physx.solver_type == 1 else "PGS"),
            ("maxPositionIterationCount", Sdf.ValueTypeNames.Int, config.physx.num_position_iterations),
            ("maxVelocityIterationCount", Sdf.ValueTypeNames.Int, config.physx.num_velocity_iterations),
        )
        for name, value_type, value in fields:
            prim.CreateAttribute(f"physxScene:{name}", value_type).Set(value)

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
        if self.options.physics_backend == "newton":
            # Newton aligns the CoM frame with the body frame. Its pose accessor
            # allocates dummy unit quaternions; DR only needs the actual XYZ view.
            data_names["coms"] = "body_com_pos_b"
        if name in data_names:
            # The public data views carry user body ordering. The native view
            # is ordered by PhysX and should only be used for shape materials.
            value = getattr(asset.data, data_names[name]).torch.cpu().clone()
            if name == "inertias" and asset.num_bodies == 1 and not hasattr(asset, "num_joints"):
                return value[:, 0]
            return value
        if self.options.physics_backend == "newton" and name == "material_properties":
            from .newton_properties import read_materials

            return read_materials(asset)
        if not self.uses_kit and name == "material_properties":
            return self.native_tensor(self.root_view(asset).get_attribute(self._ov_material_attribute(asset)))
        return self.native_tensor(getattr(self.root_view(asset), f"get_{name}")())

    @staticmethod
    def _ov_material_attribute(asset: Any) -> Any:
        from isaaclab_ov import tensor_types

        if hasattr(asset, "num_joints"):
            return tensor_types.SHAPE_FRICTION_AND_RESTITUTION
        return tensor_types.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION

    def write_physics_property(self, asset: Any, name: str, value: Any, env_ids: Any) -> None:
        import torch
        import warp as wp

        if self.options.physics_backend == "newton":
            from .newton_properties import write_body_property, write_materials

            if name == "material_properties":
                write_materials(asset, value, env_ids)
            else:
                write_body_property(asset, name, value, env_ids)
            return
        if name in ("masses", "coms", "inertias"):
            if name == "inertias" and value.ndim == 2:
                value = value.unsqueeze(1)
            # Native setters launch Warp kernels on the asset's device before
            # staging indices for PhysX. CPU Torch selectors cannot cross that
            # kernel boundary, unlike the raw material setter below.
            native_ids = wp.from_torch(env_ids.to(device=asset.device, dtype=torch.int64).contiguous(), dtype=wp.int64)
            if self.uses_kit:
                getattr(asset, f"set_{name}_index")(
                    **{name: value.to(asset.device)}, env_ids=native_ids, full_data=True
                )
            else:
                # OVPhysX's indexed asset setters consume selected rows, while
                # its raw material binding below requires the complete buffer.
                selected = value[env_ids.to(value.device)].to(asset.device)
                getattr(asset, f"set_{name}_index")(**{name: selected}, env_ids=native_ids)
        else:
            native_value = wp.from_torch(value.contiguous(), dtype=wp.float32)
            native_ids = wp.from_torch(env_ids.to(device="cpu", dtype=torch.int32).contiguous())
            if not self.uses_kit and name == "material_properties":
                self.root_view(asset).set_attribute(
                    self._ov_material_attribute(asset), native_value, indices=native_ids
                )
            else:
                getattr(self.root_view(asset), f"set_{name}")(native_value, native_ids)

    @staticmethod
    def restore_initial_articulation_state(asset: Any, env_origins: Any) -> None:
        """Set the experiment's initial state after native views are initialized.

        Kit PhysX's ``PhysxManager._warmup_and_create_views`` calls
        ``update_simulation(dt, 0.0)`` before and after creating tensor views.
        Those advance dynamics without incrementing SimulationContext's counter. Set
        defaults after reset and before startup randomization, as native
        examples do, so the first observation belongs to experiment time zero.
        """
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

    def simulation_config(self, config: Any, device: str) -> Any:
        from isaaclab.sim import SimulationCfg
        from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

        native = config.isaaclab3
        return SimulationCfg(
            dt=1.0 / config.fps,
            render_interval=config.render_interval_steps,
            device=device,
            use_fabric=native.use_fabric,
            create_stage_in_memory=native.create_stage_in_memory,
            use_newton_actuators=native.use_newton_actuators,
            physics=self.physics_config(config),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )

    @staticmethod
    def stage_context(sim: Any) -> Any:
        from isaaclab.sim import use_stage

        return use_stage(sim.stage)

    @staticmethod
    def register_scene(sim: Any, scene: Any) -> None:
        sim.register_interactive_scene(scene)

    def write_scene_data(self, sim: Any, scene: Any) -> None:
        super().write_scene_data(sim, scene)
        if self.options.physics_backend == "newton":
            # Root writers invalidate FK. Native mesh casters read body_q
            # directly, so an explicit flush must publish the authored pose
            # before a capture that does not first step physics. Newton
            # forward consumes only dirty articulations and advances no time.
            sim.forward()

    def prepare_scene(self, scene: Any) -> None:
        from isaaclab import cloner
        from isaaclab.assets import AssetBaseCfg
        from ..scene_fixtures import TERRAIN_PRIM_PATH

        # Holosoma owns a complete env subtree, including expanded scene-file
        # bodies, material bindings and lights. Declare that ownership before
        # authoring assets, so one native plan clones the whole subtree.
        if self.options.physics_backend == "newton":
            from isaaclab_newton.cloner import NewtonReplicateContext

            physics_context = NewtonReplicateContext
        elif self.uses_kit:
            from isaaclab_physx.cloner import PhysxReplicateContext

            physics_context = PhysxReplicateContext
        else:
            from isaaclab_ov.cloner import OvPhysxReplicateContext

            physics_context = OvPhysxReplicateContext
        contexts: tuple[Any, ...] = (cloner.UsdReplicateContext, physics_context)
        if self._rendering and self.options.renderer == "newton_warp" and self.options.physics_backend != "newton":
            from isaaclab_newton.cloner import NewtonReplicateContext

            # Warp's render-only Newton model must include the same scene subtree.
            contexts += (NewtonReplicateContext,)
        root = AssetBaseCfg(prim_path=scene.env_regex_ns, cloning_contexts=contexts)
        # Newton imports only declared clone-plan sources. The shared terrain
        # lives outside every env subtree and must be declared independently;
        # collision filtering alone does not import its geometry into Newton.
        ground = AssetBaseCfg(prim_path=TERRAIN_PRIM_PATH, cloning_contexts=contexts)
        cloner.clone_plan_from_env_0(scene.cfg.clone_cfg, (root, ground), scene.num_envs, scene.cfg.env_spacing)

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

        projection = getattr(kwargs.get("spawn"), "projection_type", "pinhole")
        if self.options.renderer == "newton_warp" and projection != "pinhole":
            raise ValueError(
                f"Warp cameras do not support Holosoma's USD {projection!r} projection. "
                "Choose renderer='isaac_rtx' for these fisheye models."
            )
        return CameraCfg(
            offset=CameraCfg.OffsetCfg(
                pos=offset_position, rot=self.config_quat_to_native(offset_orientation), convention="opengl"
            ),
            renderer_cfg=self.renderer_config(),
            **kwargs,
        )

    def renderer_config(self) -> Any:
        """Declare the camera renderer both at launch and at sensor construction."""
        if self.options.renderer == "ovrtx":
            from isaaclab_ov.renderers import OVRTXRendererCfg

            return OVRTXRendererCfg()
        if self.options.renderer == "newton_warp":
            from isaaclab_newton.renderers import NewtonWarpRendererCfg

            return NewtonWarpRendererCfg()
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        return IsaacRtxRendererCfg()

    def has_viewport(self, sim: Any) -> bool:
        # The native OVRTX visualizer owns its view; Kit's post-update camera
        # controller and keyboard subscriptions apply only to the Kit viewport.
        return self.uses_kit and super().has_viewport(sim)

    @property
    def material_channels(self) -> tuple[str, ...]:
        """Actual engine channels; Newton has one Coulomb friction coefficient."""
        if self.options.physics_backend == "newton":
            return ("static_friction", "restitution")
        return ("static_friction", "dynamic_friction", "restitution")

    def camera_depth(self, value: Any) -> Any:
        """Normalize native camera misses before Holosoma's shared depth policy."""
        depth = self.tensor(value)
        if self.options.renderer == "newton_warp":
            import torch

            return torch.where(depth == 0, torch.inf, depth)
        return depth

    @staticmethod
    def create_camera(cfg: Any) -> Any:
        from isaaclab.sensors import Camera

        return Camera(cfg)
