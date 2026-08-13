"""GPU-accelerated batched MuJoCo backend using mujoco_warp.

This backend provides GPU-accelerated parallel simulation of multiple environments
using mujoco_warp, which leverages Warp's kernel compilation and batched execution.

Key features:
- GPU-accelerated simulation via Warp kernels
- Batched parallel environments (1 to thousands)
- Zero-copy PyTorch tensor access via wp.to_torch()
- Automatic contact force computation (cfrc_ext)
- Efficient GPU->CPU sync for rendering only when needed

Optional Dependencies
---------------------
This module requires optional dependencies that are NOT installed by default:
  - warp-lang: GPU kernel compilation framework
  - mujoco-warp: MuJoCo integration with Warp

To enable GPU acceleration, reinstall with warp support:
  bash scripts/setup_mujoco.sh --with-warp

Or install dependencies manually:
  pip install warp-lang mujoco-warp

System Requirements:
  - CUDA-capable GPU required
  - CUDA toolkit installed

If these dependencies are not available, the system will gracefully fall back
to ClassicBackend (CPU-only simulation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from loguru import logger

from .base import IMujocoBackend, holosoma_to_mj_quat, mj_to_holosoma_quat
from .warp_bridge import WarpBridge

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.simulator.mujoco.tensor_views import BaseMujocoView
    from holosoma.simulator.shared.camera_sensor import CameraRuntime


# Quaternion helpers on MuJoCo-convention (wxyz, scalar-first) quaternions, reimplemented locally so
# the WarpBackend does not depend on mujoco_warp's private `_src.math` module (whose internal path
# varies across mujoco_warp revisions and breaks the optional import — see backends/__init__.py).
# These are byte-for-byte equivalent to mujoco_warp's rot_vec_quat / quat_to_mat / mul_quat.
@wp.func
def _mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:
    return wp.quat(
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    )


@wp.func
def _rot_vec_quat(vec: wp.vec3, quat: wp.quat) -> wp.vec3:
    s = quat[0]
    u = wp.vec3(quat[1], quat[2], quat[3])
    return 2.0 * (wp.dot(u, vec) * u) + (s * s - wp.dot(u, u)) * vec + 2.0 * s * wp.cross(u, vec)


@wp.func
def _quat_to_mat(quat: wp.quat) -> wp.mat33:
    q00 = quat[0] * quat[0]
    q01 = quat[0] * quat[1]
    q02 = quat[0] * quat[2]
    q03 = quat[0] * quat[3]
    q11 = quat[1] * quat[1]
    q12 = quat[1] * quat[2]
    q13 = quat[1] * quat[3]
    q22 = quat[2] * quat[2]
    q23 = quat[2] * quat[3]
    q33 = quat[3] * quat[3]

    return wp.mat33(
        q00 + q11 - q22 - q33,
        2.0 * (q12 - q03),
        2.0 * (q13 + q02),
        2.0 * (q12 + q03),
        q00 - q11 + q22 - q33,
        2.0 * (q23 - q01),
        2.0 * (q13 - q02),
        2.0 * (q23 + q01),
        q00 - q11 - q22 + q33,
    )


@wp.kernel
def _static_geom_local_to_global(
    worlds: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    geom_ids: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    geom_bodyid: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    geom_pos: wp.array2d(dtype=wp.vec3),  # type: ignore[valid-type]
    geom_quat: wp.array2d(dtype=wp.quat),  # type: ignore[valid-type]
    xpos: wp.array2d(dtype=wp.vec3),  # type: ignore[valid-type]
    xquat: wp.array2d(dtype=wp.quat),  # type: ignore[valid-type]
    geom_xpos: wp.array2d(dtype=wp.vec3),  # type: ignore[valid-type]
    geom_xmat: wp.array2d(dtype=wp.mat33),  # type: ignore[valid-type]
):
    """Compose world geom poses for the given geoms and worlds.

    The composition forward kinematics applies to a geom, restricted to the given geoms/worlds,
    using the local wxyz quaternion helpers above.
    """
    # wp.tid() returns a 2-tuple for a 2D launch, but Warp's stub's first overload types a 0-arg
    # call as scalar int, so mypy reports "int not iterable" on the unpack (misc) and cannot type
    # wi/gi (has-type). Comment-only ignores keep this valid in the Warp DSL (which rejects cast()).
    wi, gi = wp.tid()  # type: ignore[misc]
    worldid = worlds[wi]  # type: ignore[has-type]
    geomid = geom_ids[gi]  # type: ignore[has-type]
    bodyid = geom_bodyid[geomid]
    bpos = xpos[worldid, bodyid]
    bquat = xquat[worldid, bodyid]
    geom_xpos[worldid, geomid] = bpos + _rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], bquat)
    geom_xmat[worldid, geomid] = _quat_to_mat(_mul_quat(bquat, geom_quat[worldid % geom_quat.shape[0], geomid]))


class WarpBackend(IMujocoBackend):
    """GPU-accelerated batched MuJoCo backend using mujoco_warp.

    This backend wraps mujoco_warp for GPU-accelerated parallel simulation
    of multiple environments. It provides zero-copy access to simulation
    state via PyTorch tensors that share memory with Warp arrays.

    Key characteristics:
    - Multi-environment support (1 to thousands)
    - GPU-based computation via Warp kernels
    - Automatic contact force computation (cfrc_ext tensor)
    - Zero-copy PyTorch tensor access
    - Efficient GPU->CPU sync only for rendering

    Requirements:
    - warp-lang package
    - mujoco_warp package
    - CUDA-capable GPU
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: FullSimConfig, device: str):
        """Initialize WarpBackend with GPU context and batched data.

        Parameters
        ----------
        model : mujoco.MjModel
            Compiled MuJoCo model
        data : mujoco.MjData
            MuJoCo data structure (used for CPU rendering only)
        config : FullSimConfig
            Full simulation configuration
        device : str
            Device string (e.g., 'cuda:0')

        Raises
        ------
        ImportError
            If warp or mujoco_warp packages are not installed
        RuntimeError
            If GPU initialization fails
        """
        super().__init__(model, data, config, device)

        # Import warp packages (fail fast if not available)
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as e:
            raise ImportError(
                "WarpBackend requires 'warp-lang' and 'mujoco_warp' packages. "
                "Install with: pip install warp-lang mujoco-warp"
            ) from e

        # Initialize Warp runtime
        wp.init()
        self.mjw_device = wp.get_device(device)

        logger.info(f"Initializing WarpBackend: {self.num_envs} envs on {device}")

        # Get memory allocation config
        warp_config = config.simulator.mujoco_warp
        nconmax_per_env = warp_config.nconmax_per_env
        njmax_per_env = warp_config.njmax_per_env

        # Auto-calculate njmax if not specified
        if njmax_per_env is None:
            njmax_per_env = max(nconmax_per_env * 6, model.nv * 4)

        logger.info(f"GPU memory allocation: nconmax={nconmax_per_env} per env, njmax={njmax_per_env} per env")

        # Create Warp model and batched data within GPU context
        with wp.ScopedDevice(self.mjw_device):
            # Upload model to GPU
            self.mjw_model = mjw.put_model(model)

            # Create bridge for tensor-like access to model fields (for randomization)
            self.warp_model_bridge = WarpBridge(self.mjw_model, nworld=self.num_envs)

            # Allocate batched data for parallel environments
            # Memory allocation strategy (following mujoco_warp API):
            # - nconmax: contacts per environment (not total across all environments)
            # - njmax: constraints per environment
            # - naconmax: total contacts across ALL environments (auto-calculated internally)
            self.mjw_data = mjw.make_data(
                model,
                nworld=self.num_envs,
                nconmax=nconmax_per_env,
                njmax=njmax_per_env,
            )

            # Create zero-copy PyTorch tensors tethered to Warp arrays
            # These tensors share memory with the Warp arrays - no copying!
            self.qpos_t = wp.to_torch(self.mjw_data.qpos)  # [num_envs, nq]
            self.qvel_t = wp.to_torch(self.mjw_data.qvel)  # [num_envs, nv]
            self.qacc_t = wp.to_torch(self.mjw_data.qacc)  # [num_envs, nv]
            self.ctrl_t = wp.to_torch(self.mjw_data.ctrl)  # [num_envs, nu]
            self.cfrc_t = wp.to_torch(self.mjw_data.cfrc_ext)  # [num_envs, nbody, 6]
            self.xfrc_applied_t = wp.to_torch(self.mjw_data.xfrc_applied)  # [num_envs, nbody, 6]

            # Rigid body state tensors (for zero-copy access during refresh_sim_tensors)
            self.xpos_t = wp.to_torch(self.mjw_data.xpos)  # [num_envs, nbody, 3] - positions
            self.xquat_t = wp.to_torch(self.mjw_data.xquat)  # [num_envs, nbody, 4] - orientations [w,x,y,z]
            self.cvel_t = wp.to_torch(self.mjw_data.cvel)  # [num_envs, nbody, 6] - velocities [ang(3), lin(3)]

        # Keep reference to CPU data for rendering (synced on demand)
        self.render_data = data

        # Expand the per-world model fields BEFORE capturing the step graph, so the graph (and the
        # collision kernels that read m.geom_friction / m.body_mass etc. per world) reference the
        # SAME per-world arrays the runtime writers update. expand_model_fields reallocates these
        # arrays; doing it after capture would leave the captured kernels reading the single-world
        # copy while domain-randomization / static-move writes hit the new (orphaned) array: the
        # value would read back changed but never affect the stepped dynamics. Idempotent, so the
        # later prepare_fields()/in-place writes never reallocate.
        #
        # The list = body_pos/body_quat (set_static_body_world_pose) PLUS every field a
        # @mujoco_required_field randomization term writes (geom_friction / body_mass / body_inertia
        # / body_ipos). prepare_manager_fields() runs AFTER prepare_sim()/capture in the managed
        # flow, so a DR field expanded only there would be orphaned from the captured graph; pinning
        # the full set here keeps runtime randomization affecting the live simulation on Warp.
        if self.num_envs > 1:
            from .randomization import expand_model_fields

            _PER_WORLD_FIELDS = [
                "body_pos",
                "body_quat",  # set_static_body_world_pose (kinematic relocation)
                "geom_friction",
                "geom_solref",
                "geom_solimp",  # material DR (friction + solver vectors)
                "body_mass",
                "body_inertia",
                "body_ipos",
                "dof_damping",  # mass/inertia/com/damping DR
                # cam_pos so a world-mount camera (worldbody child, cam_bodyid==0) can be placed at
                # each env's origin. The camlight kernel (inside the captured step graph) reads
                # cam_pos per world to compute cam_xpos, so it MUST be expanded before capture;
                # offset_world_cameras() then writes each env's origin post-capture (same pre-expand /
                # post-write pattern as body_pos + set_static_body_world_pose).
                "cam_pos",
            ]
            with wp.ScopedDevice(self.mjw_device):
                expand_model_fields(self.mjw_model, nworld=self.num_envs, fields_to_expand=_PER_WORLD_FIELDS)
            self.warp_model_bridge.clear_cache()

        # Capture simulation step as CUDA graph for optimal performance
        # This eliminates per-kernel launch overhead (~20-30 kernels per step)
        # and enables GPU pipelining, providing 5-10x speedup
        logger.info("Capturing CUDA graph for simulation step...")
        with wp.ScopedDevice(self.mjw_device):
            with wp.ScopedCapture() as capture:
                mjw.step(self.mjw_model, self.mjw_data)
            self.step_graph = capture.graph
        logger.info("CUDA graph captured successfully")

        logger.info(
            f"WarpBackend initialized: {model.nbody} bodies, {model.nq} qpos, {model.nv} qvel, {model.nu} actuators"
        )

    def initialize_state(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize GPU state from CPU data after construction.

        This syncs the initial state set by _set_robot_initial_state()
        and _set_initial_joint_angles() to GPU tensors, overriding the
        default qpos0 values from the MJCF model.

        Parameters
        ----------
        model : mujoco.MjModel
            MuJoCo model (for context)
        data : mujoco.MjData
            CPU MjData with initial state to copy to GPU
        """
        import mujoco_warp as mjw
        import numpy as np
        import warp as wp

        logger.info("Syncing initial state from CPU to GPU...")

        with wp.ScopedDevice(self.mjw_device):
            # Copy qpos and qvel from CPU to GPU for all environments
            # Tile the state across all environments
            qpos_cpu = np.tile(data.qpos, (self.num_envs, 1))
            qvel_cpu = np.tile(data.qvel, (self.num_envs, 1))

            wp.copy(self.mjw_data.qpos, wp.array(qpos_cpu, dtype=float))
            wp.copy(self.mjw_data.qvel, wp.array(qvel_cpu, dtype=float))

            # Compute forward kinematics to update derived quantities
            # (body positions, orientations, etc.)
            mjw.forward(self.mjw_model, self.mjw_data)

        logger.info("Initial state synced to GPU successfully")

    def step(self) -> None:
        """Advance batched simulation by one timestep using CUDA graph.

        Launches the pre-captured CUDA graph containing all simulation kernels.
        This eliminates per-kernel launch overhead (~20-30 kernels/step) and
        provides 5-10x speedup compared to individual kernel launches.

        Note: No explicit synchronization here - allows CPU-GPU pipelining.
        GPU work completes asynchronously while CPU prepares next frame.
        Synchronization happens only when needed (e.g., in get_render_data()).
        """
        import warp as wp

        with wp.ScopedDevice(self.mjw_device):
            wp.capture_launch(self.step_graph)
            # No wp.synchronize() - let GPU work in parallel with CPU

    def get_render_data(self, world_id: int = 0) -> mujoco.MjData:
        """Sync GPU data to CPU for rendering.

        Copies state from the specified environment from GPU to CPU for
        visualization. This is the only operation that requires GPU->CPU
        synchronization.

        CRITICAL: We must synchronize BEFORE copying GPU→CPU to ensure the
        GPU has completed all pending work. Without this, we would copy
        stale/incomplete data from a previous frame.

        Parameters
        ----------
        world_id : int, default=0
            Which environment to sync for visualization (0 to num_envs-1)

        Returns
        -------
        mujoco.MjData
            CPU MjData with state from the specified environment
        """
        import mujoco_warp as mjw
        import warp as wp

        # Validate world_id
        if world_id < 0 or world_id >= self.num_envs:
            logger.warning(f"Invalid world_id {world_id}, clamping to [0, {self.num_envs - 1}]")
            world_id = max(0, min(world_id, self.num_envs - 1))

        with wp.ScopedDevice(self.mjw_device):
            # CRITICAL: Synchronize GPU before copying to CPU
            # This ensures all GPU kernels have completed and data is ready
            wp.synchronize()

            # Now safe to copy GPU→CPU with guaranteed fresh data
            mjw.get_data_into(self.render_data, self.model, self.mjw_data, world_id=world_id)

        return self.render_data

    def create_renderers(self, cameras: list[CameraRuntime]) -> None:
        # Appearance flags are GLOBAL to the shared render context (not per-camera);
        # validate_camera_dict already rejected any cross-camera conflict, so
        # collecting each set (non-None) value is unambiguous regardless of order. None => default.
        render_options: dict = {}
        for cam in cameras:
            if cam.config.mujoco is None:
                continue
            for attr in ("use_shadows", "use_textures", "use_precomputed_rays"):
                value = getattr(cam.config.mujoco, attr)
                if value is not None:
                    render_options[attr] = value

        # Per-camera modality flags in ACTIVE-camera space. cam_active is left unset, so
        # create_render_context keeps every model camera active and the active index equals the
        # MuJoCo camera id. Resolve each compiled <camera> id here into this backend's own name->id
        # map (the shared CameraRuntime holds no handle). create_render_context allocates an output
        # buffer + write address for a (camera, modality) ONLY if its flag is True at creation
        # (io.py: rgb_adr/depth_adr stay -1 otherwise). render_cameras() can then only toggle these
        # flags DOWN to the due subset, not up, so this enables the UNION of every modality each
        # camera will ever produce. render_seg stays off: holosoma cameras only produce rgb/depth.
        ncam = self.model.ncam
        self._cam_ids: dict[str, int] = {}
        rgb_flags = [False] * ncam
        depth_flags = [False] * ncam
        for cam in cameras:
            name = cam.name
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if not 0 <= cam_id < ncam:
                raise RuntimeError(
                    f"Camera '{name}' not found in compiled model (mj_name2id -> {cam_id}); "
                    "expected a <camera> element with active id in [0, ncam)."
                )
            self._cam_ids[name] = cam_id
            if "rgb" in cam.config.data_types:
                rgb_flags[cam_id] = True
            if "depth" in cam.config.data_types:
                depth_flags[cam_id] = True

        with wp.ScopedDevice(self.mjw_device):
            self._render_context = mjw.create_render_context(
                self.model,
                nworld=self.num_envs,
                render_rgb=rgb_flags,
                render_depth=depth_flags,
                render_seg=False,
                **render_options,
            )
            # Preallocate per-camera output buffers (reused each render), keyed by active id
            # (== MuJoCo cam id, from self._cam_ids). Keyed by id, not a positional list, because
            # camera ids need not be a contiguous 0..n-1 range (the model may hold non-sensor
            # cameras), and only the modalities a camera actually produces get a buffer.
            self._render_rgb_out = {
                self._cam_ids[cam.name]: wp.zeros(
                    (self.num_envs, cam.config.height, cam.config.width), dtype=wp.vec3, device=self.mjw_device
                )
                for cam in cameras
                if "rgb" in cam.config.data_types
            }
            self._render_depth_out = {
                self._cam_ids[cam.name]: wp.zeros(
                    (self.num_envs, cam.config.height, cam.config.width), dtype=wp.float32, device=self.mjw_device
                )
                for cam in cameras
                if "depth" in cam.config.data_types
            }
            self._depth_far = max(c.config.far for c in cameras)

    def render_cameras(self, cameras: list[CameraRuntime]) -> None:
        with wp.ScopedDevice(self.mjw_device):
            # Ensure the step graph's writes to mjw_data are complete before rendering reads them.
            wp.synchronize()

            # Toggle the per-camera modality flags DOWN to just the cameras due this step. These are
            # device wp.array("ncam", bool) kernel inputs (NOT wp.static), so mutating their contents
            # here (outside any graph capture) makes mjw.render render a different subset with no
            # recompile. wp.array has no item/slice assignment, so build host masks and .assign().
            # Only modalities the context was CREATED with may be enabled (create_renderers enabled
            # the union); enabling one that was False at creation would write to an unallocated (-1)
            # address. Cameras absent from `due` keep their last buffer below; render() clears the
            # shared output every call (render.py), so a non-due camera's slice is wiped, not stale.
            ncam = self.model.ncam
            rgb_mask = np.zeros(ncam, dtype=bool)
            depth_mask = np.zeros(ncam, dtype=bool)
            for cam in cameras:
                cam_id = self._cam_ids[cam.name]
                if "rgb" in cam.config.data_types:
                    rgb_mask[cam_id] = True
                if "depth" in cam.config.data_types:
                    depth_mask[cam_id] = True
            self._render_context.render_rgb.assign(rgb_mask)
            self._render_context.render_depth.assign(depth_mask)

            mjw.refit_bvh(self.mjw_model, self.mjw_data, self._render_context)
            mjw.render(self.mjw_model, self.mjw_data, self._render_context)

            for cam in cameras:
                cam_id = self._cam_ids[cam.name]
                if "rgb" in cam.config.data_types:
                    mjw.get_rgb(self._render_context, cam_id, self._render_rgb_out[cam_id])
                    rgb_f = wp.to_torch(self._render_rgb_out[cam_id])  # [N,H,W,3] float32 in [0,1]
                    cam.set_buffer("rgb", (rgb_f.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8))
                if "depth" in cam.config.data_types:
                    # mjw.get_depth writes clamp(val/depth_scale, 0, 1): a normalized [0,1] image,
                    # NOT meters. Pass depth_scale=depth_far so [0,far]->[0,1], then multiply back by
                    # far to recover meters across the full range. No-hit comes back 0 ->
                    # remap to +inf (the no-hit sentinel).
                    mjw.get_depth(self._render_context, cam_id, self._depth_far, self._render_depth_out[cam_id])
                    dep = wp.to_torch(self._render_depth_out[cam_id]) * self._depth_far  # [N,H,W] meters
                    dep = torch.where(dep <= 0, torch.full_like(dep, float("inf")), dep)
                    cam.set_buffer("depth", dep.unsqueeze(-1))  # [N,H,W,1]

    def get_ctrl_tensor(self) -> torch.Tensor:
        """Return control tensor for direct zero-copy writing.

        Returns
        -------
        torch.Tensor
            Control tensor [num_envs, nu] sharing memory with Warp array
        """
        return self.ctrl_t

    def compute_contact_forces(self) -> torch.Tensor:
        """Return net per-body contact forces for ALL model bodies (GPU, zero-copy).

        Warp computes contact forces into cfrc_ext during simulation. Returns the
        full-model-width force slice [num_envs, model.nbody, 3]; the simulator
        gathers robot-only rows and rotates the history.

        Returns
        -------
        torch.Tensor
            Contact forces [num_envs, model.nbody, 3] (cfrc_ext force components).
        """
        # cfrc_ext is [num_envs, model.nbody, 6]; take first 3 (forces, drop torque).
        return self.cfrc_t[..., :3]

    def create_root_view(self, addrs: dict) -> BaseMujocoView:
        """Create root state view using zero-copy tensors.

        Parameters
        ----------
        addrs : dict
            Address dictionary with slices for pos, quat, vel, ang_vel

        Returns
        -------
        MjwRootStateView
            Root state view with quaternion conversion and zero-copy access
        """
        from holosoma.simulator.mujoco.mjw_views import MjwRootStateView

        return MjwRootStateView(
            qpos=self.qpos_t,
            qvel=self.qvel_t,
            pos_slice=addrs["pos_indices"],
            quat_slice=addrs["quat_indices"],
            vel_slice=addrs["vel_indices"],
            ang_vel_slice=addrs["ang_vel_indices"],
            num_envs=self.num_envs,
        )

    def create_dof_pos_view(self, indices: slice, num_dof: int) -> torch.Tensor:
        """Return DOF position tensor directly (no wrapper needed).

        wp.to_torch() already returns a native PyTorch tensor that:
        - Shares memory with Warp array (zero-copy)
        - Supports all PyTorch operations natively
        - Works seamlessly on GPU

        Parameters
        ----------
        indices : slice
            Slice into qpos array
        num_dof : int
            Number of degrees of freedom (unused, for interface compatibility)

        Returns
        -------
        torch.Tensor
            DOF positions [num_envs, num_dof] - native PyTorch tensor
        """
        return self.qpos_t[:, indices]

    def create_dof_vel_view(self, indices: slice, num_dof: int) -> torch.Tensor:
        """Return DOF velocity tensor directly (no wrapper needed).

        wp.to_torch() already returns a native PyTorch tensor that:
        - Shares memory with Warp array (zero-copy)
        - Supports all PyTorch operations natively
        - Works seamlessly on GPU

        Parameters
        ----------
        indices : slice
            Slice into qvel array
        num_dof : int
            Number of degrees of freedom (unused, for interface compatibility)

        Returns
        -------
        torch.Tensor
            DOF velocities [num_envs, num_dof] - native PyTorch tensor
        """
        return self.qvel_t[:, indices]

    def create_dof_acc_view(self, indices: slice, num_dof: int) -> torch.Tensor:
        """Return DOF acceleration tensor directly (no wrapper needed).

        wp.to_torch() already returns a native PyTorch tensor that:
        - Shares memory with Warp array (zero-copy)
        - Supports all PyTorch operations natively
        - Works seamlessly on GPU

        Parameters
        ----------
        indices : slice
            Slice into qacc array
        num_dof : int
            Number of degrees of freedom (unused, for interface compatibility)

        Returns
        -------
        torch.Tensor
            DOF accelerations [num_envs, num_dof] - native PyTorch tensor
        """
        return self.qacc_t[:, indices]

    def create_dof_state_view(self, dof_addrs: dict, num_dof: int) -> BaseMujocoView:
        """Create DOF state view using zero-copy GPU tensors.

        Parameters
        ----------
        dof_addrs : dict
            Dictionary with 'dof_pos_indices' and 'dof_vel_indices' slices
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        MjwDofStateView
            DOF state view with IsaacGym flattened format [num_envs * num_dof, 2]
        """
        from holosoma.simulator.mujoco.mjw_views import MjwDofStateView

        return MjwDofStateView(
            qpos=self.qpos_t,
            qvel=self.qvel_t,
            dof_pos_indices=dof_addrs["dof_pos_indices"],
            dof_vel_indices=dof_addrs["dof_vel_indices"],
            num_envs=self.num_envs,
            num_dof=num_dof,
        )

    def get_applied_forces_view(self) -> torch.Tensor:
        """Get writable view for external applied forces (GPU tensor).

        Returns zero-copy PyTorch tensor for applying external forces and
        torques to bodies directly on the GPU.

        Returns
        -------
        torch.Tensor
            Writable GPU tensor [num_envs, num_bodies, 6]
            - [:, :, 0:3] = forces [fx, fy, fz]
            - [:, :, 3:6] = torques [tx, ty, tz]
        """
        return self.xfrc_applied_t

    def create_quaternion_view(self, quat_slice: slice):
        """Create quaternion view with format conversion.

        Parameters
        ----------
        quat_slice : slice
            Slice for extracting quaternion from qpos

        Returns
        -------
        MjwQuaternionView
            Quaternion view with [w,x,y,z] -> [x,y,z,w] conversion
        """
        from holosoma.simulator.mujoco.mjw_views import MjwQuaternionView

        return MjwQuaternionView(qpos=self.qpos_t, quat_slice=quat_slice, num_envs=self.num_envs)

    def create_angular_velocity_view(self, ang_vel_slice: slice):
        """Create angular velocity view with proper reshaping.

        Parameters
        ----------
        ang_vel_slice : slice
            Slice for extracting angular velocity from qvel

        Returns
        -------
        MjwAngularVelocityView
            Angular velocity view with proper multi-env access
        """
        from holosoma.simulator.mujoco.mjw_views import MjwAngularVelocityView

        return MjwAngularVelocityView(qvel=self.qvel_t, ang_vel_slice=ang_vel_slice, num_envs=self.num_envs)

    def get_rigid_body_state_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get zero-copy views of rigid body states from GPU.

        Returns native PyTorch tensors that share memory with Warp arrays,
        eliminating the need for CPU↔GPU synchronization or tensor allocation
        during refresh_sim_tensors().

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (positions, orientations, linear_vel, angular_vel):
            - positions: [num_envs, num_bodies, 3] - body positions
            - orientations: [num_envs, num_bodies, 4] - quaternions in [x,y,z,w] format
            - linear_vel: [num_envs, num_bodies, 3] - linear velocities
            - angular_vel: [num_envs, num_bodies, 3] - angular velocities
        """
        # Position: already in correct format
        positions = self.xpos_t  # [N, nbody, 3]

        # Orientation: convert MuJoCo [w,x,y,z] → holosoma [x,y,z,w]
        orientations = mj_to_holosoma_quat(self.xquat_t)  # [N, nbody, 4]

        # Velocities: split cvel [angular(3), linear(3)]
        angular_vel = self.cvel_t[..., 0:3]  # [N, nbody, 3]
        linear_vel = self.cvel_t[..., 3:6]  # [N, nbody, 3]

        return positions, orientations, linear_vel, angular_vel

    def set_root_state(self, env_ids: torch.Tensor, root_states: torch.Tensor, root_addrs: dict) -> None:
        """Set robot root states. The robot root is an actor freejoint, so this
        delegates to set_actor_state at the robot's qpos/qvel addresses.

        root_states is [num_selected_envs, 13] in holosoma format; root_addrs carries
        'robot_qpos_addr' and 'robot_qvel_addr'.
        """
        self.set_actor_state(env_ids, root_states, root_addrs["robot_qpos_addr"], root_addrs["robot_qvel_addr"])

    def set_dof_state(self, env_ids: torch.Tensor, dof_states: torch.Tensor, dof_addrs: dict) -> None:
        """Set DOF states via direct GPU tensor writes.

        Writes DOF positions and velocities directly to GPU tensors
        without CPU roundtrip. Supports batched updates efficiently.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update [num_selected_envs]
        dof_states : torch.Tensor
            DOF states [num_all_envs * num_dofs, 2] in IsaacGym format
            where [:, 0] = positions, [:, 1] = velocities
            NOTE: Contains states for ALL environments, we select based on env_ids
        dof_addrs : dict
            Address dictionary with 'dof_qpos_addrs' and 'dof_qvel_addrs' lists
        """
        # Parse addresses
        qpos_addrs = dof_addrs["dof_qpos_addrs"]
        qvel_addrs = dof_addrs["dof_qvel_addrs"]
        num_dof = len(qpos_addrs)

        # Vectorized selection: extract only rows for specified env_ids
        # dof_states is [num_all_envs * num_dof, 2], need [len(env_ids) * num_dof, 2]
        offsets = env_ids.unsqueeze(1) * num_dof  # [len(env_ids), 1]
        dof_offsets = torch.arange(num_dof, device=env_ids.device).unsqueeze(0)  # [1, num_dof]
        indices = (offsets + dof_offsets).flatten()  # [len(env_ids) * num_dof]

        selected_dof_states = dof_states[indices]  # [len(env_ids) * num_dof, 2]

        # Reshape from flattened IsaacGym format
        dof_pos = selected_dof_states[:, 0].view(len(env_ids), num_dof)  # [num_selected_envs, num_dof]
        dof_vel = selected_dof_states[:, 1].view(len(env_ids), num_dof)  # [num_selected_envs, num_dof]

        # Vectorized GPU tensor writes using explicit expand for shape matching
        N = len(env_ids)

        # Convert address lists to tensors
        qpos_indices = torch.tensor(qpos_addrs, device=env_ids.device)  # [num_dof]
        qvel_indices = torch.tensor(qvel_addrs, device=env_ids.device)  # [num_dof]

        # Explicit expand to [N, num_dof] for both indices
        env_idx = env_ids.unsqueeze(1).expand(N, num_dof)  # [N, num_dof]
        qpos_idx = qpos_indices.unsqueeze(0).expand(N, num_dof)  # [N, num_dof]
        qvel_idx = qvel_indices.unsqueeze(0).expand(N, num_dof)  # [N, num_dof]

        self.qpos_t[env_idx, qpos_idx] = dof_pos
        self.qvel_t[env_idx, qvel_idx] = dof_vel

        # No mj_forward call - next step() will handle forward kinematics

    def _batched_slice(self, env_ids: torch.Tensor, addr: int, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(row, col) advanced-index tensors selecting [env_ids, addr:addr+size] as [N, size]."""
        n = len(env_ids)
        rows = env_ids.unsqueeze(1).expand(n, size)
        cols = (torch.arange(size, device=env_ids.device) + addr).unsqueeze(0).expand(n, size)
        return rows, cols

    def get_actor_state(self, env_ids: torch.Tensor, qpos_addr: int, qvel_addr: int) -> torch.Tensor:
        """Read a per-env actor freejoint state from GPU storage (zero stale snapshot)."""
        pose = self.qpos_t[self._batched_slice(env_ids, qpos_addr, 7)]  # [N, 7] = [xyz, qw,qx,qy,qz]
        vel = self.qvel_t[self._batched_slice(env_ids, qvel_addr, 6)]  # [N, 6] = [v(3), w(3)] body-local ang
        quat_holo = mj_to_holosoma_quat(pose[:, 3:7])
        return torch.cat([pose[:, :3], quat_holo, vel], dim=1)  # [N, 13]

    def set_actor_state(self, env_ids: torch.Tensor, states: torch.Tensor, qpos_addr: int, qvel_addr: int) -> None:
        """Write a per-env actor freejoint state to GPU storage (not a no-op on GPU)."""
        # qpos freejoint slice is [pos(3), quat_mj(4)]; qvel is [lin(3), ang(3)], contiguous.
        pose = torch.cat([states[:, :3], holosoma_to_mj_quat(states[:, 3:7])], dim=1)  # [N, 7]
        self.qpos_t[self._batched_slice(env_ids, qpos_addr, 7)] = pose
        self.qvel_t[self._batched_slice(env_ids, qvel_addr, 6)] = states[:, 7:13]
        # No mj_forward call - next step() will handle forward kinematics

    def set_static_body_world_pose(
        self,
        body_ids: list[int],
        positions: torch.Tensor,
        quats: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Place welded (jointless) bodies at a per-env world pose.

        Welded bodies have no qpos slice; their world pose lives in the model's ``body_pos`` /
        ``body_quat``, expanded per-world (the mechanism randomization uses). ``positions`` is
        ``[len(env_ids), len(body_ids), 3]``; ``quats`` (xyzw) the matching ``[..., 4]`` or
        ``None`` to leave orientation as compiled; ``env_ids`` selects the worlds (default: all).

        ``body_pos``/``body_quat`` are expanded per-world in ``__init__`` before graph capture, so
        the writes here are in-place into the arrays the captured graph references.
        ``mjw.forward`` runs outside the graph, into the same data the graph uses, so the next
        ``step()`` sees the new pose. Forward kinematics does not refresh a welded geom's collision
        pose, so :meth:`_sync_static_geom_xpos` writes ``geom_xpos``/``geom_xmat`` for these bodies.
        """
        import warp as wp

        from .randomization import expand_model_fields

        fields = ["body_pos"] + (["body_quat"] if quats is not None else [])
        expand_model_fields(self.mjw_model, nworld=self.num_envs, fields_to_expand=fields)
        self.warp_model_bridge.clear_cache()

        body_pos = wp.to_torch(self.mjw_model.body_pos)  # [num_envs, nbody, 3], shares GPU memory
        if env_ids is None:
            body_pos[:, body_ids, :] = positions.to(body_pos.dtype)
        else:
            # Advanced index the selected worlds: env_ids[:, None] broadcasts against body_ids.
            body_pos[env_ids.to(body_pos.device)[:, None], body_ids, :] = positions.to(body_pos.dtype)
        if quats is not None:
            body_quat = wp.to_torch(self.mjw_model.body_quat)  # [num_envs, nbody, 4] wxyz
            # quats is a torch.Tensor here, but holosoma_to_mj_quat's return widens to the np|torch
            # union (only mypy sees the union; when torch is unstubbed it is Any and this is a no-op).
            quat_mj = holosoma_to_mj_quat(quats).to(body_quat.dtype)  # type: ignore[union-attr]
            if env_ids is None:
                body_quat[:, body_ids, :] = quat_mj
            else:
                body_quat[env_ids.to(body_quat.device)[:, None], body_ids, :] = quat_mj
        with wp.ScopedDevice(self.mjw_device):
            import mujoco_warp as mjw

            mjw.forward(self.mjw_model, self.mjw_data)  # refresh xpos/xquat from the new body_pos/quat

        self._sync_static_geom_xpos(body_ids, env_ids)

    def offset_world_cameras(self, cam_ids: list[int], env_origins: torch.Tensor) -> None:
        """Shift world-mount cameras' per-world ``cam_pos`` by each env's origin.

        A ``target_kind="world"`` camera is a worldbody child (``cam_bodyid==0``): unlike a
        body-attached camera, its pose does not inherit the per-env body offset, so in a multi-env
        run every world's camera would otherwise sit at the shared spec pose and see only env 0's
        scene. ``cam_pos`` is expanded per world in ``__init__`` (in ``_PER_WORLD_FIELDS``, before the
        step-graph capture), and the ``camlight`` kernel that computes ``cam_xpos`` reads it per world
        each step, so adding ``env_origins`` here — after capture — reaches the live render exactly
        like ``set_static_body_world_pose`` writes ``body_pos``. No-op at ``num_envs==1``.

        ``cam_ids`` are compiled MuJoCo camera ids (world-mount only); ``env_origins`` is
        ``[num_envs, 3]``.
        """
        if self.num_envs <= 1 or not cam_ids:
            return
        import warp as wp

        cam_pos = wp.to_torch(self.mjw_model.cam_pos)  # [num_envs, ncam, 3], shares GPU memory
        origins = env_origins.to(cam_pos.device, cam_pos.dtype)  # [num_envs, 3]
        for cam_id in cam_ids:
            cam_pos[:, cam_id, :] += origins
        with wp.ScopedDevice(self.mjw_device):
            import mujoco_warp as mjw

            mjw.forward(self.mjw_model, self.mjw_data)  # recompute cam_xpos from the shifted cam_pos

    def _sync_static_geom_xpos(self, body_ids: list[int], env_ids: torch.Tensor | None) -> None:
        """Refresh ``geom_xpos``/``geom_xmat`` for the geoms of welded ``body_ids``, per world.

        Forward kinematics skips world-static geoms, so it never recomposes their collision pose
        from a moved body. This launches the same composition over just these geoms, reading the
        per-env ``xpos``/``xquat`` forwarded above.
        """
        import warp as wp

        geom_ids = self._geom_ids_for_bodies(body_ids)
        if geom_ids.shape[0] == 0:  # type: ignore[attr-defined]
            return
        worlds = (
            wp.from_torch(torch.arange(self.num_envs, dtype=torch.int32, device=self.device), dtype=wp.int32)
            if env_ids is None
            else wp.from_torch(env_ids.to(self.device, torch.int32).contiguous(), dtype=wp.int32)
        )
        m, d = self.mjw_model, self.mjw_data
        with wp.ScopedDevice(self.mjw_device):
            wp.launch(
                _static_geom_local_to_global,
                dim=(worlds.shape[0], geom_ids.shape[0]),  # type: ignore[attr-defined]
                inputs=[worlds, geom_ids, m.geom_bodyid, m.geom_pos, m.geom_quat, d.xpos, d.xquat],
                outputs=[d.geom_xpos, d.geom_xmat],
            )

    def _geom_ids_for_bodies(self, body_ids: list[int]) -> object:
        """Warp ``int32`` array of every geom id owned by ``body_ids`` (cached; model is static)."""
        import warp as wp

        cache: dict[tuple[int, ...], object] | None = getattr(self, "_static_geom_id_cache", None)
        if cache is None:
            cache = {}
            self._static_geom_id_cache = cache
        key = tuple(body_ids)
        if key not in cache:
            ids: list[int] = []
            for body_id in body_ids:
                adr = int(self.model.body_geomadr[body_id])
                ids.extend(range(adr, adr + int(self.model.body_geomnum[body_id])))
            with wp.ScopedDevice(self.mjw_device):
                cache[key] = wp.array(ids, dtype=wp.int32)
        return cache[key]
