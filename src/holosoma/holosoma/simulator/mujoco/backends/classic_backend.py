"""CPU-based single-environment MuJoCo backend implementation.

This backend wraps the standard MuJoCo CPU simulation with manual contact
force extraction. It maintains backward compatibility with existing single-
environment simulation code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from loguru import logger

from .base import IMujocoBackend, holosoma_to_mj_quat, mj_to_holosoma_quat

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.config_types.sensor import CameraDataType
    from holosoma.simulator.mujoco.tensor_views import BaseMujocoView
    from holosoma.simulator.shared.camera_sensor import CameraRuntime


class ClassicBackend(IMujocoBackend):
    """CPU-based single-environment MuJoCo backend.

    This backend wraps the standard MuJoCo CPU simulation with manual contact
    force extraction. It maintains backward compatibility with existing single-
    environment simulation code.

    Key characteristics:
    - Single environment only (num_envs must be 1)
    - CPU-based computation
    - Manual contact force extraction via mj_contactForce
    - Numpy arrays with PyTorch tensor conversion
    - Compatible with existing tensor_views.py proxy system
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: FullSimConfig, device: str):
        """Initialize ClassicBackend with single-environment validation.

        Parameters
        ----------
        model : mujoco.MjModel
            Compiled MuJoCo model
        data : mujoco.MjData
            MuJoCo data structure (shared with frontend)
        config : FullSimConfig
            Full simulation configuration
        device : str
            Device string (typically 'cpu')

        Raises
        ------
        ValueError
            If num_envs > 1 (only single environment supported)
        """
        super().__init__(model, data, config, device)

        if self.num_envs > 1:
            raise ValueError(
                f"ClassicBackend only supports single environment, got {self.num_envs}. "
                f"Use WarpBackend (use_warp=True) for multi-environment simulation."
            )

        # Pre-allocate contact force tensor
        self._force_tensor = torch.zeros(1, model.nbody, 3, device=device)

        logger.info(f"ClassicBackend initialized: {model.nbody} bodies, device={device}")

    def step(self) -> None:
        """Advance simulation by one timestep using mj_step."""
        mujoco.mj_step(self.model, self.data)

    def get_render_data(self, world_id: int = 0) -> mujoco.MjData:
        """Return data for rendering (already on CPU).

        Parameters
        ----------
        world_id : int, default=0
            Ignored for ClassicBackend (single environment only)

        Returns
        -------
        mujoco.MjData
            The backend's data structure (no copy needed)
        """
        return self.data

    def create_renderers(self, cameras: list[CameraRuntime]) -> None:
        # Per-camera renderer (classic only), keyed by camera NAME:
        # mujoco.Renderer is RGB XOR depth, so depth cameras get a SEPARATE depth-enabled renderer
        # (per size); MuJoCo 3.x depth is already metric meters (image-plane), so no unit scaling.
        # Resolve each compiled <camera> id here (this backend's own name->id map); the shared
        # CameraRuntime holds no handle.
        self._cam_ids: dict[str, int] = {}
        self._mj_renderers: dict[str, dict[CameraDataType, mujoco.Renderer]] = {}

        for cam in cameras:
            name = cam.name
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id < 0:
                raise RuntimeError(f"Camera '{name}' not found in compiled model (expected a <camera> element).")
            self._cam_ids[name] = cam_id
            cam_renderers = self._mj_renderers[name] = {}
            for data_type in cam.config.data_types:
                renderer = mujoco.Renderer(self.model, height=cam.config.height, width=cam.config.width)
                cam_renderers[data_type] = renderer
                if data_type == "depth":
                    renderer.enable_depth_rendering()

    def render_cameras(self, cameras: list[CameraRuntime]) -> None:
        # per-world render via the size-shared mujoco.Renderer (num_envs==1).
        # No-hit reads the global far-clip plane (vis.map.zfar * extent, set across all cameras in
        # the simulator's _create_sensors); remap it to +inf. The far value comes back ~0.1% under
        # nominal (limited depth precision), so threshold at 0.99*far.
        far_clip = float(self.model.vis.map.zfar) * float(self.model.stat.extent)
        for runtime in cameras:
            name = runtime.name
            cam_id = self._cam_ids[name]
            for dt in runtime.config.data_types:
                renderer = self._mj_renderers[name][dt]
                frames = []
                for world_id in range(self.num_envs):
                    data = self.get_render_data(world_id=world_id)
                    renderer.update_scene(data, camera=cam_id)
                    frame = renderer.render()  # rgb: [H,W,3] uint8; depth: [H,W] float32 meters
                    t = torch.from_numpy((frame[..., None] if dt == "depth" else frame).copy())
                    if dt == "depth":
                        t = torch.where(t >= far_clip * 0.99, torch.full_like(t, float("inf")), t)
                    frames.append(t)
                runtime.set_buffer(dt, torch.stack(frames, dim=0).to(self.device))  # [N,H,W,C]

    def get_ctrl_tensor(self) -> None:
        """Classic backend doesn't support direct tensor writes.

        Returns
        -------
        None
            Indicates that torque application must use the loop-based method
        """
        return

    def compute_contact_forces(self) -> torch.Tensor:
        """Return net per-body contact forces for ALL model bodies (CPU extraction).

        Extracts contact forces from MuJoCo's contact system using mj_contactForce
        and accumulates them per body (Newton's 3rd law). Full-model-width
        [1, model.nbody, 3]; the simulator gathers robot-only rows and rotates the
        history. (Single environment for ClassicBackend.)

        Returns
        -------
        torch.Tensor
            Contact forces [1, model.nbody, 3].
        """
        # Reset force accumulator
        self._force_tensor.fill_(0.0)

        # Pre-allocate force/torque buffer for mj_contactForce
        forcetorque = np.zeros(6, dtype=np.float64)

        # Extract and accumulate contact forces
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get 6D force/torque vector
            mujoco.mj_contactForce(self.model, self.data, i, forcetorque)

            # Convert to torch tensor (forces only, ignore torques)
            force = torch.from_numpy(forcetorque[:3]).float().to(self.device)

            # Map geoms to bodies
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]

            # Apply Newton's 3rd law: body1 gets -force, body2 gets +force
            if b1 < self.model.nbody:
                self._force_tensor[0, b1] -= force
            if b2 < self.model.nbody:
                self._force_tensor[0, b2] += force

        return self._force_tensor

    def create_root_view(self, addrs: dict) -> BaseMujocoView:
        """Create root state view using existing tensor_views.

        Parameters
        ----------
        addrs : dict
            Address dictionary with slices for pos, quat, vel, ang_vel

        Returns
        -------
        BaseMujocoView
            MujocoRootStateView with quaternion conversion
        """
        from holosoma.simulator.mujoco.tensor_views import MujocoRootStateView

        return MujocoRootStateView(
            qpos_array=self.data.qpos,
            qvel_array=self.data.qvel,
            pos_indices=addrs["pos_indices"],
            quat_indices=addrs["quat_indices"],
            vel_indices=addrs["vel_indices"],
            ang_vel_indices=addrs["ang_vel_indices"],
            num_envs=1,
            device=self.device,
        )

    def create_dof_pos_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create DOF position view.

        Parameters
        ----------
        indices : slice
            Slice into qpos array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF positions [1, num_dof]
        """
        from holosoma.simulator.mujoco.tensor_views import create_dof_position_view

        return create_dof_position_view(self.data.qpos, indices, 1, num_dof, self.device)

    def create_dof_vel_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create DOF velocity view.

        Parameters
        ----------
        indices : slice
            Slice into qvel array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF velocities [1, num_dof]
        """
        from holosoma.simulator.mujoco.tensor_views import create_dof_velocity_view

        return create_dof_velocity_view(self.data.qvel, indices, 1, num_dof, self.device)

    def create_dof_acc_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create DOF acceleration view.

        Parameters
        ----------
        indices : slice
            Slice into qacc array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF accelerations [1, num_dof]
        """
        from holosoma.simulator.mujoco.tensor_views import create_dof_acceleration_view

        return create_dof_acceleration_view(self.data.qacc, indices, 1, num_dof, self.device)

    def create_dof_state_view(self, dof_addrs: dict, num_dof: int) -> BaseMujocoView:
        """Create DOF state view using CPU numpy arrays.

        Parameters
        ----------
        dof_addrs : dict
            Dictionary with 'dof_pos_indices' and 'dof_vel_indices' slices
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            MujocoDofStateView with IsaacGym flattened format [1 * num_dof, 2]
        """
        from holosoma.simulator.mujoco.tensor_views import MujocoDofStateView

        return MujocoDofStateView(
            qpos_array=self.data.qpos,
            qvel_array=self.data.qvel,
            dof_pos_indices=dof_addrs["dof_pos_indices"],
            dof_vel_indices=dof_addrs["dof_vel_indices"],
            num_envs=1,
            num_dof=num_dof,
            device=self.device,
        )

    def get_applied_forces_view(self) -> np.ndarray:
        """Get writable view for external applied forces.

        Returns direct view of MuJoCo's xfrc_applied array for applying
        external forces and torques to bodies.

        Returns
        -------
        np.ndarray
            Writable numpy array view [num_bodies, 6]
        """
        return self.data.xfrc_applied

    def create_quaternion_view(self, quat_slice: slice) -> BaseMujocoView:
        """Create quaternion view with format conversion.

        Delegates to tensor_views factory function for CPU numpy array views.

        Parameters
        ----------
        quat_slice : slice
            Slice for extracting quaternion from qpos

        Returns
        -------
        BaseMujocoView
            View for quaternion [1, 4] with [w,x,y,z] -> [x,y,z,w] conversion
        """
        from holosoma.simulator.mujoco.tensor_views import create_quaternion_view

        return create_quaternion_view(qpos_array=self.data.qpos, indices=quat_slice, num_envs=1, device=self.device)

    def create_angular_velocity_view(self, ang_vel_slice: slice) -> BaseMujocoView:
        """Create angular velocity view with proper reshaping.

        Delegates to tensor_views factory function for CPU numpy array views.

        Parameters
        ----------
        ang_vel_slice : slice
            Slice for extracting angular velocity from qvel

        Returns
        -------
        BaseMujocoView
            View for angular velocity [1, 3]
        """
        from holosoma.simulator.mujoco.tensor_views import (
            create_base_angular_velocity_view,
        )

        return create_base_angular_velocity_view(
            qvel_array=self.data.qvel, indices=ang_vel_slice, num_envs=1, device=self.device
        )

    def set_root_state(self, env_ids: torch.Tensor, root_states: torch.Tensor, root_addrs: dict) -> None:
        """Set robot root states. The robot root is an actor freejoint, so this
        delegates to set_actor_state at the robot's qpos/qvel addresses, then runs
        mj_forward to update derived quantities.

        root_states is [num_selected_envs, 13] in holosoma format; root_addrs carries
        'robot_qpos_addr' and 'robot_qvel_addr'.
        """
        self.set_actor_state(env_ids, root_states, root_addrs["robot_qpos_addr"], root_addrs["robot_qvel_addr"])
        if len(env_ids) > 0:
            mujoco.mj_forward(self.model, self.data)

    def set_dof_state(self, env_ids: torch.Tensor, dof_states: torch.Tensor, dof_addrs: dict) -> None:
        """Set DOF states using CPU numpy arrays.

        Converts tensors to numpy, writes to MuJoCo data arrays,
        and calls mj_forward to update derived quantities.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update (must be single environment)
        dof_states : torch.Tensor
            DOF states [num_selected_envs * num_dofs, 2] in IsaacGym format
            where [:, 0] = positions, [:, 1] = velocities
        dof_addrs : dict
            Address dictionary with 'dof_qpos_addrs' and 'dof_qvel_addrs' lists

        Raises
        ------
        AssertionError
            If multiple environments specified
        """
        # ClassicBackend only supports single environment
        assert len(env_ids) <= 1, f"ClassicBackend only supports single environment, got {len(env_ids)}"

        if len(env_ids) == 0:
            return

        # Parse addresses
        qpos_addrs = dof_addrs["dof_qpos_addrs"]
        qvel_addrs = dof_addrs["dof_qvel_addrs"]
        num_dof = len(qpos_addrs)

        # Reshape and convert to numpy
        dof_pos = dof_states[:, 0].view(len(env_ids), num_dof)[0].detach().cpu().numpy()
        dof_vel = dof_states[:, 1].view(len(env_ids), num_dof)[0].detach().cpu().numpy()

        # Write to MuJoCo data arrays
        for i, (qpos_idx, qvel_idx) in enumerate(zip(qpos_addrs, qvel_addrs)):
            self.data.qpos[qpos_idx] = dof_pos[i]
            self.data.qvel[qvel_idx] = dof_vel[i]

        # Update derived quantities
        mujoco.mj_forward(self.model, self.data)

    def get_actor_state(self, env_ids: torch.Tensor, qpos_addr: int, qvel_addr: int) -> torch.Tensor:
        """Read an actor's freejoint state from CPU qpos/qvel (single environment)."""
        assert len(env_ids) <= 1, f"ClassicBackend only supports single environment, got {len(env_ids)}"
        if len(env_ids) == 0:
            return torch.empty(0, 13, device=self.device)

        pos = self.data.qpos[qpos_addr : qpos_addr + 3]
        quat_mj = self.data.qpos[qpos_addr + 3 : qpos_addr + 7]  # [qw, qx, qy, qz]
        lin_vel = self.data.qvel[qvel_addr : qvel_addr + 3]
        ang_vel = self.data.qvel[qvel_addr + 3 : qvel_addr + 6]  # body-local
        quat_holo = mj_to_holosoma_quat(quat_mj)  # -> [qx, qy, qz, qw]
        state = torch.tensor([*pos, *quat_holo, *lin_vel, *ang_vel], dtype=torch.float32, device=self.device)
        return state.unsqueeze(0)  # [1, 13]

    def set_actor_state(self, env_ids: torch.Tensor, states: torch.Tensor, qpos_addr: int, qvel_addr: int) -> None:
        """Write an actor's freejoint state into CPU qpos/qvel (single environment)."""
        assert len(env_ids) <= 1, f"ClassicBackend only supports single environment, got {len(env_ids)}"
        if len(env_ids) == 0:
            return

        state = states[0]
        pos = state[:3].detach().cpu().numpy()
        quat_holo = state[3:7].detach().cpu().numpy()  # [qx, qy, qz, qw]
        lin_vel = state[7:10].detach().cpu().numpy()
        ang_vel = state[10:13].detach().cpu().numpy()
        quat_mj = holosoma_to_mj_quat(quat_holo)

        self.data.qpos[qpos_addr : qpos_addr + 3] = pos
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat_mj
        self.data.qvel[qvel_addr : qvel_addr + 3] = lin_vel
        self.data.qvel[qvel_addr + 3 : qvel_addr + 6] = ang_vel
        # Caller decides whether to mj_forward (set_actor_states_by_index does so once).

    def set_static_body_world_pose(
        self,
        body_ids: list[int],
        positions: torch.Tensor,
        quats: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Place welded (jointless) bodies at a world pose (single environment).

        Sibling of WarpBackend.set_static_body_world_pose: a static body's pose lives in the
        model's ``body_pos``/``body_quat`` (no qpos slice). ``positions`` is
        ``[len(env_ids)=1, len(body_ids), 3]``; ``quats`` (xyzw) is the matching ``[..., 4]`` or
        ``None`` to leave orientation as compiled. ``env_ids`` is ignored (single env). Works at
        setup and mid-rollout — ``mj_forward`` re-runs collision so the next ``mj_step`` sees the
        body at its new pose.
        """
        pos = positions[0].detach().cpu().numpy()  # [len(body_ids), 3]
        for i, body_id in enumerate(body_ids):
            self.model.body_pos[body_id] = pos[i]
        if quats is not None:
            # quats is a torch.Tensor here, but holosoma_to_mj_quat's return widens to the np|torch
            # union (only mypy sees the union; when torch is unstubbed it is Any and this is a no-op).
            quat_mj = holosoma_to_mj_quat(quats[0]).detach().cpu().numpy()  # type: ignore[union-attr]  # [len(body_ids), 4] wxyz
            for i, body_id in enumerate(body_ids):
                self.model.body_quat[body_id] = quat_mj[i]
        mujoco.mj_forward(self.model, self.data)  # refresh xpos/xquat from the new body_pos/quat
