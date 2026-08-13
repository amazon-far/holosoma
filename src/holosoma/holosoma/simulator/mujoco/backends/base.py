"""Abstract base interface for MuJoCo simulation backends.

This module defines the contract that all MuJoCo backends must implement,
providing a consistent interface for simulation control, state access, and
data synchronization.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.simulator.mujoco.tensor_views import BaseMujocoView
    from holosoma.simulator.shared.camera_sensor import CameraRuntime


def mj_to_holosoma_quat(quat_wxyz: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert MuJoCo ``[w,x,y,z]`` to holosoma ``[x,y,z,w]`` (last-axis permute; numpy or torch)."""
    return quat_wxyz[..., [1, 2, 3, 0]]


def holosoma_to_mj_quat(quat_xyzw: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert holosoma ``[x,y,z,w]`` to MuJoCo ``[w,x,y,z]`` (last-axis permute; numpy or torch)."""
    return quat_xyzw[..., [3, 0, 1, 2]]


def apply_sensor_scene_flags(show_camera_frusta: bool, opt: mujoco.MjvOption | None = None) -> mujoco.MjvOption:
    """Set MuJoCo mounted-camera scene-visualization flags and return the option.

    Draws each configured camera's view frustum in the spectator viewer / recorded video. Gated by
    the simulator's ``debug_viz`` flag (this is debug visualization, not a sink config).

    Parameters
    ----------
    show_camera_frusta : bool
        Whether to draw the camera frusta (typically ``simulator.debug_viz_enabled``).
    opt : mujoco.MjvOption | None
        Option to mutate in place; ``None`` builds a fresh defaulted one. Returned either way.
    """
    if opt is None:
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = show_camera_frusta
    return opt


class IMujocoBackend(abc.ABC):
    """Abstract interface for MuJoCo simulation backends.

    Defines the contract that all MuJoCo backends must implement, providing
    a consistent interface for simulation control, state access, and data
    synchronization.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: FullSimConfig, device: str):
        """Initialize backend with model, data, and configuration.

        Parameters
        ----------
        model : mujoco.MjModel
            Compiled MuJoCo model
        data : mujoco.MjData
            MuJoCo data structure (shared with frontend)
        config : FullSimConfig
            Full simulation configuration
        device : str
            Device string ('cpu' or 'cuda:0', etc.)
        """
        self.model = model
        self.data = data
        self.config = config
        self.device = device
        self.num_envs = config.training.num_envs

    @abc.abstractmethod
    def step(self) -> None:
        """Advance simulation by one timestep.

        Implementations should call the appropriate MuJoCo step function
        (mj_step for CPU, mjw.step for GPU).
        """
        ...

    @abc.abstractmethod
    def compute_contact_forces(self) -> torch.Tensor:
        """Return current-frame net contact forces for ALL model bodies.

        The returned tensor is full-model-width ([num_envs, model.nbody, 3]),
        including the world body (id 0) and any non-robot actor bodies. The
        simulator layer gathers the robot-only rows (via ``body_ids``) and
        owns the rolling-history rotation, keeping physics-tensor width independent
        of the robot-only ``num_bodies``.

        Returns
        -------
        torch.Tensor
            Contact forces [num_envs, model.nbody, 3] (force only, torque dropped).
        """
        ...

    @abc.abstractmethod
    def get_render_data(self, world_id: int = 0) -> mujoco.MjData:
        """Get MjData for rendering (may require GPU->CPU sync).

        Parameters
        ----------
        world_id : int, default=0
            Which environment to sync for rendering (0 to num_envs-1).
            ClassicBackend ignores this (single environment only).
            WarpBackend syncs the specified environment from GPU to CPU.

        Returns
        -------
        mujoco.MjData
            MuJoCo data structure for rendering
        """
        ...

    @abc.abstractmethod
    def get_ctrl_tensor(self) -> torch.Tensor | None:
        """Get control tensor for direct writing (None if not supported).

        Returns
        -------
        torch.Tensor | None
            Control tensor for zero-copy writes, or None if backend doesn't support it
        """
        ...

    # View factory methods
    @abc.abstractmethod
    def create_root_view(self, addrs: dict) -> BaseMujocoView:
        """Create view for robot root states.

        Parameters
        ----------
        addrs : dict
            Dictionary with keys: pos_indices, quat_indices, vel_indices, ang_vel_indices

        Returns
        -------
        BaseMujocoView
            View for 13-element root state [pos, quat, lin_vel, ang_vel]
        """
        ...

    @abc.abstractmethod
    def create_dof_pos_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create view for DOF positions.

        Parameters
        ----------
        indices : slice
            Slice into qpos array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF positions [num_envs, num_dof]
        """
        ...

    @abc.abstractmethod
    def create_dof_vel_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create view for DOF velocities.

        Parameters
        ----------
        indices : slice
            Slice into qvel array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF velocities [num_envs, num_dof]
        """
        ...

    @abc.abstractmethod
    def create_dof_acc_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create view for DOF accelerations.

        Parameters
        ----------
        indices : slice
            Slice into qacc array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF accelerations [num_envs, num_dof]
        """
        ...

    @abc.abstractmethod
    def create_dof_state_view(self, dof_addrs: dict, num_dof: int) -> BaseMujocoView:
        """Create view for DOF states in IsaacGym flattened format.

        Returns view with shape [num_envs * num_dof, 2] where:
        - [:, 0] = positions
        - [:, 1] = velocities

        Parameters
        ----------
        dof_addrs : dict
            Dictionary with 'dof_pos_indices' and 'dof_vel_indices' slices
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF states [num_envs * num_dof, 2]
        """
        ...

    @abc.abstractmethod
    def get_applied_forces_view(self) -> np.ndarray | torch.Tensor:
        """Get writable view for external applied forces.

        Returns a writable view to xfrc_applied array where forces and torques
        can be applied to bodies. Shape is [num_bodies, 6] where:
        - [:, 0:3] = forces [fx, fy, fz]
        - [:, 3:6] = torques [tx, ty, tz]

        Returns
        -------
        np.ndarray | torch.Tensor
            Writable view to applied forces array
        """
        ...

    @abc.abstractmethod
    def set_root_state(self, env_ids: torch.Tensor, root_states: torch.Tensor, root_addrs: dict) -> None:
        """Set robot root states for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update
        root_states : torch.Tensor
            Root states [num_selected_envs, 13] in holosoma format:
            [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        root_addrs : dict
            Address dictionary with 'robot_qpos_addr' and 'robot_qvel_addr'
        """
        ...

    @abc.abstractmethod
    def set_dof_state(self, env_ids: torch.Tensor, dof_states: torch.Tensor, dof_addrs: dict) -> None:
        """Set DOF states for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update
        dof_states : torch.Tensor
            DOF states [num_selected_envs * num_dofs, 2] in IsaacGym format
        dof_addrs : dict
            Address dictionary with 'dof_qpos_addrs' and 'dof_qvel_addrs'
        """
        ...

    @abc.abstractmethod
    def get_actor_state(self, env_ids: torch.Tensor, qpos_addr: int, qvel_addr: int) -> torch.Tensor:
        """Read an actor's per-environment freejoint state from live backend storage.

        Reads each env's own value from the live qpos/qvel (GPU for WarpBackend). Velocity
        is the freejoint qvel (body-local angular), matching the set path and robot proxy.

        ``qpos_addr`` / ``qvel_addr`` are the start of the actor's freejoint 7-dof pose and
        6-dof velocity slices. Returns ``[len(env_ids), 13]`` in holosoma format
        ``[x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]``.
        """
        ...

    @abc.abstractmethod
    def set_actor_state(self, env_ids: torch.Tensor, states: torch.Tensor, qpos_addr: int, qvel_addr: int) -> None:
        """Write an actor's per-environment freejoint state into live backend storage.

        Symmetric counterpart of :meth:`get_actor_state`: writes each env's state into the
        live qpos/qvel (GPU for WarpBackend), the state the simulator actually steps.
        ``states`` is ``[len(env_ids), 13]`` in holosoma format; ``qpos_addr`` / ``qvel_addr``
        are the start of the actor's freejoint pose / velocity slices.
        """
        ...

    @abc.abstractmethod
    def set_static_body_world_pose(
        self,
        body_ids: list[int],
        positions: torch.Tensor,
        quats: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Place welded (jointless) bodies at a per-environment world pose.

        Static bodies have no freejoint qpos, so :meth:`set_actor_state` cannot reach them;
        their world pose lives in the model's ``body_pos`` (and ``body_quat``). Writing those
        model fields + a forward-kinematics recompute relocates the body — a pure KINEMATIC
        teleport (a welded body has no mass/velocity in the dynamics) that the collision
        pipeline honors at the new pose, both at setup and mid-rollout. Used to spread scene
        objects across env origins and to move static obstacles at runtime.

        Parameters
        ----------
        body_ids : list[int]
            MuJoCo body ids of the static bodies to move.
        positions : torch.Tensor
            ``[len(env_ids), len(body_ids), 3]`` world positions.
        quats : torch.Tensor | None
            ``[len(env_ids), len(body_ids), 4]`` world orientations (xyzw). ``None`` leaves
            each body's orientation as compiled.
        env_ids : torch.Tensor | None
            Environments to write (default: all). Single-env ClassicBackend ignores it.
        """
        ...

    def get_rigid_body_state_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Get zero-copy views of rigid body states (optional optimization).

        This method provides efficient access to rigid body states without
        CPU↔GPU synchronization or tensor allocation overhead. Backends should
        implement this for optimal performance during refresh_sim_tensors().

        Returns
        -------
        tuple[torch.Tensor, ...] | None
            If supported, returns (positions, orientations, linear_vel, angular_vel),
            each full-model-width [num_envs, model.nbody, ...]:
            - positions: [num_envs, model.nbody, 3]
            - orientations: [num_envs, model.nbody, 4] (xyzw quaternion)
            - linear_vel: [num_envs, model.nbody, 3]
            - angular_vel: [num_envs, model.nbody, 3]
            The simulator gathers robot-only rows via ``body_ids``.

            If not supported (e.g., ClassicBackend), returns None.
        """
        return None  # Default implementation - backends can override

    def initialize_state(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Sync CPU initial state to backend storage after construction.

        Copies the initial qpos/qvel from CPU ``data`` into the backend's own storage
        (e.g. WarpBackend's GPU tensors), overriding the MJCF qpos0 defaults. Backends
        whose storage already aliases CPU ``data`` (ClassicBackend) need no sync; this
        default is a no-op.

        Parameters
        ----------
        model : mujoco.MjModel
            MuJoCo model (for context)
        data : mujoco.MjData
            CPU MjData with initial state to copy to backend storage
        """
        return  # Default implementation - backends can override

    @abc.abstractmethod
    def create_quaternion_view(self, quat_slice: slice):
        """Create quaternion view with format conversion.

        Converts between MuJoCo [w,x,y,z] and holosoma [x,y,z,w] quaternion formats.

        Parameters
        ----------
        quat_slice : slice
            Slice for extracting quaternion from qpos

        Returns
        -------
        BaseMujocoView
            View for quaternion [num_envs, 4] with format conversion
        """
        ...

    @abc.abstractmethod
    def create_angular_velocity_view(self, ang_vel_slice: slice):
        """Create angular velocity view with proper reshaping.

        Parameters
        ----------
        ang_vel_slice : slice
            Slice for extracting angular velocity from qvel

        Returns
        -------
        BaseMujocoView
            View for angular velocity [num_envs, 3]
        """
        ...

    # Camera rendering
    @abc.abstractmethod
    def create_renderers(self, cameras: list[CameraRuntime]) -> None:
        """Allocate per-backend rendering resources for the mounted cameras (called once at setup).

        Parameters
        ----------
        cameras : list[CameraRuntime]
            All registered cameras; the backend resolves each native resource from ``runtime.name``.
        """
        ...

    @abc.abstractmethod
    def render_cameras(self, cameras: list[CameraRuntime]) -> None:
        """Render the given cameras and store each one's canonical output buffer.

        Writes each modality via ``runtime.set_buffer(data_type, tensor)``:
        ``rgb`` [N,H,W,3] uint8, ``depth`` [N,H,W,1] float32 meters (+inf no-hit), on ``self.device``.

        Parameters
        ----------
        cameras : list[CameraRuntime]
            The cameras due to render this step (a subset of those passed to ``create_renderers``).
        """
        ...
