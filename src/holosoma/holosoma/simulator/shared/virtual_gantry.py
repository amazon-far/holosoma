"""Virtual gantry system (simulator-agnostic)

This module provides an 'elastic band' that implements a virtual gantry system for supporting robots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.config_types.simulator import VirtualGantryCfg
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type


class GantryCommand(Enum):
    """Virtual gantry control commands"""

    LENGTH_ADJUST = "gantry_length_adjust"
    TOGGLE = "gantry_toggle"
    FORCE_ADJUST = "gantry_force_adjust"
    FORCE_SIGN_TOGGLE = "gantry_force_sign_toggle"

    def __str__(self) -> str:
        """Backward compatibility with string-based system"""
        return self.value


@dataclass
class GantryCommandData:
    """Command with optional parameters"""

    command: GantryCommand
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}


class VirtualGantry:
    """Virtual gantry system for whole body tracking.

    Provides elastic band support that can be attached to a robot body.
    The band applies forces based on distance from a target point and
    can be controlled via keyboard inputs.

    Reference: https://github.com/unitreerobotics/unitree_mujoco
    """

    def __init__(
        self,
        sim: Any,
        body_link_id: int,
        enable: bool = True,
        cfg: VirtualGantryCfg | None = None,
        point: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize the virtual gantry system.

        Parameters
        ----------
        sim : Any
            Simulator instance with robot state access and force application methods.
        body_link_id : int
            ID of the rigid body to attach the gantry to.
        enable : bool, default=True
            Whether the gantry should be initially enabled.
        cfg : VirtualGantryCfg | None, default=None
            Configuration parameters. If None, uses default configuration.
        point : npt.NDArray[np.float64] | None, default=None
            Override for gantry anchor point position. Takes precedence over cfg.point.

        Raises
        ------
        RuntimeError
            If simulator has more than one environment (only single env supported).
        """
        if cfg is None:
            cfg = VirtualGantryCfg()

        self.sim = sim
        self.body_link_id = body_link_id
        self.stiffness = cfg.stiffness
        self.damping = cfg.damping
        self.height = cfg.height

        # Point parameter takes precedence over config
        if point is not None:
            self.point = point  # Already numpy array from direct parameter
        elif cfg.point is not None:
            self.point = np.array(cfg.point)  # Convert list[float] to numpy array
        else:
            self.point = np.array([0.0, 0.0, self.height])

        self.length = cfg.length
        self.apply_force = cfg.apply_force
        self.apply_force_sign = cfg.apply_force_sign

        # Set up simulator-specific force application method
        self._setup_force_application()

        self._enabled: bool = enable
        self.set_enable(enable)

    def _setup_force_application(self) -> None:
        """Set up simulator-specific force application and clearing methods.

        Configures the internal force application and clearing implementations
        based on the detected simulator type. This is a temporary solution until
        a unified force application interface is implemented across all simulators.

        Raises
        ------
        ValueError
            If the simulator type is not supported.
        """
        # NOTE: we need to implement a unified and generalized apply_force() to
        # the simulator interface. As a stop-gap, do so internally for the gantry
        # for a single environment and for robot only.
        simtype = get_simulator_type()
        if simtype is SimulatorType.ISAACGYM:
            self._apply_force_impl = self._apply_force_isaacgym
            self._clear_forces_impl = None  # IsaacGym doesn't need explicit clearing
        elif simtype is SimulatorType.ISAACSIM:
            logger.warning("Virtual Gantry untested in IsaacSim")
            self._apply_force_impl = self._apply_force_isaacsim
            self._clear_forces_impl = self._clear_forces_isaacsim
        elif simtype is SimulatorType.MUJOCO:
            self._apply_force_impl = self._apply_force_mujoco
            self._clear_forces_impl = self._clear_forces_mujoco
        else:
            raise ValueError(f"Unsupported simulator type: {simtype}")

    @property
    def enabled(self) -> bool:
        """Whether the virtual gantry is currently enabled.

        Returns
        -------
        bool
            True if gantry is enabled and applying forces, False otherwise.
        """
        return self._enabled

    def set_enable(self, enable: bool | None = None) -> None:
        """Enable or disable the virtual gantry system.

        Parameters
        ----------
        enable : bool | None, default=None
            If None, toggles current state. If True/False, sets state explicitly.

        Raises
        ------
        RuntimeError
            If trying to enable gantry with multiple environments (not supported).
        """
        # Store previous state to detect transitions
        was_enabled = self._enabled

        self._enabled = enable if enable is not None else not self._enabled

        # lazy check when toggled on...
        if self.enabled and self.sim.num_envs != 1:
            # ...supporting only the sim2sim use case for now
            raise RuntimeError("Virtual gantry supports num_envs=1 only")

        # Clear forces only when DISABLING (transitioning from enabled to disabled)
        # Don't clear during initialization when starting disabled
        if was_enabled and not self._enabled:
            if self._clear_forces_impl is not None:
                self._clear_forces_impl()

    def set_position_to_robot(self) -> None:
        """Reset gantry anchor point to current robot position.

        Updates the gantry anchor point to the current X,Y position of the robot
        while maintaining the configured height. This is useful for repositioning
        the gantry during runtime.
        """
        env_id = 0
        x, y = self.sim.robot_root_states[env_id, :3].detach().cpu().numpy()[:2]
        self.point = np.array([x, y, self.height])
        logger.debug(f"Virtual gantry position reset to '{self.point}'")

    def handle_command(self, command_data: GantryCommandData | GantryCommand) -> bool:
        """Handle gantry commands with optional parameters.

        Parameters
        ----------
        command_data : Union[GantryCommandData, GantryCommand]
            Command to execute, either as enum or command data with parameters

        Returns
        -------
        bool
            True if command was handled, False otherwise
        """
        # Handle enum-only case
        if isinstance(command_data, GantryCommand):
            command_data = GantryCommandData(command_data)

        command = command_data.command
        params = command_data.parameters
        assert params is not None

        if command == GantryCommand.LENGTH_ADJUST:
            amount = params.get("amount", 0.1)
            self.length += amount
            logger.info(f"Gantry length adjusted by {amount} to {self.length:.2f}")
            return True

        if command == GantryCommand.TOGGLE:
            self.set_enable(params.get("enabled"))
            status = "enabled" if self.enabled else "disabled"
            logger.info(f"Gantry {status}")
            return True

        if command == GantryCommand.FORCE_ADJUST:
            amount = params.get("amount", 10 * self.apply_force_sign)
            self.apply_force += amount
            self.apply_force = np.clip(self.apply_force, -100, 100)
            logger.info(f"Gantry apply_force adjusted by {amount} to {self.apply_force}")
            return True

        if command == GantryCommand.FORCE_SIGN_TOGGLE:
            self.apply_force_sign *= -1
            logger.info(f"Gantry force sign toggled to {self.apply_force_sign}")
            return True

        return False  # Command not handled

    def step(self) -> None:
        """Execute one simulation step of the virtual gantry system.

        Calculates and applies elastic band forces based on current robot state.
        This method should be called once per simulation timestep when the gantry
        is enabled.

        The force calculation uses a spring-damper model where:
        - Spring force is proportional to distance from rest length
        - Damping force opposes velocity in the direction of the band
        """
        if not self.enabled:
            return

        # Get robot root position and velocity
        env_id = 0
        robot_state = self.sim.robot_root_states[env_id, :]
        root_pos = robot_state[:3].detach().cpu().numpy()
        root_vel = robot_state[7:10].detach().cpu().numpy()

        # Calculate new force from robot state
        gantry_force = self._advance(root_pos, root_vel)

        # Apply force using simulator-specific implementation
        self._apply_force_impl(self.body_link_id, gantry_force)

    def draw_debug(self) -> None:
        """Draw gantry visualization for debugging purposes.

        Renders visual elements to help debug and visualize the gantry system:
        - Red sphere at the gantry anchor point
        - Blue line connecting anchor point to robot position

        Only draws when gantry is enabled. Requires the simulator to support
        the draw utilities from holosoma.utils.draw.
        """
        assert self.sim

        if not self.enabled:
            return

        from holosoma.utils.draw import draw_line, draw_sphere

        # Red sphere at gantry anchor point
        anchor_pos = torch.from_numpy(self.point).float().to(self.sim.device)
        red_color = (1.0, 0.0, 0.0)
        draw_sphere(self.sim, anchor_pos.cpu(), 0.1, color=red_color, env_id=0)

        # Draw a line from the anchor point to the robot position
        env_id = 0
        robot_pos = self.sim.robot_root_states[env_id, :3].to(self.sim.device)
        blue_color = (0.0, 0.0, 1.0)
        draw_line(self.sim, anchor_pos.cpu(), robot_pos, color=blue_color, env_id=0)

    def _advance(self, x: npt.NDArray[np.float64], vx: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate elastic band force based on current position and velocity.

        Implements a spring-damper model for the virtual gantry elastic band.
        The force is calculated as: F = k*(d - L) - c*v_radial
        where k is stiffness, d is distance, L is rest length, c is damping,
        and v_radial is velocity component along the band direction.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Current position [x, y, z] in world coordinates.
        vx : npt.NDArray[np.float64]
            Current velocity [vx, vy, vz] in world coordinates.

        Returns
        -------
        npt.NDArray[np.float64]
            Force vector [fx, fy, fz] to apply to the attached body.
        """
        dx = self.point - x
        distance = np.linalg.norm(dx)
        direction = dx / distance
        v = np.dot(vx, direction)
        return (self.stiffness * (distance - self.length) - self.damping * v) * direction

    def _apply_force_mujoco(self, link_id: int, force: npt.NDArray[np.float64]) -> None:
        """Apply force to rigid body in MuJoCo simulator.

        Uses the unified applied_forces interface for backend compatibility.
        Handles both ClassicBackend (numpy array) and WarpBackend (torch tensor).

        Parameters
        ----------
        link_id : int
            Index of the rigid body to apply force to.
        force : npt.NDArray[np.float64]
            3D force vector [fx, fy, fz] to apply.
        """
        env_id = 0  # Virtual gantry only supports single environment

        if isinstance(self.sim.applied_forces, torch.Tensor):
            # WarpBackend: GPU tensor with env dimension [num_envs, num_bodies, 6]
            force_tensor = torch.from_numpy(force).float().to(self.sim.device)
            self.sim.applied_forces[env_id, link_id, :3] = force_tensor
        else:
            # ClassicBackend: CPU numpy array without env dimension [num_bodies, 6]
            self.sim.applied_forces[link_id, :3] = force

    def _clear_forces_mujoco(self) -> None:
        """Clear forces in MuJoCo (WarpBackend only - ClassicBackend doesn't need it).

        WarpBackend requires explicit clearing of GPU tensors when disabling the gantry,
        while ClassicBackend's numpy array clearing happens naturally through the
        simulation step (xfrc_applied is automatically zeroed each step by MuJoCo).

        This method only clears forces for the specific body the gantry is attached to,
        leaving other external forces unaffected.
        """
        env_id = 0  # Virtual gantry only supports single environment

        if isinstance(self.sim.applied_forces, torch.Tensor):
            # WarpBackend: Clear GPU tensor for this body only
            # Zero out both forces [0:3] and torques [3:6] for completeness
            self.sim.applied_forces[env_id, self.body_link_id, :] = 0.0
        # ClassicBackend (numpy array): Do nothing
        # MuJoCo automatically zeros xfrc_applied each step, so no explicit clearing needed

    def _apply_force_isaacgym(self, link_id: int, force: npt.NDArray[np.float64]) -> None:
        """Apply force to rigid body in IsaacGym simulator.

        Applies force directly to the body's center of mass (similar to MuJoCo's approach).
        This provides simpler, more consistent behavior across simulators.

        Parameters
        ----------
        link_id : int
            Index of the rigid body to apply force to.
        force : npt.NDArray[np.float64]
            3D force vector [fx, fy, fz] to apply.
        """
        from isaacgym import gymapi, gymtorch

        force_tensor = torch.zeros(self.sim.num_envs, self.sim.num_bodies, 3, device=self.sim.device)
        force_tensor[:, link_id, :] = torch.tensor(force, device=self.sim.device, dtype=torch.float32)

        # Apply force directly at center of mass (matches MuJoCo behavior)
        # No torques applied (None), using ENV_SPACE coordinate frame
        self.sim.gym.apply_rigid_body_force_tensors(
            self.sim.sim, gymtorch.unwrap_tensor(force_tensor), None, gymapi.ENV_SPACE
        )

    def _apply_force_isaacsim(self, link_id: int, force: npt.NDArray[np.float64]) -> None:
        """Apply force to rigid body in IsaacSim simulator using IsaacLab API.

        Transforms forces from world frame to body-local frame since IsaacLab 2.1
        applies forces in local frame (is_global=False is hardcoded). This ensures
        the gantry forces are applied correctly regardless of body orientation.

        Parameters
        ----------
        link_id : int
            Index of the rigid body to apply force to.
        force : npt.NDArray[np.float64]
            3D force vector [fx, fy, fz] in world frame to apply.

        Raises
        ------
        RuntimeError
            If link_id is invalid or body mapping fails.
        """
        # Validate body index
        if link_id >= len(self.sim.body_ids):
            raise RuntimeError(f"Invalid link_id {link_id}, must be < {len(self.sim.body_ids)}")

        # Map body index
        isaac_body_id = self.sim.body_ids[link_id]

        # Get body orientation to transform force from world to body frame
        # IsaacLab applies forces in local frame (is_global=False hardcoded in 2.1)
        body_quat_w = self.sim._robot.data.body_quat_w[0, isaac_body_id]  # [w,x,y,z] format

        # Transform force from world frame to body frame
        from isaaclab.utils.math import quat_apply_inverse

        force_world = torch.from_numpy(force).float().to(self.sim.sim_device)
        force_body = quat_apply_inverse(body_quat_w, force_world)

        # Create force tensor for this body only
        forces = force_body.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        torques = torch.zeros_like(forces)  # [1, 1, 3] - no torques

        self.sim._robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            env_ids=torch.tensor([0], device=self.sim.sim_device),
            body_ids=torch.tensor([isaac_body_id], device=self.sim.sim_device),
            # FIXME: use is_global=True when upgrading IsaacSim/Lab
        )

    def _clear_forces_isaacsim(self) -> None:
        """Clear external forces in IsaacSim by setting zero forces.

        Sets zero force/torque values for the specific body that had forces applied.
        This properly clears the forces without causing shape mismatch errors that
        occur when using empty tensors.
        """
        # Validate body index
        if self.body_link_id >= len(self.sim.body_ids):
            return  # Body no longer exists, nothing to clear

        # Map body index
        isaac_body_id = self.sim.body_ids[self.body_link_id]

        # Create zero force/torque tensors with proper shape [1, 1, 3]
        zero_forces = torch.zeros(1, 1, 3, device=self.sim.sim_device)
        zero_torques = torch.zeros(1, 1, 3, device=self.sim.sim_device)

        # Clear forces by setting them to zero for this specific body
        self.sim._robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=zero_torques,
            env_ids=torch.tensor([0], device=self.sim.sim_device),
            body_ids=torch.tensor([isaac_body_id], device=self.sim.sim_device),
        )


def create_virtual_gantry(
    sim: Any,
    enable: bool = False,
    attachment_body_names: list[str] | None = None,
    cfg: VirtualGantryCfg | None = None,
    **kwargs: Any,
) -> VirtualGantry:
    """Factory function to create and setup virtual gantry with automatic body detection.

    Attempts to attach the virtual gantry to one of the specified body names,
    trying each name in order until a valid body is found. This provides a
    convenient way to set up the gantry without needing to know the exact
    body names used in different robot models.

    Parameters
    ----------
    sim : Any
        Simulator instance with `find_rigid_body_indice()` method for body lookup.
    enable : bool, default=False
        Whether gantry should be initially enabled.
    attachment_body_names : list[str] | None, default=None
        List of body names to try for attachment (in preference order).
        If None, uses common default body names.
    cfg : VirtualGantryCfg | None, default=None
        Configuration parameters for the gantry. If None, uses default configuration.
    **kwargs : Any
        Additional parameters passed to VirtualGantry constructor.
        These take precedence over cfg parameters.

    Returns
    -------
    VirtualGantry
        Configured virtual gantry instance attached to the first found body.

    Raises
    ------
    RuntimeError
        If no suitable attachment body is found from the provided names.

    Examples
    --------
    >>> # Basic usage with default settings
    >>> gantry = create_virtual_gantry(sim, enable=True)

    >>> # With custom configuration
    >>> cfg = VirtualGantryCfg(stiffness=300.0, damping=150.0)
    >>> gantry = create_virtual_gantry(sim, cfg=cfg, enable=True)

    >>> # With specific body names
    >>> gantry = create_virtual_gantry(
    ...     sim,
    ...     attachment_body_names=["torso", "base_link"],
    ...     enable=True
    ... )
    """
    if attachment_body_names is None:
        # Default names from holosoma_inference, likely needs updating or removing to force users to specify
        attachment_body_names = ["torso_link", "torso", "base_link", "pelvis", "Trunk", "Waist", "base"]

    logger.info("=== Setting up virtual gantry system ===")

    for body_name in attachment_body_names:
        try:
            # Attempt to find a body, need a cleaner API to avoid exception handling as happy path...
            body_id = sim.find_rigid_body_indice(body_name)
            if body_id >= 0:
                logger.info(f"Virtual gantry attached to body '{body_name}' (ID: {body_id})")
                gantry = VirtualGantry(sim=sim, enable=enable, body_link_id=body_id, cfg=cfg, **kwargs)
                logger.info("=== Virtual gantry system setup completed ===")
                return gantry
        except (RuntimeError, ValueError):  # noqa: PERF203
            continue

    available_bodies = getattr(sim, "body_names", "unknown")
    raise RuntimeError(
        f"Could not find suitable attachment body from {attachment_body_names}. Available bodies: {available_bodies}"
    )
