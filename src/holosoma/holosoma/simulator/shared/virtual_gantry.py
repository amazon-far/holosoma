"""Virtual gantry system (simulator-agnostic)

This module provides an 'elastic band' that implements a virtual gantry system for supporting robots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.config_types.simulator import VirtualGantryCfg
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.hooks import HookRegistry


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

        self._enabled: bool = enable
        self.set_enable(enable)

    def register_hooks(self, hooks: HookRegistry) -> None:
        """Register virtual gantry lifecycle hooks with the simulator loop.

        step() is PRE_STEP: gantry forces must be applied before the substep so they act within it,
        not reactively after.
        """
        hooks.add(Phase.PRE_STEP, self.step, name="virtual_gantry.step")
        hooks.add(Phase.EPISODE_START, self.on_episode_start, name="virtual_gantry.on_episode_start")

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
        self._enabled = enable if enable is not None else not self._enabled

        # lazy check when toggled on...
        if self.enabled and self.sim.num_envs != 1:
            # ...supporting only the sim2sim use case for now
            raise RuntimeError("Virtual gantry supports num_envs=1 only")

        # No explicit clearing on disable: apply_external_force auto-zeroes each step, so the
        # `not self.enabled` early-return in step() drops the gantry force on the next substep.

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

    def on_episode_start(self, env_id: int) -> None:
        """Reset the gantry anchor when the tracked environment starts."""
        if env_id == 0:
            self.set_position_to_robot()

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

        return False  # type: ignore[unreachable]  # Command not handled

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

        # Calculate new force from robot state (world frame)
        gantry_force = self._advance(root_pos, root_vel)

        # Re-applied every PRE_STEP (this method) since the accumulator auto-zeros each substep.
        # apply_external_force addresses by name; body_link_id is a holosoma body index.
        body_name = self.sim.body_names[self.body_link_id]
        force_world = torch.as_tensor(gantry_force, dtype=torch.float32, device=self.sim.device)
        self.sim.apply_external_force(
            "robot",
            forces=force_world,
            body_names=[body_name],
            env_ids=torch.tensor([env_id], device=self.sim.device),
        )

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
        force: npt.NDArray[np.float64] = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return force


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
    # A disabled gantry applies no force and reads no body, so it should not require a humanoid
    # attachment body to exist; otherwise a non-humanoid robot (e.g. a single-link test body) cannot
    # boot with the gantry off. Return a disabled instance attached to body 0, a valid index that is
    # never read while disabled. Only the enabled path needs a real attachment body.
    if not enable:
        logger.info("Virtual gantry disabled; skipping attachment-body resolution.")
        return VirtualGantry(sim=sim, enable=False, body_link_id=0, cfg=cfg, **kwargs)

    if attachment_body_names is None:
        # Default names from holosoma_inference, likely needs updating or removing to force users to specify
        attachment_body_names = ["torso_link", "torso", "base_link", "pelvis", "Trunk", "Waist", "base"]

    logger.info("=== Setting up virtual gantry system ===")

    # Resolve the first candidate name that exists on the robot. The except covers only the
    # body-lookup miss (find_rigid_body_indice raises RuntimeError/ValueError when a name is absent)
    # so the next candidate can be tried. It does not wrap VirtualGantry construction, whose own
    # errors (e.g. the num_envs=1 guard) would otherwise be swallowed and mis-reported below as a
    # "could not find attachment body".
    body_id = None
    for body_name in attachment_body_names:
        try:
            body_id = sim.find_rigid_body_indice(body_name)
        except (RuntimeError, ValueError):
            continue
        if body_id >= 0:
            break
    else:
        available_bodies = getattr(sim, "body_names", "unknown")
        raise RuntimeError(
            f"Could not find suitable attachment body from {attachment_body_names}. "
            f"Available bodies: {available_bodies}"
        )

    logger.info(f"Virtual gantry attached to body '{body_name}' (ID: {body_id})")
    gantry = VirtualGantry(sim=sim, enable=enable, body_link_id=body_id, cfg=cfg, **kwargs)
    logger.info("=== Virtual gantry system setup completed ===")
    return gantry
