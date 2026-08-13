"""State adapter for IsaacSim with quaternion format conversion.

This module provides a lightweight adapter that handles object state access
with automatic quaternion format conversion between IsaacSim's native wxyz
format and holosoma's standard xyzw format.
"""

from __future__ import annotations

import torch
from .state_utils import fullstate_wxyz_to_xyzw, fullstate_xyzw_to_wxyz


class IsaacSimStateAdapter:
    """Lightweight adapter for IsaacSim state access with quaternion conversion.

    This adapter provides a clean interface for object state management while
    handling the quaternion format conversion between IsaacSim (wxyz) and
    holosoma (xyzw) automatically.

    Parameters
    ----------
    device : torch.device
        Device for tensor operations
    object_registry : ObjectRegistry
        Registry for object index resolution
    scene : InteractiveScene
        IsaacLab scene containing rigid objects
    robot : Articulation
        Robot articulation handle used by write_object_states
    robot_states : RootStatesProxy
        Robot states proxy (already handles wxyz->xyzw conversion)
    """

    def __init__(self, device: torch.device, object_registry, scene, robot, robot_states):
        self.device = device
        self._object_registry = object_registry
        self._scene = scene
        self._robot = robot
        self._robot_states = robot_states
        self._objects_dirty = False

    def resolve_indices(self, indices: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Resolve flat indices to (object_name, env_ids) pairs.

        Parameters
        ----------
        indices : torch.Tensor
            Flat tensor indices to resolve

        Returns
        -------
        list[tuple[str, torch.Tensor]]
            List of (object_name, env_ids) pairs
        """
        return self._object_registry.resolve_indices(indices)

    def get_states_by_index(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather unified [N, 13] xyzw world states for flat actor indices.

        Resolves ``indices`` (robot + object actors) to (object, env_ids) groups and reads each
        via :meth:`get_object_states` (which applies wxyz->xyzw), concatenating in resolution
        order. Empty ``indices`` -> empty ``[0, 13]``.
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        if len(indices) == 0:
            return torch.empty(0, 13, device=self.device, dtype=torch.float32)
        resolved_objects = self.resolve_indices(indices)
        if not resolved_objects:
            return torch.empty(0, 13, device=self.device, dtype=torch.float32)
        results = [self.get_object_states(obj_name, env_ids) for obj_name, env_ids in resolved_objects]
        return torch.cat(results, dim=0) if len(results) > 1 else results[0]

    def write_states_by_index(self, indices: torch.Tensor, states: torch.Tensor) -> None:
        """Scatter unified [N, 13] xyzw world states into per-object sim buffers (sets dirty).

        Resolves ``indices`` to (object, env_ids) groups and writes each via
        :meth:`write_object_states` (xyzw->wxyz, per-object pose/velocity), consuming ``states``
        row-for-row in resolution order. Expects full 13-vec states. Empty ``indices`` -> no-op.
        Does not flush; the caller syncs via ``write_state_updates``.
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        if len(indices) == 0:
            return
        resolved_objects = self.resolve_indices(indices)
        offset = 0
        for obj_name, env_ids in resolved_objects:
            num_envs_for_obj = len(env_ids)
            self.write_object_states(obj_name, states[offset : offset + num_envs_for_obj], env_ids)
            offset += num_envs_for_obj

    def get_object_states(self, obj_name: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object states with automatic wxyz->xyzw conversion.

        Parameters
        ----------
        obj_name : str
            Name of the object to get states for
        env_ids : torch.Tensor
            Environment IDs to query

        Returns
        -------
        torch.Tensor
            Object states in xyzw format, shape [len(env_ids), 13]

        Raises
        ------
        ValueError
            If object type is unknown
        """
        obj_type = self._object_registry.get_object_type(obj_name)

        if obj_type == "robot":
            # Robot states already converted to xyzw via RootStatesProxy
            return self._robot_states[env_ids]

        elif obj_type in ("individual", "scene") and obj_name in self._scene.rigid_objects:
            # A scene rigid object — free (INDIVIDUAL) or kinematic/static (SCENE). Every
            # object, including multi-body scene-file bodies, lives under its own name in
            # scene.rigid_objects; read directly (wxyz->xyzw).
            rigid_object = self._scene.rigid_objects[obj_name]
            raw_states = rigid_object.data.root_state_w[env_ids]
            return fullstate_wxyz_to_xyzw(raw_states)

        else:
            raise ValueError(f"Cannot resolve object '{obj_name}' (type '{obj_type}')")

    def write_object_states(self, obj_name: str, states: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Write object states with automatic xyzw->wxyz conversion.

        Parameters
        ----------
        obj_name : str
            Name of the object to update
        states : torch.Tensor
            New states in xyzw format, shape [len(env_ids), 13]
        env_ids : torch.Tensor
            Environment IDs to update
        """
        obj_type = self._object_registry.get_object_type(obj_name)

        if obj_type == "robot":
            # Write the robot directly via write_root_pose/velocity_to_sim (xyzw->wxyz).
            # NOTE: Intentionally does NOT apply env origins offsets for backwards compatibilty
            #       with existing robot root state setters
            converted_states = fullstate_xyzw_to_wxyz(states)
            self._robot.write_root_pose_to_sim(converted_states[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(converted_states[:, 7:], env_ids)

        elif obj_type in ("individual", "scene") and obj_name in self._scene.rigid_objects:
            # A scene rigid object — free (INDIVIDUAL) or static (SCENE). Every object,
            # including multi-body scene-file bodies, lives under its own name.
            rigid_object = self._scene.rigid_objects[obj_name]
            converted_states = fullstate_xyzw_to_wxyz(states)
            # For now, do NOT apply env origins as WBT does this itself. We need to update WBT first
            # before uncommenting this.
            # converted_states[:, 0:3] += self._scene.env_origins[env_ids]  # Apply environment origins
            rigid_object.write_root_pose_to_sim(converted_states[:, :7], env_ids)
            rigid_object.write_root_velocity_to_sim(converted_states[:, 7:], env_ids)

        else:
            raise ValueError(f"Cannot resolve object '{obj_name}' (type '{obj_type}')")

        # Mark states as dirty for batch synchronization
        self.mark_dirty()

    def is_dirty(self) -> bool:
        """Check if states have been modified and need synchronization.

        Returns
        -------
        bool
            True if states are dirty and need sync
        """
        return self._objects_dirty

    def clear_dirty(self) -> None:
        """Clear the dirty flag after synchronization."""
        self._objects_dirty = False

    def mark_dirty(self) -> None:
        """Explicitly mark states as dirty."""
        self._objects_dirty = True
