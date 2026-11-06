"""Actor state adapter serving Holosoma's XYZW convention on both Isaac backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .backends import get_backend

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.state import SimState
    from holosoma.simulator.shared.object_registry import ObjectRegistry
    from .backends.base import IsaacLabBackend


class IsaacSimStateAdapter:
    """Lightweight adapter for IsaacSim state access with quaternion conversion.

    This adapter provides a clean interface for object state management while
    handling Lab 2's quaternion conversion and Lab 3's native XYZW arrays through
    the backend interface.

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
    state : SimState
        The simulator's state registry, serving ``robot_root_states`` in xyzw
    """

    def __init__(
        self,
        device: torch.device,
        object_registry: ObjectRegistry,
        scene: Any,
        robot: Any,
        state: SimState,
        backend: IsaacLabBackend | None = None,
    ) -> None:
        self.device = device
        self._object_registry = object_registry
        self._scene = scene
        self._robot = robot
        self._state = state
        self._backend = get_backend() if backend is None else backend
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
        via :meth:`get_object_states` in the backend's XYZW interface, concatenating in resolution
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
        :meth:`write_object_states` (per-object pose/velocity), consuming ``states``
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
        """Get object states in Holosoma's XYZW convention.

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
            return self._state.get("robot_root_states")[env_ids]

        elif obj_type in ("individual", "scene") and obj_name in self._scene.rigid_objects:
            # A scene rigid object — free (INDIVIDUAL) or kinematic/static (SCENE). Every
            # object, including multi-body scene-file bodies, lives under its own name in
            # scene.rigid_objects; the backend serves XYZW state.
            rigid_object = self._scene.rigid_objects[obj_name]
            return self._backend.root_state(rigid_object)[env_ids]

        else:
            raise ValueError(f"Cannot resolve object '{obj_name}' (type '{obj_type}')")

    def write_object_states(self, obj_name: str, states: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Write object states in Holosoma's XYZW convention.

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
            # Robot states are world-frame with env origins included, as robot_root_states serves them.
            self._backend.write_root_state(self._robot, states, env_ids)

        elif obj_type in ("individual", "scene") and obj_name in self._scene.rigid_objects:
            # A scene rigid object — free (INDIVIDUAL) or static (SCENE). Every object,
            # including multi-body scene-file bodies, lives under its own name.
            rigid_object = self._scene.rigid_objects[obj_name]
            # States already contain world positions; origins must not be added again.
            self._backend.write_root_state(rigid_object, states, env_ids)

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
