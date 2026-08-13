"""
Unified Object Registry for Cross-Simulator Object Management
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from holosoma.utils.safe_torch_import import torch


class RegisteredObject(NamedTuple):
    """One object's registry record.

    Attributes
    ----------
    name : str
        Actor name (e.g. ``"box"``, ``"scene_free_box"``).
    type : str
        ObjectType value (``"robot"``/``"scene"``/``"individual"``); stored as the str
        value for backward compatibility.
    position_in_type : int
        Position within its type band, used for index calculation.
    indices : torch.Tensor
        Per-env flat indices into the interleaved address space.
    initial_poses : torch.Tensor
        Initial poses [num_envs, 7] in world coordinates, [x,y,z,qx,qy,qz,qw].
    initial_velocities : torch.Tensor
        Initial world-frame velocities [num_envs, 6], [vx,vy,vz,wx,wy,wz].
    """

    name: str
    type: str
    position_in_type: int
    indices: torch.Tensor
    initial_poses: torch.Tensor
    initial_velocities: torch.Tensor


class ObjectType(Enum):
    """Enumeration of object types in the unified simulator interface.

    This enum defines the three main categories of objects that can be managed
    by the unified object registry system across different simulators.

    Attributes
    ----------
    ROBOT : str
        Robot objects (typically one per environment).
    SCENE : str
        Scene objects (static environment elements like furniture).
    INDIVIDUAL : str
        Individual objects (dynamic objects that can be manipulated).
    """

    ROBOT = "robot"
    SCENE = "scene"
    INDIVIDUAL = "individual"

    def __str__(self) -> str:
        """Return the string value for backward compatibility.

        Returns
        -------
        str
            The string value of the enum member.
        """
        return self.value


class ObjectRegistry:
    """Object registry for mapping objects by name and index to access their state tensors

    This class manages object registration and index calculation across different
    simulators using an interleaved layout. Objects are organized by environment
    rather than by type, enabling O(1) lookups and vectorized operations for
    common use cases.

    This assumes the same number of objects for each environment instance.

    Interleaved Layout:
    [robot_env0, scene0_env0, scene1_env0, indiv0_env0,  # env 0 objects
     robot_env1, scene0_env1, scene1_env1, indiv0_env1,  # env 1 objects
     ...]

    Index Space Layout
    ------------------
    The flat index space uses an interleaved layout organized by environment:
    [env0_objects][env1_objects][env2_objects]...

    Example with 2 environments, 1 robot, 2 scene objects, 1 individual object:
    - Indices 0,4:   Robot in envs 0,1
    - Indices 1,5:   Scene object 0 (table) in envs 0,1
    - Indices 2,6:   Scene object 1 (chair) in envs 0,1
    - Indices 3,7:   Individual object 0 (box) in envs 0,1

    Object Resolution
    -----------------
    resolve_indices() converts flat indices back to (object_name, env_ids) pairs:
    - Input:  [0, 1, 4, 5]
    - Output: [("robot", [0]), ("table", [0]), ("robot", [1]), ("table", [1])]

    This enables efficient batch operations on mixed object types while maintaining
    consistent ordering across different simulator backends.

    Parameters
    ----------
    device : str
        Device identifier for tensor operations (e.g., 'cuda:0', 'cpu').

    Attributes
    ----------
    device : str
        Device for tensor operations.
    objects : list[RegisteredObject]
        List of registered objects (see RegisteredObject).
    name_to_index : dict[str, int]
        Mapping from object names to indices in the objects list.
    num_envs : int
        Number of environments in the simulation.
    objects_per_env : int
        Total objects per environment in interleaved layout.
    robot_offset_in_env : int
        Offset of robot objects within each environment block.
    scene_offset_in_env : int
        Offset of scene objects within each environment block.
    individual_offset_in_env : int
        Offset of individual objects within each environment block.
    """

    def __init__(self, device: str):
        self.device = device

        # Object storage
        self.objects: list[RegisteredObject] = []
        self.name_to_index: dict[str, int] = {}

        # Interleaved layout parameters
        self.num_envs = 0
        self.objects_per_env = 0
        self.individual_count = 0

        # Offsets within each environment block
        self.robot_offset_in_env = 0
        self.scene_offset_in_env = 0
        self.individual_offset_in_env = 0

        # Lookup array: position_in_env -> object_name
        self._position_to_name: list[str] = []

        self._resolved_objects_cache: list[tuple[str, torch.Tensor]] | None = None

        self._finalized = False

    def register_object(
        self,
        name: str,
        object_type: ObjectType,
        position_in_type: int,
        initial_poses: torch.Tensor,
        initial_velocities: torch.Tensor | None = None,
    ):
        """Register object with pre-calculated initial poses (and velocities) for all environments.

        Registers a new object in the registry with its type, position within that type,
        and initial poses for all environments. The caller is responsible for providing
        world coordinates with any necessary transformations (like env_origins) already applied.

        Parameters
        ----------
        name : str
            Direct actor name (e.g., 'obj0_0', 'obj0_1').
        object_type : ObjectType
            Type of object (ObjectType enum).
        position_in_type : int
            Position within the object type (for index calculation).
        initial_poses : torch.Tensor
            Initial poses for ALL environments [num_envs, 7] in world coordinates.
            Format: [x, y, z, qx, qy, qz, qw] per environment.
            Caller is responsible for applying env_origins if needed.
        initial_velocities : torch.Tensor | None, default=None
            Initial world-frame velocities for ALL environments [num_envs, 6], format
            [vx, vy, vz, wx, wy, wz] per environment. ``None`` (the common case — static
            bodies and most free bodies) stores zeros.

        Raises
        ------
        ValueError
            If object with the same name is already registered or tensor shape is incorrect.
        """
        if self._finalized:
            raise RuntimeError(f"ObjectRegistry already finalized, cannot register object '{name}'")

        if name in self.name_to_index:
            raise ValueError(f"Object '{name}' already registered")

        # Validate tensor shape
        if self.num_envs > 0 and initial_poses.shape != (self.num_envs, 7):
            raise ValueError(f"initial_poses must have shape [{self.num_envs}, 7], got {initial_poses.shape}")

        if initial_velocities is None:
            initial_velocities = torch.zeros(self.num_envs or 1, 6, device=self.device, dtype=torch.float32)
        elif self.num_envs > 0 and initial_velocities.shape != (self.num_envs, 6):
            raise ValueError(f"initial_velocities must have shape [{self.num_envs}, 6], got {initial_velocities.shape}")

        # Create placeholder indices (will be updated during finalization)
        indices = torch.arange(self.num_envs or 1, device=self.device)

        # Store type as its string value for backward compatibility with existing consumers.
        self.objects.append(
            RegisteredObject(name, object_type.value, position_in_type, indices, initial_poses, initial_velocities)
        )
        self.name_to_index[name] = len(self.objects) - 1

    def setup_ranges(self, num_envs: int, robot_count: int, scene_count: int, individual_count: int):
        """Set up interleaved layout parameters.

        Configures the interleaved layout where objects are organized by environment
        rather than by type for maximum performance.

        Parameters
        ----------
        num_envs : int
            Number of environments in the simulation.
        robot_count : int
            Number of robot objects per environment.
        scene_count : int
            Number of scene objects per environment.
        individual_count : int
            Number of individual objects per environment.
        """
        self.num_envs = num_envs

        # Interleaved layout: all objects for env0, then all objects for env1, etc.
        self.objects_per_env = robot_count + scene_count + individual_count

        # Offsets within each environment block
        self.robot_offset_in_env = 0
        self.scene_offset_in_env = robot_count
        self.individual_offset_in_env = robot_count + scene_count

    def finalize_registration(self):
        """Finalize registration and build lookup structures.

        Updates the indices for all registered objects and builds the direct
        position-to-name lookup array for O(1) reverse lookups.
        """
        # Update indices for all registered objects
        for i, obj in enumerate(self.objects):
            indices = torch.arange(self.num_envs, device=self.device)
            self.objects[i] = obj._replace(indices=indices)

        self._build_position_lookup()
        self._finalized = True

    def _build_position_lookup(self):
        """Build direct position-to-name array for O(1) lookup."""
        # Initialize array with empty strings
        self._position_to_name = [""] * self.objects_per_env

        for obj in self.objects:
            if obj.type == ObjectType.ROBOT.value:
                array_pos = self.robot_offset_in_env + obj.position_in_type
            elif obj.type == ObjectType.SCENE.value:
                array_pos = self.scene_offset_in_env + obj.position_in_type
            elif obj.type == ObjectType.INDIVIDUAL.value:
                array_pos = self.individual_offset_in_env + obj.position_in_type
            else:
                raise ValueError(f"Unknown object type '{obj.type}'")

            if array_pos >= self.objects_per_env:
                raise ValueError(f"Position {array_pos} exceeds objects_per_env {self.objects_per_env}")

            if self._position_to_name[array_pos]:
                existing_name = self._position_to_name[array_pos]
                raise ValueError(f"Duplicate object at position {array_pos}: '{existing_name}' and '{obj.name}'")
            self._position_to_name[array_pos] = obj.name

    def get_object_indices(self, names: str | list[str], env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get object indices by object names

        Converts (object_names, env_ids) pairs into flat indices in the unified address space.
        This is the primary method for converting semantic object references into tensor indices.

        Returns:
            torch.Tensor: Flat indices for interleaved virtual address space
        """
        if not self._finalized:
            raise RuntimeError("ObjectRegistry must be finalized before generating indices")

        if isinstance(names, str):
            names = [names]

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if env_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        all_indices = []

        for name in names:
            if name not in self.name_to_index:
                available = list(self.name_to_index.keys())
                raise KeyError(f"Object '{name}' not found. Available: {available}")

            obj = self.objects[self.name_to_index[name]]

            # Direct calculation using interleaved layout
            if obj.type == ObjectType.ROBOT.value:
                indices = env_ids * self.objects_per_env + self.robot_offset_in_env + obj.position_in_type
            elif obj.type == ObjectType.SCENE.value:
                indices = env_ids * self.objects_per_env + self.scene_offset_in_env + obj.position_in_type
            elif obj.type == ObjectType.INDIVIDUAL.value:
                indices = env_ids * self.objects_per_env + self.individual_offset_in_env + obj.position_in_type
            else:
                raise ValueError(f"Unknown object type '{obj.type}' for object '{name}'")

            all_indices.append(indices)

        return torch.cat(all_indices) if len(all_indices) > 1 else all_indices[0]

    def get_initial_pose(self, name: str, env_id: int = 0) -> torch.Tensor:
        """Get initial pose for object by name for a specific environment

        Args:
            name: Object name
            env_id: Environment ID (default: 0)

        Returns:
            torch.Tensor: Initial pose [7] in format [x, y, z, qx, qy, qz, qw]
        """
        if name not in self.name_to_index:
            available = list(self.name_to_index.keys())
            raise KeyError(f"Object '{name}' not found. Available: {available}")

        initial_poses = self.objects[self.name_to_index[name]].initial_poses  # [num_envs, 7]
        return initial_poses[env_id]  # [7]

    def get_initial_poses_batch(self, names: list[str], env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get initial poses for multiple objects across environments

        Args:
            names: List of object names
            env_ids: Environment IDs to get poses for. If None, gets poses for all environments.

        Returns:
            torch.Tensor: Initial poses [num_objects * num_envs, 7] in format [x,y,z,qx,qy,qz,qw]
        """
        if not names:
            return torch.empty(0, 7, device=self.device, dtype=torch.float32)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        all_poses = []
        for name in names:
            initial_poses = self.objects[self.name_to_index[name]].initial_poses  # [num_envs, 7]
            selected_poses = initial_poses[env_ids]  # [len(env_ids), 7]
            all_poses.append(selected_poses)

        # Flatten to match expected format [num_objects * num_envs, 7]
        return torch.cat(all_poses, dim=0)

    def get_initial_velocity(self, name: str, env_id: int = 0) -> torch.Tensor:
        """Get initial velocity for object by name for a specific environment.

        The velocity sibling of get_initial_pose.

        Args:
            name: Object name
            env_id: Environment ID (default: 0)

        Returns:
            torch.Tensor: Initial velocity [6] in format [vx, vy, vz, wx, wy, wz]
        """
        if name not in self.name_to_index:
            available = list(self.name_to_index.keys())
            raise KeyError(f"Object '{name}' not found. Available: {available}")

        initial_velocities = self.objects[self.name_to_index[name]].initial_velocities  # [num_envs, 6]
        return initial_velocities[env_id]  # [6]

    def get_initial_velocities_batch(self, names: list[str], env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get initial velocities for multiple objects across environments.

        The velocity sibling of get_initial_poses_batch; same row order (object-major,
        then env-minor) so the two can be concatenated column-wise into a 13-vector.

        Args:
            names: List of object names
            env_ids: Environment IDs to get velocities for. If None, gets all environments.

        Returns:
            torch.Tensor: Initial velocities [num_objects * num_envs, 6] in format
            [vx, vy, vz, wx, wy, wz]
        """
        if not names:
            return torch.empty(0, 6, device=self.device, dtype=torch.float32)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        all_velocities = []
        for name in names:
            initial_velocities = self.objects[self.name_to_index[name]].initial_velocities  # [num_envs, 6]
            all_velocities.append(initial_velocities[env_ids])  # [len(env_ids), 6]

        return torch.cat(all_velocities, dim=0)

    def resolve_indices(self, indices: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Reverse lookup object indices back to (object_name, env_ids) pairs

        This method is the inverse of get_object_indices(). It takes flat indices from the
        unified address space and groups them back into per-object environment lists.

        Parameters
        ----------
        indices : torch.Tensor
            Flat indices to resolve, e.g., [0, 2, 3, 6, 7]

        Returns
        -------
        List[Tuple[str, torch.Tensor]]
            List of (object_name, env_ids) pairs, e.g.:
            [("robot", tensor([0])),
             ("table", tensor([0, 1])),
             ("box", tensor([0, 1]))]

        Example
        -------
        >>> # Setup: 2 envs, robot + table + box (interleaved layout)
        >>> registry.resolve_indices(torch.tensor([0, 1, 3, 4]))
        [("robot", tensor([0])),      # Index 0 -> robot in env 0
         ("table", tensor([0])),      # Index 1 -> table in env 0
         ("robot", tensor([1])),      # Index 3 -> robot in env 1
         ("table", tensor([1]))]      # Index 4 -> table in env 1
        """
        if not self._finalized:
            raise RuntimeError("ObjectRegistry must be finalized before resolving indices")

        if len(indices) == 0:
            return []

        # Vectorized calculation of env_ids and positions
        env_ids = indices // self.objects_per_env
        pos_in_env = indices % self.objects_per_env

        # After calculating env_ids, validate they're in range
        if env_ids.max() >= self.num_envs or env_ids.min() < 0:
            raise ValueError(f"Environment IDs {env_ids} out of range [0, {self.num_envs})")

        # Group by position (which maps directly to object name), preserving the
        # FIRST-OCCURRENCE order of positions in `indices` rather than sorted order.
        # get_object_indices / get_initial_poses_batch emit indices in request
        # (name-list) order; callers that pair a get_actor_initial_poses result with
        # set_actor_states walk both sequentially, so this must group in the same
        # order or poses get applied to the wrong actor. (torch.unique would return
        # sorted positions, swapping poses when request order != position order.)
        # dict.fromkeys keeps first-occurrence order while de-duplicating.
        ordered_positions = list(dict.fromkeys(pos_in_env.tolist()))

        results = []
        for pos_val in ordered_positions:
            if pos_val >= len(self._position_to_name):
                raise ValueError(f"Position {pos_val} out of range [0, {len(self._position_to_name)})")

            object_name = self._position_to_name[pos_val]
            if not object_name:
                raise ValueError(f"No object registered at position {pos_val}")

            # Find all env_ids for this position (env order preserved from `indices`)
            mask = pos_in_env == pos_val
            matching_env_ids = env_ids[mask]

            results.append((object_name, matching_env_ids))

        return results

    def resolved_objects(self) -> list[tuple[str, torch.Tensor]]:
        """Get cached resolution of ALL objects (full-range gather optimization).

        Returns the same result as resolve_indices(torch.arange(total_objects)) but
        cached for performance, for callers that need every object in every environment
        (e.g. a full-state clone/gather).

        Note:
        ------------------------------
        This method MUST return objects in the EXACT same order that resolve_indices()
        produces when given torch.arange(total_objects), so a cached full gather matches a
        direct resolve_indices(arange(total)) gather. Any deviation in ordering causes the two
        to disagree.

        The ordering is determined by the interleaved layout where objects are organized
        by environment: [env0_objects][env1_objects][env2_objects]... with consistent
        positioning within each environment block.

        Returns
        -------
        List[Tuple[str, torch.Tensor]]
            Pre-computed (object_name, env_ids) pairs for all registered objects.
            Each object is paired with ALL environment IDs.

        Example
        -------
        >>> # With 1000 envs, robot + table + box:
        >>> registry.resolved_objects()
        [("robot", tensor([0, 1, 2, ..., 999])),
         ("table", tensor([0, 1, 2, ..., 999])),
         ("box", tensor([0, 1, 2, ..., 999]))]
        """
        if not self._finalized:
            raise RuntimeError("ObjectRegistry must be finalized before using resolved objects")

        if self._resolved_objects_cache is not None:
            return self._resolved_objects_cache

        total_objects = len(self.objects) * self.num_envs
        if total_objects == 0:
            self._resolved_objects_cache = []
        else:
            all_indices = torch.arange(total_objects, device=self.device)
            self._resolved_objects_cache = self.resolve_indices(all_indices)

        return self._resolved_objects_cache

    def _find_object_by_type_and_position(self, obj_type: str, position: int) -> str:
        """Find object name by type and position"""
        for obj in self.objects:
            if obj.type == obj_type and obj.position_in_type == position:
                return obj.name
        raise KeyError(f"No {obj_type} object found at position {position}")

    def get_object_type(self, name: str) -> str:
        """Get object type by name"""
        if name not in self.name_to_index:
            raise KeyError(f"Object '{name}' not found")
        return self.objects[self.name_to_index[name]].type

    def get_scene_position(self, name: str) -> int:
        """Get scene object position by name"""
        if name not in self.name_to_index:
            raise KeyError(f"Object '{name}' not found")
        obj = self.objects[self.name_to_index[name]]
        if obj.type != ObjectType.SCENE.value:
            raise ValueError(f"Object '{name}' is not a scene object")
        return obj.position_in_type

    def list_all_objects(self) -> list[str]:
        """List all registered object names"""
        return list(self.name_to_index.keys())

    def get_names_by_type(self, object_type: ObjectType) -> list[str]:
        """List registered actor names of a given ObjectType, in registration order.

        The registry is the single source of truth for which actors exist and what
        type each is, so callers needing e.g. the rigid (INDIVIDUAL) objects or the
        scene elements query here rather than reaching into per-backend config.
        """
        return [obj.name for obj in self.objects if obj.type == object_type.value]
