"""Type aliases for simulator interfaces.

This module provides clear type aliases for simulator-related data structures,
improving code readability and type safety across the IsaacSim and MuJoCo implementations.

The type aliases defined here are used throughout the simulator framework to provide
consistent, self-documenting interfaces that work across different simulator backends.

Examples
--------
>>> from holosoma.simulator.types import ActorNames, EnvIds, ActorStates
>>>
>>> # Clear, typed function signatures
>>> def reset_objects(names: ActorNames, env_ids: EnvIds) -> ActorStates:
>>>     # Implementation here
>>>     pass
>>>
>>> # Usage with clear intent
>>> actor_names: ActorNames = ["obj0", "table_pantene", "robot"]
>>> env_ids: EnvIds = torch.tensor([0, 1, 2], device=device)
>>> states: ActorStates = sim.get_actor_states(actor_names, env_ids)
"""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch

# Core simulator types for actor/object management
ActorNames = list[str]
"""List of actor names for identification.

Examples: ["obj0", "table_pantene", "robot"]

Used for specifying which actors to operate on in name-based simulator methods.
Provides semantic meaning and enables dynamic object specification.
"""

ActorIndices = torch.Tensor
"""Flat tensor indices for direct actor access.

Shape: [num_objects * num_envs], dtype: torch.long

Pre-computed indices that can be used for direct tensor access without name lookup overhead.
These indices account for the virtual address space layout and multi-environment setup.
Obtained via simulator.get_actor_indices(names, env_ids).
"""

EnvIds = torch.Tensor
"""Environment IDs tensor.

Shape: [num_envs], dtype: torch.long

Specifies which environments to operate on in multi-environment setups.
Used to select specific environments for state queries and updates.
"""

ActorStates = torch.Tensor
"""Actor state tensor containing full pose and velocity information.

Shape: [num_objects * num_envs, 13], dtype: torch.float32
Format: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]

Where:
- [x, y, z]: Position in world coordinates (meters)
- [qx, qy, qz, qw]: Quaternion orientation in xyzw format (unitless)
- [vx, vy, vz]: Linear velocity (meters/second)
- [wx, wy, wz]: Angular velocity (radians/second)
"""

ActorPoses = torch.Tensor
"""Actor pose tensor containing position and orientation only.

Shape: [num_objects * num_envs, 7], dtype: torch.float32
Format: [x, y, z, qx, qy, qz, qw]

Where:
- [x, y, z]: Position in world coordinates (meters)
- [qx, qy, qz, qw]: Quaternion orientation in xyzw format (unitless)

Used for initial poses, target poses, and other scenarios where velocity is not needed.
Subset of ActorStates containing only the pose components.
"""
