"""Configuration types for the terrain manager."""

from __future__ import annotations

from dataclasses import field
from enum import Enum

from pydantic.dataclasses import dataclass


class MeshType(str, Enum):
    """Terrain mesh type enumeration.

    Defines the different types of terrain representations available in the simulator.
    Each type has different computational costs and fidelity levels.
    """

    NONE = "none"
    """No terrain mesh.

    Used when terrain height measurement is not needed. Raises errors if height measurement is attempted.
    """

    PLANE = "plane"
    """Simple flat plane terrain.

    Minimal computational cost with uniform height. Good for basic locomotion tasks without terrain variation.
    """

    TRIMESH = "trimesh"
    """Triangle mesh terrain representation.

    Highest fidelity and computational cost. Supports complex terrain geometry and detailed collision detection.
    """

    LOAD_OBJ = "load_obj"
    """Load an OBJ file as a terrain. This is temporary."""

    FAKE = "fake"
    """Mock terrain for testing purposes.

    Behaves like plane but with different internal handling for development/testing scenarios.
    """

    def __str__(self) -> str:
        """Return the string value for backward compatibility with existing configs."""
        return self.value

    def __hash__(self) -> int:
        """Return hash of the enum value for use in sets and as dict keys."""
        return hash(self.value)

    def __eq__(self, other) -> bool:
        """Enable comparison with strings for backward compatibility."""
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


@dataclass(frozen=True)
class SpawnCfg:
    """Configuration for robot spawning behavior."""

    randomize_tiles: bool = True
    """Randomize terrain tile selection for curriculum learning.
    When True, robots spawn at random tiles across the terrain grid.
    When False, all robots spawn at tile (0,0) for deterministic evaluation."""

    xy_offset_range: float = 1.0
    """Random XY offset range within terrain tile (±meters from tile center).
    Adds positional variety within the selected tile.
    Set to 0.0 for deterministic spawning at exact tile center."""

    query_terrain_height: bool = False
    """Query terrain height after applying XY offset to set correct spawn Z position.

    When False (default - faster):
      - Uses original simple spawning: applies XY offset but keeps original Z from env_origins
      - Fastest option, works well for plane/flat terrain
      - May cause robots to spawn slightly below/above terrain on rough terrain

    When True (More accurate for rough terrain):
      - Queries actual terrain height at new XY position via ray casting
      - Updates Z position to place robot correctly on terrain surface
      - Slower due to ray casting, but necessary for uneven terrain
      - Use with use_grid_sampling for maximum safety on rough terrain"""

    use_grid_sampling: bool = False
    """Use grid sampling when querying terrain height (requires query_terrain_height=True).

    Only applies if query_terrain_height=True:
      - False: Single ray cast at new XY position (1 ray per robot)
                Good for smooth terrain with gentle slopes
      - True:  Sample grid of points around position, take max height (9 rays per robot)
               Ensures robot clears all terrain within footprint
               Essential for rough terrain with obstacles/steep features

    If query_terrain_height=False, this parameter is ignored."""

    grid_size: int = 3
    """Grid sampling: number of sample points per dimension (e.g., 3 = 3x3 = 9 points).
    Only used if query_terrain_height=True and use_grid_sampling=True."""

    grid_spacing: float = 0.2
    """Grid sampling: spacing between sample points in meters.
    Should roughly match robot footprint size.
    Only used if query_terrain_height=True and use_grid_sampling=True."""


@dataclass(frozen=True)
class TerrainTermCfg:
    """Configuration for the terrain manager.

    This class defines all parameters needed to configure the terrain manager.
    """

    func: str
    """Import path for the terrain hook (function or callable class)."""

    static_friction: float
    """Static friction coefficient between robot and terrain."""

    dynamic_friction: float
    """Dynamic friction coefficient between robot and terrain."""

    restitution: float
    """Bounce/elasticity coefficient (0=no bounce, 1=perfect bounce)."""

    mesh_type: MeshType = MeshType.PLANE
    """Type of terrain mesh representation.

    See MeshType enum for options:
    - "trimesh": Triangle mesh (default, highest fidelity)
    - "plane": Simple flat plane
    - "none": No terrain mesh
    - "fake": Mock terrain for testing
    """

    hide_visual: bool = False
    """Render the terrain invisible while keeping its collider (default ``False``). Set ``True`` when
    a scene supplies its own visible floor; the robot still collides with the terrain."""

    horizontal_scale: float = 0.1
    """Horizontal resolution of terrain grid in meters."""

    vertical_scale: float = 0.005
    """Vertical scaling factor for height variations."""

    border_size: int = 40
    """Size of border around terrain in meters."""

    terrain_move_down_ratio: float = 0.0
    """Ratio for moving to easier terrain levels."""

    terrain_move_up_ratio: float = 0.0
    """Ratio for moving to harder terrain levels."""

    terrain_length: float = 8.0
    """Length of each terrain tile in meters."""

    terrain_width: float = 8.0
    """Width of each terrain tile in meters."""

    num_rows: int = 10
    """Number of terrain rows (difficulty levels)."""

    num_cols: int = 20
    """Number of terrain columns (terrain types)."""

    spawn: SpawnCfg = field(default_factory=SpawnCfg)
    """Spawn behavior configuration."""

    # Dictionary keys must match terrain generation function names (without '_terrain_func' suffix)
    # See terrain.py for available terrain types and their corresponding functions
    terrain_config: dict[str, float] = field(default_factory=dict)
    """Dictionary mapping terrain types to their proportions.

    This implements a dynamic function dispatch pattern where each key
    corresponds to a terrain generation function with the naming pattern
    '{terrain_type}_terrain_func()'. For example:
    - "flat" → flat_terrain_func()
    - "rough" → rough_terrain_func()
    - "low_obst" → low_obstacles_terrain_func()
    Values represent the proportion of terrain tiles that should use
    that terrain type (must sum to ≤ 1.0).
    """

    max_slope: float = 0.3
    """Maximum slope angle allowed before correction to vertical."""

    platform_size: float = 2.0
    """Size of flat platforms in complex terrain."""

    step_width_range: list[float] = field(default_factory=lambda: [0.30, 0.40])
    """Range of step widths for stair-like terrain."""

    amplitude_range: list[float] = field(default_factory=lambda: [0.01, 0.05])
    """Range of height variations for rough terrain."""

    slope_treshold: float = 0.75
    """Slope threshold for trimesh correction to vertical surfaces."""

    obj_file_path: str = ""
    """Path to OBJ file for custom terrain mesh."""

    scale_factor: float = 1.0
    """Use for performance to scale border_size, terrain_length, terrain_width, num_ros and num_cols."""

    name: str = "floor"
    """Mujoco-only: object (geom) name to assign to terrain. Use to match MJCF files <contacts> e.g, 'floor'."""


@dataclass(frozen=True)
class TerrainManagerCfg:
    terrain_term: TerrainTermCfg
    """Configuration for the terrain term. Only one terrain term is supported."""
