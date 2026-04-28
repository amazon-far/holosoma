"""Terrain generation and management"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import trimesh

from holosoma.config_types.terrain import TerrainTermCfg
from holosoma.simulator.shared.terrain_types import TerrainInterface
from holosoma.utils import terrain_utils
from holosoma.utils.path import resolve_data_file_path


class Terrain(TerrainInterface):
    """Procedural terrain generator

    This class generates heightfield-based terrains with various types of obstacles,
    slopes, and challenging features. Supports random terrain generation. The terrain
    is represented as a 2D heightfield that can be converted to trimesh format for
    physics simulation.

    The terrain system divides the world into a grid of sub-terrains, each containing
    a different terrain type or difficulty level. Robots can be spawned at specific
    locations within this grid for training or evaluation.
    """

    def __init__(self, cfg: TerrainTermCfg, num_robots: int) -> None:
        """Initialize the terrain generator.

        Parameters
        ----------
        cfg : TerrainConfig
            Terrain configuration object. See TerrainConfig for detailed parameter documentation.
        num_robots : int
            Number of robots that will use this terrain (affects layout planning).
        """
        self._cfg: TerrainTermCfg = cfg
        self._num_robots: int = num_robots
        self._type = self._cfg.mesh_type

        self._num_rows: int = int(max(1, self._cfg.num_rows * self._cfg.scale_factor))
        self._num_cols: int = int(max(1, self._cfg.num_cols * self._cfg.scale_factor))

        self._env_length: int = max(1, int(self._cfg.terrain_length * self._cfg.scale_factor))
        self._env_width: int = max(1, int(self._cfg.terrain_width * self._cfg.scale_factor))

        if self._type in ["none"]:
            # a fully-managed terrain isn't supported for these types, so just return
            return

        if self._type == "load_obj":
            mesh = self._initialize_obj_config()
        else:
            mesh = self._initialize_terrain_config()

        self._mesh: trimesh.Trimesh = mesh

    def _initialize_obj_config(self) -> trimesh.Trimesh:
        if self._cfg.obj_dir:
            return self._initialize_obj_config_multi()
        return self._initialize_obj_config_single()

    def _initialize_obj_config_single(self) -> trimesh.Trimesh:
        terrain_path = pathlib.Path(resolve_data_file_path(self._cfg.obj_file_path))
        if not terrain_path.exists():
            raise FileNotFoundError(f"Terrain file not found: {terrain_path}")
        print(f"[INFO] Loading custom terrain from: {terrain_path}")

        base = self._load_mesh_auto(terrain_path)

        print(
            f"[INFO] Loaded terrain mesh with {len(base.vertices)} vertices and {len(base.faces)} faces"
        )

        if not self._cfg.tile_mesh:
            self._obj_tile_stride = np.zeros(3)
            return base

        gap = 1e-4 + float(self._cfg.obj_tile_gap)  # 1e-4 keeps tiles from intersecting; obj_tile_gap adds visible separation
        stride = (base.bounds[1] - base.bounds[0]) + gap
        self._obj_tile_stride = stride

        tiles = []
        for r in range(self._num_rows):
            for c in range(self._num_cols):
                tile = base.copy()
                tile.apply_translation([c * stride[0], r * stride[1], 0.0])
                tiles.append(tile)

        return trimesh.util.concatenate(tiles)

    def _initialize_obj_config_multi(self) -> trimesh.Trimesh:
        """Load one mesh per file in `obj_dir` as a separate tile column."""
        obj_dir = pathlib.Path(resolve_data_file_path(self._cfg.obj_dir))
        if not obj_dir.is_dir():
            raise FileNotFoundError(f"Terrain obj_dir not found or not a directory: {obj_dir}")

        mesh_paths = sorted(
            p for p in obj_dir.rglob("*.npz") if p.is_file()
        ) + sorted(
            p for p in obj_dir.rglob("*.obj") if p.is_file()
        )
        if not mesh_paths:
            raise FileNotFoundError(f"No .npz or .obj mesh files found under {obj_dir}")

        cap = int(self._cfg.obj_max_tiles)
        if cap > 0 and len(mesh_paths) > cap:
            print(f"[INFO] Capping per-tile meshes: {len(mesh_paths)} -> {cap}")
            mesh_paths = mesh_paths[:cap]

        print(f"[INFO] Loading {len(mesh_paths)} per-tile meshes from: {obj_dir}")
        bases: list[trimesh.Trimesh] = [self._load_mesh_auto(p) for p in mesh_paths]
        self._obj_mesh_paths = [str(p) for p in mesh_paths]

        max_faces = int(self._cfg.obj_max_faces_per_tile)
        if max_faces > 0:
            # Use quadric decimation from open3d so we keep a continuous surface
            # (random face sampling leaves holes that USD renders as point clouds).
            import open3d as o3d

            total_in, total_out = 0, 0
            for i, m in enumerate(bases):
                total_in += len(m.faces)
                if len(m.faces) > max_faces:
                    o3d_mesh = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(np.asarray(m.vertices, dtype=np.float64)),
                        o3d.utility.Vector3iVector(np.asarray(m.faces, dtype=np.int32)),
                    )
                    o3d_mesh = o3d_mesh.simplify_quadric_decimation(
                        target_number_of_triangles=max_faces
                    )
                    bases[i] = trimesh.Trimesh(
                        vertices=np.asarray(o3d_mesh.vertices),
                        faces=np.asarray(o3d_mesh.triangles),
                        process=False,
                    )
                total_out += len(bases[i].faces)
            print(
                f"[INFO] Decimated per-tile meshes: {total_in} -> {total_out} faces "
                f"(cap={max_faces} per tile, quadric)"
            )

        # Per-tile bbox-aware placement: each tile's "cube" = its own collision footprint.
        # X: cumulative — tile c starts where tile c-1 ended (+ gap).
        # Y (rows): uniform stride = max Y span across meshes (so all rows align).
        # 1e-4 keeps adjacent tiles from intersecting; obj_tile_gap adds visible separation.
        gap = 1e-4 + float(self._cfg.obj_tile_gap)
        tile_spans = np.stack([(m.bounds[1] - m.bounds[0]) for m in bases], axis=0)  # (n_cols, 3)
        tile_mins = np.stack([m.bounds[0] for m in bases], axis=0)                   # (n_cols, 3)

        # X start (left bbox edge in world) for each column, cumulative.
        x_starts = np.zeros(len(bases), dtype=np.float64)
        for c in range(1, len(bases)):
            x_starts[c] = x_starts[c - 1] + tile_spans[c - 1, 0] + gap

        # Per-tile translation so mesh's bbox.min lands at (x_start, 0, 0). Y/Z untouched
        # at this stage (row offset added below).
        per_tile_translation = np.zeros((len(bases), 3), dtype=np.float64)
        per_tile_translation[:, 0] = x_starts - tile_mins[:, 0]

        # Uniform Y stride across rows (rare in current usage; num_rows is usually 1).
        y_stride = float(tile_spans[:, 1].max()) + gap

        self._obj_tile_translations = per_tile_translation  # (n_cols, 3) base translations
        self._obj_tile_y_stride = y_stride
        # Keep _obj_tile_stride populated for legacy callers (uses max X span).
        self._obj_tile_stride = np.array(
            [float(tile_spans[:, 0].max()) + gap, y_stride, 0.0], dtype=np.float64
        )

        n_cols = len(bases)
        n_rows = self._num_rows
        # Override _num_cols so env_origin grid computations line up.
        self._num_cols = n_cols

        print(
            f"[INFO] Multi-mesh grid: {n_rows} rows x {n_cols} cols (per-tile bbox placement, "
            f"total X span = {x_starts[-1] + tile_spans[-1, 0]:.2f}m, y_stride = {y_stride:.2f}m, "
            f"obj_tile_gap = {self._cfg.obj_tile_gap:.3f}m)"
        )

        tiles = []
        total = n_rows * n_cols
        for r in range(n_rows):
            for c, base in enumerate(bases):
                tile = base.copy()
                t = per_tile_translation[c].copy()
                t[1] += r * y_stride
                tile.apply_translation(t)
                tiles.append(tile)
            print(f"[INFO] Tiling progress: {(r + 1) * n_cols}/{total}", flush=True)

        print(f"[INFO] Concatenating {len(tiles)} tiles into one mesh...", flush=True)
        merged = trimesh.util.concatenate(tiles)
        # Ensure normals are populated so USD shades the surface properly.
        _ = merged.face_normals
        _ = merged.vertex_normals
        print(
            f"[INFO] Merged terrain mesh: {len(merged.vertices)} verts, "
            f"{len(merged.faces)} faces",
            flush=True,
        )
        return merged

    def _load_mesh_auto(self, path: pathlib.Path) -> trimesh.Trimesh:
        if path.suffix == ".npz":
            return self._load_mesh_from_npz(path)
        return self._load_mesh_from_obj(path)

    @staticmethod
    def _load_mesh_from_npz(path: pathlib.Path) -> trimesh.Trimesh:
        """Load mesh vertices/faces embedded in an NPZ motion file."""
        with np.load(str(path)) as data:
            if "mesh_vertices" not in data or "mesh_faces" not in data:
                raise ValueError(f"NPZ file {path} does not contain mesh_vertices/mesh_faces keys")
            vertices = data["mesh_vertices"]
            faces = data["mesh_faces"]
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    @staticmethod
    def _load_mesh_from_obj(path: pathlib.Path) -> trimesh.Trimesh:
        """Load a mesh from an OBJ (or other trimesh-supported) file."""
        base = trimesh.load(str(path), process=False)
        if isinstance(base, trimesh.Scene):
            base = base.dump(concatenate=True)  # type: ignore[assignment]
        if not isinstance(base, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid Trimesh: {type(base)}")
        return base

    def _initialize_terrain_config(self) -> trimesh.Trimesh:
        terrain_config = self._cfg.terrain_config
        if self._type == "plane":
            assert len(terrain_config) == 0, "Plane terrain does not support terrain_config"
            terrain_config = {"flat": 1.0}

        # Filter out terrain types with 0.0 probability to avoid generating them
        terrain_config = {k: v for k, v in terrain_config.items() if v > 0.0}
        assert len(terrain_config) > 0, "At least one terrain type must have non-zero probability"

        self._terrain_types: list[str] = list(terrain_config.keys())
        self._terrain_proportions: list[float] = list(terrain_config.values())
        self._proportions: list[float] = [
            np.sum(self._terrain_proportions[: i + 1]) for i in range(len(self._terrain_proportions))
        ]
        self._slope_threshold: float = self._cfg.slope_treshold * self._cfg.scale_factor

        self._border_size: float = self._cfg.border_size * self._cfg.scale_factor
        self._num_sub_terrains: int = self._num_rows * self._num_cols
        self._env_origins: np.ndarray = np.zeros((self._num_rows, self._num_cols, 3))

        # don't apply scale factor to horizontal scale, it's applied indirectly through other params
        self._horizontal_scale: float = self._cfg.horizontal_scale

        # apply scale factor so vertical scale is proportional to horizontal scaling
        self._vertical_scale: float = self._cfg.vertical_scale * self._cfg.scale_factor

        self._width_per_env_pixels: int = int(self._env_width / self._horizontal_scale)
        self._length_per_env_pixels: int = int(self._env_length / self._horizontal_scale)

        self._border: int = max(1, int(self._border_size / self._horizontal_scale))
        self._tot_cols: int = max(1, int(self._num_cols * self._width_per_env_pixels + 2 * self._border))
        self._tot_rows: int = max(1, int(self._num_rows * self._length_per_env_pixels + 2 * self._border))

        self._total_width: float = self._num_cols * self._env_width + 2 * self._border_size
        self._total_length: float = self._num_rows * self._env_length + 2 * self._border_size

        self._height_field_raw: np.ndarray = np.zeros((self._tot_rows, self._tot_cols), dtype=np.int16)
        self._max_slope: float = self._cfg.max_slope
        self.randomized_terrain()

        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            self._height_field_raw, self._horizontal_scale, self._vertical_scale, self._slope_threshold
        )
        mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.vertices[..., :2] -= self._border_size
        return mesh

    def sample_env_origins(self) -> np.ndarray:
        if self._type == "load_obj":
            origin_grid = self._get_load_obj_env_origin_grid()
            # Deterministic round-robin: evenly distribute envs across tiles
            flat_origins = origin_grid.reshape(-1, 3)  # [num_tiles, 3]
            num_tiles = flat_origins.shape[0]
            tile_indices = np.arange(self._num_robots) % num_tiles
            return flat_origins[tile_indices]
        else:
            origin_grid = self._env_origins
            terrain_levels = np.random.randint(0, self._num_rows, (self._num_robots,))
            terrain_types = np.floor_divide(
                np.arange(self._num_robots),
                (self._num_robots / self._num_cols),
            ).astype(np.int32)
            return origin_grid[terrain_levels, terrain_types]

    @property
    def mesh(self) -> trimesh.Trimesh:
        return self._mesh

    def _get_load_obj_env_origin_grid(self) -> np.ndarray:
        grid = getattr(self, "_load_obj_origin_grid", None)
        if grid is None:
            grid = self._build_load_obj_env_origin_grid()
            self._load_obj_origin_grid = grid
        return grid

    def _build_load_obj_env_origin_grid(self) -> np.ndarray:
        """Compute per-tile origins for OBJ terrains."""
        if not hasattr(self, "_mesh"):
            raise RuntimeError("Mesh must be initialized before computing load_obj env origins.")

        # Multi-mesh mode always tiles in num_rows x num_cols grid.
        is_multi = bool(self._cfg.obj_dir)

        if not is_multi and not self._cfg.tile_mesh:
            offset = np.array(self._cfg.env_origin_in_tile or [0.0, 0.0, 0.0], dtype=np.float32)
            return offset.reshape(1, 1, 3)

        if is_multi or self._cfg.env_origin_in_tile is not None:
            offset = np.array(
                self._cfg.env_origin_in_tile or [0.0, 0.0, 0.0], dtype=np.float32
            )
            origins = np.zeros((self._num_rows, self._num_cols, 3), dtype=np.float32)
            # Multi-mesh mode uses per-tile translations (different X offset per col);
            # legacy single-mesh tiling falls back to a uniform stride.
            if is_multi and hasattr(self, "_obj_tile_translations"):
                per_tile = self._obj_tile_translations.astype(np.float32)
                y_stride = float(self._obj_tile_y_stride)
                for r in range(self._num_rows):
                    for c in range(self._num_cols):
                        t = per_tile[c].copy()
                        t[1] += r * y_stride
                        origins[r, c] = t + offset
            else:
                stride = self._obj_tile_stride
                for r in range(self._num_rows):
                    for c in range(self._num_cols):
                        origins[r, c] = np.array([c * stride[0], r * stride[1], 0.0], dtype=np.float32) + offset
            return origins

        bounds = self._mesh.bounds.astype(np.float64)
        min_corner, max_corner = bounds
        span = max_corner - min_corner
        eps = 1e-9
        tile_length = span[0] / max(self._num_rows, 1)
        tile_width = span[1] / max(self._num_cols, 1)
        if tile_length <= eps or tile_width <= eps:
            raise ValueError("Loaded OBJ mesh must span positive X and Y extents to place env origins.")

        row_centers = min_corner[0] + (np.arange(self._num_rows) + 0.5) * tile_length
        col_centers = min_corner[1] + (np.arange(self._num_cols) + 0.5) * tile_width
        grid_x, grid_y = np.meshgrid(row_centers, col_centers, indexing="ij")

        heights = np.full((self._num_rows, self._num_cols), min_corner[2], dtype=np.float64)
        vertices = self._mesh.vertices
        if vertices.size > 0:
            row_indices = np.clip(
                ((vertices[:, 0] - min_corner[0]) / tile_length).astype(np.int64), 0, self._num_rows - 1
            )
            col_indices = np.clip(
                ((vertices[:, 1] - min_corner[1]) / tile_width).astype(np.int64), 0, self._num_cols - 1
            )
            np.maximum.at(heights, (row_indices, col_indices), vertices[:, 2])

        return np.stack(
            (
                grid_x.astype(np.float32, copy=False),
                grid_y.astype(np.float32, copy=False),
                heights.astype(np.float32, copy=False),
            ),
            axis=-1,
        )

    def randomized_terrain(self) -> None:
        """Generate randomized terrain layout with mixed terrain types.

        Creates a grid of sub-terrains where each cell is randomly assigned
        a terrain type based on the configured proportions. Difficulty levels
        are also randomly assigned except for slope terrains which use
        progressive difficulty based on row position.
        """
        proportions = np.array(self._terrain_proportions) / np.sum(self._terrain_proportions)
        for k in range(self._num_sub_terrains):
            print(f"generating randomized terrains {k} / {self._num_sub_terrains}     ", end="\r")
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self._num_rows, self._num_cols))

            terrain_type = np.random.choice(self._terrain_types, p=proportions)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            if terrain_type in {"smooth_slope", "rough_slope", "slope"}:
                difficulty = i / self._num_rows
            terrain = self.make_terrain(terrain_type, difficulty)
            self.add_terrain_to_map(terrain, int(i), int(j))
        print("\n generated all randomized terrains!")

    def make_terrain(self, terrain_type: str, difficulty: float) -> Any:
        """Create a single sub-terrain of the specified type and difficulty.

        Parameters
        ----------
        terrain_type : str
            Type of terrain to generate (e.g., "flat", "rough", "slope", "stairs").
        difficulty : float
            Difficulty level in range [0, 1] where 0 is easiest and 1 is hardest.

        Returns
        -------
        terrain_utils.SubTerrain
            Generated sub-terrain object with populated heightfield data.
        """
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self._length_per_env_pixels,
            length=self._width_per_env_pixels,
            vertical_scale=self._vertical_scale,
            horizontal_scale=self._horizontal_scale,
        )

        terrain_func = getattr(self, f"_{terrain_type}_terrain_func")
        terrain_func(terrain, difficulty)
        return terrain

    def add_terrain_to_map(self, terrain: Any, row: int, col: int) -> None:
        """Add a sub-terrain to the global heightfield map at specified position.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Sub-terrain object to add to the global map.
        row : int
            Row index in the terrain grid where to place the sub-terrain.
        col : int
            Column index in the terrain grid where to place the sub-terrain.
        """
        i = row
        j = col
        # map coordinate system
        start_x = self._border + i * self._length_per_env_pixels
        end_x = self._border + (i + 1) * self._length_per_env_pixels
        start_y = self._border + j * self._width_per_env_pixels
        end_y = self._border + (j + 1) * self._width_per_env_pixels
        self._height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # Origin is center of the tile, not the corner.
        env_origin_x = (i + 0.5) * self._env_length
        env_origin_y = (j + 0.5) * self._env_width

        x1 = int((self._env_length / 2.0 - 0.5) / terrain.horizontal_scale)
        x2 = int((self._env_length / 2.0 + 0.5) / terrain.horizontal_scale)
        y1 = int((self._env_width / 2.0 - 0.5) / terrain.horizontal_scale)
        y2 = int((self._env_width / 2.0 + 0.5) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self._env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def _flat_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Create a completely flat terrain with zero height everywhere.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `height_field_raw` attribute to modify.
        difficulty : float
            Difficulty level in range [0, 1], unused for flat terrain.
        """
        terrain.height_field_raw[:] = 0.0

    def _rough_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate rough terrain with random height variations below ground level.

        Creates uniform random heights between -max_height and -0.025 meters,
        where max_height increases with difficulty. All terrain is below ground level.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `height_field_raw` and `vertical_scale` attributes.
        difficulty : float
            Difficulty level in range [0, 1], controls maximum height variation.
        """
        max_height = 0.025 * difficulty / 0.9
        terrain.height_field_raw = (
            np.random.uniform(-max_height * 2 - 0.025, -0.025, terrain.height_field_raw.shape) / terrain.vertical_scale
        )

    def _smooth_slope_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate a smooth sloped terrain using pyramid slope pattern.

        Creates a sloped terrain where the slope angle increases with difficulty.
        Randomly chooses between upward and downward slopes. Uses terrain_utils
        to create a pyramid-shaped slope with a central platform.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `height_field_raw` attribute to modify.
        difficulty : float
            Difficulty level in range [0, 1], controls slope steepness.
        """
        slope = difficulty * self._max_slope
        random_01 = np.random.randint(0, 2)
        down = random_01 * 2 - 1
        slope *= down
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=self._cfg.platform_size)

    def _rough_slope_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate a rough sloped terrain with random height variations on the slope.

        First creates a pyramid-shaped slope (same as _smooth_slope_terrain_func),
        then adds random uniform terrain variations on top. The amplitude of
        roughness is randomly chosen from the configured amplitude range.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `height_field_raw` attribute to modify.
        difficulty : float
            Difficulty level in range [0, 1], controls slope steepness.
        """
        slope = difficulty * self._max_slope
        amplitude = np.random.uniform(self._cfg.amplitude_range[0], self._cfg.amplitude_range[1])
        random_01 = np.random.randint(0, 2)
        down = random_01 * 2 - 1
        slope *= down
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=self._cfg.platform_size)
        terrain_utils.random_uniform_terrain(
            terrain, min_height=-amplitude, max_height=amplitude, step=terrain.vertical_scale, downsampled_scale=0.2
        )

    def _smooth_stairs_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate smooth stair terrain with increasing step height based on difficulty.

        Creates pyramid-shaped stairs with random step width from configured range
        and step height that increases with difficulty (0.05m to 0.23m). Randomly
        chooses between upward and downward stairs. Uses terrain_utils to create
        the stair pattern with a central platform.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `height_field_raw` attribute to modify.
        difficulty : float
            Difficulty level in range [0, 1], controls step height.
        """
        step_width = np.random.uniform(self._cfg.step_width_range[0], self._cfg.step_width_range[1])
        step_height = 0.05 + 0.18 * difficulty
        random_01 = np.random.randint(0, 2)
        down = random_01 * 2 - 1
        step_height *= down
        terrain_utils.pyramid_stairs_terrain(
            terrain, step_width=step_width, step_height=step_height, platform_size=self._cfg.platform_size
        )

    def _rough_stairs_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate rough stair terrain with random height variations on the stairs.

        First creates pyramid-shaped stairs (same as _smooth_stairs_terrain_func),
        then adds random uniform terrain variations on top. The amplitude of
        roughness is randomly chosen from the configured amplitude range.
        Step height increases with difficulty and can be positive or negative.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `height_field_raw` attribute to modify.
        difficulty : float
            Difficulty level in range [0, 1], controls step height.
        """
        step_width = np.random.uniform(self._cfg.step_width_range[0], self._cfg.step_width_range[1])
        step_height = 0.05 + 0.18 * difficulty
        amplitude = np.random.uniform(self._cfg.amplitude_range[0], self._cfg.amplitude_range[1])
        random_01 = np.random.randint(0, 2)
        down = random_01 * 2 - 1
        step_height *= down
        terrain_utils.pyramid_stairs_terrain(
            terrain, step_width=step_width, step_height=step_height, platform_size=self._cfg.platform_size
        )
        terrain_utils.random_uniform_terrain(
            terrain, min_height=-amplitude, max_height=amplitude, step=terrain.vertical_scale, downsampled_scale=0.2
        )

    def _slope_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate a linear slope terrain with peak in the middle.

        Creates a terrain that slopes up linearly from left to center, then
        slopes down linearly from center to right, forming a triangular ridge.
        The slope steepness increases with difficulty. This is different from
        the pyramid slopes as it creates smooth linear transitions.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `width`, `length`, `horizontal_scale`, `vertical_scale`,
            and `height_field_raw` attributes.
        difficulty : float
            Difficulty level in range [0, 1], controls slope steepness.
        """
        slope = difficulty * self._max_slope
        width = int(terrain.width / 2)
        length = terrain.length
        x = np.arange(0, width)
        y = np.arange(0, length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(width, 1)
        max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * width)
        terrain.height_field_raw[:width, np.arange(length)] += (max_height * xx / width).astype(
            terrain.height_field_raw.dtype
        )
        terrain.height_field_raw[width:, np.arange(length)] += (max_height * (width - xx) / width).astype(
            terrain.height_field_raw.dtype
        )

    def _low_obstacles_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate terrain with randomly placed square depressions (obstacles).

        Creates 30 square obstacles of fixed size (terrain.width//10) placed at
        random positions. All obstacles are at the same depth below ground level,
        with depth increasing with difficulty. The terrain starts flat and then
        obstacles are carved out as negative height regions.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `width`, `length`, `vertical_scale`, and
            `height_field_raw` attributes.
        difficulty : float
            Difficulty level in range [0, 1], controls obstacle depth.
        """
        max_height = 0.03 * difficulty / 0.9
        obst_size = terrain.width // 10
        obst_num = 30
        xs = np.random.randint(0, terrain.length - obst_size, (obst_num,))
        ys = np.random.randint(0, terrain.width - obst_size, (obst_num,))
        terrain.height_field_raw[:] = 0.0
        for i in range(obst_num):
            terrain.height_field_raw[xs[i] : xs[i] + obst_size, ys[i] : ys[i] + obst_size] = (
                -max_height / terrain.vertical_scale
            )

    def _gap_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate terrain with gaps and platforms in the center.

        Creates a central platform at ground level surrounded by gaps (depressions)
        at random depth (0.4-0.6m). Gap size increases with difficulty from 0 to 0.5m.
        May create additional outer gaps/platforms if terrain is large enough.
        The pattern is symmetric around the center of the terrain.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `length`, `width`, `horizontal_scale`, `vertical_scale`,
            and `height_field_raw` attributes.
        difficulty : float
            Difficulty level in range [0, 1], controls gap size.
        """
        gap_size = 0.5 * difficulty
        platform_size = self._cfg.platform_size
        gap_size = math.ceil(gap_size / terrain.horizontal_scale)
        platform_size = math.ceil(platform_size / terrain.horizontal_scale)

        center_x = terrain.length // 2
        center_y = terrain.width // 2

        depth = np.random.uniform(0.4, 0.6)
        depth = depth / terrain.vertical_scale

        # First gap rectangle
        x1 = platform_size // 2
        x2 = x1 + gap_size
        y1 = platform_size // 2
        y2 = y1 + gap_size

        # Second gap rectangle
        x_margin = (terrain.length - platform_size) // 4 - gap_size
        y_margin = (terrain.width - platform_size) // 4 - gap_size
        if x_margin > 8 and y_margin > 8:
            x3 = x2 + x_margin
            x4 = x3 + gap_size
            y3 = y2 + y_margin
            y4 = y3 + gap_size
            terrain.height_field_raw[center_x - x4 : center_x + x4, center_y - y4 : center_y + y4] = -depth
            terrain.height_field_raw[center_x - x3 : center_x + x3, center_y - y3 : center_y + y3] = 0

        terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -depth
        terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0

    def _stepping_stone_terrain_func(self, terrain: Any, difficulty: float) -> None:
        """Generate terrain with stepping stones and gaps with a central platform.

        Creates a grid of stepping stones separated by gaps, with stone density
        and size decreasing as difficulty increases. The background (gaps) is set
        to a random depth between 0.4-0.6m below ground level. A central square
        platform is always maintained at ground level for robot spawning.

        Parameters
        ----------
        terrain : terrain_utils.SubTerrain
            Terrain object with `width`, `length`, `horizontal_scale`, `vertical_scale`,
            and `height_field_raw` attributes.
        difficulty : float
            Difficulty level in range [0, 1] where 0 creates dense large stones
            and 1 creates sparse small stones.
        """
        H, W = terrain.length, terrain.width

        # Convert meters -> grid cells for depth and platform
        depth = np.random.uniform(0.4, 0.6)
        gap_val = -depth / terrain.vertical_scale
        cp_cells = max(1, int(self._cfg.platform_size / terrain.horizontal_scale))

        # 1. Fill entire field with gap depth
        terrain.height_field_raw[:] = gap_val

        # 2. Divide into fixed grid
        n = 10
        stride_x = W / n
        stride_y = H / n

        # 3. Compute stone size in cells: shrink with difficulty
        max_cell = int(min(stride_x, stride_y))
        stone_shrink_ratio = 0.4
        stone_cells = max(1, int((1 - difficulty * stone_shrink_ratio) * max_cell))

        # 4) Place stepping stones at random offsets within each cell
        for i in range(n):
            for j in range(n):
                # cell boundaries
                cell_x1 = int(i * stride_x)
                cell_x2 = int((i + 1) * stride_x)
                cell_y1 = int(j * stride_y)
                cell_y2 = int((j + 1) * stride_y)
                # random top-left within cell such that stone fits
                if cell_x2 - cell_x1 >= stone_cells:
                    x1 = np.random.randint(cell_x1, cell_x2 - stone_cells + 1)
                else:
                    x1 = cell_x1
                if cell_y2 - cell_y1 >= stone_cells:
                    y1 = np.random.randint(cell_y1, cell_y2 - stone_cells + 1)
                else:
                    y1 = cell_y1
                x2 = min(W, x1 + stone_cells)
                y2 = min(H, y1 + stone_cells)
                terrain.height_field_raw[y1:y2, x1:x2] = 0

        # 5. Carve out the central flat platform
        cx, cy = W // 2, H // 2
        half = cp_cells // 2
        x1 = max(0, cx - half)
        x2 = min(W, cx + half)
        y1 = max(0, cy - half)
        y2 = min(H, cy + half)
        terrain.height_field_raw[y1:y2, x1:x2] = 0
