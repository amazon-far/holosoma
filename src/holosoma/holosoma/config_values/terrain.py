from holosoma.config_types.terrain import MeshType, SpawnCfg, TerrainManagerCfg, TerrainTermCfg
from holosoma.utils.config_registry import ConfigRegistry

TERRAIN_REGISTRY = ConfigRegistry(TerrainManagerCfg, group="holosoma.config.terrain")

terrain_locomotion_plane = TERRAIN_REGISTRY.add(
    "terrain_locomotion_plane",
    TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            mesh_type=MeshType.PLANE,
            horizontal_scale=1.0,
            vertical_scale=0.005,
            border_size=40,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            terrain_length=8.0,
            terrain_width=8.0,
            num_rows=10,
            num_cols=20,
            max_slope=0.3,
            platform_size=2.0,
            step_width_range=[0.30, 0.40],
            amplitude_range=[0.01, 0.05],
            slope_treshold=0.75,
        )
    ),
)

terrain_locomotion_mix = TERRAIN_REGISTRY.add(
    "terrain_locomotion_mix",
    TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            mesh_type=MeshType.TRIMESH,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            border_size=40,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            terrain_length=8.0,
            terrain_width=8.0,
            num_rows=10,
            num_cols=20,
            terrain_config={
                "flat": 0.2,
                "rough": 0.6,
                "low_obstacles": 0.2,
                "smooth_slope": 0.0,
                "rough_slope": 0.0,
            },
            max_slope=0.3,
            slope_treshold=0.75,
        )
    ),
)

terrain_load_obj = TERRAIN_REGISTRY.add(
    "terrain_load_obj",
    TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            mesh_type=MeshType.LOAD_OBJ,
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            obj_file_path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_parkour.obj",
        )
    ),
)

# Stepped-block course for vision (depth-distillation) locomotion: an 80m x 80m ground plane with
# a line of raised blocks of varying height along +X. Registered so it is selectable as
# ``terrain:terrain-load-step``; override the mesh with
# ``--terrain.terrain-term.obj-file-path <abs path>``.
terrain_load_step = TERRAIN_REGISTRY.add(
    "terrain_load_step",
    TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            mesh_type=MeshType.LOAD_OBJ,
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            obj_file_path="holosoma/data/terrains/terrain.obj",
            num_rows=1,
            num_cols=1,
            spawn=SpawnCfg(
                randomize_tiles=False,
                query_terrain_height=True,
                use_grid_sampling=True,
            ),
        )
    ),
)

terrain_locomotion_stairs_and_slope_eval = TERRAIN_REGISTRY.add(
    "terrain_locomotion_stairs_and_slope_eval",
    TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            mesh_type=MeshType.TRIMESH,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            border_size=1,
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            terrain_length=8.0,
            terrain_width=8.0,
            num_rows=2,
            num_cols=2,
            terrain_config={
                "smooth_stairs": 0.5,
                "smooth_slope": 0.5,
            },
            step_width_range=[0.25, 0.25],
            fixed_step_height=0.15,
            max_slope=0.3,
            slope_treshold=0.75,
            spawn=SpawnCfg(
                randomize_tiles=False,
                xy_offset_range=0.0,
                query_terrain_height=True,
                use_grid_sampling=True,
                grid_size=20,
                grid_spacing=0.01,
            ),
        )
    ),
)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, TERRAIN_REGISTRY)
