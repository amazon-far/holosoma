from holosoma.config_types.terrain import MeshType, TerrainManagerCfg, TerrainTermCfg
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

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, TERRAIN_REGISTRY)
