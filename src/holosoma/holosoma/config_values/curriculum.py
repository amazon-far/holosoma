"""Default curriculum manager configurations."""

from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_values.loco.g1.curriculum import g1_29dof_curriculum, g1_29dof_curriculum_fast_sac
from holosoma.config_values.loco.t1.curriculum import t1_29dof_curriculum, t1_29dof_curriculum_fast_sac
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum
from holosoma.utils.config_registry import ConfigRegistry

CURRICULUM_REGISTRY = ConfigRegistry(CurriculumManagerCfg, group="holosoma.config.curriculum")

none = CURRICULUM_REGISTRY.add("none", None)
CURRICULUM_REGISTRY.add("t1_29dof", t1_29dof_curriculum)
CURRICULUM_REGISTRY.add("g1_29dof", g1_29dof_curriculum)
CURRICULUM_REGISTRY.add("t1_29dof_fast_sac", t1_29dof_curriculum_fast_sac)
CURRICULUM_REGISTRY.add("g1_29dof_fast_sac", g1_29dof_curriculum_fast_sac)
CURRICULUM_REGISTRY.add("g1_29dof_wbt_curriculum", g1_29dof_wbt_curriculum)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, CURRICULUM_REGISTRY)
