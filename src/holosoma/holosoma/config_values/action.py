"""Default action manager configurations."""

from holosoma.config_types.action import ActionManagerCfg
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos as _g1_29dof_joint_pos
from holosoma.config_values.loco.t1.action import t1_29dof_joint_pos as _t1_29dof_joint_pos
from holosoma.utils.config_registry import ConfigRegistry

ACTION_REGISTRY = ConfigRegistry(ActionManagerCfg, group="holosoma.config.action")

none = ACTION_REGISTRY.add("none", None)
t1_29dof_joint_pos = ACTION_REGISTRY.add("t1_29dof_joint_pos", _t1_29dof_joint_pos)
g1_29dof_joint_pos = ACTION_REGISTRY.add("g1_29dof_joint_pos", _g1_29dof_joint_pos)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, ACTION_REGISTRY)
