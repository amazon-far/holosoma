"""Validate exact robot names and establish native indices in configured order."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


def validate_asset_names(
    names: Sequence[str],
    asset_names: Sequence[str],
    *,
    kind: str,
    require_all: bool = False,
) -> None:
    """Require each configured name to identify exactly one asset element.

    Ordering may differ. Bodies may be a subset; articulated DOFs must be fully covered.
    """
    if len(set(asset_names)) != len(asset_names):
        raise ValueError(f"Asset {kind} names are not unique: {list(asset_names)}.")
    available = set(asset_names)
    for name in names:
        if name not in available:
            raise ValueError(
                f"Robot {kind} {name!r} has no asset matches; names must match exactly. "
                f"Asset names: {list(asset_names)}."
            )
    if require_all and available != set(names):
        raise ValueError(f"Asset has unconfigured {kind}s: {sorted(available - set(names))}.")


def configure_robot_names(
    simulator: BaseSimulator, joint_names: list[str], body_names: list[str], *, root_body: str | None = None
) -> None:
    """Establish canonical tensor order and explicit native mappings after loading an asset.

    MuJoCo replaces body IDs with compiled world IDs after attaching the robot to the scene.
    """
    robot = simulator.robot_config
    validate_asset_names(robot.dof_names, joint_names, kind="joint", require_all=True)
    validate_asset_names(robot.body_names, body_names, kind="body")
    simulator.body_name_map = {name: name for name in robot.body_names}
    # PhysX articulation body 0 is the root; MuJoCo supplies its captured root explicitly.
    physical_root = body_names[0] if root_body is None else root_body
    simulator.root_body_name = physical_root if physical_root in robot.body_names else None
    simulator.dof_ids = [joint_names.index(name) for name in robot.dof_names]
    simulator.body_ids = [body_names.index(name) for name in robot.body_names]
