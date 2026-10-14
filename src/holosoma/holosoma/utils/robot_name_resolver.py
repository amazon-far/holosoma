"""Task-local resolution of canonical robot selections, independent of task or simulator classes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from holosoma.utils.name_tags import expand_name_spec
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig


class RobotNameResolver:
    """Resolve selections into canonical state-tensor order, caching indices for per-step consumers.

    Created once per task from its robot config. Copies the name and tag tables so a caller cannot
    change the meaning of cached selections by mutating a config after task initialization.
    """

    def __init__(self, robot: RobotConfig, device: str):
        self.body_names = list(robot.body_names)
        self.dof_names = list(robot.dof_names)
        self.body_tags = {key: list(names) for key, names in robot.body_tags.items()}
        self.joint_tags = {key: list(names) for key, names in robot.joint_tags.items()}
        self.device = device
        self._cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def _resolve_cached(self, kind: str, spec: Sequence[str], compute: Callable[[], Any]) -> Any:
        key = (kind, tuple(spec))
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def resolve_body_names(self, spec: Sequence[str]) -> list[str]:
        """Expand ``spec`` (exact body names and/or ``#tag`` references) to an ordered, deduped body-name list.

        Uses the copied robot tag table. Cached by spec; unknown names/tags raise ValueError.
        """
        names = self._resolve_cached(
            "body_names",
            spec,
            lambda: expand_name_spec(spec, self.body_tags, self.body_names, kind="body"),
        )
        return list(names)

    def resolve_joint_names(self, spec: Sequence[str]) -> list[str]:
        """Expand ``spec`` (exact DOF names and/or ``#tag`` references) to an ordered, deduped DOF-name list.

        Uses the copied robot tag table. Cached by spec; unknown names/tags raise ValueError.
        """
        names = self._resolve_cached(
            "joint_names",
            spec,
            lambda: expand_name_spec(spec, self.joint_tags, self.dof_names, kind="joint"),
        )
        return list(names)

    def resolve_body_indices(self, spec: Sequence[str]) -> torch.Tensor:
        """Return the canonical body index of each body in ``resolve_body_names(spec)``.

        LongTensor on ``self.device``, one entry per resolved body (shape ``[0]`` for an empty spec).
        Cached by ``tuple(spec)``: repeated calls return the same tensor for use in per-step terms.
        """

        def compute() -> torch.Tensor:
            indices = [self.body_names.index(name) for name in self.resolve_body_names(spec)]
            return torch.tensor(indices, dtype=torch.long, device=self.device)

        return cast("torch.Tensor", self._resolve_cached("body_indices", spec, compute))

    def resolve_joint_indices(self, spec: Sequence[str]) -> torch.Tensor:
        """Return the position in ``self.dof_names`` of each DOF in ``resolve_joint_names(spec)``.

        LongTensor on ``self.device``, one entry per resolved DOF (shape ``[0]`` for an empty spec).
        Cached by ``tuple(spec)``: repeated calls return the same tensor.
        """

        def compute() -> torch.Tensor:
            indices = [self.dof_names.index(name) for name in self.resolve_joint_names(spec)]
            return torch.tensor(indices, dtype=torch.long, device=self.device)

        return cast("torch.Tensor", self._resolve_cached("joint_indices", spec, compute))
