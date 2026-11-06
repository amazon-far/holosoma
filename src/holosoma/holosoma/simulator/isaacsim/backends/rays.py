"""Mutable scan patterns on the native Lab 3 multi-mesh ray caster."""

from functools import lru_cache
from typing import Any, cast

import torch

from . import as_tensor, current


@lru_cache(maxsize=None)
def mutable_ray_caster() -> type[Any]:
    from isaaclab.sensors import MultiMeshRayCaster
    from isaaclab.utils.math import quat_apply

    native_class: Any = MultiMeshRayCaster.resolve_class()

    class MutableRayCaster(native_class):
        def _register_sites_for_expr(self, prim_expr: str) -> list[str]:
            if prim_expr == current().env_regex:
                import warp as wp
                from isaaclab_newton.physics import NewtonManager

                # Sensors are built before Holosoma installs the clone plan.
                # The native Newton resolver cannot recognize an environment
                # root until then, so register its public per-world site here.
                return [NewtonManager.cl_register_site(None, wp.transform(), per_world=True)]
            return cast(list[str], super()._register_sites_for_expr(prim_expr))

        @property
        def ray_starts_w(self) -> torch.Tensor:
            return as_tensor(self._ray_starts_w)

        def set_ray_directions(self, directions: torch.Tensor) -> None:
            target = as_tensor(self.ray_directions)
            if directions.shape != (self.num_rays, 3):
                raise ValueError(f"Scan shape must remain {(self.num_rays, 3)}, got {directions.shape}.")
            rotation = torch.tensor(self.cfg.offset.rot, device=target.device, dtype=target.dtype)
            rotated = quat_apply(rotation.expand(self.num_rays, -1), directions.to(target))
            target.copy_(rotated.unsqueeze(0).expand_as(target))
            self.mark_outdated()

        def mark_outdated(self) -> None:
            self.update(0.0, force_recompute=True)

    return MutableRayCaster
