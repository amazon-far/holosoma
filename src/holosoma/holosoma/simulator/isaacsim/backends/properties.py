"""Lab 3 property access preserving Holosoma's existing randomization contract."""

from __future__ import annotations

from typing import Any
from importlib import import_module

import torch
import warp as wp

from . import as_tensor


class NativeProperties:
    """Keep property sampling independent of native storage and writer conventions.

    Property reads are CPU copies, matching the 5.1 property API. Mass/inertia
    writes use Lab 3 setters so the engine updates derived data. OVPhysX body
    damping uses its live control stage because it has no tensor setter.
    """

    def __init__(self, asset: Any, physics: str) -> None:
        self.asset = asset
        self.physics = physics
        self.articulation = hasattr(asset, "num_joints")

    def get_masses(self) -> torch.Tensor:
        return as_tensor(self.asset.data.body_mass).cpu().clone()

    def set_masses(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(self.asset.device)
        self.asset.set_masses_index(masses=values[env_ids].to(self.asset.device), env_ids=ids)

    def get_inertias(self) -> torch.Tensor:
        value = as_tensor(self.asset.data.body_inertia).cpu().clone()
        return value if self.articulation else value[:, 0]

    def set_inertias(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        value = values if values.ndim == 3 else values[:, None]
        self.asset.set_inertias_index(
            inertias=value[env_ids].to(self.asset.device), env_ids=env_ids.to(self.asset.device)
        )

    def get_coms(self) -> torch.Tensor:
        return as_tensor(self.asset.data.body_com_pose_b).cpu().clone()

    def set_coms(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.asset.set_coms_index(coms=values[env_ids].to(self.asset.device), env_ids=env_ids.to(self.asset.device))

    def set_body_damping(self, axis: str, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Write live OVPhysX body drag through its sealed control-stage protocol."""
        if self.physics != "ovphysx":
            raise NotImplementedError(
                "Newton does not expose PhysX per-body linear/angular damping through this adapter. "
                "Disable body-damping randomization or select PhysX."
            )
        from contextlib import ExitStack

        from isaaclab_ov.physics import OvPhysxManager

        ovstage = import_module("ovstage")
        paths = self.asset.root_view.prim_paths
        selected = [paths[index] for index in env_ids.cpu().tolist()]
        if not selected:
            return
        stage = OvPhysxManager.backend.stage
        # Share the manager's ordinal allocator with native gravity updates.
        ordinal = OvPhysxManager._next_control_ordinal
        OvPhysxManager._next_control_ordinal += 1
        with ExitStack() as cleanup:
            dictionary = cleanup.enter_context(ovstage.PathDictionary(stage))
            path_list = dictionary.create_path_list_from_strings(selected)
            cleanup.callback(dictionary.destroy_path_list, path_list)
            query = cleanup.enter_context(stage.query_from_path_list(path_list))
            stage.write_attribute(
                query,
                f"physxRigidBody:{axis}Damping",
                ordinal,
                values.to(device="cpu", dtype=torch.float32).contiguous().numpy(),
                is_array=False,
            ).wait()
            stage.advance_write_floor(ordinal=ordinal).wait()
            OvPhysxManager.backend.physx.update_from_ovstage(ordinal, ordinal)

    def _ov_material_type(self) -> Any:
        from isaaclab_ov import tensor_types as types

        return (
            types.SHAPE_FRICTION_AND_RESTITUTION
            if self.articulation
            else types.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION
        )

    def _newton_materials(self) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        view = self.asset.root_view
        friction = as_tensor(view.get_attribute("shape_material_mu", model)[:, 0])
        restitution = as_tensor(view.get_attribute("shape_material_restitution", model)[:, 0])
        return friction, restitution

    def get_material_properties(self) -> torch.Tensor:
        if self.physics == "isaacsim_physx":
            value = as_tensor(self.asset.root_view.get_material_properties())
        elif self.physics == "ovphysx":
            value = as_tensor(self.asset.root_view.get_attribute(self._ov_material_type()))
        else:
            friction, restitution = self._newton_materials()
            value = torch.stack((friction, friction, restitution), dim=-1)
        return value.cpu().clone()

    @property
    def max_shapes(self) -> int:
        return int(self.get_material_properties().shape[1])

    def set_material_properties(self, values: torch.Tensor, env_ids: torch.Tensor) -> None:
        device = self.asset.device
        if self.physics == "isaacsim_physx":
            # The PhysX tensor view uses the CPU frontend for material properties.
            self.asset.root_view.set_material_properties(
                wp.from_torch(values.contiguous()), wp.from_torch(env_ids.to(dtype=torch.int32).contiguous())
            )
        elif self.physics == "ovphysx":
            # Material attributes live on the CPU, even when dynamics use CUDA.
            self.asset.root_view.set_attribute(
                self._ov_material_type(),
                wp.from_torch(values.cpu().contiguous()),
                indices=wp.from_torch(env_ids.to(device="cpu", dtype=torch.int32).contiguous()),
            )
        else:
            from isaaclab_newton.physics import NewtonManager
            from newton import ModelFlags

            if "Kamino" in type(NewtonManager._solver).__name__:
                raise NotImplementedError(
                    "Kamino shares materials across environments; per-environment material randomization is unsupported."
                )
            if not torch.equal(values[env_ids, :, 0], values[env_ids, :, 1]):
                raise ValueError("Newton requires equal static and dynamic friction samples.")
            friction, restitution = self._newton_materials()
            ids = env_ids.to(device)
            friction[ids] = values[env_ids, :, 0].to(device)
            restitution[ids] = values[env_ids, :, 2].to(device)
            NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
