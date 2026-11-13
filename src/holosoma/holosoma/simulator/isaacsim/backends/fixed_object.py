"""Newton world-fixed actors without a free joint or an articulation view.

Newton imports a USD FixedJoint to world as a standalone zero-DOF body. Isaac
Lab's RigidObject instead requires an ArticulationView, which excludes these
bodies. This AssetBase implementation uses the ordinary SDK lifecycle and
public Newton model/state arrays. The kinematic flag selects MuJoCo's pose
settable mocap representation; the fixed joint contributes no dynamic DOFs,
locking armature, or equality constraint.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import torch
import warp as wp
from newton import JointType, ModelFlags
from isaaclab.assets import AssetBase
from isaaclab.sim import SimulationContext
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def spawn_fixed(prim_path: str, cfg: Any, *args: Any, original: Any, **kwargs: Any) -> Any:
    """Author a world attachment over the spawned reference, before replication."""
    prim = original(prim_path, cfg, *args, **kwargs)
    stage = prim.GetStage()
    # A compound fixture has one enclosure pose. Keep its geometry/materials
    # under that enclosure rather than referencing only a nested body prim.
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.ArticulationRootAPI):
            child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
            if child.HasAPI(UsdPhysics.MassAPI) and not prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI.Apply(prim)
                for name in ("mass", "density"):
                    source = child.GetAttribute(f"physics:{name}")
                    if source.HasAuthoredValue():
                        prim.GetAttribute(f"physics:{name}").Set(source.Get())
            child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if child.IsA(UsdPhysics.Joint):
            child.SetActive(False)
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(True)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    joint = UsdPhysics.FixedJoint.Define(stage, str(prim.GetPath()) + "/holosoma_world_attachment")
    joint.CreateBody1Rel().SetTargets([prim.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(transform.ExtractTranslation()))
    joint.CreateLocalRot0Attr(Gf.Quatf(transform.ExtractRotationQuat()))
    return prim


class FixedObject(AssetBase):
    """A fixed fixture with per-environment state and explicit pose relocation."""

    num_bodies = 1
    is_world_fixed = True

    @property
    def num_instances(self) -> int:
        return len(self.body_ids)

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False, *, as_proxy: bool = False
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        """Resolve the single enclosure body for scene entities and randomization."""
        indices, names = resolve_matching_names(name_keys, self.body_names, preserve_order)
        if as_proxy:
            return ProxyArray(wp.array(indices, dtype=wp.int32, device=self.device)), names
        return indices, names

    @property
    def data(self) -> Any:
        # Read the current state, including writes made without a physics step.
        model = self.manager.get_model()
        poses = wp.to_torch(self.manager.get_state_0().body_q)
        self._data.root_link_pose_w.torch.copy_(poses.index_select(0, self.body_ids))
        self._data.body_link_pose_w.torch.copy_(self._data.root_link_pose_w.torch[:, None])
        for name, native in (
            ("body_mass", model.body_mass),
            ("body_com_pos_b", model.body_com),
            ("body_inertia", model.body_inertia),
        ):
            value = wp.to_torch(native).index_select(0, self.body_ids)
            target = getattr(self._data, name).torch
            target.copy_(value.reshape(target.shape))
        return self._data

    def _initialize_impl(self) -> None:
        """Bind standalone fixed bodies after cloning and on each hard reset."""
        self.manager = SimulationContext.instance().physics_manager
        model = self.manager.get_model()
        worlds = model.body_world.numpy()
        bodies = sorted(
            (int(worlds[i]), i) for i, path in enumerate(model.body_label) if re.fullmatch(self.cfg.prim_path, path)
        )
        if [world for world, _ in bodies] != list(range(model.world_count)):
            raise RuntimeError(f"Expected one fixed body per world for {self.cfg.prim_path!r}: {bodies}")
        self.body_ids = torch.tensor([body for _, body in bodies], device=self.device, dtype=torch.int64)
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        types, dimensions = model.joint_type.numpy(), model.joint_dof_dim.numpy()
        joints = []
        for _, body in bodies:
            matches = [i for i, c in enumerate(child) if c == body and parent[i] == -1]
            if len(matches) != 1 or types[matches[0]] != JointType.FIXED or dimensions[matches[0]].sum():
                raise RuntimeError(f"Fixed fixture {model.body_label[body]!r} must have a zero-DOF joint to world.")
            joints.append(matches[0])
        self.joint_ids = torch.tensor(joints, device=self.device, dtype=torch.int64)
        self.body_names = [self.cfg.prim_path.rsplit("/", 1)[-1]]

        def buffer(shape: tuple[int, ...]) -> ProxyArray:
            return ProxyArray(wp.zeros(shape, dtype=wp.float32, device=self.device))

        n = self.num_instances
        self._data = SimpleNamespace(
            root_link_pose_w=buffer((n, 7)),
            root_com_vel_w=buffer((n, 6)),
            body_link_pose_w=buffer((n, 1, 7)),
            body_mass=buffer((n, 1)),
            body_com_pos_b=buffer((n, 1, 3)),
            body_inertia=buffer((n, 1, 9)),
            default_root_pose=buffer((n, 7)),
            default_root_vel=buffer((n, 6)),
        )
        # InteractiveScene's public snapshots use the root aliases.
        self._data.root_pose_w = self._data.root_link_pose_w
        self._data.root_vel_w = self._data.root_com_vel_w
        self._data.default_root_pose.torch[:] = torch.tensor(
            (*self.cfg.init_state.pos, *self.cfg.init_state.rot), device=self.device
        )
        # Materials are selected by shape ownership, not an artificial articulation.
        shape_bodies = model.shape_body.numpy()
        self.shape_ids = torch.tensor(
            [[i for i, owner in enumerate(shape_bodies) if owner == body] for _, body in bodies],
            device=self.device,
            dtype=torch.int64,
        )

    def write_root_link_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: Any = None) -> None:
        """Relocate selected fixtures and publish their physics and rendered poses."""
        ids = self._indices(env_ids)
        pose = root_pose.to(device=self.device, dtype=torch.float32)
        if pose.shape != (len(ids), 7):
            raise ValueError(f"Expected {len(ids)} fixed-object poses, got {tuple(pose.shape)}.")
        body_ids = self.body_ids.index_select(0, ids)
        model = self.manager.get_model()
        wp.to_torch(model.joint_X_p).index_copy_(0, self.joint_ids.index_select(0, ids), pose)
        wp.to_torch(model.body_q).index_copy_(0, body_ids, pose)
        for state in (self.manager.get_state_0(), self.manager.get_state_1()):
            if state is not None:
                wp.to_torch(state.body_q).index_copy_(0, body_ids, pose)
        self.manager.add_model_change(ModelFlags.JOINT_PROPERTIES)
        self.manager.invalidate_body_state(env_ids=wp.from_torch(ids.to(torch.int32)))

    def write_root_com_velocity_to_sim_index(self, *, root_velocity: torch.Tensor, env_ids: Any = None) -> None:
        """A fixed fixture cannot carry velocity; accept ordinary zero-velocity resets."""
        if torch.count_nonzero(root_velocity):
            raise ValueError("A fixed fixture has zero dynamic DOFs and cannot have a nonzero velocity.")

    # InteractiveScene's public state restoration uses these root aliases.
    write_root_pose_to_sim_index = write_root_link_pose_to_sim_index
    write_root_velocity_to_sim_index = write_root_com_velocity_to_sim_index

    def _indices(self, env_ids: Any) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_instances, device=self.device)
        if isinstance(env_ids, wp.array):
            env_ids = wp.to_torch(env_ids)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)

    def reset(self, env_ids: Any = None) -> None:
        """No force/history buffers to reset; actor pose restoration uses the state API."""

    def write_data_to_sim(self) -> None:
        """Pose writes already target native buffers and are committed at the next SDK boundary."""

    def update(self, dt: float) -> None:
        """Fixed actors have no integrated state or numerical derivative history."""

    def set_masses_index(self, *, masses: torch.Tensor, env_ids: Any) -> None:
        """Set selected fixture mass metadata without adding dynamic coordinates."""
        self._write_property("body_mass", masses, env_ids, ModelFlags.BODY_INERTIAL_PROPERTIES)

    def set_coms_index(self, *, coms: torch.Tensor, env_ids: Any) -> None:
        """Set selected fixture centers of mass without changing their world attachment."""
        self._write_property("body_com", coms, env_ids, ModelFlags.BODY_INERTIAL_PROPERTIES)

    def set_inertias_index(self, *, inertias: torch.Tensor, env_ids: Any) -> None:
        """Set selected fixture inertia metadata without changing their world attachment."""
        self._write_property("body_inertia", inertias, env_ids, ModelFlags.BODY_INERTIAL_PROPERTIES)

    def _write_property(self, name: str, value: torch.Tensor, env_ids: Any, flag: ModelFlags) -> None:
        """Retain property writes without giving a fixed body dynamic coordinates."""
        target = wp.to_torch(getattr(self.manager.get_model(), name))
        selected = self.body_ids.index_select(0, self._indices(env_ids))
        target.index_copy_(0, selected, value.reshape(len(selected), *target.shape[1:]))
        self.manager.add_model_change(flag)
