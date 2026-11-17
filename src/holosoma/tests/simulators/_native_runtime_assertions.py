"""Inspect actual process dependencies after native scene/sensor initialization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict


class NativeRuntimeDependencies(TypedDict):
    uses_kit: bool
    kit_libraries: list[str]
    physx_libraries: list[str]
    libraries: list[str]


def snapshot_native_physics(sim):
    """Read engine state directly so renderer clock bookkeeping cannot hide integration."""
    import warp as wp

    if sim.backend.options.physics_backend == "newton":
        state = sim.sim.physics_manager.get_state_0()
        return tuple(wp.to_torch(getattr(state, name)).clone() for name in ("body_q", "body_qd", "joint_q", "joint_qd"))
    view = sim._robot.root_view
    if sim.backend.uses_kit:
        names = ("get_root_transforms", "get_root_velocities", "get_dof_positions", "get_dof_velocities")
        return tuple(wp.to_torch(getattr(view, name)()).clone() for name in names)
    from isaaclab_ov import tensor_types as TT

    return tuple(
        wp.to_torch(view.get_attribute(name)).clone()
        for name in (TT.ROOT_POSE, TT.ROOT_VELOCITY, TT.DOF_POSITION, TT.DOF_VELOCITY)
    )


def assert_newton_fixed_objects(sim, names):
    """Require actual world-fixed topology, including native zero-velocity state."""
    import numpy as np
    from newton import JointType

    model = sim.sim.physics_manager.get_model()
    state = sim.sim.physics_manager.get_state_0()
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    dimensions = model.joint_dof_dim.numpy()
    types = model.joint_type.numpy()
    starts = model.joint_qd_start.numpy()
    velocities = state.body_qd.numpy()
    for name in names:
        asset = sim.scene.rigid_objects[name]
        assert asset.is_world_fixed, f"{name} still uses the SDK's free-joint kinematic workaround"
        assert asset.cfg.class_type is type(asset), f"{name} config cannot recreate its fixed asset handle"
        for body in asset.body_ids.cpu().tolist():
            joints = np.flatnonzero(child == body)
            assert len(joints) == 1, (name, joints)
            joint = joints[0]
            assert parent[joint] == -1 and types[joint] == JointType.FIXED, (name, joint)
            assert dimensions[joint].sum() == 0 and starts[joint] == starts[joint + 1], name
            assert np.array_equal(velocities[body], np.zeros(6)), (name, velocities[body])
        print(f"FIXED {name}: {asset.num_instances} world-attached bodies, zero dynamic DOFs, no locking armature")


def assert_native_runtime(*, physics: str, uses_kit: bool) -> NativeRuntimeDependencies:
    from isaaclab.utils.version import has_kit

    assert has_kit() == uses_kit, (uses_kit, has_kit())
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and "/" in line.split()[-1]
        }
    )
    # Carbonite is also a dependency of standalone OV/importer wheels. The
    # activated image may supply its libcarb.so from the bundled SDK directory;
    # loading that shared framework library does not initialize the Kit app.
    kit_libraries = [
        path
        for path in libraries
        if "omni.kit" in path.lower() or ("/kit/lib" in path.lower() and Path(path).name != "libcarb.so")
    ]
    physx_libraries = [path for path in libraries if "physx" in Path(path).name.lower()]
    if not uses_kit:
        assert "omni.kit.app" not in sys.modules, "Standalone startup imported the Kit app"
        assert not kit_libraries, f"Standalone startup loaded Kit libraries: {kit_libraries}"
        assert "isaaclab_physx.physics.physx_manager" not in sys.modules
        if physics == "newton":
            assert "ovphysx._bindings" not in sys.modules, "Newton loaded OVPhysX native bindings"
            assert not physx_libraries, f"Newton loaded PhysX libraries: {physx_libraries}"
        else:
            assert physx_libraries, "OVPhysX scene did not load its selected physics engine"
    return {
        "uses_kit": uses_kit,
        "kit_libraries": kit_libraries,
        "physx_libraries": physx_libraries,
        "libraries": libraries,
    }
