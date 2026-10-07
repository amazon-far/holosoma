"""
Utility functions for working with USD prims in IsaacSim.
"""

from __future__ import annotations

import fnmatch
from typing import Any

import numpy as np
import numpy.typing as npt
import omni
import omni.log
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def get_current_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


def is_rigid_body_enabled(prim: Usd.Prim) -> bool:
    """Whether a prim has an enabled USD rigid-body API."""
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return False
    attr = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr()
    return not (attr and attr.HasAuthoredValue()) or bool(attr.Get())


def _find_robot_link_prim(link_name: str, stage: Usd.Stage) -> Usd.Prim:
    robot_root_path = "/World/envs/env_0/Robot"
    robot_root = stage.GetPrimAtPath(robot_root_path)
    if not robot_root.IsValid():
        raise ValueError(f"Cannot resolve robot link '{link_name}': prim '{robot_root_path}' does not exist.")

    direct = stage.GetPrimAtPath(f"{robot_root_path}/{link_name}")
    if direct.IsValid() and UsdGeom.Xformable(direct):
        return direct

    matches = [
        prim
        for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())
        if prim.GetName() == link_name and UsdGeom.Xformable(prim)
    ]
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise ValueError(
            f"Cannot resolve robot link '{link_name}' below '{robot_root_path}': "
            f"expected one matching prim, found {len(matches)} ({paths})."
        )
    return matches[0]


def resolve_robot_link_prim_expression(link_name: str, stage: Usd.Stage | None = None) -> str:
    """Resolve a configured body name to its per-environment Isaac prim expression.

    URDF importers may preserve an extra root link, placing robot bodies below paths such as
    ``Robot/import_root/base_link`` instead of directly below ``Robot``. Holosoma body names
    remain plain link names across backends, so Isaac adapters resolve the imported hierarchy
    from env 0 and return the equivalent expression for every environment.
    """
    stage = stage or get_current_stage()
    robot_root_path = "/World/envs/env_0/Robot"
    suffix = str(_find_robot_link_prim(link_name, stage).GetPath())[len(robot_root_path) :]
    return f"/World/envs/env_.*/Robot{suffix}"


def resolve_robot_link_rigid_body_prim_expression(link_name: str, stage: Usd.Stage | None = None) -> str:
    """Resolve a source robot link to the Isaac rigid body that owns its geometry."""
    stage = stage or get_current_stage()
    robot_root_path = "/World/envs/env_0/Robot"
    prim = _find_robot_link_prim(link_name, stage)
    while prim.IsValid() and str(prim.GetPath()).startswith(robot_root_path):
        if is_rigid_body_enabled(prim):
            suffix = str(prim.GetPath())[len(robot_root_path) :]
            return f"/World/envs/env_.*/Robot{suffix}"
        prim = prim.GetParent()
    raise ValueError(f"Robot link '{link_name}' is not owned by a rigid body below '{robot_root_path}'.")


def print_prim_tree(
    prim_path: str, max_depth: int | None = None, indent: int = 0, stage: Usd.Stage | None = None
) -> None:
    """Print a tree visualization of a prim and its descendants.

    Args:
        prim_path: Path to the root prim to start printing from
        max_depth: Maximum depth to traverse (None for unlimited)
        indent: Current indentation level (used recursively)
        stage: Optional USD stage to use (defaults to current stage)

    Example:
        ```python
        print_prim_tree("/World/envs/env_0/robot")

        # Output:
        /World/envs/env_0/robot
        ├── base
        │   ├── link_base
        │   │   └── collision_base
        ├── right_arm
        │   ├── link_arm_0
        │   │   └── collision_arm_0
        │   ├── link_arm_1
        │   │   └── collision_arm_1
        └── camera
            ├── ZED_X
            │   ├── base_link
            │   └── CameraLeft
        ```
    """
    # Get stage if not provided
    if stage is None:
        stage = omni.usd.get_context().get_stage()

    # Get prim at path
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Invalid prim path: {prim_path}")
        return

    # Check max depth
    if max_depth is not None and indent > max_depth:
        return

    # Print current prim with indentation
    prefix = "│   " * (indent - 1) + "├── " if indent > 0 else ""
    print(f"{prefix}{prim.GetPath()}")

    # Recursively print children
    children = prim.GetChildren()
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        # Change prefix for last child
        if is_last and indent > 0:
            print("│   " * (indent - 1) + "└── " + str(child.GetPath()))
        else:
            print_prim_tree(str(child.GetPath()), max_depth, indent + 1, stage)


def find_matching_prims(
    root_path: str, pattern: str = "*", include_root: bool = False, stage: Usd.Stage | None = None
) -> list[Usd.Prim]:
    """Find all prims under a root path that match a pattern.

    Args:
        stage: USD stage to search
        root_path: Base path to start searching from
        pattern: Pattern to match against relative paths (using fnmatch)
        include_root: Whether to include the root prim in results if it matches

    Returns:
        List of matching Usd.Prim objects
    """
    stage = stage or get_current_stage()

    matching_prims: list[Usd.Prim] = []
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return matching_prims

    # Process the root prim if requested
    if include_root and fnmatch.fnmatch("", pattern):
        matching_prims.append(root_prim)

    # Recursively traverse and match pattern, but only return top-level matches
    def _traverse_prims(prim: Usd.Prim, base_path: str, matched_ancestors: set[str] | None = None) -> None:
        if matched_ancestors is None:
            matched_ancestors = set()

        for child in prim.GetChildren():
            path = str(child.GetPath())
            rel_path = path[len(base_path) :]

            # Check if this prim matches the pattern
            if fnmatch.fnmatch(rel_path, pattern):
                # Only add if none of its ancestors have already matched
                is_top_level_match = True
                for ancestor_path in matched_ancestors:
                    if path.startswith(ancestor_path + "/"):
                        is_top_level_match = False
                        break

                if is_top_level_match:
                    matching_prims.append(child)
                    # Add this path to matched ancestors for its descendants
                    new_matched_ancestors = matched_ancestors | {path}
                    _traverse_prims(child, base_path, new_matched_ancestors)
                else:
                    # This is a descendant of an already matched prim, skip it
                    # but continue traversing in case there are other matches
                    _traverse_prims(child, base_path, matched_ancestors)
            else:
                # This prim doesn't match, continue traversing with same matched ancestors
                _traverse_prims(child, base_path, matched_ancestors)

    _traverse_prims(root_prim, root_path)
    return matching_prims


def log_robot_properties(robot_path: str, pattern: str = "*", stage: Usd.Stage | None = None) -> None:
    """Log mass properties and velocity limits of robot links matching a pattern.

    Args:
        stage: USD stage
        robot_path: Base path to the robot prim
        pattern: Pattern to match link names (e.g. "*" for all, "*/hand/*" for hand links)
    """

    from prettytable import PrettyTable
    from pxr import PhysxSchema, UsdPhysics

    # Default to current stage
    stage = stage or get_current_stage()

    # Find all matching prims under the robot
    matching_prims = find_matching_prims(robot_path, pattern, stage=stage)

    # Create table for mass properties
    mass_table = PrettyTable()
    mass_table.title = f"Robot Mass Properties (Pattern: {pattern})"
    mass_table.field_names = ["Name", "Mass (kg)", "Center of Mass", "Diagonal Inertia", "Principal Axes"]
    mass_table.align["Name"] = "l"

    # Create table for velocity limits
    velocity_table = PrettyTable()
    velocity_table.title = f"Robot Velocity Limits (Pattern: {pattern})"
    velocity_table.field_names = ["Name", "Max Linear Velocity", "Max Angular Velocity", "Max Joint Velocity"]
    velocity_table.align["Name"] = "l"

    # Process each matching prim
    for prim in matching_prims:
        name = prim.GetName()

        # Check if prim has mass properties
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)

            # Get mass value - convert to float
            mass: Any = mass_api.GetMassAttr().Get() if mass_api.GetMassAttr() else "-"
            if mass != "-":
                mass = float(mass)

            # Get center of mass - convert Gf.Vec3f to list of floats
            com: Any = mass_api.GetCenterOfMassAttr().Get() if mass_api.GetCenterOfMassAttr() else "-"
            if com != "-":
                com = [float(x) for x in com]

            # Get diagonal inertia - convert Gf.Vec3f to list of floats
            inertia: Any = mass_api.GetDiagonalInertiaAttr().Get() if mass_api.GetDiagonalInertiaAttr() else "-"
            if inertia != "-":
                inertia = [float(x) for x in inertia]

            # Get principal axes - convert Gf.Quatf to list of floats
            axes: Any = mass_api.GetPrincipalAxesAttr().Get() if mass_api.GetPrincipalAxesAttr() else "-"
            if axes != "-":
                # Convert quaternion to list of floats
                axes = [float(axes.GetReal())] + [float(x) for x in axes.GetImaginary()]

            mass_table.add_row(
                [
                    name,
                    f"{mass}" if isinstance(mass, float) else mass,
                    f"[{com[0]}, {com[1]}, {com[2]}]" if isinstance(com, list) else com,
                    f"[{inertia[0]}, {inertia[1]}, {inertia[2]}]" if isinstance(inertia, list) else inertia,
                    f"[{axes[0]}, {axes[1]}, {axes[2]}, {axes[3]}]" if isinstance(axes, list) else axes,
                ]
            )

        # Get velocity limits from both APIs
        max_linear_vel: Any = "-"
        max_angular_vel: Any = "-"
        max_joint_vel: Any = "-"

        # Check PhysxRigidBodyAPI for linear/angular velocity limits
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)

            # Get max linear velocity
            if physx_api.GetMaxLinearVelocityAttr():
                max_linear_vel = physx_api.GetMaxLinearVelocityAttr().Get()
                if isinstance(max_linear_vel, float):
                    max_linear_vel = f"{max_linear_vel}"

            # Get max angular velocity
            if physx_api.GetMaxAngularVelocityAttr():
                max_angular_vel = physx_api.GetMaxAngularVelocityAttr().Get()
                if isinstance(max_angular_vel, float):
                    max_angular_vel = f"{max_angular_vel}"

        # Check PhysxJointAPI for joint velocity limit
        if prim.HasAPI(PhysxSchema.PhysxJointAPI):
            joint_api = PhysxSchema.PhysxJointAPI(prim)
            if joint_api.GetMaxJointVelocityAttr():
                max_joint_vel = joint_api.GetMaxJointVelocityAttr().Get()
                if isinstance(max_joint_vel, float):
                    max_joint_vel = f"{max_joint_vel}"

        # Add velocity limits to table if any limits are defined
        if max_linear_vel != "-" or max_angular_vel != "-" or max_joint_vel != "-":
            velocity_table.add_row([name, max_linear_vel, max_angular_vel, max_joint_vel])

    # Print tables
    if mass_table.rows:
        omni.log.info("\n" + mass_table.get_string())
    else:
        omni.log.info(f"No prims with mass properties found matching pattern: {pattern}")

    if velocity_table.rows:
        omni.log.info("\n" + velocity_table.get_string())
    else:
        omni.log.info(f"No prims with velocity limits found matching pattern: {pattern}")


def list_prims(usd_path: str, path: str = "/", recurse: bool = True) -> list[str]:
    """List prims in a USD file at the specified path.

    Args:
        usd_path: Path to the USD file
        path: Root path to start listing from
        recurse: Whether to recursively list children

    Returns:
        List of prim paths as strings
    """
    stage = Usd.Stage.Open(usd_path)
    return list_prims_in_stage(stage, path, recurse)


def list_prims_in_stage(stage: Usd.Stage, path: str = "/", recurse: bool = True) -> list[str]:
    """List prims in a USD stage at the specified path.

    Args:
        stage: USD stage
        path: Root path to start listing from
        recurse: Whether to recursively list children

    Returns:
        List of prim paths as strings
    """
    root_prim = stage.GetPrimAtPath(path)

    if not root_prim.IsValid():
        omni.log.warn(f"Path {path} not found")
        return []

    # List direct children
    children = []
    for child in root_prim.GetChildren():
        children.append(str(child.GetPath()))
        if recurse:
            children.extend(list_prims_in_stage(stage, str(child.GetPath()), recurse))

    return children


def compute_world_transform(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """Compute the world transform matrix for a prim.

    Args:
        stage: USD stage
        prim_path: Path to the prim

    Returns:
        Gf.Matrix4d: World transform matrix
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        omni.log.warn(f"Invalid prim path for transform computation: {prim_path}")
        return Gf.Matrix4d(1.0)  # Return identity matrix

    # UsdGeom.Xformable works for all transformable prims and handles hierarchy automatically
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    return Gf.Matrix4d(1.0)  # Fallback to identity


def get_pose(transform: Gf.Transform) -> tuple[npt.NDArray[np.float64], tuple[float, float, float, float]]:
    """Extract position and rotation from a transform.

    Args:
        transform: Gf.Transform object

    Returns:
        Tuple of (position_array, rotation_tuple) where:
        - position_array: numpy array of (x, y, z)
        - rotation_tuple: (w, x, y, z) quaternion
    """
    translation: Gf.Vec3d = transform.GetTranslation()
    rotation: Gf.Quatd = transform.GetRotation().GetQuat()
    rot_tuple = (rotation.real, rotation.imaginary[0], rotation.imaginary[1], rotation.imaginary[2])
    return np.array(translation), rot_tuple


def set_instanceable(stage: Usd.Stage, prim_path: str, instanceable: bool = True) -> bool:
    """Set the instanceable flag on a prim and (when clearing) its whole subtree.

    Clearing must cover descendants too: a nested instanceable prim hides its children from
    ``Usd.PrimRange``/``stage.Traverse`` (instance proxies), so physics edits and the composition
    validator would silently miss geometry living inside it. Paths are collected before mutating
    (flipping instanceable mid-traversal is undefined).

    Args:
        stage: The USD stage containing the prim
        prim_path: Path to the prim to modify
        instanceable: True to make instanceable (root only), False to clear root + subtree

    Returns:
        bool: True if flag was set, False if prim not found
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return False

    prim.SetInstanceable(instanceable)
    if not instanceable:
        # TraverseInstanceProxies so prims INSIDE still-instanced subtrees are found too.
        paths = [p.GetPath() for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if p.IsInstanceable()]
        for path in paths:
            sub = stage.GetPrimAtPath(path)
            if sub:
                sub.SetInstanceable(False)
    return True
