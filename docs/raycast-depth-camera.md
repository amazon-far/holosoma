# Ray-cast depth from the loaded Isaac Sim scene

`IsaacSimCameraConfig(renderer="raycast")` uses the bespoke Warp depth kernel
with geometry from the scene Holosoma has already loaded. It supports depth
images, pinhole projection, and mounts on a robot link listed in the robot
config's `body_names`. Scene objects contribute
depth along with the ground and robot.

Select geometry with `raycast_meshes`:

```python
IsaacSimCameraConfig(
    renderer="raycast",
    raycast_meshes=IsaacSimRaycastMeshConfig(
        include_paths=["/World/ground", "/World/envs/env_.*/Robot", "/World/envs/env_.*/obstacle"],
        exclude_paths=["/World/envs/env_.*/Robot/torso_link"],
    ),
)
```

Rules use the same absolute USD prim-path expressions as Isaac Sim LiDAR,
resolved by IsaacLab's native matcher. Each path component is matched separately;
`.*` matches one component, and expressions cannot span multiple components.
Matching an object
includes its descendant meshes; matching a mesh selects that mesh. Exclusions
take precedence over inclusions. Environment paths must use `/World/envs/env_.*`:
selection applies to every environment and resolves against the representative
`env_0` scene during initialization. Omit `include_paths` to include all eligible
ground, robot, and registered scene-actor geometry except the camera's mount body,
which usually encloses the camera. Every explicit include must match eligible geometry,
and the final selection must contain at least one mesh. Exclusions may match
nothing, allowing optional objects to be omitted across different scenes; each
unmatched exclusion is logged as a warning.

Discovery follows the same optical-surface rules as Holosoma's LiDAR: geometry
must be supported by IsaacLab and visible, with USD purpose `default` or
`render`. Inherited invisibility and `proxy`/`guide` purpose remove geometry,
even with an explicit include; the ground is the exception to invisibility, since a
terrain with `hide_visual` still collides. Neither a `collisions` path name nor CollisionAPI
automatically excludes geometry. A whole-link include therefore also selects
eligible colliders; use a named visual subtree to restrict the optical surface.
Shader opacity is not interpreted; hide or exclude helper surfaces explicitly.

The G1 D435i preset explicitly includes terrain and the 30 named leg, waist and
arm visual meshes that can occlude the head camera. Both pelvis meshes, the waist meshes, and all
listed arm and leg meshes remain included. Collision subtrees, torso/head
housing, rubber hands, contact points, helper boxes and scene actors remain
outside this allowlist. These paths use the G1 URDF import's canonical named
visual roots, including the pelvis contour merged into the pelvis body.
Override `include_paths` for a different USD hierarchy or additional scene
objects. Every include must resolve; there is no fallback to a broader inventory.
An explicit include list omits other geometry without an exclude-all rule:
exclusions win and would remove the included geometry too.

Native mesh views are registered with `InteractiveScene` before environment
cloning. When physics starts, they query the completed scene and use
IsaacLab's `MultiMeshRayCaster` extraction, mesh cache, and transform views.
Exact mesh paths keep independently moving bodies separate and let cameras
and LiDAR reuse the same cached Warp meshes. Shared-mesh mode references the
geometry cloned from `env_0` across environments. USD geometry carries its
authored scale and local transforms; the sensor has no STL/OBJ file inventory.

For managed `trimesh` and `load_obj` terrain, after native sensors have been
constructed and before cloning or physics initialization, Holosoma registers the
terrain manager's existing Warp mesh under both `/World/ground` and
`/World/ground/mesh`. Terrain queries, the native height scanner, LiDAR, and
depth share that mesh's vertex/index buffers and BVH. Ground still passes
through path selection. Flat planes retain
native conversion because the loaded plane differs from the terrain
manager's finite grid. Holosoma spawns that plane's visual mesh 4 km across
(IsaacLab's default is 2,000 km), so float32 ray hits on it stay sub-millimetre;
its physics collider is a separate infinite plane.

`prepare_sim()` initializes each depth renderer once body indices and scene
sensors are ready; the renderer prepares its mesh provider's views then. Robot
meshes cache their enabled physics owners, Holosoma body indices, and
body-to-mesh offsets during initialization. Each
capture gathers refreshed robot body poses in a batch and composes the local
offsets; other scene objects, and robot meshes whose body is not in the robot
config, retain native transform tracking. Initialization
seeds poses from native views before Holosoma's first body-tensor refresh.
Selected rigid actors can translate and rotate during physics stepping without
reloading their geometry. Selection is fixed at initialization; this does not
discover newly spawned actors or update deforming meshes.
Partial rerenders require body tensors refreshed after resets or teleports,
as does the camera mount pose. Pose buffers are updated in place before the
fused depth kernel launches on Torch's current stream.
The kernel retains pixel-center rays, image-plane depth, shrinking query
bounds, persistent buffers, and per-environment rerenders. The native geometry
view does not perform an additional LiDAR ray-casting pass.

The renderer still queries each selected mesh per pixel. Including more
geometry can increase capture cost, and the native cache chooses the mesh
BVH builder for newly converted geometry; managed terrain retains its existing
BVH. Performance figures are only comparable between identical scene
selections.
