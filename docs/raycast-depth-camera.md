# Ray-cast depth from the loaded Isaac Sim scene

`IsaacSimCameraConfig(renderer="raycast")` uses the bespoke Warp depth kernel
with geometry from the scene Holosoma has already loaded. It supports depth
images, pinhole projection, and robot-link mounts. Scene objects contribute
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
take precedence over inclusions. Use `/World/envs/env_.*` for cloned objects;
selection resolves against the representative `env_0` scene during initialization.
Omit `include_paths` to include all eligible ground, robot, and registered
scene-actor geometry. Every explicit include must match eligible geometry,
and the final selection must contain at least one mesh. Exclusions may match
nothing, allowing optional objects to be omitted across different scenes.

Discovery follows the same optical-surface rules as Holosoma's LiDAR: geometry
must be supported by IsaacLab and visible, with USD purpose `default` or
`render`. Inherited invisibility and `proxy`/`guide` purpose remove geometry,
even with an explicit include. Neither a `collisions` path name nor CollisionAPI
automatically excludes geometry. A whole-link include therefore also selects
eligible colliders; use a named visual subtree to restrict the optical surface.
Shader opacity is not interpreted; hide or exclude helper surfaces explicitly.

The G1 D435i preset explicitly includes the original #281 inventory: 30 named
robot visual meshes and terrain. Both pelvis meshes, the waist meshes, and all
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
through the same visibility and path selection rules. Flat planes retain
native conversion because the loaded plane differs from the terrain
manager's finite grid.

`prepare_sim()` asks sensor setup to prepare these views, then initializes each
depth renderer with its ready mesh buffers once body indices and scene sensors
are ready. Cameras do not initialize their mesh providers. Robot meshes cache
their enabled physics owners, Holosoma body indices, and body-to-mesh offsets
during initialization. Each
capture gathers refreshed robot body poses in a batch and composes the local
offsets; other scene objects retain native transform tracking. Initialization
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
BVH. Existing performance figures need to be remeasured with identical
scene selections; this change makes no throughput claim.
