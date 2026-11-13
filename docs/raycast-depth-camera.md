# Ray-cast depth from the loaded Isaac Sim scene

`IsaacSimCameraConfig(renderer="raycast")` uses the bespoke Warp depth kernel
with geometry from the scene Holosoma has already loaded. It supports depth
images, pinhole projection, and mounts on a robot link listed in the robot
config's `body_names`. Scene objects contribute
depth along with the ground and robot.

Geometry selection uses the same two fields as Isaac Sim LiDAR:

```python
IsaacSimCameraConfig(
    renderer="raycast",
    body_filter=LidarBodyFilterConfig(target_kind="robot"),  # default: the mount body
)
IsaacSimCameraConfig(
    renderer="raycast",
    mesh_prim_paths=["/World/ground", "/World/envs/env_.*/obstacle"],
)
```

With `mesh_prim_paths=None`, discovery selects the ground, every registered
scene object and the robot, minus `body_filter`. The default filter leaves out
the mount body, which usually encloses the camera. Discovery follows the same
optical-surface rules as LiDAR: geometry must be supported by IsaacLab and
visible, with USD purpose `default` or `render`. Neither a `collisions` path
name nor CollisionAPI excludes geometry, and shader opacity is not interpreted.

An explicit `mesh_prim_paths` list is passed to IsaacLab's `MultiMeshRayCaster`
unchanged and bypasses discovery and `body_filter`, exactly as for LiDAR. Each
target is merged into one mesh per environment, including every supported
descendant regardless of visibility or purpose, so list named visual subtrees
rather than whole links when colliders sit beside them.

The G1 D435i preset lists terrain and the 30 named leg, waist and arm visual
meshes that can occlude the head camera. Collision subtrees, the torso and
head housing, rubber hands, contact points and scene objects stay out. These
paths use the G1 URDF import's named visual roots, including the pelvis
contour merged into the pelvis body; override `mesh_prim_paths` for a
different USD hierarchy or additional scene objects.

`/World/ground` is the terrain's ray-cast mesh for every terrain type, shared
by depth cameras, LiDAR and the height scanner through IsaacLab's mesh cache.
Managed `trimesh` and `load_obj` terrain registers the terrain manager's
existing Warp mesh, sharing its buffers and BVH. Plane terrain registers a flat
square 4 km across, centred on the world origin: IsaacLab's own plane is
2,000 km across, too large for float32 ray hits to stay sub-millimetre. The
physics collider remains IsaacLab's infinite plane, and a terrain with
`hide_visual` is still hit by ray sensors.

Native mesh views are registered with `InteractiveScene` before environment
cloning. When physics starts, they query the completed scene and use
IsaacLab's `MultiMeshRayCaster` extraction, mesh cache, and transform views.
Exact mesh paths keep independently moving bodies separate and let cameras
and LiDAR reuse the same cached Warp meshes. Shared-mesh mode references the
geometry cloned from `env_0` across environments. USD geometry carries its authored
scale and local transforms; the sensor has no STL/OBJ file inventory.

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
