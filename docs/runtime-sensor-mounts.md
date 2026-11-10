# Runtime sensor mounting poses

Every configured camera and LiDAR supports runtime offset changes on Isaac Sim, classic
MuJoCo, and MuJoCo Warp. Use the sensor's configuration dictionary key as its name:

```python
env_ids = torch.tensor([0, 2], device=sim.sim_device)
poses = sim.get_sensor_mount_pose("head_camera", env_ids)
poses[:, :3] += torch.empty_like(poses[:, :3]).uniform_(-0.02, 0.02)
sim.set_sensor_mount_pose("head_camera", poses, env_ids)
```

`get_sensor_mount_pose` returns a writable copy with shape `[len(env_ids), 7]`. Runtime poses
use **`[x, y, z, qx, qy, qz, qw]`**: meters followed by a unit quaternion in XYZW order, matching
Holosoma's runtime actor pose APIs. `SensorMountConfig.orientation` uses **WXYZ**, so convert
its component order explicitly when constructing a runtime pose from a config.

Both methods select all environments when `env_ids=None`. The setter also accepts a single
`[7]` pose to broadcast to the selected environments. For example, a world-mounted sensor can
be placed at the same position relative to each environment's origin:

```python
sim.set_sensor_mount_pose(
    "overview",
    torch.tensor([2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0], device=sim.sim_device),
)
```

Offsets are relative to the **original configured mount**:

| `mount.target_kind` | Runtime position and orientation frame |
| --- | --- |
| `robot_link` | The configured robot link, including links fused into another body |
| `actor` | The configured scene actor |
| `world` | The environment frame; Holosoma adds that environment's origin |

The sensor keeps following its parent after an update. Runtime reparenting is not supported;
choose the mounting object or link in `SensorMountConfig`. All sensors retain the optical
convention `-Z` forward and `+Y` up.

Updates do not advance physics, change the original configuration, or clear published output
buffers. Camera and LiDAR reads return their last published outputs until the next capture at
their normal update cadence. For Isaac Sim cameras, the next native rendering pass must occur
before that capture. Offsets persist across environment resets; apply new random offsets in
your reset event when desired. Classic MuJoCo supports one environment, as before.

Save and restore offsets explicitly when randomization should start from a nominal pose:

```python
nominal = sim.get_sensor_mount_pose("head_camera")
# Later, in a reset event:
poses = nominal[env_ids].clone()
poses[:, :3] += torch.randn_like(poses[:, :3]) * 0.01
sim.set_sensor_mount_pose("head_camera", poses, env_ids)
```

Invalid shapes, non-finite values, non-unit quaternions, out-of-range environment ids, and
duplicate environment ids in a write raise `ValueError` before a backend write. Quaternion
norm deviations up to `1e-4` are normalized. Unknown sensor names raise `KeyError`.
