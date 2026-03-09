import numpy as np
import mujoco
import imageio

model = mujoco.MjModel.from_xml_path(
    'src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml'
)
data = mujoco.MjData(model)

# Make ankle/foot body geoms semi-transparent
for i in range(model.ngeom):
    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if body_name and ('ankle' in body_name or 'foot' in body_name):
        model.geom_rgba[i, 3] = 0.15

mujoco.mj_forward(model, data)

H, W = 480, 640
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

def add_markers(scene, data):
    markers = [
        ('left_foot_contact_point',  [1.0, 0.0, 0.0, 1.0], 0.018),
        ('right_foot_contact_point', [0.0, 0.2, 1.0, 1.0], 0.018),
        ('left_ankle_roll_link',     [1.0, 0.6, 0.0, 1.0], 0.015),
        ('right_ankle_roll_link',    [0.0, 0.8, 0.8, 1.0], 0.015),
    ]
    for name, color, radius in markers:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[bid].copy()
        if scene.ngeom < scene.maxgeom:
            g = scene.ngeom
            scene.geoms[g].type = mujoco.mjtGeom.mjGEOM_SPHERE
            scene.geoms[g].size[:] = [radius]*3
            scene.geoms[g].pos[:] = pos
            scene.geoms[g].mat[:] = np.eye(3)
            scene.geoms[g].rgba[:] = color
            scene.ngeom += 1
    
    for side, color in [('left', [1,0,0,0.9]), ('right', [0,0.2,1,0.9])]:
        ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{side}_ankle_roll_link')
        cp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{side}_foot_contact_point')
        p1 = data.xpos[ankle_id]
        p2 = data.xpos[cp_id]
        mid = (p1 + p2) / 2
        half_len = np.linalg.norm(p1 - p2) / 2
        if scene.ngeom < scene.maxgeom:
            g = scene.ngeom
            scene.geoms[g].type = mujoco.mjtGeom.mjGEOM_CAPSULE
            scene.geoms[g].size[:] = [0.003, half_len, 0.003]
            scene.geoms[g].pos[:] = mid
            scene.geoms[g].mat[:] = np.eye(3)
            scene.geoms[g].rgba[:] = color
            scene.ngeom += 1

renderer = mujoco.Renderer(model, H, W)

# View 1: Close-up
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.0, 0.0, 0.02]
cam.distance = 0.55
cam.elevation = -15.0
cam.azimuth = 160.0
renderer.update_scene(data, camera=cam, scene_option=scene_option)
add_markers(renderer.scene, data)
pixels = renderer.render()
imageio.imwrite('g1_contact_points_closeup.png', pixels)
print("Saved g1_contact_points_closeup.png")

# View 2: Side
cam.lookat[:] = [0.0, 0.11, 0.02]
cam.distance = 0.35
cam.elevation = 0.0
cam.azimuth = 180.0
renderer.update_scene(data, camera=cam, scene_option=scene_option)
add_markers(renderer.scene, data)
pixels = renderer.render()
imageio.imwrite('g1_contact_points_side.png', pixels)
print("Saved g1_contact_points_side.png")

# View 3: Full body
cam.lookat[:] = [0.0, 0.0, 0.45]
cam.distance = 2.2
cam.elevation = -20.0
cam.azimuth = 160.0
renderer.update_scene(data, camera=cam, scene_option=scene_option)
add_markers(renderer.scene, data)
pixels = renderer.render()
imageio.imwrite('g1_contact_points_full.png', pixels)
print("Saved g1_contact_points_full.png")

renderer.close()
print("\nRED=left_contact_point, BLUE=right_contact_point, ORANGE=left_ankle_roll, CYAN=right_ankle_roll")
print("Vertical lines show 37mm offset between ankle_roll_link and foot_contact_point")
