# I2RT YAM (Yet Another Manipulator)

6-DOF arm with a two-finger parallel gripper (8 DOF), fixed base. holosoma's reference fixed-base robot;
preset `yam` in `holosoma/config_values/robot.py`.

## Source

- `yam.urdf`, `assets/*.stl`: <https://github.com/i2rt-robotics/i2rt>, `i2rt/robot_models/arm/yam/v1/`
  at commit `5b72c47239bd056d0fa6c1a39edeb0537c89443c` (2026-09-07). MIT License, copyright I2RT Robotics
  (`LICENSE`).
- Product page: <https://i2rt.com/products/yam-manipulator>.

## Modifications

- `yam.urdf`: a `<collision>` element with the link's visual mesh was added to every link (the source ships
  visual meshes only); the CAD export's per-link colours were replaced by two materials matching the real arm
  (`yam_black` 0.25 grey for the base, motors, wrist and gripper; `yam_white` 0.9 for the `link2` / `link3`
  tubes), the values the MuJoCo Menagerie model uses.
- `yam.xml`: generated from the modified URDF with MuJoCo 3.9 (`MjSpec.from_string` with
  `balanceinertia="true"`, `fusestatic="false"` so the jointless root body `base` survives), the URDF's
  placeholder `actuatorfrcrange="-1 1"` removed, and the geom colours replaced by the two named materials (with
  a specular highlight). No actuators: holosoma adds a torque motor per DOF at load.

## Joints and bodies

| DOF | Joint | Type | Parent -> child | Range |
|---|---|---|---|---|
| 0 | joint1 | revolute | base -> link1 | [-2.618, 3.142] |
| 1 | joint2 | revolute | link1 -> link2 | [0, 3.665] |
| 2 | joint3 | revolute | link2 -> link3 | [0, 3.142] |
| 3 | joint4 | revolute | link3 -> link4 | [-1.693, 1.571] |
| 4 | joint5 | revolute | link4 -> link5 | [-1.571, 1.571] |
| 5 | joint6 | revolute | link5 -> gripper | [-2.094, 2.094] |
| 6 | joint7 | prismatic | gripper -> tip_left | [-0.04695, 0] m (0 = open) |
| 7 | joint8 | prismatic | gripper -> tip_right | [-0.04695, 0] m (0 = open) |

Body order: `base, link1, link2, link3, link4, link5, gripper, tip_left, tip_right`. The base plate sits on
the floor when the robot spawns at z = 0 (its geometry spans z in [0, 0.073] m).

The URDF's `effort="1" velocity="1"` limits are placeholders. The preset supplies effort
28 Nm (joints 1-3, DM4340 motors) / 10 Nm (joints 4-6, DM4310) / 20 N (fingers), velocity
10 / 30 rad/s / 1 m/s, armature 0.032 / 0.0018 / 0.1 (effort and armature from the MuJoCo Menagerie `i2rt_yam`
model, velocity limits from mjlab's `yam_constants.py`) and no joint friction (Menagerie's `frictionloss="0.1"`
is a MuJoCo constant torque; PhysX reads a joint friction value as a coefficient of the constraint force, so
the backends only agree at 0). MuJoCo reports velocity limits without enforcing them.
