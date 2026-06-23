#!/usr/bin/env python
"""Generate the MenteeBot v3.1 MJCF for holosoma MuJoCo sim2sim from the URDF.

The holosoma MuJoCo backend (scene_manager.py) loads robot.asset.xml_file via
MjSpec.from_file, attaches it under a "robot_" prefix to a generated world
(ground + lights added separately), and drives it with one TORQUE actuator per
DOF looked up by name `robot_<dof_name>` (mujoco.py:882). So the MJCF must:

  * have a floating base (freejoint on the root body),
  * keep the URDF joint names (== holosoma dof_names),
  * expose one <motor> actuator per actuated joint, named == joint name,
  * resolve meshes via meshdir.

We build it from menteebot_v3_1_29dof.urdf (the processed 29-DOF asset) with
MjSpec, add the freejoint + actuators, compile to validate, and save XML.

Run with the hsmujoco env python (needs mujoco>=3).
"""

import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROBOT_DIR = os.path.join(os.path.dirname(__file__), "..", "src/holosoma/holosoma/data/robots/menteebot_v3_1")
ROBOT_DIR = os.path.abspath(ROBOT_DIR)
URDF = os.path.join(ROBOT_DIR, "menteebot_v3_1_29dof.urdf")
OUT = os.path.join(ROBOT_DIR, "menteebot_v3_1_29dof.xml")


# --- URDF preprocessing: fix MjSpec's dropped fixed-joint transforms -----------
# This produces the PHYSICS model. MuJoCo fuses fixed-joint links into their
# parent and, when such a link has a *moving* child, DROPS the fixed joint's
# origin transform (the limbs mount via fixed "ee" links with a 90° rotation:
# base_link -[left_leg_ee rpy=90°]-> left_leg_link -[left_hip_yaw]). The
# hips/shoulders then attach at the raw origin and the legs splay sideways.
# Fix: reparent every moving joint whose parent is a welded link to its nearest
# non-welded ancestor, composing the skipped fixed transforms into the joint
# origin. Reproduces URDF forward kinematics exactly (0 mismatch).
#
# NOTE: we keep MuJoCo's default fusing (fast, well-conditioned inertia, clean
# foot contact — validated: the holosoma policy walks on this model). The
# downside is that the welded links' *visual* meshes are mis-placed by the same
# fuse-time transform drop, so this model renders poorly. Rendering therefore
# uses a separate unfused visual model (see make_visual_model); physics uses
# this one. fusestatic="false" would fix the visuals but splits the feet/fingers
# into many ill-conditioned bodies and breaks the contact dynamics.
def _rpy_to_mat(rpy):
    rr, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(rr), np.sin(rr), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _mat_to_rpy(R):  # ZYX Euler, matching _rpy_to_mat
    sy = -R[2, 0]
    cy = np.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-8:
        return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(sy), np.arctan2(R[1, 0], R[0, 0])])
    return np.array([np.arctan2(-R[1, 2], R[1, 1]), np.arcsin(sy), 0.0])


def _origin_T(o):
    rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()] if o is not None else [0, 0, 0]
    xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()] if o is not None else [0, 0, 0]
    T = np.eye(4)
    T[:3, :3] = _rpy_to_mat(rpy)
    T[:3, 3] = xyz
    return T


def _fix_dropped_fixed_transforms(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = root.findall("joint")
    incoming = {j.find("child").get("link"): j for j in joints}

    def anchor(link):
        T = np.eye(4)
        while link in incoming and incoming[link].get("type") == "fixed":
            j = incoming[link]
            T = _origin_T(j.find("origin")) @ T
            link = j.find("parent").get("link")
        return link, T

    fixed = 0
    for j in joints:
        if j.get("type") == "fixed":
            continue
        parent = j.find("parent").get("link")
        if parent in incoming and incoming[parent].get("type") == "fixed":
            a, t_ap = anchor(parent)
            new_T = t_ap @ _origin_T(j.find("origin"))
            o = j.find("origin")
            if o is None:
                o = ET.SubElement(j, "origin")
            o.set("xyz", " ".join(f"{v:.8g}" for v in new_T[:3, 3]))
            o.set("rpy", " ".join(f"{v:.8g}" for v in _mat_to_rpy(new_T[:3, :3])))
            j.find("parent").set("link", a)
            fixed += 1
    print(f"reparented {fixed} moving joints across welded fixed-joint links")
    fd, out = tempfile.mkstemp(suffix=".urdf", dir=os.path.dirname(urdf_path))
    os.close(fd)
    tree.write(out)
    return out


print(f"loading URDF: {URDF}")
_corrected_urdf = _fix_dropped_fixed_transforms(URDF)
try:
    spec = mujoco.MjSpec.from_file(_corrected_urdf)
finally:
    os.unlink(_corrected_urdf)
spec.modelname = "menteebot_v31_29dof"
spec.compiler.degree = False  # radians (URDF is radians)

# Implicit-fast integrator handles stiff PD (esp. torso kp=1300) far better than
# the default semi-implicit Euler.
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

# Per-joint armature (reflected motor inertia). Critical for stability: without
# it, the stiff PD gains used in training blow up under MuJoCo. Values come from
# the training robot config (robot_config_v3_1.yaml dof_props/armature).
from holosoma.config_values.robot import menteebot_v31_29dof as _rc  # noqa: E402

_armature = dict(zip(_rc.dof_names, _rc.dof_armature_list))
_friction = dict(zip(_rc.dof_names, _rc.dof_joint_friction_list))

# --- floating base: add a freejoint to the root body --------------------------
# The root body is the one directly under worldbody (base_link).
world = spec.worldbody
root_bodies = world.bodies
assert len(root_bodies) == 1, f"expected 1 root body, got {[b.name for b in root_bodies]}"
root = root_bodies[0]
print(f"root body: {root.name}; adding freejoint")
root.add_freejoint(name="floating_base_joint")

# --- one torque actuator per actuated (hinge) joint, named == joint name ------
hinge_joints = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
print(f"adding {len(hinge_joints)} motor actuators; setting armature/frictionloss")
for j in hinge_joints:
    j.armature = float(_armature[j.name])
    j.frictionloss = float(_friction[j.name])
    act = spec.add_actuator()
    act.name = j.name
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.target = j.name
    # default gaintype=fixed, gainprm[0]=1 -> ctrl is interpreted as torque (N·m)

# --- collision filtering: feet/body vs ground only, no self-collision ---------
# The URDF imports every collision geom as contype=1/conaffinity=1, so all robot
# parts collide with each other (~50 spurious contacts at the standing pose,
# which overwhelm the PD and topple the robot). IsaacLab trains with self-
# collision filtered. Put robot collision geoms on contype=2/conaffinity=1 and
# the ground (added by the sim interface) on contype=1/conaffinity=1: robot-vs-
# robot = (2&1)|(2&1)=0 (no self-collision); robot-vs-ground = (2&1)|(1&1)=1
# (feet and any body part still contact the floor). Visual-only geoms (0/0) are
# left untouched.
n_coll = 0
for g in spec.geoms:
    if g.contype or g.conaffinity:
        g.contype = 2
        g.conaffinity = 1
        n_coll += 1
print(f"set {n_coll} collision geoms to contype=2/conaffinity=1 (no self-collision)")

# --- compile to validate, then save -------------------------------------------
model = spec.compile()
print(f"compiled OK: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")
assert model.nu == 29, f"expected 29 actuators, got {model.nu}"
assert model.njnt == 30, f"expected 30 joints (1 free + 29 hinge), got {model.njnt}"

xml = spec.to_xml()
with open(OUT, "w") as f:
    f.write(xml)
print(f"wrote {OUT}")
