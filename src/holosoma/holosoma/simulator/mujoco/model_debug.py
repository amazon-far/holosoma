"""Read-only diagnostic dump of a compiled MuJoCo model.

Pretty-prints the compiled model's structure (bodies, joints, DOF-candidate joints,
actuator->joint transmission map, and a qpos/qvel/ctrl snapshot) to stdout. Pure
reporting: it reads the model/data and writes nothing back, so it lives outside the
simulator class.
"""

from __future__ import annotations

import mujoco


def print_mujoco_model_tree(model: mujoco.MjModel, data: mujoco.MjData, model_path: str | None) -> None:
    """Print a comprehensive MuJoCo model-structure report for debugging.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model to analyze.
    data : mujoco.MjData
        The associated data (read for the current-state snapshot only).
    model_path : str | None
        Source path of the robot model, shown in the report header.
    """
    print(f"Analyzing compiled model (robot source: {model_path})")

    print("=" * 80)
    print("MUJOCO MODEL STRUCTURE ANALYSIS")
    print("=" * 80)

    # 1. BASIC MODEL INFO
    print("\n📊 MODEL OVERVIEW:")
    print(f"   Model file: {model_path}")
    print(f"   Total bodies: {model.nbody}")
    print(f"   Total joints: {model.njnt}")
    print(f"   Total DOFs: {model.nv}")
    print(f"   Total qpos elements: {model.nq}")
    print(f"   Total actuators: {model.nu}")
    print(f"   Total geoms: {model.ngeom}")

    # 2. BODY LIST (Simple, no hierarchy to avoid infinite loops)
    print("\n🏗️  BODY LIST:")
    print(f"   {'ID':<3} {'Name':<30} {'Parent ID':<9} {'Parent Name'}")
    print(f"   {'-' * 3} {'-' * 30} {'-' * 9} {'-' * 20}")

    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        parent_id = model.body_parentid[body_id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) if parent_id != -1 else "WORLD"
        print(f"   {body_id:<3} {body_name:<30} {parent_id:<9} {parent_name}")

    # 3. JOINT DETAILS (This is the most important part!)
    print("\n🔗 JOINT STRUCTURE:")
    print(f"   {'ID':<3} {'Name':<30} {'Type':<8} {'Body':<20} {'qpos_addr':<9} {'qvel_addr':<9}")
    print(f"   {'-' * 3} {'-' * 30} {'-' * 8} {'-' * 20} {'-' * 9} {'-' * 9}")

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        joint_type = model.jnt_type[joint_id]
        body_id = model.jnt_bodyid[joint_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        qpos_addr = model.jnt_qposadr[joint_id]
        qvel_addr = model.jnt_dofadr[joint_id]

        # Joint type names
        type_names = {0: "FREE", 1: "BALL", 2: "SLIDE", 3: "HINGE"}
        type_name = type_names.get(joint_type, f"TYPE_{joint_type}")

        print(f"   {joint_id:<3} {joint_name:<30} {type_name:<8} {body_name:<20} {qpos_addr:<9} {qvel_addr:<9}")

    # 4. DOF ANALYSIS (What holosoma expects)
    print("\n🎯 DOF ANALYSIS (holosoma perspective):")

    # Get all non-freejoint joints
    dof_joints = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        joint_type = model.jnt_type[joint_id]

        # Skip freejoint (type 0) and floating_base joints
        if joint_type != 0 and "floating_base" not in joint_name.lower():
            dof_joints.append((joint_id, joint_name))

    print(f"   Expected DOF count: {len(dof_joints)}")
    print(f"\n   {'Idx':<3} {'DOF Name':<30} {'MJ_ID':<5} {'qpos_addr':<9} {'qvel_addr':<9}")
    print(f"   {'-' * 3} {'-' * 30} {'-' * 5} {'-' * 9} {'-' * 9}")

    for idx, (joint_id, joint_name) in enumerate(dof_joints):
        qpos_addr = model.jnt_qposadr[joint_id]
        qvel_addr = model.jnt_dofadr[joint_id]
        print(f"   {idx:<3} {joint_name:<30} {joint_id:<5} {qpos_addr:<9} {qvel_addr:<9}")

    # 5. ACTUATOR MAPPING
    print("\n⚙️  ACTUATOR MAPPING:")
    print(f"   {'ID':<3} {'Name':<30} {'Joint':<30}")
    print(f"   {'-' * 3} {'-' * 30} {'-' * 30}")

    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or f"actuator_{actuator_id}"
        # Get the joint this actuator controls
        joint_id = model.actuator_trnid[actuator_id, 0]  # First transmission element
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        print(f"   {actuator_id:<3} {actuator_name:<30} {joint_name:<30}")

    # 6. CURRENT STATE SNAPSHOT
    print("\n📸 CURRENT STATE SNAPSHOT:")
    print(f"   qpos (first 10): {data.qpos[:10]}")
    print(f"   qvel (first 10): {data.qvel[:10]}")
    print(f"   ctrl (all): {data.ctrl}")

    print("\n" + "=" * 80)
    print("END OF MODEL ANALYSIS")
    print("=" * 80)
