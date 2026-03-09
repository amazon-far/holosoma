"""Verify motion data ↔ robot joint/body mapping for statue_v1_rmd."""
import numpy as np
import sys

# ── 1. Load motion NPZ ───────────────────────────────────────────────────────
motion_file = "src/holosoma/holosoma/data/motions/statue_v1_rmd/whole_body_tracking/unreal_engine/walk_statue_v1_rmd_50fps.npz"
data = np.load(motion_file, allow_pickle=True)

npz_body_names = data["body_names"].tolist()
npz_joint_names = data["joint_names"].tolist()
body_pos_w = data["body_pos_w"]  # [T, num_bodies, 3]
body_quat_w = data["body_quat_w"]  # [T, num_bodies, 4] wxyz
joint_pos = data["joint_pos"][:, 7:]  # skip first 7 (root pos+quat)

print("=" * 80)
print("MOTION NPZ INFO")
print("=" * 80)
print(f"File: {motion_file}")
print(f"Timesteps: {body_pos_w.shape[0]}, FPS: {data['fps']}")
print(f"Body count: {len(npz_body_names)}, Joint count: {len(npz_joint_names)}")
print(f"\nNPZ body_names ({len(npz_body_names)}):")
for i, name in enumerate(npz_body_names):
    pos = body_pos_w[0, i]
    print(f"  [{i:2d}] {name:30s}  pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

print(f"\nNPZ joint_names ({len(npz_joint_names)}):")
for i, name in enumerate(npz_joint_names):
    print(f"  [{i:2d}] {name:30s}  val={joint_pos[0, i]:+.4f}")

# ── 2. Robot config body/joint names ─────────────────────────────────────────
# From robot.py statue_v1_rmd config
robot_body_names = [
    "pelvis_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_pitch_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_pitch_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_pitch_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_pitch_link", "left_wrist_roll_link", "left_wrist_yaw_link", "left_wrist_pitch_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_pitch_link", "right_wrist_roll_link", "right_wrist_yaw_link", "right_wrist_pitch_link",
]

# From robot.py dof_names_list
robot_joint_names = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
]

# ── 3. Simulate MotionLoader mapping ─────────────────────────────────────────
print("\n" + "=" * 80)
print("MAPPING VERIFICATION")
print("=" * 80)

# Add fake body aliases (from wbt.py)
FAKE_BODY_NAME_ALIASES = {
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}
# The actual robot may have contact point bodies - add them
robot_body_names_with_contacts = robot_body_names + ["left_foot_contact_point", "right_foot_contact_point"]
robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names_with_contacts]

# Simulate _get_index_of_a_in_b for bodies
print("\n--- Body Mapping (robot → NPZ) ---")
body_errors = []
for i, (robot_name, alias_name) in enumerate(zip(robot_body_names_with_contacts, robot_body_names_alias)):
    if alias_name in npz_body_names:
        npz_idx = npz_body_names.index(alias_name)
        alias_note = f" (alias→{alias_name})" if robot_name != alias_name else ""
        print(f"  robot[{i:2d}] {robot_name:35s}{alias_note} → NPZ[{npz_idx:2d}] {npz_body_names[npz_idx]}")
    else:
        body_errors.append(robot_name)
        print(f"  robot[{i:2d}] {robot_name:35s} → *** NOT FOUND IN NPZ ***")

# Simulate _get_index_of_a_in_b for joints
print("\n--- Joint Mapping (robot → NPZ) ---")
joint_errors = []
for i, robot_name in enumerate(robot_joint_names):
    if robot_name in npz_joint_names:
        npz_idx = npz_joint_names.index(robot_name)
        print(f"  robot[{i:2d}] {robot_name:35s} → NPZ[{npz_idx:2d}] {npz_joint_names[npz_idx]}")
    else:
        joint_errors.append(robot_name)
        print(f"  robot[{i:2d}] {robot_name:35s} → *** NOT FOUND IN NPZ ***")

if body_errors:
    print(f"\n*** BODY MAPPING ERRORS: {body_errors}")
if joint_errors:
    print(f"\n*** JOINT MAPPING ERRORS: {joint_errors}")

# ── 4. Left/Right sanity check using Y position ──────────────────────────────
print("\n" + "=" * 80)
print("LEFT/RIGHT SANITY CHECK (Y position at t=0)")
print("=" * 80)
print("Convention: Y > 0 = left side, Y < 0 = right side")

check_pairs = [
    ("left_ankle_roll_link", "right_ankle_roll_link"),
    ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    ("left_knee_pitch_link", "right_knee_pitch_link"),
    ("left_hip_roll_link", "right_hip_roll_link"),
]

for left_name, right_name in check_pairs:
    if left_name in npz_body_names and right_name in npz_body_names:
        left_idx = npz_body_names.index(left_name)
        right_idx = npz_body_names.index(right_name)
        left_y = body_pos_w[0, left_idx, 1]
        right_y = body_pos_w[0, right_idx, 1]
        status = "OK" if left_y > right_y else "*** SWAPPED ***"
        print(f"  {left_name:30s} Y={left_y:+.4f}  |  {right_name:30s} Y={right_y:+.4f}  [{status}]")

# ── 5. Check after remapping (simulating what MotionLoader returns) ──────────
print("\n" + "=" * 80)
print("AFTER REMAPPING: body_pos_w[:, body_indexes] at t=0")
print("=" * 80)

body_indexes = []
for alias_name in robot_body_names_alias[:len(robot_body_names)]:  # exclude contact points for this check
    if alias_name in npz_body_names:
        body_indexes.append(npz_body_names.index(alias_name))

remapped_body_pos = body_pos_w[:, body_indexes]  # [T, num_robot_bodies, 3]
print(f"Remapped shape: {remapped_body_pos.shape}")
print("\nRemapped body positions at t=0:")
for i, name in enumerate(robot_body_names):
    pos = remapped_body_pos[0, i]
    print(f"  [{i:2d}] {name:35s} pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

# Verify left/right after remapping
print("\nLeft/Right check AFTER remapping:")
for left_name, right_name in check_pairs:
    if left_name in robot_body_names and right_name in robot_body_names:
        left_i = robot_body_names.index(left_name)
        right_i = robot_body_names.index(right_name)
        left_y = remapped_body_pos[0, left_i, 1]
        right_y = remapped_body_pos[0, right_i, 1]
        status = "OK" if left_y > right_y else "*** SWAPPED ***"
        print(f"  {left_name:30s} Y={left_y:+.4f}  |  {right_name:30s} Y={right_y:+.4f}  [{status}]")

# ── 6. body_names_to_track mapping check ─────────────────────────────────────
print("\n" + "=" * 80)
print("body_names_to_track MAPPING")
print("=" * 80)
body_names_to_track = [
    "pelvis_link", "left_hip_roll_link", "left_knee_pitch_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_pitch_link", "right_ankle_roll_link",
    "waist_pitch_link", "left_shoulder_roll_link", "left_elbow_pitch_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_pitch_link", "right_wrist_yaw_link",
]
print("tracked body → robot_body index → NPZ body index")
for name in body_names_to_track:
    robot_i = robot_body_names.index(name) if name in robot_body_names else "NOT FOUND"
    npz_i = npz_body_names.index(name) if name in npz_body_names else "NOT FOUND"
    pos = body_pos_w[0, npz_i] if isinstance(npz_i, int) else [0,0,0]
    print(f"  {name:30s} robot[{robot_i:>2}] → NPZ[{npz_i:>2}]  pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
if not body_errors and not joint_errors:
    print("All joint and body names matched successfully.")
else:
    print(f"ERRORS: {len(body_errors)} body mismatches, {len(joint_errors)} joint mismatches")
