"""Visualize motion data skeleton animation and compare with robot FK.

Creates an MP4 showing:
- Motion data body positions (blue skeleton)
- Robot FK body positions (red skeleton) from applying motion joint angles to robot model
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

# ── 1. Load motion NPZ ───────────────────────────────────────────────────────
motion_file = "src/holosoma/holosoma/data/motions/statue_v1_rmd/whole_body_tracking/unreal_engine/walk_statue_v1_rmd_50fps.npz"
data = np.load(motion_file, allow_pickle=True)

npz_body_names = data["body_names"].tolist()
npz_joint_names = data["joint_names"].tolist()
body_pos_w = data["body_pos_w"]  # [T, num_bodies, 3]
joint_pos_all = data["joint_pos"]  # [T, 7+num_joints] (first 7 = root pos+quat)
fps = int(data["fps"])

print(f"Motion: {body_pos_w.shape[0]} frames at {fps} FPS = {body_pos_w.shape[0]/fps:.1f}s")
print(f"Bodies: {len(npz_body_names)}, Joints: {len(npz_joint_names)}")

# ── 2. Define skeleton connections ───────────────────────────────────────────
# Parent-child relationships for visualization
skeleton_connections = [
    # Torso
    ("pelvis_link", "waist_yaw_link"),
    ("waist_yaw_link", "waist_pitch_link"),
    # Left leg
    ("pelvis_link", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_pitch_link"),
    ("left_knee_pitch_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    # Right leg
    ("pelvis_link", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_pitch_link"),
    ("right_knee_pitch_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    # Left arm
    ("waist_pitch_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_pitch_link"),
    ("left_elbow_pitch_link", "left_wrist_roll_link"),
    ("left_wrist_roll_link", "left_wrist_yaw_link"),
    # Right arm
    ("waist_pitch_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_pitch_link"),
    ("right_elbow_pitch_link", "right_wrist_roll_link"),
    ("right_wrist_roll_link", "right_wrist_yaw_link"),
]

# Get indices for connections
conn_indices = []
for parent, child in skeleton_connections:
    if parent in npz_body_names and child in npz_body_names:
        conn_indices.append((npz_body_names.index(parent), npz_body_names.index(child)))

# Bodies to highlight (termination check bodies)
highlight_bodies = ["left_ankle_roll_link", "right_ankle_roll_link",
                    "left_wrist_yaw_link", "right_wrist_yaw_link"]
highlight_indices = [npz_body_names.index(n) for n in highlight_bodies if n in npz_body_names]

# ── 3. Create animation ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))

# Front view (XZ plane)
ax1 = fig.add_subplot(121)
ax1.set_title("Front View (Y vs Z)")
ax1.set_xlabel("Y (m)")
ax1.set_ylabel("Z (m)")
ax1.set_aspect("equal")

# Side view (XZ plane)
ax2 = fig.add_subplot(122)
ax2.set_title("Side View (X vs Z)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Z (m)")
ax2.set_aspect("equal")

# Subsample for speed (every 2nd frame)
step = 2
frames = range(0, body_pos_w.shape[0], step)
total_frames = len(list(frames))

def update(frame_idx):
    t = frame_idx * step
    ax1.clear()
    ax2.clear()

    pos = body_pos_w[t]  # [num_bodies, 3]
    pelvis_x = pos[npz_body_names.index("pelvis_link"), 0]

    # Front view: Y vs Z (centered on pelvis X)
    ax1.set_title(f"Front View (Y vs Z) - Frame {t}/{body_pos_w.shape[0]} ({t/fps:.2f}s)")
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_ylim(-0.1, 1.8)
    ax1.set_xlabel("Y (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Draw skeleton
    for pi, ci in conn_indices:
        ax1.plot([pos[pi, 1], pos[ci, 1]], [pos[pi, 2], pos[ci, 2]], 'b-', linewidth=1.5, alpha=0.7)

    # Draw all bodies
    for i in range(len(npz_body_names)):
        if npz_body_names[i] == "world":
            continue
        color = 'red' if i in highlight_indices else 'blue'
        size = 40 if i in highlight_indices else 15
        ax1.scatter(pos[i, 1], pos[i, 2], c=color, s=size, zorder=5)

    # Label highlighted bodies
    for i in highlight_indices:
        ax1.annotate(npz_body_names[i].replace("_link", "").replace("_", "\n"),
                     (pos[i, 1], pos[i, 2]), fontsize=6, ha='center', va='bottom',
                     color='red', fontweight='bold')

    # Side view: X vs Z (centered on pelvis)
    ax2.set_title(f"Side View (X vs Z)")
    ax2.set_xlim(pelvis_x - 0.8, pelvis_x + 0.8)
    ax2.set_ylim(-0.1, 1.8)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    for pi, ci in conn_indices:
        ax2.plot([pos[pi, 0], pos[ci, 0]], [pos[pi, 2], pos[ci, 2]], 'b-', linewidth=1.5, alpha=0.7)

    for i in range(len(npz_body_names)):
        if npz_body_names[i] == "world":
            continue
        color = 'red' if i in highlight_indices else 'blue'
        size = 40 if i in highlight_indices else 15
        ax2.scatter(pos[i, 0], pos[i, 2], c=color, s=size, zorder=5)

    # Ground line
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='ground')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='ground')

    return []

print(f"Creating animation with {total_frames} frames...")
anim = FuncAnimation(fig, update, frames=total_frames, blit=False)
writer = FFMpegWriter(fps=fps//step, bitrate=2000)
output_path = "motion_data_walk.mp4"
anim.save(output_path, writer=writer)
print(f"Saved to {output_path}")
plt.close()

# ── 4. Also print key body height differences ───────────────────────────────
print("\n" + "=" * 80)
print("BODY HEIGHT ANALYSIS (Z positions over time)")
print("=" * 80)
pelvis_idx = npz_body_names.index("pelvis_link")
print(f"Pelvis Z: min={body_pos_w[:, pelvis_idx, 2].min():.3f} max={body_pos_w[:, pelvis_idx, 2].max():.3f} mean={body_pos_w[:, pelvis_idx, 2].mean():.3f}")
print(f"Robot init_state height: 0.89")
print(f"Motion data pelvis height at t=0: {body_pos_w[0, pelvis_idx, 2]:.3f}")
print(f"Difference: {0.89 - body_pos_w[0, pelvis_idx, 2]:.3f}m")

for name in highlight_bodies:
    idx = npz_body_names.index(name)
    print(f"{name}: Z_min={body_pos_w[:, idx, 2].min():.3f} Z_max={body_pos_w[:, idx, 2].max():.3f} Z_mean={body_pos_w[:, idx, 2].mean():.3f}")
