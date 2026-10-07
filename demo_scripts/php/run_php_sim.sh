#!/usr/bin/env bash
# MuJoCo sim for the D435i depth-distillation (PHP) policy: spawns the G1 with the RealSense D435i
# torso depth camera and publishes preprocessed depth to shared memory for run_php_inference.sh.
#
# Start this BEFORE the policy — the plugin creates the shared-memory block.
#
# Differs from run_stair_sim.sh only in the camera/plugin rig: the ZED 2i presets there carry a
# 71deg-down mount and a 2.0m far clip, while these match the D435i training rig (27deg down,
# [0.3, 3.0]m, training's [2:, 4:-4] pre-resize crop).
#
# See src/holosoma_inference/docs/workflows/sim-to-sim-depth-distillation.md
set -e

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

source scripts/source_mujoco_setup.sh

# MuJoCo opens a window whenever it can reach a display; without one it runs headless and the
# viewer-only gantry keys (7/8/9) are unavailable. Set DISPLAY yourself if your X server isn't on :0.
if [ -z "${DISPLAY:-}" ]; then
    echo "[run_php_sim] DISPLAY is unset — the MuJoCo window will not open and the" >&2
    echo "              gantry keys (7/8/9) will be unavailable. Export DISPLAY to fix." >&2
fi

# A stale block from a crashed run has the wrong size if the resolution changed.
rm -f /dev/shm/depth_img_shm 2>/dev/null || true

# --robot.asset.xml-file selects the half-sphere-hand MJCF, matching the reference sim2sim setup.
# Set here rather than in the robot preset on purpose: training ran on an Isaac backend and so read
# only `urdf_file` (`main_mesh_arm_collision_halfspherehand.urdf`), leaving `xml_file` at the plain
# `g1_29dof.xml` it never opens. Baking the half-sphere XML into the preset would therefore change
# every MuJoCo consumer of `g1-29dof`, including ones that want the training pairing.
# 500 Hz physics / 125 Hz control (control_decimation=4): the reference sim2sim rate the
# depth-distillation checkpoints are validated against. Set here, not as the global mujoco
# default, whose 2000 Hz other configs rely on. A camera's update_decimation resolves against
# the CONTROL rate, which is why the depth presets say ">50Hz" (125/50 = 2.5 is inexact).
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    sensor.d435i_front_depth:g1-d435i-front-depth \
    plugin.depth:depth-shm-d435i \
    terrain:terrain-load-step \
    --robot.asset.xml-file g1/g1_29dof_halfspherehand.xml \
    --simulator.config.bridge.enabled=True \
    --simulator.config.sim.fps=500 \
    "$@"

# Notes:
#   - `sensor.<key>:<preset>` and `plugin.<key>:<preset>` are positional declarations, not --flags.
#     The key becomes the sensor/plugin name, and the plugin's `camera` field must match the sensor
#     key — the `depth-shm-d435i` preset defaults to `d435i_front_depth`, as used above.
#   - Expect on startup:
#       [DepthShmPlugin] created 'depth_img_shm' (20184 bytes) shape=(1, 1, 58, 87)
#   - The sim runs 500Hz physics / 125Hz control. A camera's update_decimation resolves against the
#     CONTROL rate, so the presets use ">50Hz" (a bare "50Hz" is not exactly achievable from 125Hz).
#   - Override the depth contract per-run if a checkpoint needs it, e.g.
#       demo_scripts/php/run_php_sim.sh --plugin.depth.far-clip 2.0 --plugin.depth.latency-frames 3
#   - Swap the terrain mesh with
#       --terrain.terrain-term.obj-file-path /abs/path/to/terrain.obj
#     or use terrain:terrain-locomotion-plane for flat ground.
#   - To mirror the object-carrying WBT rig instead, pass robot:g1-29dof-w-object.
#
# MuJoCo window keys: 8 lower gantry (hold ~25 presses), 9 release, backspace reset.
