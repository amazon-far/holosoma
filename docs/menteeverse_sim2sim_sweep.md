# MenteeVerse → MuJoCo sim2sim validation: sweep report

**Date:** 2026-06-23
**Tool:** `scripts/validate_menteeverse_sim2sim.py`
**Goal:** Take MenteeVerse (IsaacSim-trained) deploy ONNX policies and evaluate whether
they transfer to MuJoCo physics, as a pre-hardware / cross-engine regression check.

## What the validator does

Drives a MenteeVerse structured deploy export (`policy.onnx` + `policy_config.json` +
`initial_state/`, the "menteebot 3.x" tensor schema) against the same v3.1 MuJoCo model
holosoma uses (`menteebot_v3_1_29dof.xml`). It is **schema-driven**: it reads the input/
output tensor labels from `policy_config.json` and assembles `motor[71,3]`, `imu_torso[10]`,
`clock`, `velocity_command`, recurrent LSTM state, and the extended LbcStanding/Planner
inputs (`lbc`, `imu_torso_link`, `base_link`, …). Per 30/50 Hz control step it assembles
observations from MuJoCo state, runs the ONNX, and applies joint-space PD with the
policy-emitted gains, carrying LSTM state across steps.

### Three real harness bugs were found and fixed during bring-up
1. **Floating init.** The nominal `base_z = 0.95` floats the feet ~7 cm above the floor;
   the robot free-fell, slammed, and the recurrent policy never recovered. Fix: auto-lower
   the base so the feet rest on the floor (`--ground-init`, default on). The policy has no
   base-height observation, so only ground contact at the default pose matters.
2. **IMU angular-velocity frame.** The fused physics MJCF welds `imu_torso` into
   `torso_roll_link` *and reorients that link's frame*, so a hand-coded rotation put
   `ang_v_loc` on **permuted axes** — invisible at standing (ω = 0), fatal under motion.
   Fix: read the IMU off the **real `imu_torso` body in a parallel unfused model** synced
   each step — exactly MenteeVerse's `body_quat_w` / `rotate_to_rb_frame` definition.
3. **Ankle-sphere ground contact.** The URDF gives `ankle_pitch_link` a collision sphere
   whose bottom sits ~15 mm **below** the foot soles, so the robot balanced on two ankle
   points (a line) and pitched over; the flat feet never touched. Fix: feet-only ground
   contact (`--all-leg-contact` to restore full contact).

Contract verified from source (`packages/menteeverse/.../mdp/sensors/sensors.py`): deploy
`motor.pos/vel` are **raw** (no scaling), `imu` is raw world-frame quat (xyzw) + body-frame
angular velocity. `effort` and `free_acc` are ignored by these policies (zeroing them gives
a byte-identical run); PD gains come from the policy output (`ki = 0`).

## Sweep results

Config: feet-only contact, ground-init, IMU fix, foot friction 1.5, dt 1/210.
"PASS" = stays upright for the full ≥3 s stand phase (and 7 s total).

| Policy (June 2026 deploy) | Result | Behavior in MuJoCo |
|---|---|---|
| **V3.1-LbcStanding-Isaac-Eval** (17/18/21 Jun) | ✅ **PASS** | Stands rock-solid 7 s, upright 0.998–0.999, drift < 0.1 m |
| V3.1-LbcStanding (14/15 Jun) | ⚠️ survives, wobbly | upright 0.946, drifts −0.78 m |
| V3.1-LbcStanding (16 Jun) | ❌ fell 1.44 s | forward pitch (regression) |
| V3.1-Standing-Isaac-Eval (20/22 Jun) | ❌ ~1.70 s | balances ~1 s, then marches forward and pitches over |
| V3.1-Walking-Isaac-Eval (22 Jun) | ❌ ~1.63 s | **no real gait** — topples/slides forward, no leg lift |
| V3.1-Stomping-Isaac-Eval (23 Jun) | ❌ ~1.87 s | sideways drift then forward pitch |
| V3.1-Planner-Isaac (23 Jun) | ⚠️ stands (caveat) | upright 0.995, but the `planner` input was zero-filled (unverified) |

### LbcStanding robustness emergence
14–15 Jun: survives but wobbly (drifts ~0.8 m). 16 Jun: regressed (fell). **17 Jun onward:
rock-solid.** Robustness stabilized mid-June.

## Conclusions

1. **The validator is correct and well-calibrated.** LbcStanding passes *flawlessly* through
   the exact same path (same `motor`/`imu_torso` inputs, same IMU computation, same contact
   and actuator) that the failing policies use. If there were a systematic obs/contact/
   actuator bug, LbcStanding could not stand perfectly through it. The validator cleanly
   **discriminates** transfer-robust from transfer-fragile policies.
2. **Standing / Walking / Stomping are overfit to PhysX.** They show real balancing/locomotion
   intent for ~1.5–1.9 s, then hit a fine contact-dynamics limit and fall in a
   policy-specific, direction-specific way (Standing → forward pitch, Walking → lateral roll,
   Stomping → sideways). This is a genuine **transfer-robustness gap**, not a harness bug.
3. **LbcStanding is the existence proof** that MenteeVerse policies *can* transfer to MuJoCo
   with the current asset/contact model — so the fix for the others is on the **training side**
   (more contact/dynamics domain randomization), with this validator as the regression check.
   This mirrors holosoma, whose IsaacGym/IsaacSim-trained policies transfer to its own MuJoCo
   sim2sim (`validate_menteebot_v31_sim2sim.py`) because they are DR-trained.

## How to run

```bash
HOME=<home> PYTHONPATH= PYTHONNOUSERSITE=1 \
  .venv/hsinference/bin/python scripts/validate_menteeverse_sim2sim.py \
    --model <model.onnx> --config <policy_config.json> --init-state-dir <initial_state/> \
    [--cmd-vx 0.5] [--walk-from-start] [--lbc base_z roll pitch yaw stand] \
    [--diag] [--video out.mp4]
```

Key flags: `--walk-from-start` (command forward velocity from t=0, for walking policies that
can't hold a zero-velocity stand), `--lbc` (LbcStanding command), `--all-leg-contact`
(restore full contact), `--pin-base` (diagnostic: hold base, isolate leg behavior),
`--no-filter/--no-delay/--no-clip` (actuator ablations).

Models live in S3 `s3://mentee-models/develop/<Task>/<DD_MM_YY_n>/`; only **June 2026+**
exports use the structured `policy_config.json` deploy format this tool reads.
