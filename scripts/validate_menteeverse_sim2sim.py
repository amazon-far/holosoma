#!/usr/bin/env python
"""Sim2sim validator: run a menteeverse (IsaacSim-trained) policy in MuJoCo.

Counterpart to validate_menteebot_v31_sim2sim.py (which validates holosoma
policies). This one consumes a *menteeverse* deploy export — the structured
"menteebot 3.x" tensor-schema ONNX (e.g. V3.1-Walking-Isaac-Eval) produced by
`play.py --deploy` and posted to the model bot / S3 — and drives the same v3.1
MuJoCo model we built for the holosoma validator.

The contract is read from the deploy's `policy_config.json` (schema-driven), so
this generalizes to other menteeverse deploy ONNXes, not just walking. For the
walking policy the contract is:

  inputs:  motor[1,1,71,3]  (features: effort,pos,vel; 71 joints incl. fingers)
           imu_torso[1,1,10] (ang_v_loc.xyz, free_acc_lab.xyz, quat.xyzw)
           clock[1,1,1]       (episode time, seconds)
           velocity_command[1,1,3] (lin_vel_x, lin_vel_y, heading)
           input_hidden_state[1,1,256], input_cell_state[1,1,256]  (LSTM)
  output:  motor_action[1,1,41,4] (features: target_pos,kp,kd,ki; 41 joints)
           output_hidden_state, output_cell_state

Control: per 30 Hz step assemble inputs from MuJoCo state, run ONNX, then apply
joint-space PD with the policy-EMITTED gains (target_pos absolute) for the
decimated physics substeps, carrying the LSTM state across steps.

Joints in the schema but absent from the MJCF (the 12 fingers) are held at their
default pose for the motor input and skipped for actuation.

Usage:
  HOME=/home/ddn/galn PYTHONPATH= PYTHONNOUSERSITE=1 \
    .venv/hsinference/bin/python scripts/validate_menteeverse_sim2sim.py \
      --model /tmp/walking_model/policy.onnx \
      --config /tmp/walking_model/policy_config.json \
      --init-state-dir /tmp/walking_model/initial_state \
      [--mjcf <path>] [--cmd-vx 0.5]
"""

import argparse
import json
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort

DEFAULT_MJCF = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src/holosoma/holosoma/data/robots/menteebot_v3_1/menteebot_v3_1_29dof.xml",
)

# v3.1 default standing pose (matches both holosoma and menteeverse init_state).
DEFAULT_POSE = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.3,
    "left_knee": 0.6, "left_ankle_pitch": 0.3, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.3,
    "right_knee": -0.6, "right_ankle_pitch": -0.3, "right_ankle_roll": 0.0,
    "torso_yaw": 0.0, "torso_pitch": 0.0, "torso_roll": 0.0,
    "left_shoulder_pitch": 0.0, "left_shoulder_roll": 1.3, "left_shoulder_yaw": 0.0,
    "left_elbow": 0.0, "left_wrist_roll": 0.0, "left_wrist_yaw": 0.0, "left_wrist_pitch": 0.0,
    "right_shoulder_pitch": 0.0, "right_shoulder_roll": 1.3, "right_shoulder_yaw": 0.0,
    "right_elbow": 0.0, "right_wrist_roll": 0.0, "right_wrist_yaw": 0.0, "right_wrist_pitch": 0.0,
}
# NOTE: the IMU is NOT read off the fused physics model. MjSpec welds the
# imu_torso fixed-joint chain into torso_roll_link AND reorients that link's frame,
# so a hand-coded torso_roll->imu rotation does NOT compose correctly for the
# angular velocity (the quat matched but ang_v landed on permuted axes — a bug that
# is invisible at standing but corrupts balance feedback under motion). Instead we
# read the real imu_torso body off a parallel UNFUSED model synced each step (see
# build_imu_model / assemble_inputs), which reproduces menteeverse's sensors exactly.

# menteeverse V3_1 actuator groups: (effort_limit, velocity_limit, velocity_peak).
# The policy was trained with this actuator model (DC-motor torque-speed clip +
# 8 ms command delay + 2nd-order 4 Hz Butterworth target filter), so reproducing
# it is required for the recurrent policy to stay in-distribution. RFI / efficiency
# / voltage-curve / current-limiter are domain randomization (nominal at eval).
GROUP_PARAMS = {
    "MGA": (150.0, 7.19, 5.81),   # torso_yaw, hip_pitch, knee
    "MGB": (130.0, 7.07, 4.96),   # hip_yaw, hip_roll, shoulder_pitch, shoulder_roll, ankle_pitch
    "MGC": (70.0, 7.51, 6.35),    # ankle_roll, shoulder_yaw, elbow
    "MGD": (36.0, 13.64, 11.46),  # wrists
    "TORSO": (150.0, 7.19, 5.81),  # torso_roll, torso_pitch
}
ACT_DELAY_S = 0.008
FILTER_ORDER = 2
FILTER_CUTOFF_HZ = 4.0


def _joint_group(name):
    if name in ("torso_roll", "torso_pitch"):
        return "TORSO"
    if name == "torso_yaw" or name.endswith("hip_pitch") or name.endswith("knee"):
        return "MGA"
    if (name.endswith("hip_yaw") or name.endswith("hip_roll") or name.endswith("shoulder_pitch")
            or name.endswith("shoulder_roll") or name.endswith("ankle_pitch")):
        return "MGB"
    if name.endswith("ankle_roll") or name.endswith("shoulder_yaw") or name.endswith("elbow"):
        return "MGC"
    if "wrist" in name:
        return "MGD"
    return "MGB"

# menteeverse V3_1_RobotCfg actuator PD gains (kp, kd), per joint. Used only for
# the pre-policy settle; during the policy phase the gains come from the policy
# output (the deploy_only action_gains).
SETTLE_GAINS = {
    "left_hip_pitch": (500, 60), "right_hip_pitch": (500, 60),
    "left_knee": (400, 50), "right_knee": (400, 50),
    "left_hip_yaw": (300, 30), "right_hip_yaw": (300, 30),
    "left_hip_roll": (600, 50), "right_hip_roll": (600, 50),
    "left_ankle_pitch": (200, 10), "right_ankle_pitch": (200, 10),
    "left_ankle_roll": (150, 15), "right_ankle_roll": (150, 15),
    "torso_yaw": (600, 50), "torso_roll": (1300, 300), "torso_pitch": (1300, 300),
    "left_shoulder_pitch": (300, 15), "right_shoulder_pitch": (300, 15),
    "left_shoulder_roll": (300, 15), "right_shoulder_roll": (300, 15),
    "left_shoulder_yaw": (100, 10), "right_shoulder_yaw": (100, 10),
    "left_elbow": (100, 10), "right_elbow": (100, 10),
    "left_wrist_roll": (35, 3), "left_wrist_yaw": (35, 3), "left_wrist_pitch": (35, 3),
    "right_wrist_roll": (35, 3), "right_wrist_yaw": (35, 3), "right_wrist_pitch": (35, 3),
}


def _feature_labels(node):
    for dim in node["dimensions"]:
        if dim["name"] == "features":
            return dim["labels"]
    return None


def _instance_labels(node):
    for dim in node["dimensions"]:
        if dim["name"] == "instance":
            return dim["labels"]
    return None


class MenteeverseContract:
    """Parses policy_config.json into the index maps needed to talk to the ONNX."""

    def __init__(self, config_path, model, init_state_dir):
        cfg = json.load(open(config_path))
        self.freq = float(cfg.get("inference_frequency", 30))
        sensors, actions = cfg["sensors"], cfg["actions"]

        self.motor_joints = _instance_labels(sensors["motor"])           # 71
        self.motor_feats = _feature_labels(sensors["motor"])             # [effort,pos,vel]
        self.imu_feats = _feature_labels(sensors["imu_torso"])           # 10
        self.act_joints = _instance_labels(actions["motor_action"])      # 41
        self.act_feats = _feature_labels(actions["motor_action"])        # [target_pos,kp,kd,ki]

        # MJCF address maps (None where the joint is absent, e.g. fingers).
        self.q_adr, self.v_adr = {}, {}
        self.act_id = {}
        for j in range(model.njnt):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                self.q_adr[n] = model.jnt_qposadr[j]
                self.v_adr[n] = model.jnt_dofadr[j]
        for a in range(model.nu):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            self.act_id[n] = a

        # Recurrent state init (256-float32 blobs).
        rt = cfg.get("recurrent_tensors", {})
        self.hidden0 = self._load_state(init_state_dir, rt, "hidden_state")
        self.cell0 = self._load_state(init_state_dir, rt, "cell_state")

    @staticmethod
    def _load_state(init_state_dir, rt, key):
        info = rt.get(key, {})
        shape = tuple(info.get("shape", [1, 1, 256]))
        path = os.path.join(init_state_dir, os.path.basename(info.get("initial_state_file", key)))
        if os.path.isfile(path):
            return np.fromfile(path, dtype=np.float32).reshape(shape)
        return np.zeros(shape, dtype=np.float32)


class JointActuator:
    """Vectorized reproduction of menteeverse PDActuator over the MJCF's actuated
    joints: per-substep command delay, 2nd-order Butterworth target filtering, PD
    with policy-supplied kp/kd, and DC-motor torque-speed clipping.
    """

    def __init__(self, names, q_adr, v_adr, act_id, sim_dt):
        from scipy import signal
        self.names = names
        self.q_adr = np.array([q_adr[n] for n in names])
        self.v_adr = np.array([v_adr[n] for n in names])
        self.act_id = np.array([act_id[n] for n in names])
        el, vl, vp = (np.array(x) for x in zip(*[GROUP_PARAMS[_joint_group(n)] for n in names]))
        self.effort_limit, self.velocity_limit, self.velocity_peak = el, vl, vp
        self.saturation = el * vl / (vl - vp)
        # Butterworth coeffs at the sim rate (4 Hz cutoff in real time).
        b, a = signal.butter(FILTER_ORDER, FILTER_CUTOFF_HZ / (0.5 / sim_dt), btype="low")
        self.b, self.a = b.astype(np.float64), a.astype(np.float64)
        self.n = len(names)
        self.x = np.zeros((self.n, FILTER_ORDER + 1))  # input history
        self.y = np.zeros((self.n, FILTER_ORDER))      # output history
        self.delay_steps = max(0, round(ACT_DELAY_S / sim_dt))
        self.delay_buf = None

    def reset(self, init_target):
        self.x[:] = init_target[:, None]
        self.y[:] = init_target[:, None]
        self.delay_buf = [init_target.copy() for _ in range(self.delay_steps + 1)]

    def _filter(self, u):
        self.x = np.roll(self.x, 1, axis=1); self.x[:, 0] = u
        out = self.x @ self.b - self.y @ self.a[1:]
        self.y = np.roll(self.y, 1, axis=1); self.y[:, 0] = out
        return out

    use_delay = True
    use_filter = True
    use_clip = True

    def substep(self, data, target, kp, kd):
        # command delay (on target position), then Butterworth filter
        if self.use_delay:
            self.delay_buf.append(target.copy())
            delayed = self.delay_buf.pop(0)
        else:
            delayed = target
        filt = self._filter(delayed) if self.use_filter else delayed
        q = data.qpos[self.q_adr]
        qd = data.qvel[self.v_adr]
        tau = kp * (filt - q) - kd * qd
        if self.use_clip:
            # DC-motor torque-speed clip
            max_e = np.clip(np.minimum(self.saturation * (1.0 - qd / self.velocity_limit), self.effort_limit), 0.0, None)
            min_e = np.clip(np.maximum(self.saturation * (-1.0 - qd / self.velocity_limit), -self.effort_limit), None, 0.0)
            tau = np.clip(np.clip(tau, min_e, max_e), -self.effort_limit, self.effort_limit)
        else:
            tau = np.clip(tau, -self.effort_limit, self.effort_limit)
        data.ctrl[self.act_id] = tau


def build_imu_model(mjcf_path):
    """Build an UNFUSED model (fusestatic=false) that keeps the real `imu_torso`
    body, used purely as a sensor reference. The fused physics model welds
    imu_torso into torso_roll_link AND reorients that link's frame, so neither a
    fixed R_OFFSET nor mj_objectVelocity composes cleanly for the IMU angular
    velocity. Instead we run physics on the fused model and, each control step,
    sync this unfused model's qpos/qvel by joint name, then read imu_torso exactly
    as menteeverse does (body_quat_w xyzw; world angular velocity rotated into the
    body frame; body-origin world linear velocity for the free-acc finite diff).
    """
    urdf = mjcf_path[:-4] + ".urdf" if mjcf_path.endswith(".xml") else mjcf_path
    tree = ET.parse(urdf)
    comp = ET.SubElement(ET.SubElement(tree.getroot(), "mujoco"), "compiler")
    comp.set("discardvisual", "true")
    comp.set("fusestatic", "false")
    fd, tmp = tempfile.mkstemp(suffix=".urdf", dir=os.path.dirname(urdf))
    os.close(fd)
    tree.write(tmp)
    try:
        spec = mujoco.MjSpec.from_file(tmp)
    finally:
        os.unlink(tmp)
    spec.compiler.degree = False
    spec.worldbody.bodies[0].add_freejoint()
    return spec.compile()


def build_visual_model(mjcf_path):
    """Build a render-only model from the URDF with fusestatic=false +
    discardvisual=false, so the visual meshes are correctly placed (the fused
    physics model mis-places welded-link visuals). Same 29 joints + freejoint as
    the physics model, so qpos can be synced by joint name for rendering."""
    urdf = mjcf_path[:-4] + ".urdf" if mjcf_path.endswith(".xml") else mjcf_path
    tree = ET.parse(urdf)
    comp = ET.SubElement(ET.SubElement(tree.getroot(), "mujoco"), "compiler")
    comp.set("discardvisual", "false")
    comp.set("fusestatic", "false")
    fd, tmp = tempfile.mkstemp(suffix=".urdf", dir=os.path.dirname(urdf))
    os.close(fd)
    tree.write(tmp)
    try:
        spec = mujoco.MjSpec.from_file(tmp)
    finally:
        os.unlink(tmp)
    spec.compiler.degree = False
    spec.worldbody.bodies[0].add_freejoint()
    fl = spec.worldbody.add_geom()
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [8, 8, 0.1]
    fl.rgba = [0.16, 0.17, 0.2, 1.0]
    for p in ([3, -3, 4], [-3, 3, 4]):
        spec.worldbody.add_light().pos = p
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-state-dir", required=True)
    ap.add_argument("--mjcf", default=os.path.abspath(DEFAULT_MJCF))
    ap.add_argument("--cmd-vx", type=float, default=0.5, help="forward velocity command for walk phase")
    ap.add_argument("--lbc", type=float, nargs=5, default=[0.95, 0.0, 0.0, 0.0, 1.0],
                    metavar=("base_z", "roll", "pitch", "yaw", "stand"),
                    help="LbcStanding command [base_z, torso_roll, torso_pitch, torso_yaw, stand]")
    ap.add_argument("--base-z", type=float, default=0.95)
    ap.add_argument("--ground-init", action="store_true", default=True,
                    help="auto-lower base so feet rest on the floor at init (default on)")
    ap.add_argument("--no-ground-init", dest="ground_init", action="store_false")
    ap.add_argument("--settle-steps", type=int, default=0,
                    help="pre-policy PD settle steps (0 = hand off from clean IsaacLab-style init)")
    ap.add_argument("--dt", type=float, default=1.0 / 210.0)
    ap.add_argument("--foot-friction", type=float, default=1.0, help="floor tangential friction")
    ap.add_argument("--ground-pen", type=float, default=0.001,
                    help="init penetration: feet pushed this far below floor (engages all foot boxes)")
    ap.add_argument("--solref", type=str, default=None, help="contact solref, e.g. '0.01 1'")
    ap.add_argument("--solimp", type=str, default=None, help="contact solimp, e.g. '0.95 0.99 0.001'")
    ap.add_argument("--no-delay", action="store_true", help="disable actuator command delay")
    ap.add_argument("--no-filter", action="store_true", help="disable Butterworth target filter")
    ap.add_argument("--no-clip", action="store_true", help="disable DC-motor torque-speed clip")
    ap.add_argument("--walk-from-start", action="store_true",
                    help="command forward velocity from t=0 (single walk phase) — for walking policies "
                         "that can't hold a zero-velocity stand")
    ap.add_argument("--pin-base", action="store_true",
                    help="diagnostic: hold the floating base fixed/upright so only legs move")
    ap.add_argument("--all-leg-contact", action="store_true",
                    help="keep collision on all leg geoms (default: feet-only — the ankle_pitch_link "
                         "spheres hang ~15mm below the soles and would be the only ground contact, "
                         "making the robot balance on 2 points and pitch over)")
    ap.add_argument("--zero-effort", action="store_true", help="feed 0 for motor.effort")
    ap.add_argument("--zero-freeacc", action="store_true", help="feed 0 for imu free_acc_lab")
    ap.add_argument("--diag", action="store_true", help="print per-step base roll/pitch/xy diagnostics")
    ap.add_argument("--video", default=None, help="path to write an mp4 of the run")
    args = ap.parse_args()

    # --- build MuJoCo world: robot MJCF + ground plane (collision filtering already
    # baked into the MJCF: robot geoms contype=2/conaffinity=1) ---
    spec = mujoco.MjSpec.from_file(args.mjcf)
    spec.compiler.degree = False
    spec.option.timestep = args.dt
    g = spec.worldbody.add_geom()
    g.name = "floor"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [0, 0, 0.05]
    g.contype = 1
    g.conaffinity = 1
    g.friction = [args.foot_friction, 0.005, 0.0001]
    if args.solref is not None:
        g.solref = [float(x) for x in args.solref.split()]
    if args.solimp is not None:
        g.solimp = [float(x) for x in args.solimp.split()]
    # Feet-only ground contact: the URDF gives ankle_pitch_link a collision sphere
    # whose bottom sits ~15 mm BELOW the foot soles, so it (not the flat foot) is the
    # only thing touching the floor — the robot ends up balancing on two points (a
    # line) and pitches over. Disable collision on every robot geom except the foot
    # bodies so the 4 sole boxes per foot form the real support polygon.
    model = spec.compile()
    if not args.all_leg_contact:
        n_off = 0
        for gid in range(model.ngeom):
            if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
                continue
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue  # the floor
            if "foot" not in bname:
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
                n_off += 1
        print(f"  feet-only contact: disabled {n_off} non-foot robot collision geoms")
    data = mujoco.MjData(model)

    C = MenteeverseContract(args.config, model, args.init_state_dir)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}

    control_dt = 1.0 / C.freq
    decimation = max(1, round(control_dt / args.dt))
    feat_idx = {f: i for i, f in enumerate(C.motor_feats)}

    # --- unfused IMU sensor model: keeps the real imu_torso body. Sync map from
    # fused physics model -> unfused sensor model, by joint name (the freejoint +
    # all hinges; fixed joints carry no DOF so the sets match). ---
    imu_model = build_imu_model(args.mjcf)
    imu_data = mujoco.MjData(imu_model)
    imu_body_id = mujoco.mj_name2id(imu_model, mujoco.mjtObj.mjOBJ_BODY, "imu_torso")
    assert imu_body_id >= 0, "imu_torso body missing from unfused model"
    # joint qpos/qvel address pairs (fused_adr, imu_adr) for each shared hinge
    imu_q_pairs, imu_v_pairs = [], []
    for j in range(imu_model.njnt):
        n = mujoco.mj_id2name(imu_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if imu_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and n in C.q_adr:
            imu_q_pairs.append((C.q_adr[n], imu_model.jnt_qposadr[j]))
            imu_v_pairs.append((C.v_adr[n], imu_model.jnt_dofadr[j]))
    # freejoint of the unfused base (first 7 qpos / first 6 qvel)
    imu_free_qadr = imu_model.jnt_qposadr[0]
    imu_free_vadr = imu_model.jnt_dofadr[0]

    # actuated joints in action order that exist in the MJCF (29 of the 41)
    act_names = [jn for jn in C.act_joints if jn in C.act_id]
    actuator = JointActuator(act_names, C.q_adr, C.v_adr, C.act_id, args.dt)
    actuator.use_delay = not args.no_delay
    actuator.use_filter = not args.no_filter
    actuator.use_clip = not args.no_clip
    default_targets = np.array([DEFAULT_POSE.get(n, 0.0) for n in act_names])

    # --- initial pose: base upright, joints at default, feet resting on the floor.
    # The policy has NO base-height observation, so absolute base_z is irrelevant to
    # it — what matters is starting in ground contact at the default pose (as in the
    # IsaacSim reset that produced the initial LSTM state). The nominal base_z=0.95
    # actually floats the feet ~7 cm above the floor for this MJCF, so we auto-lower
    # the base until the lowest collision geom just touches (1 mm penetration so the
    # contact is active). Avoids a free-fall/slam transient that topples the policy.
    def _lowest_contact_z():
        zmin = 1e9
        for g in range(model.ngeom):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue  # the floor itself sits at z=0; only consider robot geoms
            R = data.geom_xmat[g].reshape(3, 3)
            sz = model.geom_size[g]
            p = data.geom_xpos[g]
            t = model.geom_type[g]
            if t in (mujoco.mjtGeom.mjGEOM_BOX,):
                corners = np.array([[sx, sy, sz2] for sx in (-sz[0], sz[0])
                                    for sy in (-sz[1], sz[1]) for sz2 in (-sz[2], sz[2])])
                zz = (p + corners @ R.T).min(0)[2]
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                zz = p[2] - sz[0]
            elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                zz = p[2] - sz[0] - sz[1]
            else:
                zz = p[2]
            zmin = min(zmin, zz)
        return zmin

    def reset_pose():
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qpos[:3] = [0.0, 0.0, args.base_z]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for n, adr in C.q_adr.items():
            data.qpos[adr] = DEFAULT_POSE.get(n, 0.0)
        mujoco.mj_forward(model, data)
        if args.ground_init:
            data.qpos[2] -= _lowest_contact_z() + args.ground_pen  # push feet ground_pen below floor
            mujoco.mj_forward(model, data)

    reset_pose()
    actuator.reset(default_targets)  # init target filter + delay buffers to default pose

    # --- input assembly --------------------------------------------------------
    prev_v_world = np.zeros(3)
    _warned_inputs = set()

    def assemble_inputs(clock_t, cmd, hidden, cell):
        nonlocal prev_v_world
        # motor [1,1,71,3]
        motor = np.zeros((1, 1, len(C.motor_joints), 3), dtype=np.float32)
        for i, jn in enumerate(C.motor_joints):
            pos = DEFAULT_POSE.get(jn, 0.0)
            vel = 0.0
            eff = 0.0
            if jn in C.q_adr:
                pos = float(data.qpos[C.q_adr[jn]])
                vel = float(data.qvel[C.v_adr[jn]])
                if jn in C.act_id:
                    eff = float(data.actuator_force[C.act_id[jn]])
            motor[0, 0, i, feat_idx["effort"]] = 0.0 if args.zero_effort else eff
            motor[0, 0, i, feat_idx["pos"]] = pos
            motor[0, 0, i, feat_idx["vel"]] = vel

        # imu_torso [1,1,10]: ang_v_loc.xyz, free_acc_lab.xyz, quat.xyzw, read off
        # the real imu_torso body in the synced unfused model — exactly menteeverse's
        # bodies_ang_v_local / bodies_quat / bodies_free_acc_lab sensors.
        imu_data.qpos[imu_free_qadr:imu_free_qadr + 7] = data.qpos[:7]
        imu_data.qvel[imu_free_vadr:imu_free_vadr + 6] = data.qvel[:6]
        for fa, ia in imu_q_pairs:
            imu_data.qpos[ia] = data.qpos[fa]
        for fv, iv in imu_v_pairs:
            imu_data.qvel[iv] = data.qvel[fv]
        mujoco.mj_forward(imu_model, imu_data)
        R_imu = imu_data.xmat[imu_body_id].reshape(3, 3)  # imu orientation in world
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(imu_model, imu_data, mujoco.mjtObj.mjOBJ_BODY, imu_body_id, v6, 1)
        ang_v_loc = v6[:3]  # body-frame angular velocity (rotate_to_rb_frame)
        v_world = np.zeros(6)
        mujoco.mj_objectVelocity(imu_model, imu_data, mujoco.mjtObj.mjOBJ_BODY, imu_body_id, v_world, 0)
        free_acc_lab = (v_world[3:] - prev_v_world) / control_dt  # world-frame linear accel
        prev_v_world = v_world[3:].copy()
        if args.zero_freeacc:
            free_acc_lab = np.zeros(3)
        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, R_imu.flatten())
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        imu = np.concatenate([ang_v_loc, free_acc_lab, quat_xyzw]).reshape(1, 1, 10).astype(np.float32)

        inp = {
            "motor": motor,
            "imu_torso": imu,
            "clock": np.array([[[clock_t]]], dtype=np.float32),
            "velocity_command": np.array([[cmd]], dtype=np.float32),
            "input_hidden_state": hidden,
            "input_cell_state": cell,
        }
        # Extended-schema inputs (LbcStanding / Planner variants), assembled only if
        # the ONNX wants them. Names changed across checkpoints, so alias them:
        #  imu_torso_link / imu_torso_transform: TF [pos.xyz, quat.xyzw] of imu body.
        #  base_link / base_link_transform:      TF [pos.xyz, quat.xyzw] of base.
        #  lbc / lbc_commands: standing command [base_z, torso_rpy, stand].
        imu_tf = np.concatenate([imu_data.xpos[imu_body_id], quat_xyzw]).reshape(1, 1, 7).astype(np.float32)
        base_q = data.qpos[3:7]  # wxyz
        base_tf = np.concatenate([data.qpos[:3],
                                  [base_q[1], base_q[2], base_q[3], base_q[0]]]).reshape(1, 1, 7).astype(np.float32)
        lbc_v = np.array(args.lbc, dtype=np.float32).reshape(1, 1, len(args.lbc))
        for nm in ("imu_torso_link", "imu_torso_transform"):
            if nm in in_names:
                inp[nm] = imu_tf
        for nm in ("base_link", "base_link_transform"):
            if nm in in_names:
                inp[nm] = base_tf
        for nm in ("lbc", "lbc_commands"):
            if nm in in_names:
                inp[nm] = lbc_v
        # Any other required input we don't know → zero-fill with a warning (once).
        for nm in in_names:
            if nm not in inp and not nm.startswith("input_"):
                shp = next(i.shape for i in sess.get_inputs() if i.name == nm)
                shp = [1 if (not isinstance(d, int)) else d for d in shp]
                if nm not in _warned_inputs:
                    print(f"  WARNING: unknown required input '{nm}' shape={shp} → zero-filled (result unreliable)")
                    _warned_inputs.add(nm)
                inp[nm] = np.zeros(shp, dtype=np.float32)
        return {k: v for k, v in inp.items() if k in in_names}

    def run_policy_step(clock_t, cmd, hidden, cell):
        out = sess.run(None, assemble_inputs(clock_t, cmd, hidden, cell))
        names = [o.name for o in sess.get_outputs()]
        od = dict(zip(names, out))
        action = od["motor_action"][0, 0]  # [41,4]
        hidden = od.get("output_hidden_state", hidden)
        cell = od.get("output_cell_state", cell)
        # map to per-actuator arrays (target_pos, kp, kd) in act_names order
        ti = C.act_feats.index("target_pos")
        kpi = C.act_feats.index("kp")
        kdi = C.act_feats.index("kd")
        action_by_joint = {jn: action[i] for i, jn in enumerate(C.act_joints)}
        tp = np.empty(len(act_names)); kp = np.empty(len(act_names)); kd = np.empty(len(act_names))
        for k, jn in enumerate(act_names):
            row = action_by_joint[jn]
            t_, p_, d_ = row[ti], row[kpi], row[kdi]
            # The walking policy was trained on the fixed-wrist asset and emits
            # NaN for those non-controlled joints; hold them at default instead.
            if not (np.isfinite(t_) and np.isfinite(p_) and np.isfinite(d_)):
                t_, (p_, d_) = DEFAULT_POSE.get(jn, 0.0), SETTLE_GAINS.get(jn, (80, 8))
            tp[k], kp[k], kd[k] = t_, p_, d_
        return (tp, kp, kd), hidden, cell

    pinned_qpos7 = data.qpos[:7].copy()

    def apply_and_step(target, kp, kd):
        for _ in range(decimation):
            actuator.substep(data, target, kp, kd)
            mujoco.mj_step(model, data)
            if args.pin_base:
                # diagnostic: hold the floating base fixed & upright so only the
                # legs move — isolates whether marching is base-instability-driven.
                data.qpos[:7] = pinned_qpos7
                data.qvel[:6] = 0.0
                mujoco.mj_forward(model, data)

    def base_metrics():
        z = float(data.qpos[2])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.qpos[3:7])
        up = R.reshape(3, 3)[2, 2]  # +1 = upright
        return z, up, float(data.qpos[0])

    def base_diag():
        # roll/pitch of the base (deg), and xy drift from origin.
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.qpos[3:7])
        R = R.reshape(3, 3)
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
        return roll, pitch, float(data.qpos[0]), float(data.qpos[1])

    def com_contact():
        # whole-robot CoM x, and per-foot ground-contact count + mean contact x.
        com_x = float(data.subtree_com[1][0])
        ncL = ncR = 0
        sxL = sxR = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
            body = b1 if "foot" in b1 else b2
            if "left_foot" in body:
                ncL += 1; sxL += c.pos[0]
            elif "right_foot" in body:
                ncR += 1; sxR += c.pos[0]
        cxL = sxL / ncL if ncL else float("nan")
        cxR = sxR / ncR if ncR else float("nan")
        return com_x, ncL, cxL, ncR, cxR

    # Rendering uses a separate unfused visual model (correct mesh placement);
    # physics stays on the fused `model`. Sync qpos by joint name each frame.
    renderer = vis_model = vis_data = None
    vis_qmap = []
    frames = []
    if args.video:
        vis_model = build_visual_model(args.mjcf)
        vis_data = mujoco.MjData(vis_model)
        renderer = mujoco.Renderer(vis_model, height=480, width=640)
        for jn, pa in C.q_adr.items():
            vj = mujoco.mj_name2id(vis_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if vj >= 0:
                vis_qmap.append((pa, vis_model.jnt_qposadr[vj]))

    def grab_frame():
        if renderer is None:
            return
        vis_data.qpos[:7] = data.qpos[:7]  # floating base
        for pa, va in vis_qmap:
            vis_data.qpos[va] = data.qpos[pa]
        mujoco.mj_forward(vis_model, vis_data)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [vis_data.qpos[0], vis_data.qpos[1], 0.85]
        cam.distance, cam.azimuth, cam.elevation = 3.0, 130, -7
        renderer.update_scene(vis_data, cam)
        frames.append(renderer.render())

    # --- Optional pre-settle. The walking policy is recurrent and its initial
    # LSTM state assumes IsaacLab's exact reset (base at base_z, default pose,
    # upright, zero velocity), so by default we hand off from that clean state
    # directly (settle_steps=0). A short settle is available for non-recurrent /
    # dropped-start policies via --settle-steps. ---
    settle_kp = np.array([SETTLE_GAINS.get(n, (80, 8))[0] for n in act_names])
    settle_kd = np.array([SETTLE_GAINS.get(n, (80, 8))[1] for n in act_names])
    for _ in range(args.settle_steps):
        apply_and_step(default_targets, settle_kp, settle_kd)
    z0, up0, _ = base_metrics()
    print(f"  [init] base_z {z0:.3f} | upright {up0:.3f} (settle_steps={args.settle_steps})")

    hidden, cell = C.hidden0.copy(), C.cell0.copy()
    prev_v_world = np.zeros(3)
    clock_t = 0.0

    def run_phase(label, cmd, n_steps):
        nonlocal clock_t, hidden, cell
        zs, ups = [], []
        for k in range(n_steps):
            (tp, kp, kd), hidden, cell = run_policy_step(clock_t, cmd, hidden, cell)
            apply_and_step(tp, kp, kd)
            clock_t += control_dt
            grab_frame()
            z, up, x = base_metrics()
            zs.append(z); ups.append(up)
            if args.diag and label == "stand" and k < 14:
                roll, pitch, bx, by = base_diag()
                cx, ncL, cxL, ncR, cxR = com_contact()
                print(f"  [{label}] t={k/C.freq:4.2f}s pitch={pitch:+6.1f} "
                      f"CoMx={cx:+.4f} baseX={bx:+.4f} | Lfoot n={ncL} cx={cxL:+.3f} "
                      f"Rfoot n={ncR} cx={cxR:+.3f}")
            elif args.diag and (k % max(1, int(round(C.freq / 6))) == 0 or up < 0.85):
                roll, pitch, bx, by = base_diag()
                print(f"  [{label}] t={k/C.freq:4.2f}s z={z:.3f} up={up:.3f} "
                      f"roll={roll:+6.1f} pitch={pitch:+6.1f} xy=({bx:+.3f},{by:+.3f})")
            if z < 0.4:
                roll, pitch, bx, by = base_diag()
                fdir = ("forward" if pitch > 10 else "backward" if pitch < -10
                        else "right" if roll > 10 else "left" if roll < -10 else "?")
                print(f"  [{label}] FELL at step {len(zs)} t={len(zs)/C.freq:.2f}s (z={z:.2f}) "
                      f"-> {fdir} (roll={roll:+.1f} pitch={pitch:+.1f} xy=({bx:+.3f},{by:+.3f}))")
                return False, zs, ups, x
        print(f"  [{label}] {n_steps} steps | base_z {np.mean(zs):.3f}±{np.std(zs):.3f} "
              f"| upright {np.mean(ups):.3f} (want ~1) | x={x:.2f}")
        return True, zs, ups, x

    print(f"model: {args.model}")
    print(f"freq={C.freq}Hz dt={args.dt} decimation={decimation} | "
          f"motor={len(C.motor_joints)} action={len(C.act_joints)} mapped_act={len(C.act_id)}")

    if args.walk_from_start:
        # walking policies can't hold a zero-velocity stand; command forward vel
        # immediately and run a single longer walk phase.
        x_before = base_metrics()[2]
        ok_stand = True; ups = [1.0]
        ok_walk, _, _, x_after = run_phase("walk", [args.cmd_vx, 0.0, 0.0], int(round(7.0 * C.freq)))
    else:
        ok_stand, _, ups, _ = run_phase("stand", [0.0, 0.0, 0.0], int(round(3.0 * C.freq)))
        x_before = base_metrics()[2]
        ok_walk, _, _, x_after = run_phase("walk", [args.cmd_vx, 0.0, 0.0], int(round(4.0 * C.freq)))

    print("\n=== RESULT ===")
    print(f"stand upright: {'PASS' if ok_stand and np.mean(ups) > 0.9 else 'CHECK'}")
    print(f"walk forward dx = {x_after - x_before:+.2f} m (cmd vx={args.cmd_vx})")
    print(f"survived: stand={ok_stand} walk={ok_walk}")

    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=int(C.freq))
        print(f"wrote {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
