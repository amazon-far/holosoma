from holosoma_inference.config.config_values import inference, observation, robot


def test_h1_motor_mapping_matches_sdk_slots():
    """H1: 19 joints, 20 physical motor slots (ID 9 unused, matching Beyondmimic)."""
    cfg = robot.h1_19dof

    assert cfg.num_joints == 19
    assert cfg.num_motors == 20
    assert len(cfg.dof_names) == cfg.num_joints
    assert len(cfg.joint2motor) == cfg.num_joints
    assert len(cfg.motor2joint) == cfg.num_motors
    assert len(cfg.default_dof_angles) == cfg.num_joints
    assert len(cfg.default_motor_angles) == cfg.num_motors
    assert len(cfg.motor_kp) == cfg.num_motors
    assert len(cfg.motor_kd) == cfg.num_motors

    # Physical motor IDs 0..19, ID 9 excluded from joint2motor
    motor_ids = set(cfg.joint2motor)
    assert 9 not in motor_ids, "Motor ID 9 should be unused"
    assert min(motor_ids) == 0
    assert max(motor_ids) == 19

    # motor2joint: physical ID 9 → -1 (unused)
    assert cfg.motor2joint[9] == -1
    assert cfg.motor_kp[9] == 0.0
    assert cfg.motor_kd[9] == 0.0

    for joint_id, motor_id in enumerate(cfg.joint2motor):
        assert cfg.motor2joint[motor_id] == joint_id


def test_h1_named_physical_motor_indices():
    """Verify each joint maps to the correct physical motor ID (Beyondmimic dof_idx)."""
    cfg = robot.h1_19dof
    by_name = dict(zip(cfg.dof_names, cfg.joint2motor, strict=True))

    # Left leg: L_hip_yaw(7), L_hip_roll(3), L_hip_pitch(4), L_knee(5), L_ankle(10)
    assert by_name["left_hip_yaw_joint"] == 7
    assert by_name["left_hip_roll_joint"] == 3
    assert by_name["left_hip_pitch_joint"] == 4
    assert by_name["left_knee_joint"] == 5
    assert by_name["left_ankle_joint"] == 10

    # Right leg: R_hip_yaw(8), R_hip_roll(0), R_hip_pitch(1), R_knee(2), R_ankle(11)
    assert by_name["right_hip_yaw_joint"] == 8
    assert by_name["right_hip_roll_joint"] == 0
    assert by_name["right_hip_pitch_joint"] == 1
    assert by_name["right_knee_joint"] == 2
    assert by_name["right_ankle_joint"] == 11

    # Waist: waist_yaw(6)
    assert by_name["torso_joint"] == 6

    # Left arm: L_shoulder_pitch(16), L_shoulder_roll(17), L_shoulder_yaw(18), L_elbow(19)
    assert by_name["left_shoulder_pitch_joint"] == 16
    assert by_name["left_shoulder_roll_joint"] == 17
    assert by_name["left_shoulder_yaw_joint"] == 18
    assert by_name["left_elbow_joint"] == 19

    # Right arm: R_shoulder_pitch(12), R_shoulder_roll(13), R_shoulder_yaw(14), R_elbow(15)
    assert by_name["right_shoulder_pitch_joint"] == 12
    assert by_name["right_shoulder_roll_joint"] == 13
    assert by_name["right_shoulder_yaw_joint"] == 14
    assert by_name["right_elbow_joint"] == 15


def test_h1_wbt_observation_dimensions():
    cfg = observation.wbt_h1_19dof

    assert cfg.obs_dims["motion_command"] == 38
    assert cfg.obs_dims["dof_pos"] == 19
    assert cfg.obs_dims["dof_vel"] == 19
    assert cfg.obs_dims["actions"] == 19


def test_h1_wbt_inference_default_uses_dance_primary_walk_secondary():
    cfg = inference.h1_19dof_wbt

    assert cfg.task.model_path == "/home/pjm/Desktop/holosoma/logs/dance_h1.onnx"
    assert cfg.task.velocity_input == "interface"
    assert cfg.task.state_input == "interface"
    assert cfg.secondary is not None
    assert cfg.secondary.task.model_path == "/home/pjm/Desktop/holosoma/logs/walk_h1.onnx"
    assert cfg.secondary.observation is observation.loco_h1_19dof
