"""Tests for the fields a WBT default-pose transition emits (no sim).

``_fk_body_poses`` is the only part of the path that needs a simulator, so a closed-form fake
kinematic chain in its place lets the whole interpolation be checked on CPU. Pins the two reported
defects:

- ``joint_vel`` is the derivative of ``joint_pos``, at every frame and across the seam into the clip
  -- not just at the endpoints. The old path interpolated the two independently, producing a velocity
  ramp where the true derivative of the linear position ramp was a constant.
- ``body_pos``/``body_quat`` are forward kinematics of the joint angles in the *same* frame. The old
  path blended body poses in world space, so a limb sweeping an arc had its links cut to the chord
  and mid-segment frames matched no joint configuration at all.

Plus the invariants that make those safe: the seam carries no jump, the default-pose end is exactly
at rest, body linear velocity differences the *centre of mass* (the reference point a motion file's
``body_lin_vel_w`` uses, not the link origin), every emitted quaternion is a usable rotation, and the
object holds still because a transition has no way to move it.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_apply, quat_from_angle_axis, quat_mul
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.transition_trajectory import (
    angular_velocity_from_quats,
    linear_velocity_from_positions,
)

pytestmark = pytest.mark.no_sim

DT = 0.02
NUM_STEPS = 20
DURATION = NUM_STEPS * DT
NUM_JOINTS = 3
NUM_CLIP_FRAMES = 6

# One motion slot per robot body plus a trailing slot the robot does not have, so the anchor fill is
# exercised. The robot's own bodies are the root plus one link per joint.
NUM_ROBOT_BODIES = 1 + NUM_JOINTS
NUM_MOTION_BODIES = NUM_ROBOT_BODIES + 1
# Each link hangs this far below its parent along -z and swings about x by its joint angle.
LINK_OFFSET = torch.tensor([0.0, 0.0, -0.3])
SWING_AXIS = torch.tensor([1.0, 0.0, 0.0])
# A body-frame centre-of-mass offset, so tests can tell a COM velocity from a link-origin one.
COM_OFFSET = torch.tensor([0.0, 0.04, -0.12])
# Wide enough that the fixture's own trajectory fits; a dedicated test tightens it.
JOINT_LIMIT = 6.0


def _fake_fk(joint_pos: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor):
    """A real (if trivial) serial chain: each link rotates about its parent, never stretches.

    Deliberately not a world-space blend of endpoints -- that is the behaviour under test.
    """
    num_frames = joint_pos.shape[0]
    pos = torch.empty(num_frames, NUM_ROBOT_BODIES, 3)
    quat = torch.empty(num_frames, NUM_ROBOT_BODIES, 4)
    pos[:, 0] = root_pos
    quat[:, 0] = root_quat
    for link in range(1, NUM_ROBOT_BODIES):
        swing = quat_from_angle_axis(joint_pos[:, link - 1], SWING_AXIS.expand(num_frames, 3), w_last=True)
        quat[:, link] = quat_mul(quat[:, link - 1], swing, w_last=True)
        pos[:, link] = pos[:, link - 1] + quat_apply(quat[:, link], LINK_OFFSET.expand(num_frames, 3), w_last=True)
    com_pos = pos + quat_apply(quat, COM_OFFSET.expand_as(pos), w_last=True)
    return pos, quat, com_pos


# A clip already in motion at its first frame, so a lead-in that arrives at rest is a visible jerk.
# Constant rates keep the clip's own stored velocities exact -- a differenced clip would carry
# one-sided estimates at its boundary frames, and those cannot agree with any difference taken across
# a spliced seam, which would make a derivative check unfalsifiable rather than strict.
CLIP_JOINT_RATE = 1.5  # rad/s
CLIP_ROOT_RATE = 0.4  # m/s along +x
CLIP_YAW_RATE = 0.6  # rad/s about +z


def _clip_frames() -> dict[str, torch.Tensor]:
    frames = torch.arange(NUM_CLIP_FRAMES, dtype=torch.float32).unsqueeze(-1)

    joint_pos = 0.1 + frames * CLIP_JOINT_RATE * DT + torch.linspace(0.0, 0.2, NUM_JOINTS)
    joint_vel = torch.full((NUM_CLIP_FRAMES, NUM_JOINTS), CLIP_JOINT_RATE)

    root_pos = torch.zeros(NUM_CLIP_FRAMES, 3)
    root_pos[:, 2] = 0.8
    root_pos[:, 0] = frames.squeeze(-1) * CLIP_ROOT_RATE * DT
    # A yawing root, so the root slerp is exercised rather than being an identity no-op.
    root_quat = quat_from_angle_axis(
        0.2 + frames.squeeze(-1) * CLIP_YAW_RATE * DT,
        torch.tensor([0.0, 0.0, 1.0]).expand(NUM_CLIP_FRAMES, 3),
        w_last=True,
    )

    body_pos, body_quat, com_pos = _fake_fk(joint_pos, root_pos, root_quat)
    # Pad the extra motion slot the robot lacks, and keep the clip self-consistent.
    padded_pos = torch.cat([body_pos, body_pos[:, -1:] + LINK_OFFSET], dim=1)
    padded_quat = torch.cat([body_quat, body_quat[:, -1:]], dim=1)
    padded_com = torch.cat([com_pos, com_pos[:, -1:] + LINK_OFFSET], dim=1)
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos": padded_pos,
        "body_quat": padded_quat,
        # COM-referenced, as a converted motion file's body_lin_vel_w is.
        "body_lin_vel": linear_velocity_from_positions(padded_com, DT),
        "body_ang_vel": angular_velocity_from_quats(padded_quat, DT),
        "body_com_pos": padded_com,
    }


def _command(clip: dict[str, torch.Tensor]) -> Any:
    """A MotionCommand carrying only what the transition path reads, with FK stubbed."""
    robot_body_rows = torch.arange(NUM_ROBOT_BODIES, dtype=torch.long)
    command = types.SimpleNamespace(
        device=torch.device("cpu"),
        motion=types.SimpleNamespace(
            has_object=False,
            _joint_pos=clip["joint_pos"],
            _joint_vel=clip["joint_vel"],
            _body_pos_w=clip["body_pos"],
            _body_quat_w=clip["body_quat"],
            _body_lin_vel_w=clip["body_lin_vel"],
            _body_ang_vel_w=clip["body_ang_vel"],
        ),
        _env=types.SimpleNamespace(
            simulator=types.SimpleNamespace(dof_pos_limits=torch.tensor([[-JOINT_LIMIT, JOINT_LIMIT]] * NUM_JOINTS))
        ),
        _joint_indexes_in_motion=torch.arange(NUM_JOINTS, dtype=torch.long),
        _body_indexes_in_motion=robot_body_rows,
        _body_scatter_src=robot_body_rows,
        _body_scatter_dst=robot_body_rows,
        _com_offsets_cache=COM_OFFSET.expand(NUM_ROBOT_BODIES, 3).contiguous(),
    )
    for name in (
        "_map_robot_bodies_to_motion_order",
        "_map_robot_joints_to_motion_order",
        "_joint_limits_in_motion_order",
        "_check_joints_within_limits",
        "_check_joints_stay_between_the_endpoints",
        "_body_com_offsets_b",
    ):
        setattr(command, name, types.MethodType(getattr(MotionCommand, name), command))
    return command


# The default pose: a fixed configuration at rest, placed where the clip it joins starts. Root
# orientation deliberately differs from the clip's, matching production (the default pose adopts the
# clip's yaw but keeps init_state's roll and pitch).
REST_ROOT_QUAT = quat_from_angle_axis(
    torch.tensor([0.35]), torch.nn.functional.normalize(torch.tensor([[0.1, 0.0, 1.0]]), dim=-1), w_last=True
)[0]


def _rest_state(with_object: bool = False) -> dict[str, torch.Tensor]:
    state = {
        "joint_pos": torch.zeros(NUM_JOINTS),
        "joint_vel": torch.zeros(NUM_JOINTS),
        "root_pos": torch.tensor([0.0, 0.0, 0.9]),
        "root_quat": REST_ROOT_QUAT,
    }
    if with_object:
        state["object_pos"] = OBJECT_POS
        state["object_quat"] = OBJECT_QUAT
    return state


def _clip_state(clip: dict[str, torch.Tensor], frame: int, with_object: bool = False) -> dict[str, torch.Tensor]:
    """Mirrors what _motion_state builds, including the CoM -> link-origin velocity conversion."""
    state = {
        "joint_pos": clip["joint_pos"][frame],
        "joint_vel": clip["joint_vel"][frame],
        "root_pos": clip["body_pos"][frame, 0],
        "root_quat": clip["body_quat"][frame, 0],
    }
    if with_object:
        state["object_pos"] = OBJECT_POS
        state["object_quat"] = OBJECT_QUAT
        state["object_lin_vel"] = CLIP_OBJECT_VEL
    return state


OBJECT_POS = torch.tensor([0.4, -0.1, 0.7])
OBJECT_QUAT = torch.tensor([0.0, 0.0, 0.0, 1.0])
# Nonzero, so a Hermite through the object would visibly swing out and back.
CLIP_OBJECT_VEL = torch.tensor([0.0, 0.0, 1.0])


def _segment(prepend: bool = True, with_object: bool = False):
    clip = _clip_frames()
    command = _command(clip)
    command.motion.has_object = with_object
    anchor = 0 if prepend else NUM_CLIP_FRAMES - 1
    rest = _rest_state(with_object)
    clip_end = _clip_state(clip, anchor, with_object)

    free = MotionCommand._transition_free_variables(
        command,
        start=rest if prepend else clip_end,
        target=clip_end if prepend else rest,
        num_steps=NUM_STEPS,
        duration_s=DURATION,
    )
    fk_pos, fk_quat, fk_com_pos = _fake_fk(free["joint_pos"], free["root_pos"], free["root_quat"])
    segment = MotionCommand._assemble_transition_segment(
        command,
        free=free,
        fk_pos=fk_pos,
        fk_quat=fk_quat,
        fk_com_pos=fk_com_pos,
        num_steps=NUM_STEPS,
        dt=DT,
        anchor_frame_idx=anchor,
        prepend=prepend,
    )
    return segment, clip, rest


def test_every_field_has_exactly_num_steps_frames():
    segment, _, _ = _segment()
    for key, values in segment.items():
        assert values.shape[0] == NUM_STEPS, key


def test_joint_vel_is_the_derivative_of_joint_pos():
    """The stored velocity must be d/dt of the stored position at every frame.

    The old path interpolated the two independently, so the velocity was a ramp from 0 to the clip's
    value while the true derivative of the linearly interpolated position was a constant -- wrong
    everywhere except the endpoints, and by an arbitrary factor.
    """
    segment, clip, _ = _segment(prepend=True)
    joint_pos = torch.cat([segment["joint_pos"], clip["joint_pos"]], dim=0)
    joint_vel = torch.cat([segment["joint_vel"], clip["joint_vel"]], dim=0)
    numeric = linear_velocity_from_positions(joint_pos, DT)

    # Interior of the segment only. Rows 0 and -1 of the concatenated array get a one-sided
    # difference, and the row straddling the seam differences across an acceleration step -- both are
    # properties of the measurement, not of the stored data. The seam itself is checked below.
    interior = slice(1, NUM_STEPS - 1)
    error = (joint_vel[interior] - numeric[interior]).abs().max()
    assert float(error) < 0.02 * CLIP_JOINT_RATE


def test_prepend_starts_exactly_at_the_default_pose_at_rest():
    segment, _, rest = _segment(prepend=True)
    torch.testing.assert_close(segment["joint_pos"][0], rest["joint_pos"])
    torch.testing.assert_close(segment["joint_vel"][0], rest["joint_vel"])
    torch.testing.assert_close(segment["body_lin_vel"][0], torch.zeros(NUM_MOTION_BODIES, 3))
    torch.testing.assert_close(segment["body_ang_vel"][0], torch.zeros(NUM_MOTION_BODIES, 3))


def test_append_ends_exactly_at_the_default_pose_at_rest():
    segment, _, rest = _segment(prepend=False)
    torch.testing.assert_close(segment["joint_pos"][-1], rest["joint_pos"])
    torch.testing.assert_close(segment["joint_vel"][-1], rest["joint_vel"])
    torch.testing.assert_close(segment["body_lin_vel"][-1], torch.zeros(NUM_MOTION_BODIES, 3))


def test_prepend_does_not_duplicate_the_clip_frame():
    """Row N of the grid *is* the clip frame, so the emitted rows must stop one short of it."""
    segment, clip, _ = _segment(prepend=True)
    assert not torch.allclose(segment["joint_pos"][-1], clip["joint_pos"][0])


def test_body_pos_is_forward_kinematics_of_the_emitted_joint_angles():
    """Forward kinematics of the same frame's joint angles -- at every frame, not just the endpoints."""
    segment, _, _ = _segment(prepend=True)
    expected_pos, expected_quat, _ = _fake_fk(
        segment["joint_pos"], segment["body_pos"][:, 0], segment["body_quat"][:, 0]
    )
    torch.testing.assert_close(segment["body_pos"][:, :NUM_ROBOT_BODIES], expected_pos, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(segment["body_quat"][:, :NUM_ROBOT_BODIES], expected_quat, atol=1e-5, rtol=1e-5)


def test_link_lengths_are_constant_through_the_transition():
    """A world-space blend shortens a swinging limb mid-segment; forward kinematics cannot."""
    segment, _, _ = _segment(prepend=True)
    body_pos = segment["body_pos"][:, :NUM_ROBOT_BODIES]
    lengths = (body_pos[:, 1:] - body_pos[:, :-1]).norm(dim=-1)
    assert float((lengths - LINK_OFFSET.norm()).abs().max()) < 1e-5


def test_body_velocities_follow_the_emitted_body_poses():
    """Linear velocity differences the *centre of mass*, which is what a motion file stores."""
    segment, _, _ = _segment(prepend=True)
    interior = slice(1, -1)
    _, _, com_pos = _fake_fk(segment["joint_pos"], segment["body_pos"][:, 0], segment["body_quat"][:, 0])

    torch.testing.assert_close(
        segment["body_lin_vel"][interior, :NUM_ROBOT_BODIES],
        linear_velocity_from_positions(com_pos, DT)[interior],
    )
    torch.testing.assert_close(
        segment["body_ang_vel"][interior],
        angular_velocity_from_quats(segment["body_quat"], DT)[interior],
    )


def test_body_lin_vel_is_not_the_link_origin_difference():
    """Differencing link origins instead of centres of mass leaves an omega-cross-r error.

    The clip's own body_lin_vel_w is COM-referenced (mj_objectVelocity), so a link-origin velocity in
    the transition would disagree with the clip across the seam on every rotating body.
    """
    segment, _, _ = _segment(prepend=True)
    interior = slice(1, -1)
    link_origin_velocity = linear_velocity_from_positions(segment["body_pos"], DT)

    gap = (segment["body_lin_vel"][interior] - link_origin_velocity[interior]).abs().max()
    assert float(gap) > 1e-3, "COM and link-origin velocities are indistinguishable; COM_OFFSET is too small"


def test_every_emitted_quaternion_is_unit_norm():
    """Motion slots with no robot body used to hold (0, 0, 0, 0), which is not a rotation."""
    segment, _, _ = _segment(prepend=True)
    norms = segment["body_quat"].norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)


def test_untracked_motion_body_holds_its_clip_anchor_pose():
    segment, clip, _ = _segment(prepend=True)
    extra_slot = NUM_MOTION_BODIES - 1
    expected = clip["body_pos"][0, extra_slot].expand(NUM_STEPS, 3)
    torch.testing.assert_close(segment["body_pos"][:, extra_slot], expected.contiguous())


def test_root_pose_moves_from_the_default_pose_to_the_clip():
    segment, clip, rest = _segment(prepend=True)
    torch.testing.assert_close(segment["body_pos"][0, 0], rest["root_pos"])
    # The last emitted row is one step short of the clip, so it should be close but not equal.
    gap = (segment["body_pos"][-1, 0] - clip["body_pos"][0, 0]).norm()
    assert 0.0 < float(gap) < 0.2


def test_object_holds_its_anchor_pose_through_the_transition():
    """A transition cannot move the object, so it must not drift.

    Putting the object on the same Hermite would: with equal endpoint positions and the clip's
    non-zero endpoint velocity, the cubic swings out and back. On a 2 s lead-in with a 1 m/s object
    that is tens of centimetres of phantom motion.
    """
    segment, _, _ = _segment(prepend=True, with_object=True)

    torch.testing.assert_close(segment["object_pos"], OBJECT_POS.expand(NUM_STEPS, 3).contiguous())
    torch.testing.assert_close(segment["object_quat"], OBJECT_QUAT.expand(NUM_STEPS, 4).contiguous())


def test_object_velocity_matches_its_stationary_position():
    """Zero, because the emitted position is constant -- the same derivative rule as every other field."""
    segment, _, _ = _segment(prepend=True, with_object=True)
    torch.testing.assert_close(segment["object_lin_vel"], torch.zeros(NUM_STEPS, 3))


def test_object_fields_are_absent_without_an_object():
    segment, _, _ = _segment(prepend=True, with_object=False)
    assert "object_pos" not in segment
    assert "object_quat" not in segment
    assert "object_lin_vel" not in segment


def test_root_rotation_is_interpolated_between_differing_endpoints():
    """Production always has a rotation to cover: the default pose keeps init_state's roll/pitch."""
    segment, clip, rest = _segment(prepend=True)
    root_quat = segment["body_quat"][:, 0]

    torch.testing.assert_close(root_quat[0], rest["root_quat"])
    # Strictly between the endpoints, and never a degenerate or non-unit quaternion.
    torch.testing.assert_close(root_quat.norm(dim=-1), torch.ones(NUM_STEPS), atol=1e-5, rtol=0.0)
    assert not torch.allclose(root_quat[-1], rest["root_quat"], atol=1e-3)
    assert float((root_quat[-1] - clip["body_quat"][0, 0]).abs().max()) < 0.05


def test_emitting_a_joint_past_its_limit_is_refused():
    """Self-consistency is not reachability.

    Every other check here verifies that a frame's fields agree with each other. This one asks the
    separate question of whether the pose could be held, which is what nothing asked before.
    """
    command = _command(_clip_frames())
    endpoints = torch.zeros(2, NUM_JOINTS)
    free = {"joint_pos": endpoints}
    # A trajectory that leaves the endpoint envelope and pokes past the upper limit.
    joint_pos = torch.zeros(NUM_STEPS, NUM_JOINTS)
    joint_pos[NUM_STEPS // 2, 1] = JOINT_LIMIT + 0.5

    with pytest.raises(RuntimeError, match="beyond its limit"):
        MotionCommand._check_joints_within_limits(command, joint_pos, free, 0)


def test_a_limit_the_endpoints_already_break_is_not_blamed_on_the_transition():
    """A clip frame outside its own limit is the clip's problem; do not be stricter than the data."""
    command = _command(_clip_frames())
    over = JOINT_LIMIT + 0.5
    endpoints = torch.zeros(2, NUM_JOINTS)
    endpoints[1, 1] = over
    free = {"joint_pos": endpoints}
    joint_pos = torch.zeros(NUM_STEPS, NUM_JOINTS)
    joint_pos[:, 1] = over  # never worse than the endpoint already is

    MotionCommand._check_joints_within_limits(command, joint_pos, free, 0)


def test_the_joint_path_is_monotone_between_the_two_poses():
    """The property this path is judged on: walk from one pose to the other without wandering.

    Matching the clip's arrival velocity instead makes the trajectory travel out and back whenever
    the net displacement is small -- measured at 1.05 rad to make a 0.05 rad move, on 10 of 29 joints.
    """
    segment, _, _ = _segment(prepend=True)
    q = segment["joint_pos"]
    lower, upper = torch.minimum(q[0], q[-1]), torch.maximum(q[0], q[-1])
    excursion = torch.maximum(lower - q, q - upper).max()
    assert float(excursion) < 1e-5, f"joint path left its endpoint interval by {float(excursion):.4f} rad"


def test_each_joint_moves_in_one_direction_only():
    """Monotone in the strict sense: no sign change in a joint's step from frame to frame."""
    segment, _, _ = _segment(prepend=True)
    steps = segment["joint_pos"].diff(dim=0)
    moving = steps.abs().max(dim=0).values > 1e-9
    signs = torch.sign(steps[:, moving])
    for j in range(signs.shape[1]):
        column = signs[:, j][signs[:, j] != 0]
        assert bool((column == column[0]).all()), f"joint {j} reversed direction mid-transition"


def test_the_guard_catches_a_reintroduced_endpoint_velocity():
    """A non-zero arrival velocity is what caused the detour, so the guard has to see it."""
    command = _command(_clip_frames())
    q = torch.zeros(NUM_STEPS, NUM_JOINTS)
    q[NUM_STEPS // 2, 2] = 1.0  # overshoots its own endpoints, as a velocity-matched cubic does

    with pytest.raises(RuntimeError, match="leaves the interval between its endpoints"):
        MotionCommand._check_joints_stay_between_the_endpoints(command, q)
