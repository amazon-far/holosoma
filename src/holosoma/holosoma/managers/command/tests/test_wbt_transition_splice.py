"""Tests for splicing default-pose transitions around WBT motion clips (no sim).

The user-visible defect these pin: a transition used to be registered as its own motion, and
``MotionCommand.step()`` resets and resamples the moment ``time_steps`` reaches a motion's end
index. So the robot played a lead-in and was teleported to a random clip at exactly the instant it
should have handed off to the reference motion. Pins:

- the boundary arithmetic folds each segment into the clip it joins, leaving the clip count alone;
- a robot advancing one frame per step walks from a lead-in into its clip without a reset;
- the frames land in the right order in the raw arrays, for every clip, both directions;
- clip-count and frame-count mismatches are rejected rather than silently mis-spliced;
- the setup-time consistency log measures the row the hand-off actually falls on, since that index
  arithmetic runs once at startup and would otherwise never be exercised.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from loguru import logger

from holosoma.managers.command.terms.wbt import (
    _SEGMENT_CONCAT_TARGETS,
    MotionCommand,
    _splice_motion_frames,
    splice_transition_boundaries,
)
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_BODIES = 4
NUM_JOINTS = 3


class _FakeLoader:
    """Minimal stand-in for MotionLoader/MultiMotionLoader's raw array storage.

    Frame ``f`` of clip ``c`` is tagged with the scalar ``c * 100 + f`` in every field, so a test
    can read the splice order straight off the values.
    """

    def __init__(self, clip_lengths: list[int], has_object: bool = False):
        self.has_object = has_object
        self.clip_lengths = clip_lengths
        tags = torch.cat(
            [torch.arange(length, dtype=torch.float32) + 100.0 * clip for clip, length in enumerate(clip_lengths)]
        )
        total = int(tags.shape[0])
        self._joint_pos = tags.reshape(total, 1).expand(total, NUM_JOINTS).clone()
        self._joint_vel = self._joint_pos.clone()
        self._body_pos_w = tags.reshape(total, 1, 1).expand(total, NUM_BODIES, 3).clone()
        self._body_quat_w = tags.reshape(total, 1, 1).expand(total, NUM_BODIES, 4).clone()
        self._body_lin_vel_w = self._body_pos_w.clone()
        self._body_ang_vel_w = self._body_pos_w.clone()
        self._object_pos_w = tags.reshape(total, 1).expand(total, 3).clone()
        self._object_quat_w = tags.reshape(total, 1).expand(total, 4).clone()
        self._object_lin_vel_w = self._object_pos_w.clone()

    def boundaries(self) -> tuple[torch.Tensor, torch.Tensor]:
        ends = torch.tensor(self.clip_lengths, dtype=torch.long).cumsum(dim=0)
        starts = torch.cat([torch.zeros(1, dtype=torch.long), ends[:-1]])
        return starts, ends


def _segment(num_frames: int, tag: float, has_object: bool = False) -> dict[str, torch.Tensor]:
    """A transition segment whose every field carries ``tag``, for order assertions."""
    segment = {
        "joint_pos": torch.full((num_frames, NUM_JOINTS), tag),
        "joint_vel": torch.full((num_frames, NUM_JOINTS), tag),
        "body_pos": torch.full((num_frames, NUM_BODIES, 3), tag),
        "body_quat": torch.full((num_frames, NUM_BODIES, 4), tag),
        "body_lin_vel": torch.full((num_frames, NUM_BODIES, 3), tag),
        "body_ang_vel": torch.full((num_frames, NUM_BODIES, 3), tag),
    }
    if has_object:
        segment["object_pos"] = torch.full((num_frames, 3), tag)
        segment["object_quat"] = torch.full((num_frames, 4), tag)
        segment["object_lin_vel"] = torch.full((num_frames, 3), tag)
    return segment


def _tags(loader: _FakeLoader) -> list[float]:
    return loader._joint_pos[:, 0].tolist()


def test_boundaries_grow_the_clip_that_owns_the_segment():
    starts = torch.tensor([0, 10, 30])
    ends = torch.tensor([10, 30, 60])
    added = torch.tensor([2, 3, 4])

    new_starts, new_ends = splice_transition_boundaries(starts, ends, added)

    # Clip 0 grows by 2; clip 1 shifts by 2 and grows by 3; clip 2 shifts by 5 and grows by 4.
    torch.testing.assert_close(new_starts, torch.tensor([0, 12, 35]))
    torch.testing.assert_close(new_ends, torch.tensor([12, 35, 69]))


def test_boundaries_leave_no_gaps_and_keep_the_clip_count():
    starts = torch.tensor([0, 7, 11, 20])
    ends = torch.tensor([7, 11, 20, 24])
    added = torch.tensor([5, 5, 5, 5])

    new_starts, new_ends = splice_transition_boundaries(starts, ends, added)

    assert new_starts.shape == starts.shape
    assert int(new_starts[0]) == 0
    torch.testing.assert_close(new_starts[1:], new_ends[:-1])
    assert int(new_ends[-1]) == int(ends[-1]) + int(added.sum())


def test_boundaries_of_a_single_clip_absorb_the_whole_segment():
    new_starts, new_ends = splice_transition_boundaries(torch.tensor([0]), torch.tensor([50]), torch.tensor([12]))
    torch.testing.assert_close(new_starts, torch.tensor([0]))
    torch.testing.assert_close(new_ends, torch.tensor([62]))


def test_prepended_transition_flows_into_its_clip_without_a_reset():
    """Replays MotionCommand.step()'s advance rule: +1 per step, reset when the end index is hit.

    Before the fix the lead-in was its own motion, so the counter hit an end index -- and thus a
    resample -- exactly at the hand-off. Now it must reach the clip's first real frame untouched.
    """
    clip_lengths = [20, 30]
    added = 6
    starts, ends = _FakeLoader(clip_lengths).boundaries()
    new_starts, new_ends = splice_transition_boundaries(starts, ends, torch.tensor([added, added]))

    motion_id = 0
    time_step = int(new_starts[motion_id])
    resets = 0
    # The clip's original first frame now sits `added` rows into the motion.
    first_clip_frame = int(new_starts[motion_id]) + added
    while time_step < first_clip_frame:
        time_step += 1
        if time_step >= int(new_ends[motion_id]):
            resets += 1
            break

    assert resets == 0
    assert time_step == first_clip_frame


def test_appended_transition_is_reached_before_the_clip_ends():
    clip_lengths = [20]
    added = 6
    starts, ends = _FakeLoader(clip_lengths).boundaries()
    new_starts, new_ends = splice_transition_boundaries(starts, ends, torch.tensor([added]))

    # The lead-out occupies the last `added` rows, so the settle frames are inside the motion.
    assert int(new_ends[0]) - int(new_starts[0]) == clip_lengths[0] + added
    assert int(new_ends[0]) - added == clip_lengths[0]


def test_prepend_puts_each_segment_immediately_before_its_clip():
    loader = _FakeLoader([3, 2])
    starts, ends = loader.boundaries()

    added = _splice_motion_frames(loader, [_segment(2, -1.0), _segment(2, -2.0)], starts, ends, prepend=True)

    torch.testing.assert_close(added, torch.tensor([2, 2]))
    assert _tags(loader) == [-1.0, -1.0, 0.0, 1.0, 2.0, -2.0, -2.0, 100.0, 101.0]


def test_append_puts_each_segment_immediately_after_its_clip():
    loader = _FakeLoader([3, 2])
    starts, ends = loader.boundaries()

    _splice_motion_frames(loader, [_segment(2, -1.0), _segment(2, -2.0)], starts, ends, prepend=False)

    assert _tags(loader) == [0.0, 1.0, 2.0, -1.0, -1.0, 100.0, 101.0, -2.0, -2.0]


def test_splice_rewrites_every_field_consistently():
    loader = _FakeLoader([3], has_object=True)
    starts, ends = loader.boundaries()

    _splice_motion_frames(loader, [_segment(2, -1.0, has_object=True)], starts, ends, prepend=True)

    for _, attr_name in _SEGMENT_CONCAT_TARGETS:
        assert getattr(loader, attr_name).shape[0] == 5, attr_name
    for attr_name in ("_object_pos_w", "_object_quat_w", "_object_lin_vel_w"):
        assert getattr(loader, attr_name).shape[0] == 5, attr_name


def test_splice_ignores_object_fields_when_the_clip_has_no_object():
    loader = _FakeLoader([3], has_object=False)
    starts, ends = loader.boundaries()

    # A segment without object fields must be accepted, and object arrays left untouched.
    _splice_motion_frames(loader, [_segment(2, -1.0)], starts, ends, prepend=True)

    assert loader._joint_pos.shape[0] == 5
    assert loader._object_pos_w.shape[0] == 3


def test_splice_rejects_a_segment_whose_fields_disagree_on_frame_count():
    loader = _FakeLoader([3])
    starts, ends = loader.boundaries()
    segment = _segment(2, -1.0)
    segment["body_pos"] = torch.zeros(5, NUM_BODIES, 3)

    with pytest.raises(ValueError, match="disagree on frame count"):
        _splice_motion_frames(loader, [segment], starts, ends, prepend=True)


def test_splice_leaves_untouched_clips_alone_when_segments_differ_in_length():
    loader = _FakeLoader([2, 2, 2])
    starts, ends = loader.boundaries()

    added = _splice_motion_frames(
        loader, [_segment(1, -1.0), _segment(3, -2.0), _segment(2, -3.0)], starts, ends, prepend=True
    )

    torch.testing.assert_close(added, torch.tensor([1, 3, 2]))
    assert _tags(loader) == [
        -1.0, 0.0, 1.0,
        -2.0, -2.0, -2.0, 100.0, 101.0,
        -3.0, -3.0, 200.0, 201.0,
    ]  # fmt: skip


def _consistency_command(num_clips: int, clip_len: int, spike_row: int) -> Any:
    """A spliced motion whose only frame-to-frame step is at ``spike_row`` of clip 0.

    Lets a test read back which row ``_log_transition_consistency`` treats as the seam.
    """
    total = num_clips * clip_len
    num_bodies = 3
    joint_pos = torch.zeros(total, 2)
    joint_pos[spike_row:] = 1.0
    ends = torch.arange(1, num_clips + 1, dtype=torch.long) * clip_len
    body_rows = torch.arange(num_bodies, dtype=torch.long)
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(total, num_bodies, 4).contiguous()
    command = types.SimpleNamespace(
        motion=types.SimpleNamespace(
            motion_start_idx=ends - clip_len,
            motion_end_idx=ends,
            _joint_pos=joint_pos,
            _joint_vel=torch.zeros(total, 2),
            _body_pos_w=torch.zeros(total, num_bodies, 3),
            _body_quat_w=identity_quat,
            _body_lin_vel_w=torch.zeros(total, num_bodies, 3),
            _body_ang_vel_w=torch.zeros(total, num_bodies, 3),
        ),
        _body_scatter_src=body_rows,
        _body_scatter_dst=body_rows,
        _com_offsets_cache=torch.zeros(num_bodies, 3),
    )
    command._body_com_offsets_b = types.MethodType(MotionCommand._body_com_offsets_b, command)
    return command


def _logged(command, num_steps: int, prepend: bool) -> str:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        MotionCommand._log_transition_consistency(command, num_steps, 0.02, prepend=prepend)
    finally:
        logger.remove(sink_id)
    return messages[-1] if messages else ""


@pytest.mark.parametrize("num_clips", [1, 3])
def test_consistency_log_finds_the_prepend_seam(num_clips):
    """The lead-in occupies rows 0..num_steps-1, so the hand-off step lands on row num_steps."""
    num_steps = 4
    command = _consistency_command(num_clips, clip_len=10, spike_row=num_steps)

    message = _logged(command, num_steps, prepend=True)

    assert "Seam step 1.0000 rad" in message


@pytest.mark.parametrize("num_clips", [1, 3])
def test_consistency_log_finds_the_append_seam(num_clips):
    """The lead-out occupies the clip's last num_steps rows, so the step lands where it starts."""
    num_steps = 4
    clip_len = 10
    command = _consistency_command(num_clips, clip_len=clip_len, spike_row=clip_len - num_steps)

    message = _logged(command, num_steps, prepend=False)

    assert "Seam step 1.0000 rad" in message


def test_consistency_log_skips_a_segment_too_short_to_difference():
    # num_steps=2 leaves no interior row once the one-sided ends are trimmed.
    command = _consistency_command(1, clip_len=6, spike_row=1)
    assert _logged(command, 2, prepend=True) == ""
