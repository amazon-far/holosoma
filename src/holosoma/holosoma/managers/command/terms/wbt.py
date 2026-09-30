from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.file_cache import cached_open
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType
from holosoma.utils.transition_trajectory import (
    angular_velocity_from_quats,
    hermite_rotation_series,
    hermite_segment,
    linear_velocity_from_positions,
)

#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################

# Segment key -> raw loader attribute. The raw arrays are in *motion* body/joint order; the
# same-named properties re-index them into robot order.
_SEGMENT_CONCAT_TARGETS: tuple[tuple[str, str], ...] = (
    ("joint_pos", "_joint_pos"),
    ("joint_vel", "_joint_vel"),
    ("body_pos", "_body_pos_w"),
    ("body_quat", "_body_quat_w"),
    ("body_lin_vel", "_body_lin_vel_w"),
    ("body_ang_vel", "_body_ang_vel_w"),
)
_OBJECT_CONCAT_TARGETS: tuple[tuple[str, str], ...] = (
    ("object_pos", "_object_pos_w"),
    ("object_quat", "_object_quat_w"),
    ("object_lin_vel", "_object_lin_vel_w"),
)


def _concat_targets(has_object: bool) -> tuple[tuple[str, str], ...]:
    return _SEGMENT_CONCAT_TARGETS + _OBJECT_CONCAT_TARGETS if has_object else _SEGMENT_CONCAT_TARGETS


def _segment_frame_count(segment: dict[str, torch.Tensor], targets: tuple[tuple[str, str], ...]) -> int:
    """Frame count of a transition segment, raising if any field disagrees."""
    counts = {segment[seg_key].shape[0] for seg_key, _ in targets}
    if len(counts) != 1:
        per_key = {seg_key: segment[seg_key].shape[0] for seg_key, _ in targets}
        raise ValueError(f"Transition segment fields disagree on frame count: {per_key}")
    return counts.pop()


def splice_transition_boundaries(
    start_idx: torch.Tensor, end_idx: torch.Tensor, added_frames: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grow each clip's interval to swallow the frames inserted next to it.

    The transition must belong to the clip it leads into (or out of) rather than becoming a clip
    of its own: ``step()`` resets and resamples the moment ``time_steps`` reaches a clip's end
    index, so a standalone transition gets the robot teleported at exactly the instant it should
    have handed off to the reference motion.

    The arithmetic is the same whether the segment goes before or after each clip's frames --
    only the interleaving differs -- because either way clip ``i`` is preceded by every segment
    inserted for clips ``j < i`` and grows by its own.
    """
    inclusive = added_frames.cumsum(dim=0)
    return start_idx + inclusive - added_frames, end_idx + inclusive


def _splice_motion_frames(
    loader: MotionLoader | MultiMotionLoader,
    segments_per_motion: list[dict[str, torch.Tensor]],
    start_idx: torch.Tensor,
    end_idx: torch.Tensor,
    prepend: bool,
) -> torch.Tensor:
    """Interleave one transition segment per clip into the loader's raw arrays.

    Returns the per-clip added frame counts, for :func:`splice_transition_boundaries`.
    """
    targets = _concat_targets(loader.has_object)
    added = [_segment_frame_count(segment, targets) for segment in segments_per_motion]

    for seg_key, attr_name in targets:
        existing = getattr(loader, attr_name)
        pieces: list[torch.Tensor] = []
        for segment, clip_start, clip_end in zip(segments_per_motion, start_idx.tolist(), end_idx.tolist()):
            clip = existing[clip_start:clip_end]
            pieces.extend((segment[seg_key], clip) if prepend else (clip, segment[seg_key]))
        setattr(loader, attr_name, torch.cat(pieces, dim=0))

    return torch.tensor(added, dtype=torch.long, device=start_idx.device)


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    # Expected holosoma NPZ keys
    _REQUIRED_KEYS = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
    }

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            # Sanity check: warn if not in expected holosoma format
            keys = set(data.files)
            missing = self._REQUIRED_KEYS - keys
            if missing:
                logger.warning(
                    f"Motion NPZ '{motion_file}' is missing expected holosoma keys: {missing}. "
                    f"All motion data should be in holosoma format (with body_names, joint_names, "
                    f"and root DOFs in joint_pos). Convert from TML/BeyondMimic first."
                )
                raise ValueError(
                    f"Unsupported motion format in '{motion_file}': missing keys {missing}. "
                    f"Please convert to holosoma format."
                )

            self.fps = data["fps"]

            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            joint_pos_raw = data["joint_pos"]
            joint_vel_raw = data["joint_vel"]
            body_pos_w_raw = data["body_pos_w"]
            body_quat_w_raw = data["body_quat_w"]
            body_lin_vel_w_raw = data["body_lin_vel_w"]
            body_ang_vel_w_raw = data["body_ang_vel_w"]

            # Holosoma format: joint_pos includes root DOFs [xyz, wxyz] as first 7 values
            # joint_vel includes root velocity [vel_xyz, vel_wxyz] as first 6 values
            num_joint_cols = joint_pos_raw.shape[1]
            num_vel_cols = joint_vel_raw.shape[1]
            num_bodies = body_pos_w_raw.shape[1]

            if num_joint_cols != len(joint_names) + 7:
                logger.warning(
                    f"Unexpected joint_pos columns: got {num_joint_cols}, expected {len(joint_names) + 7} "
                    f"(= {len(joint_names)} joints + 7 root DOFs). File: {motion_file}"
                )
            if num_vel_cols != len(joint_names) + 6:
                logger.warning(
                    f"Unexpected joint_vel columns: got {num_vel_cols}, expected {len(joint_names) + 6} "
                    f"(= {len(joint_names)} joints + 6 root DOFs). File: {motion_file}"
                )
            if num_bodies != len(body_names):
                logger.warning(
                    f"Body count mismatch: body_pos_w has {num_bodies} bodies but body_names has "
                    f"{len(body_names)}. File: {motion_file}"
                )

            # Strip root DOFs
            self._joint_pos = torch.tensor(joint_pos_raw[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel_raw[:, 6:], dtype=torch.float32, device=device)

            assert len(joint_names) == self._joint_pos.shape[1], (
                f"Joint names ({len(joint_names)}) != joint_pos columns ({self._joint_pos.shape[1]}) in {motion_file}"
            )
            assert len(body_names) == body_pos_w_raw.shape[1], (
                f"Body names ({len(body_names)}) != body_pos_w bodies ({body_pos_w_raw.shape[1]}) in {motion_file}"
            )

            self._body_pos_w = torch.tensor(body_pos_w_raw, dtype=torch.float32, device=device)

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(body_quat_w_raw, dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w_raw, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w_raw, dtype=torch.float32, device=device)

            # add object pos and quat
            self.has_object: bool = "object_pos_w" in data
            if self.has_object:
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                # NOTE: wxyz after loading from npz
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def num_motions(self) -> int:
        return 1

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return torch.tensor([self.time_step_total], dtype=torch.long, device=self._joint_pos.device)

    def splice_transition_segments(
        self, segments_per_motion: list[dict[str, torch.Tensor]], prepend: bool
    ) -> MotionLoader:
        """Splice one transition segment around this loader's single clip, mutating in place."""
        if len(segments_per_motion) != 1:
            raise ValueError(f"MotionLoader holds one clip, got {len(segments_per_motion)} transition segments")

        _splice_motion_frames(self, segments_per_motion, self.motion_start_idx, self.motion_end_idx, prepend)
        # motion_start_idx/motion_end_idx are derived from time_step_total, so they follow along.
        self.time_step_total = self._joint_pos.shape[0]
        return self


class MultiMotionLoader:
    """Loads multiple NPZ motion files from a directory and concatenates them at runtime.

    Tracks per-motion boundaries so environments can sample within individual clips.
    Compatible with the same interface as MotionLoader.
    """

    def __init__(
        self,
        motion_dir: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ):
        # Support comma-separated directories for combining multiple datasets
        dirs = [d.strip() for d in motion_dir.split(",")]
        motion_files = []
        for d in dirs:
            expanded = os.path.expanduser(d)
            files = sorted(str(p) for p in Path(expanded).glob("*.npz"))
            logger.info(f"MultiMotionLoader: found {len(files)} .npz files in {expanded}")
            motion_files.extend(files)
        assert len(motion_files) > 0, f"No .npz files found in {motion_dir}"
        logger.info(f"MultiMotionLoader: loading {len(motion_files)} total motion files")

        loaders = []
        skipped = 0
        for mf in motion_files:
            try:
                loader = MotionLoader(mf, robot_body_names, robot_joint_names, device=device)
                loaders.append(loader)
            except (KeyError, AssertionError, ValueError) as e:  # noqa: PERF203
                # Skip files with incompatible format (e.g., missing body_names, wrong body count)
                skipped += 1
                if skipped <= 3:
                    logger.warning(f"MultiMotionLoader: skipping {mf}: {e}")
        if skipped > 3:
            logger.warning(f"MultiMotionLoader: skipped {skipped} files total due to format issues")
        assert len(loaders) > 0, f"No compatible motion files found (skipped {skipped})"

        # Track per-motion boundaries
        lengths = [loader.time_step_total for loader in loaders]
        cumulative = torch.tensor(lengths, dtype=torch.long, device=device).cumsum(dim=0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=device), cumulative[:-1]])
        self._motion_end_idx = cumulative
        self._num_motions = len(loaders)

        # Concatenate all motion data
        self._joint_pos = torch.cat([ld._joint_pos for ld in loaders], dim=0)
        self._joint_vel = torch.cat([ld._joint_vel for ld in loaders], dim=0)
        self._body_pos_w = torch.cat([ld._body_pos_w for ld in loaders], dim=0)
        self._body_quat_w = torch.cat([ld._body_quat_w for ld in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([ld._body_lin_vel_w for ld in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([ld._body_ang_vel_w for ld in loaders], dim=0)

        # Use indexes from first loader (all loaders share the same robot)
        self._joint_indexes = loaders[0]._joint_indexes
        self._body_indexes = loaders[0]._body_indexes
        self.fps = loaders[0].fps
        self.time_step_total = self._joint_pos.shape[0]

        # Object support: only if ALL motions have objects
        self.has_object: bool = all(ld.has_object for ld in loaders)
        n_with_object = sum(ld.has_object for ld in loaders)
        if 0 < n_with_object < len(loaders):
            # A subset of files carry object data but not all — we disable object tracking for the
            # whole run and discard the object data those files DID contain. Warn rather than drop
            # silently, since the user authored that object motion and it looks like it "worked".
            logger.warning(
                f"MultiMotionLoader: {n_with_object}/{len(loaders)} motion files contain object "
                f"tracking data, but not all do; object tracking is DISABLED for the whole run and "
                f"the object data from those files is discarded. Provide object data in all files or none."
            )
        if self.has_object:
            self._object_pos_w = torch.cat([ld._object_pos_w for ld in loaders], dim=0)
            self._object_quat_w = torch.cat([ld._object_quat_w for ld in loaders], dim=0)
            self._object_lin_vel_w = torch.cat([ld._object_lin_vel_w for ld in loaders], dim=0)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)

        logger.info(f"MultiMotionLoader: {self._num_motions} motions, {self.time_step_total} total frames")

    @property
    def num_motions(self) -> int:
        return self._num_motions

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return self._motion_start_idx

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return self._motion_end_idx

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    def splice_transition_segments(
        self, segments_per_motion: list[dict[str, torch.Tensor]], prepend: bool
    ) -> MultiMotionLoader:
        """Splice one transition segment around each clip, mutating in place.

        The clip count is unchanged: each segment joins the clip it leads into (or out of), so
        the robot walks straight from a lead-in into its clip's frames instead of being
        resampled at the hand-off.
        """
        if len(segments_per_motion) != self._num_motions:
            raise ValueError(
                f"Expected one transition segment per clip ({self._num_motions}), got {len(segments_per_motion)}"
            )

        added_frames = _splice_motion_frames(
            self, segments_per_motion, self._motion_start_idx, self._motion_end_idx, prepend
        )
        self._motion_start_idx, self._motion_end_idx = splice_transition_boundaries(
            self._motion_start_idx, self._motion_end_idx, added_frames
        )
        self.time_step_total = self._joint_pos.shape[0]
        return self


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ):
        self.device = device
        # length of the motion in rl environment time steps
        self.motion_time_step_total = motion_time_step_total
        # fps of the rl environment
        self.env_fps = env_fps

        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha

        # Match BeyondMimic binning: ~1 second bins at env FPS, with +1 tail bin.
        self.num_bins = int(self.motion_time_step_total // max(self.env_fps, 1)) + 1

        # Match BeyondMimic non-causal kernel.
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def init_buffers(self):
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor):
        """Update the current bin failed count with terminated time steps."""
        failed_bin = torch.clamp(
            (failed_at_time_step * self.num_bins) // max(self.motion_time_step_total, 1),
            0,
            self.num_bins - 1,
        ).long()
        assert failed_bin.min() >= 0 and failed_bin.max() < self.num_bins, "Failed bin is out of range"
        # Accumulate (not overwrite): reset() may be called more than once per env
        # step — once for termination-driven resets and again for clip-ended resets
        # in MotionCommand.step() — before update_bin_failed_count() folds + zeroes
        # this buffer. Overwriting clobbered the earlier wave's failures.
        self.current_bin_failed_count += torch.bincount(failed_bin, minlength=self.num_bins).float()

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.adaptive_alpha * self.current_bin_failed_count) + (
            1 - self.adaptive_alpha
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = self.bin_failed_count + self.adaptive_uniform_ratio / float(self.num_bins)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        return sampling_probabilities / sampling_probabilities.sum()

    def sample(self, num_samples: int) -> torch.Tensor:
        sampled_bins = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        # inside of each bin, randomly sample a time step, ignoring the borders
        return (sampled_bins + torch.rand(num_samples, device=self.device)) / self.num_bins

    def sample_global_time_steps(self, num_samples: int) -> torch.Tensor:
        """Sample absolute (global) frame indices in [0, motion_time_step_total).

        The bins live in the GLOBAL concatenated-motion frame space, so a sampled
        phase must be mapped back to a global frame index — NOT reinterpreted as a
        per-motion fraction of an unrelated clip. The caller derives the motion id
        from which clip's [start, end) interval the returned index falls into, so
        the failure-prioritized location stays attached to the motion it came from.
        """
        phase = self.sample(num_samples)
        global_idx = (phase * self.motion_time_step_total).long()
        return global_idx.clamp_(0, self.motion_time_step_total - 1)

    def get_stats(self):
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum()
        H_norm = H / np.log(max(self.num_bins, 2))  # guard num_bins==1 (log(1)=0 -> nan)
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = imax.float() / self.num_bins


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose
        # Resolved in setup(); declared here so their types are visible to callers and to mypy.
        self._body_scatter_src: torch.Tensor
        self._body_scatter_dst: torch.Tensor
        # Per-body centre-of-mass offsets, read from the articulation on first use.
        self._com_offsets_cache: torch.Tensor | None = None

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names

        # 1. load motion data
        assert self.motion_cfg.motion_file or self.motion_cfg.motion_dir, (
            "Either motion_file or motion_dir must be set in MotionConfig"
        )
        self.motion: MotionLoader | MultiMotionLoader
        if self.motion_cfg.motion_dir:
            self.motion = MultiMotionLoader(
                self.motion_cfg.motion_dir,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
            )
        else:
            self.motion = MotionLoader(
                self.motion_cfg.motion_file,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
            )

        # Store body and joint indexes for interpolation. Two index spaces coexist here: the raw
        # loader arrays (motion._body_pos_w and friends) are in *motion* order, while the
        # same-named properties (motion.body_pos_w) re-index them into *robot* order. These
        # indexes map robot -> motion, so they gather for reads and scatter for writes.
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes
        self._build_motion_scatter_map(robot_body_names, robot_body_names_alias)

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            # Derive the object name from the registered rigid object (object names vary
            # per scene, e.g. "box").
            rigid_object_names = self._env.simulator.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
            if not rigid_object_names:
                raise RuntimeError("Set 'has_object' to true, but loaded no rigid bodies in the scene.")
            self.object_name = rigid_object_names[0]

        # 4. get the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total, self.device, int(1 / (self._env.dt))
            )

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self.init_buffers()

        # 6. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        num_motions = self.motion.num_motions

        # 0. Sample the time steps (and, for the adaptive sampler, the motion id).
        adaptive_global_idx = None
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            # Match BeyondMimic behavior: update failed bins from environments
            # that terminated before this reset, then sample new phases.
            # Gate the failure-stat update on training mode so evaluation episodes
            # don't contaminate the training sampler's failure distribution
            # (the is_evaluating phase-zeroing below only affects sampling, not stats).
            if not self._env.is_evaluating:
                episode_failed = self._env.termination_manager.terminated[env_ids]
                if torch.any(episode_failed):
                    failed_at_time_step = self.time_steps[env_ids][episode_failed]
                    self.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_at_time_step)
            # The sampler bins failures over the GLOBAL concatenated-motion frame
            # axis, so it must return a global frame index here. The motion id is
            # then derived from that index (NOT chosen independently), keeping the
            # failure-prioritized phase attached to the motion it was recorded on.
            adaptive_global_idx = self.adaptive_timesteps_sampler.sample_global_time_steps(n)
            phase = None
        else:
            phase = torch.rand(n, device=self.device)

        if self._env.is_evaluating:
            # Eval forces every env through the uniform/else branch below, which
            # indexes `phase`, so it must be a real zero tensor even when the
            # adaptive sampler left it as None.
            phase = torch.zeros(n, device=self.device)
            adaptive_global_idx = None  # eval starts every env at its motion's first frame

        if adaptive_global_idx is not None:
            # Map global frame index -> (motion_id, time_step). searchsorted on the
            # per-motion end indices yields the clip whose [start, end) contains it.
            motion_ids = torch.searchsorted(self.motion.motion_end_idx, adaptive_global_idx, right=True)
            motion_ids = motion_ids.clamp_(0, num_motions - 1)
            self.motion_ids[env_ids] = motion_ids
            start_idx = self.motion.motion_start_idx[motion_ids]
            end_idx = self.motion.motion_end_idx[motion_ids]
            self.time_steps[env_ids] = adaptive_global_idx.clamp(start_idx, end_idx - 1)
        else:
            # Uniform path (or eval): randomly assign each env to a motion, sample
            # a phase within that motion's range.
            self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)
            start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
            end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]
            motion_len = end_idx - start_idx
            self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()

        # Handle start_at_timestep_zero_prob (reset to start of assigned motion)
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = start_idx
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, start_idx, subset)
            self.time_steps[env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        already_last_timestep_mask = self.time_steps[env_ids] >= end_idx - 1
        self.time_steps[env_ids] = torch.where(already_last_timestep_mask, end_idx - 2, self.time_steps[env_ids])

        # 1. Get the root/body poses from the motion data
        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        # 2. Adding noise
        # 2.1 prepare the noise scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_vel_noise = (
            torch.tensor(
                self.init_pose_cfg.root_lin_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_ang_vel_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_ang_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (3,)
            target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise

            # object_ang_vel_w exists in the npz but MotionLoader never loads it, so it resets to zero.
            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 13): pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # Handle freeze_at_timestep_zero_prob: for envs at their motion's start, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == self.motion.motion_start_idx[self.motion_ids]
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        self.time_steps += advance_mask.long()

        # BeyondMimic-style behavior: when the clip ends, resample motion and
        # reset robot/object state without terminating the whole episode.
        per_motion_end = self.motion.motion_end_idx[self.motion_ids]
        ended_env_ids = torch.where(self.time_steps >= per_motion_end)[0]
        if ended_env_ids.numel() > 0:
            self.reset(ended_env_ids)
            # Flush the mutated root/dof state into the simulator so that
            # rigid-body positions are up-to-date for downstream consumers
            # (termination checks, observations, rewards).
            sim = self._env.simulator
            sim.set_actor_root_state_tensor_robots(ended_env_ids, sim.robot_root_states)
            sim.set_dof_state_tensor_robots(ended_env_ids, sim.dof_state)  # type: ignore[attr-defined]
            sim.refresh_sim_tensors()

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

        ### 1.3 update the adaptive timesteps sampler (training only — eval episodes
        ### must not decay/fold failure stats into the training sampler).
        if self.motion_cfg.use_adaptive_timesteps_sampler and not self._env.is_evaluating:
            self.adaptive_timesteps_sampler.update_bin_failed_count()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps][:, self.tracked_body_indexes]
            + self._env.simulator.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.ref_body_index] + self._env.simulator.scene.env_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.ref_body_index]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.ref_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, 0] + self._env.simulator.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, 0]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, 0]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, 0]

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        # Applies env origins, but ideally we should rely on the simulator
        return self.motion.object_pos_w[self.time_steps] + self._env.simulator.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.motion.object_quat_w[self.time_steps]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.motion.object_lin_vel_w[self.time_steps]

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    def _simulator_object_states(self) -> torch.Tensor:
        """Current object states [num_envs, 13] via the unified actor API.

        Reads through ``get_actor_states`` rather than indexing ``all_root_states``
        directly: on MuJoCo ``all_root_states`` is robot-only, so indexing it with an
        object index would be out of bounds. The unified API resolves the object's own
        per-env state on every backend.
        """
        return self._env.simulator.get_actor_states([self.object_name], env_ids=None)

    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._simulator_object_states()[:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._simulator_object_states()[:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._simulator_object_states()[:, 7:10]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.init_buffers()

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.get_stats()
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        action = "prepend" if prepend else "append"
        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                action,
                duration,
                self._env.dt,
            )
            return

        # Frames are consumed one per policy step, so the grid spacing is env.dt and the polynomial
        # duration has to be the grid's, not the configured one -- otherwise the analytic velocities
        # are scaled by duration / effective_duration relative to the spacing they are replayed at.
        effective_duration = num_steps * self._env.dt
        if abs(effective_duration - duration) > 1e-9:
            logger.warning(
                "Default pose {} duration {}s is not a multiple of dt {}; using {}s ({} frames).",
                action,
                duration,
                self._env.dt,
                effective_duration,
                num_steps,
            )

        num_clips = self.motion.num_motions
        try:
            self._add_transition_to_motion(num_steps, effective_duration, prepend=prepend)
            logger.info(
                f"{action} {num_steps} interpolated frames ({effective_duration}s) between the "
                f"default pose and each of {num_clips} clip(s)"
            )
        except Exception as exc:
            # Do not name a cause: this wraps the backend check, the at-rest init_state check, the
            # body-slot collision check and the FK freshness check, and the inner error already says
            # which fired. Add only the configuration that produced it.
            raise RuntimeError(
                f"Failed to {action} a {effective_duration}s default-pose transition "
                f"({num_steps} frames) onto {num_clips} clip(s): {exc}"
            ) from exc

    def _build_default_pose_state(self, anchor_frame_idx: int) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        Root x/y and yaw are adopted from ``anchor_frame_idx``, the clip frame this transition
        joins, so the default pose is placed where that clip starts (or ends) rather than at the
        world origin.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        # Robot body 0 is the floating base; index the raw array rather than the re-indexed view, so
        # reading one row does not gather the whole thing.
        root_slot = int(self._body_indexes_in_motion[0])
        motion_root_pos = self.motion._body_pos_w[anchor_frame_idx, root_slot].to(self.device)
        motion_root_quat = self.motion._body_quat_w[anchor_frame_idx, root_slot].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        )
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)
        # The segment forces the default-pose frame's body velocities to zero, which only holds for a
        # robot spawned at rest. No shipped robot config does otherwise; fail loudly if one starts to.
        if default_root_lin_vel.any() or default_root_ang_vel.any():
            raise NotImplementedError(
                "Default-pose transitions assume the robot's init_state is at rest, but this robot "
                f"config sets lin_vel={init_state.lin_vel}, ang_vel={init_state.ang_vel}."
            )

        state = {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
        }
        if self.motion.has_object:
            # See _transition_free_variables: the object holds its anchor pose.
            state["object_pos"] = self.motion._object_pos_w[anchor_frame_idx].to(self.device)
            state["object_quat"] = self.motion._object_quat_w[anchor_frame_idx].to(self.device)
        return state

    def _joint_limits_in_motion_order(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Lower and upper joint limits scattered into motion joint order, or None if unavailable."""
        limits = getattr(self._env.simulator, "dof_pos_limits", None)
        if limits is None:
            return None
        limits = limits.to(device=self.device)
        lower = self._map_robot_joints_to_motion_order(limits[:, 0], fill=torch.tensor(-torch.inf))
        upper = self._map_robot_joints_to_motion_order(limits[:, 1], fill=torch.tensor(torch.inf))
        return lower, upper

    def _add_transition_to_motion(self, num_steps: int, duration_s: float, prepend: bool) -> None:
        """Add interpolated frames before or after every clip the loader holds.

        Three passes rather than one loop, because forward kinematics is the expensive step: every
        clip's free variables are built first, put through FK in a single batched call, and only then
        assembled back into per-clip segments.
        """
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None
        assert num_steps > 0, f"Caller must reject non-positive step counts, got {num_steps}"

        device = self.device
        dtype = self.motion._joint_pos.dtype
        dt = float(self._env.dt)

        # Snapshot the boundaries before splicing; the splice recomputes them.
        clip_starts = self.motion.motion_start_idx.tolist()
        clip_ends = self.motion.motion_end_idx.tolist()
        # The frame each transition joins: the clip's first frame for a lead-in, its last for a
        # lead-out. Absolute indices into the raw concatenated arrays.
        anchors = [start if prepend else end - 1 for start, end in zip(clip_starts, clip_ends)]

        free_variables = []
        for anchor_frame_idx in anchors:
            default_state = self._build_default_pose_state(anchor_frame_idx)
            default_motion_state = self._default_motion_state(
                default_state, dtype=dtype, device=device, anchor_frame_idx=anchor_frame_idx
            )
            clip_motion_state = self._motion_state(anchor_frame_idx, dtype=dtype, device=device)
            free_variables.append(
                self._transition_free_variables(
                    start=default_motion_state if prepend else clip_motion_state,
                    target=clip_motion_state if prepend else default_motion_state,
                    num_steps=num_steps,
                    duration_s=duration_s,
                )
            )

        fk_pos, fk_quat, fk_com_pos = self._fk_body_poses(
            torch.cat([free["joint_pos"] for free in free_variables])[:, self._joint_indexes_in_motion],
            torch.cat([free["root_pos"] for free in free_variables]),
            torch.cat([free["root_quat"] for free in free_variables]),
        )

        frames_per_clip = num_steps + 1
        segments_per_motion = [
            self._assemble_transition_segment(
                free=free,
                fk_pos=fk_pos[clip * frames_per_clip : (clip + 1) * frames_per_clip],
                fk_quat=fk_quat[clip * frames_per_clip : (clip + 1) * frames_per_clip],
                fk_com_pos=fk_com_pos[clip * frames_per_clip : (clip + 1) * frames_per_clip],
                num_steps=num_steps,
                dt=dt,
                anchor_frame_idx=anchor_frame_idx,
                prepend=prepend,
            )
            for clip, (free, anchor_frame_idx) in enumerate(zip(free_variables, anchors))
        ]

        self.motion = self.motion.splice_transition_segments(segments_per_motion, prepend=prepend)
        try:
            self._log_transition_consistency(num_steps, dt, prepend=prepend)
        except Exception as exc:
            # A diagnostic must not fail the splice it reports on -- the caller turns anything
            # raised here into "critical error during motion interpolation setup", which would be
            # a lie once the splice itself has succeeded.
            logger.warning(f"Could not measure default-pose transition consistency: {exc}")

    def _log_transition_consistency(self, num_steps: int, dt: float, prepend: bool) -> None:
        """Log how well the spliced segment honours the derivative contract, once per transition.

        Measured over the segment's own rows only. A clip's ``body_lin_vel_w`` comes from
        ``mj_objectVelocity`` -- an analytic velocity from qvel -- while its ``body_pos_w`` is a
        sampled position, so on fast motion the two disagree by metres per second with nothing wrong.
        Including clip rows here made every healthy run print that number and look broken.

        The seam steps are reported next to the clip's own typical frame-to-frame step, because the
        absolute value is meaningless without that scale: a transition into a fast clip should step
        about as much as the clip does.
        """
        clip_start = int(self.motion.motion_start_idx[0])
        clip_end = int(self.motion.motion_end_idx[0])
        # The segment occupies the leading (prepend) or trailing (append) num_steps rows of the clip's
        # interval. Trim one row at each end, where differencing is one-sided.
        if prepend:
            first, last, seam = clip_start + 1, clip_start + num_steps - 1, clip_start + num_steps
        else:
            seam = clip_end - num_steps
            first, last = seam + 1, clip_end - 1
        if last - first < 2 or seam <= clip_start or seam >= clip_end:
            return
        segment = slice(first, last)

        joint_pos = self.motion._joint_pos
        body_pos = self.motion._body_pos_w
        joint_residual = (
            (self.motion._joint_vel[segment] - linear_velocity_from_positions(joint_pos, dt)[segment]).abs().max()
        )

        # Differentiate the centre of mass, not the link origin -- body_lin_vel_w is CoM-referenced,
        # so differencing body_pos_w would report a healthy transition as an omega-cross-r failure.
        offsets = self._body_com_offsets_b().to(device=body_pos.device, dtype=body_pos.dtype)
        com_pos = body_pos.clone()
        tracked = self._body_scatter_dst
        com_pos[:, tracked] += quat_apply(
            self.motion._body_quat_w[:, tracked],
            offsets[self._body_scatter_src].expand_as(body_pos[:, tracked]),
            w_last=True,
        )
        body_residual = (
            (self.motion._body_lin_vel_w[segment] - linear_velocity_from_positions(com_pos, dt)[segment]).abs().max()
        )

        # Scale for the seam steps: how much the clip itself moves between neighbouring frames.
        clip_rows = slice(clip_start + num_steps, clip_end) if prepend else slice(clip_start, seam)
        clip_joint_step = self.motion._joint_pos[clip_rows].diff(dim=0).abs().max()
        ang_vel = self.motion._body_ang_vel_w
        clip_ang_step = ang_vel[clip_rows].diff(dim=0).abs().max()

        logger.info(
            "Default-pose {} consistency over its {} frames: max |joint_vel - d(joint_pos)/dt| "
            "{:.5f} rad/s, max |body_lin_vel - d(body_com_pos)/dt| {:.5f} m/s. Seam step "
            "{:.4f} rad / {:.4f} m / {:.4f} rad/s, against a clip whose own frame-to-frame step "
            "reaches {:.4f} rad / {:.4f} rad/s.",
            "prepend" if prepend else "append",
            num_steps,
            float(joint_residual),
            float(body_residual),
            float((joint_pos[seam] - joint_pos[seam - 1]).abs().max()),
            float((body_pos[seam] - body_pos[seam - 1]).norm(dim=-1).max()),
            float((ang_vel[seam] - ang_vel[seam - 1]).abs().max()),
            float(clip_joint_step),
            float(clip_ang_step),
        )

    def _fk_body_poses(
        self, joint_pos: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward kinematics for a batch of frames, by writing each into its own environment.

        Args:
            joint_pos: ``(num_frames, num_robot_dofs)`` in robot joint order.
            root_pos: ``(num_frames, 3)`` relative to the env origin.
            root_quat: ``(num_frames, 4)`` xyzw.

        Returns:
            ``(body_pos, body_quat, body_com_pos)`` shaped ``(num_frames, num_robot_bodies, ·)``,
            relative to the env origin, in ``simulator._body_list`` order. ``body_quat`` is xyzw,
            matching both the simulator buffers and the loader's in-memory arrays. ``body_pos`` is
            the link frame origin; ``body_com_pos`` is the centre of mass, which is the reference
            point a motion file's body velocities use -- differencing the wrong one leaves an
            omega-cross-r error on every rotating body.

        One write and one read per chunk, rather than one per frame. Besides being far cheaper than
        a per-frame loop, it removes a correctness hazard: IsaacLab's ``body_*_w`` are
        timestamp-gated lazy buffers, so a loop that writes and reads without advancing the sim
        clock could return the same stale pose for every frame and yield a body trajectory frozen
        at frame 0.

        Velocities are deliberately not read back. Callers derive body velocities by differencing
        these poses, which keeps them consistent with the poses actually stored.
        """
        simulator = self._env.simulator
        if simulator.get_simulator_type() != SimulatorType.ISAACSIM:
            raise NotImplementedError(
                "Default-pose transitions need a backend whose state write runs forward kinematics. "
                "IsaacGym applies state writes immediately but does not run FK, so the rigid-body "
                "buffers stay stale until the next simulate() (see reset()'s flush). MuJoCo's "
                "ClassicBackend does run mj_forward and could be supported, but its body ordering "
                "and the Warp backend (which skips forward on state writes) need validating first."
            )

        num_frames = joint_pos.shape[0]
        chunk_size = min(num_frames, self.num_envs)
        env_origins = simulator.scene.env_origins.to(self.device)

        body_pos = torch.empty(
            (num_frames, simulator._rigid_body_pos.shape[1], 3), device=self.device, dtype=joint_pos.dtype
        )
        body_quat = torch.empty((num_frames, body_pos.shape[1], 4), device=self.device, dtype=joint_pos.dtype)

        # setup() runs before init_buffers() and the first reset(), so nothing observes these
        # writes -- but an exception mid-loop must not leave every env parked in a bogus pose.
        root_backup = simulator.robot_root_states[:].clone()
        dof_pos_backup = simulator.dof_pos[:].clone()
        dof_vel_backup = simulator.dof_vel[:].clone()
        try:
            for chunk_start in range(0, num_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_frames)
                env_ids = torch.arange(chunk_end - chunk_start, device=self.device)

                simulator.robot_root_states[env_ids, :3] = root_pos[chunk_start:chunk_end] + env_origins[env_ids]
                simulator.robot_root_states[env_ids, 3:7] = root_quat[chunk_start:chunk_end]
                simulator.robot_root_states[env_ids, 7:13] = 0.0
                # refresh_sim_tensors rebinds dof_pos/dof_vel to fresh tensors, so these writes have
                # to be redone on every chunk rather than hoisted.
                simulator.dof_pos[env_ids] = joint_pos[chunk_start:chunk_end]
                simulator.dof_vel[env_ids] = 0.0

                simulator.set_actor_root_state_tensor_robots(env_ids, simulator.robot_root_states)
                simulator.set_dof_state_tensor_robots(env_ids, simulator.dof_state)  # type: ignore[attr-defined]
                simulator.write_state_updates()
                simulator.refresh_sim_tensors()

                chunk_pos = simulator._rigid_body_pos[env_ids] - env_origins[env_ids].unsqueeze(1)
                body_pos[chunk_start:chunk_end] = chunk_pos.to(body_pos.dtype)
                body_quat[chunk_start:chunk_end] = simulator._rigid_body_rot[env_ids].to(body_quat.dtype)

                self._check_fk_poses_are_fresh(
                    body_pos[chunk_start:chunk_end], root_pos[chunk_start:chunk_end], joint_pos[chunk_start:chunk_end]
                )
        finally:
            simulator.robot_root_states[:] = root_backup
            simulator.dof_pos[:] = dof_pos_backup
            simulator.dof_vel[:] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        com_offset = self._body_com_offsets_b().to(device=body_pos.device, dtype=body_pos.dtype)
        body_com_pos = body_pos + quat_apply(body_quat, com_offset.expand_as(body_pos), w_last=True)
        return body_pos, body_quat, body_com_pos

    def _body_com_offsets_b(self) -> torch.Tensor:
        """Per-body centre-of-mass offset in the body frame, ``(num_robot_bodies, 3)``.

        Constant for the run, so read once. ``get_coms`` rows are ``(pos, quat)`` per body over all
        environments; the transition only ever uses env 0, and CoM randomisation happens later.
        """
        if self._com_offsets_cache is None:
            simulator = self._env.simulator
            coms = simulator._robot.root_physx_view.get_coms()  # type: ignore[attr-defined]
            self._com_offsets_cache = coms[0, simulator.body_ids, :3].to(self.device)  # type: ignore[attr-defined]
        return self._com_offsets_cache

    def _check_fk_poses_are_fresh(
        self, body_pos: torch.Tensor, root_pos: torch.Tensor, joint_pos: torch.Tensor
    ) -> None:
        """Catch the simulator handing back stale poses instead of the state we just wrote.

        Raises rather than asserts: this is the one guard against a silent failure mode (a body
        trajectory frozen at frame 0), and ``python -O`` would strip an assert.
        """
        stale_hint = (
            "The simulator returned body poses that do not reflect the state just written. This is "
            "what a stale lazy pose buffer looks like; the sim clock may need advancing before the "
            "read (e.g. simulator.scene.update(dt))."
        )
        # Robot body 0 is the floating base, the same assumption root_pos_w makes of motion body 0.
        root_error = float((body_pos[:, 0] - root_pos).abs().max())
        if root_error >= 1e-3:
            raise RuntimeError(f"Root pose did not round-trip through FK (max error {root_error:.6f}). {stale_hint}")

        if body_pos.shape[0] > 1 and not torch.allclose(joint_pos[0], joint_pos[-1]):
            # Root-relative, or the root's own motion would mask frozen joints.
            relative = body_pos - body_pos[:, :1]
            if float((relative - relative[0]).abs().max()) == 0.0:
                raise RuntimeError(f"FK returned identical body poses for distinct joint angles. {stale_hint}")

    def _build_motion_scatter_map(self, robot_body_names: list[str], robot_body_names_alias: list[str]) -> None:
        """Resolve which robot body writes each motion body slot.

        ``_body_indexes_in_motion`` is fine to *gather* with -- two robot bodies reading the same
        motion slot is exactly what an alias is for -- but scattering with it is last-write-wins.
        The fake foot contact points alias onto their ankle links and sit immediately after them in
        ``body_names``, so the contact point used to overwrite the ankle's own pose with one taken
        a few centimetres lower down the URDF; those ankle links are tracked bodies, so the error
        went straight into the tracking reward.

        Drop an aliased row when the body it aliases onto is itself present, and refuse to guess if
        any collision survives that rule.
        """
        assert self._body_indexes_in_motion is not None
        real_slots = {
            int(self._body_indexes_in_motion[i])
            for i, (name, alias) in enumerate(zip(robot_body_names, robot_body_names_alias))
            if name == alias
        }
        keep, dropped = [], []
        for i, (name, alias) in enumerate(zip(robot_body_names, robot_body_names_alias)):
            if name != alias and int(self._body_indexes_in_motion[i]) in real_slots:
                dropped.append(f"{name} -> {alias}")
            else:
                keep.append(i)

        if dropped:
            logger.info(f"Motion body scatter ignores aliased bodies already covered by a real body: {dropped}")

        keep_tensor = torch.tensor(keep, dtype=torch.long, device=self.device)
        self._body_scatter_src = keep_tensor
        self._body_scatter_dst = self._body_indexes_in_motion[keep_tensor]
        if len(set(self._body_scatter_dst.tolist())) != len(keep):
            raise RuntimeError(
                "Multiple robot bodies map to the same motion body slot after alias resolution; "
                "scattering would silently keep only one of them. "
                f"Robot bodies: {[robot_body_names[i] for i in keep]}, "
                f"motion slots: {self._body_scatter_dst.tolist()}"
            )

    def _map_robot_bodies_to_motion_order(
        self, robot_tensor: torch.Tensor, fill: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Scatter a robot-ordered body tensor into motion body order.

        Args:
            robot_tensor: ``(..., num_robot_bodies, C)``. Leading dims (e.g. time) pass through.
            fill: value for motion slots no robot body covers, broadcast over the leading dims.
                Motion files legitimately carry bodies the robot does not have (including the
                MuJoCo ``world`` row), and zeros are not a valid rotation. Passing the clip's anchor
                frame keeps those slots constant and correct at the seam. ``None`` keeps zeros.
        """
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = robot_tensor.shape[:-2] + (num_motion_bodies,) + robot_tensor.shape[-1:]
        if fill is None:
            motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        else:
            motion_tensor = fill.to(device=robot_tensor.device, dtype=robot_tensor.dtype).expand(motion_shape).clone()
        motion_tensor[..., self._body_scatter_dst, :] = robot_tensor[..., self._body_scatter_src, :]
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None, fill: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Scatter a robot-ordered joint tensor into motion joint order. See the body variant."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        if fill is None:
            motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        else:
            motion_tensor = fill.to(device=robot_tensor.device, dtype=robot_tensor.dtype).expand(motion_shape).clone()
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict.

        The root is read at robot body 0, the same way ``root_pos_w`` and friends do at runtime --
        the motion file's own body 0 is MuJoCo's ``world`` row, not the floating base.
        """
        root_slot = int(self._body_indexes_in_motion[0])
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "root_pos": self.motion._body_pos_w[idx, root_slot].to(device=device, dtype=dtype),
            "root_quat": self.motion._body_quat_w[idx, root_slot].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self,
        default_state: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        anchor_frame_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        # Motion joint columns with no robot counterpart (a 29-DoF robot against a 31-DoF clip) are
        # never read by either field below -- motion.joint_pos gathers only the covered ones -- so
        # whatever they interpolate to is inert. Filled from the anchor to keep the array honest.
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
                fill=self.motion._joint_pos[anchor_frame_idx],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
                fill=self.motion._joint_vel[anchor_frame_idx],
            ),
            "root_pos": default_state["root_pos"].to(device=device, dtype=dtype),
            "root_quat": default_state["root_quat"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
        return state

    def _transition_free_variables(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        num_steps: int,
        duration_s: float,
    ) -> dict[str, torch.Tensor]:
        """Interpolate the quantities a transition is free to choose, over ``num_steps + 1`` frames.

        Joint angles and the root pose are those quantities; everything else in a motion frame is
        determined by them through kinematics. They get a cubic Hermite and its exact analytic
        derivative, so position and velocity come from one polynomial and therefore agree at every
        frame rather than only at the endpoints.
        """
        # Both endpoints at rest, which makes the cubic a smoothstep: monotone between the two
        # poses, so no joint can wander past either of them, let alone past its limit. The cost is a
        # velocity step into the clip. Matching the clip's arrival velocity instead removes that step
        # but forces the trajectory to travel out and back whenever the net displacement is small --
        # on an ordinary clip that was a 1.05 rad detour to make a 0.05 rad move, on 10 of 29 joints.
        # A monotone path with one bad frame beats a hundred frames of wandering.
        at_rest = torch.zeros_like(start["joint_vel"])
        joint_pos, joint_vel = hermite_segment(
            start["joint_pos"], at_rest, target["joint_pos"], at_rest, duration_s, num_steps
        )
        root_pos, _ = hermite_segment(
            start["root_pos"],
            torch.zeros_like(start["root_pos"]),
            target["root_pos"],
            torch.zeros_like(start["root_pos"]),
            duration_s,
            num_steps,
        )
        root_quat = hermite_rotation_series(
            start["root_quat"],
            torch.zeros(3, device=start["root_quat"].device, dtype=start["root_quat"].dtype),
            target["root_quat"],
            torch.zeros(3, device=start["root_quat"].device, dtype=start["root_quat"].dtype),
            duration_s,
            num_steps,
        )
        free = {"joint_pos": joint_pos, "joint_vel": joint_vel, "root_pos": root_pos, "root_quat": root_quat}
        if self.motion.has_object:
            # The object holds its anchor pose for the whole segment. A lead-in has nothing driving
            # it yet; a lead-out freezes an object the robot may still be holding, which is an
            # approximation. The alternative is worse: a Hermite between equal positions with the
            # clip's nonzero endpoint velocity bulges out and back, i.e. phantom object motion the
            # policy would be rewarded for tracking.
            free["object_pos"] = start["object_pos"].expand(num_steps + 1, -1).clone()
            free["object_quat"] = start["object_quat"].expand(num_steps + 1, -1).clone()
            free["object_lin_vel"] = torch.zeros_like(free["object_pos"])
        return free

    def _assemble_transition_segment(
        self,
        free: dict[str, torch.Tensor],
        fk_pos: torch.Tensor,
        fk_quat: torch.Tensor,
        fk_com_pos: torch.Tensor,
        num_steps: int,
        dt: float,
        anchor_frame_idx: int,
        prepend: bool,
    ) -> dict[str, torch.Tensor]:
        """Turn one clip's free variables plus its FK result into the frames that get spliced in.

        Body poses come from forward kinematics of the free variables, so a frame's body poses and
        its joint angles describe the same configuration. Linear velocity differences ``fk_com_pos``
        rather than ``fk_pos`` -- see :meth:`_fk_body_poses`.

        The grid spans both endpoints; the one coinciding with the clip frame is dropped so the
        splice does not duplicate it. A lead-in keeps rows 0..N-1 (row N *is* the clip frame) and a
        lead-out keeps rows 1..N (row 0 is). Either way exactly ``num_steps`` rows survive.
        """
        body_pos = self._map_robot_bodies_to_motion_order(fk_pos, fill=self.motion._body_pos_w[anchor_frame_idx])
        body_quat = self._map_robot_bodies_to_motion_order(fk_quat, fill=self.motion._body_quat_w[anchor_frame_idx])
        com_pos = self._map_robot_bodies_to_motion_order(fk_com_pos, fill=self.motion._body_pos_w[anchor_frame_idx])

        grid = {
            "joint_pos": free["joint_pos"],
            "joint_vel": free["joint_vel"],
            "body_pos": body_pos,
            "body_quat": body_quat,
            "body_lin_vel": linear_velocity_from_positions(com_pos, dt),
            "body_ang_vel": angular_velocity_from_quats(body_quat, dt),
        }
        for key in ("object_pos", "object_quat", "object_lin_vel"):
            if key in free:
                grid[key] = free[key]

        # Differencing is one-sided at the grid ends, so the default-pose row gets an estimate where
        # the answer is known exactly: the robot is at rest there.
        default_pose_row = 0 if prepend else num_steps
        for key in ("body_lin_vel", "body_ang_vel"):
            grid[key][default_pose_row] = 0.0

        emitted = slice(None, -1) if prepend else slice(1, None)
        segment = {key: values[emitted].contiguous() for key, values in grid.items()}
        for key, values in segment.items():
            assert values.shape[0] == num_steps, (
                f"Transition field {key} has {values.shape[0]} frames, expected {num_steps}"
            )
        self._check_joints_stay_between_the_endpoints(free["joint_pos"])
        self._check_joints_within_limits(segment["joint_pos"], free, anchor_frame_idx)
        return segment

    def _check_joints_stay_between_the_endpoints(self, joint_pos: torch.Tensor) -> None:
        """Refuse a joint path that leaves the interval between the poses it connects.

        Monotonicity is the property this path is judged on: a lead-in exists to walk the robot from
        one pose to another, and a joint that overshoots and comes back is doing something the clip
        never asked for. With both endpoint velocities at rest the cubic is a smoothstep and cannot
        overshoot, so this only fires if someone reintroduces a non-zero endpoint velocity -- which is
        exactly when it needs to fire, since that is what caused a 1.05 rad detour on a 0.05 rad move.
        """
        lower = torch.minimum(joint_pos[0], joint_pos[-1])
        upper = torch.maximum(joint_pos[0], joint_pos[-1])
        excursion = torch.maximum(lower - joint_pos, joint_pos - upper).amax(dim=0)
        worst = float(excursion.max())
        if worst > 1e-4:
            joint = int(excursion.argmax())
            raise RuntimeError(
                f"Default-pose transition leaves the interval between its endpoints: motion joint "
                f"{joint} travels {worst:.4f} rad beyond them, spanning "
                f"{float(joint_pos[:, joint].min()):.3f}..{float(joint_pos[:, joint].max()):.3f} "
                f"between {float(joint_pos[0, joint]):.3f} and {float(joint_pos[-1, joint]):.3f}."
            )

    def _check_joints_within_limits(
        self, joint_pos: torch.Tensor, free: dict[str, torch.Tensor], anchor_frame_idx: int
    ) -> None:
        """Refuse to emit a frame that commands a joint past its limit.

        Self-consistency is not reachability. Everything else here checks that a frame's fields agree
        with each other; nothing checked whether the pose could be held. A cubic that arrives at the
        clip's velocity overshoots its endpoints, and at the default duration that overshoot ran a
        joint 0.6 rad past its limit without anything noticing.
        """
        limits = self._joint_limits_in_motion_order()
        if limits is None:
            return
        lower, upper = limits
        # The grid's own endpoints inherit whatever the clip and the default pose already violate by.
        endpoints = free["joint_pos"][[0, -1]]
        allowed = torch.maximum(
            (lower - endpoints).clamp(min=0.0).amax(dim=0), (endpoints - upper).clamp(min=0.0).amax(dim=0)
        )
        excess = torch.maximum(lower - joint_pos, joint_pos - upper).clamp(min=0.0) - allowed
        worst = float(excess.max())
        if worst > 1e-3:
            joint = int(excess.amax(dim=0).argmax())
            raise RuntimeError(
                f"Default-pose transition commands motion joint {joint} {worst:.4f} rad beyond its "
                f"limit ({float(lower[joint]):.3f}..{float(upper[joint]):.3f}); the segment spans "
                f"{float(joint_pos[:, joint].min()):.3f}..{float(joint_pos[:, joint].max()):.3f} "
                f"between endpoints {float(endpoints[0, joint]):.3f} and {float(endpoints[1, joint]):.3f}. "
                "The duration cap should have prevented this."
            )

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
