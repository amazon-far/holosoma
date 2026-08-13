"""Tests for retargeting Viser configuration."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from holosoma_retargeting.config_types.viser import ViserConfig, XsensViserConfig
from holosoma_retargeting.config_values.viser import get_default_xsens_viser_config
from holosoma_retargeting.src.viser_utils import CameraFollowController, resolve_frame_times
from holosoma_retargeting.viser_player import (
    add_tennis_racket_control,
    compute_camera_follow_target,
    compute_ground_plane_bounds,
    compute_initial_camera_view,
    load_npz,
    offset_qpos_positions,
    resolve_actor_modes,
    resolve_actor_offsets,
    resolve_record_output_path,
)


def test_load_npz_missing_root_file_explains_shell_expansion() -> None:
    missing_path = "/missing-qpos-file-for-viser-player.npz"

    with pytest.raises(FileNotFoundError) as error:
        load_npz(missing_path)

    message = str(error.value)
    assert "QPOS NPZ file not found" in message
    assert missing_path in message
    assert "SAVE_DIR" in message
    assert "TASK" in message


def test_camera_follow_is_opt_in_by_default() -> None:
    assert ViserConfig().camera_follow is False


def test_camera_follow_target_centers_each_actor_equally() -> None:
    target = compute_camera_follow_target(
        robot_position_m=np.array([0.0, 0.0, 1.0]),
        avatar_positions_m=(
            np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 2.0]]),
            np.array([[6.0, 3.0, 1.0]]),
        ),
    )

    np.testing.assert_allclose(target, [3.0, 1.0, 1.0])


def test_camera_follow_target_rejects_missing_actors() -> None:
    with pytest.raises(ValueError, match="At least one robot or avatar"):
        compute_camera_follow_target()


def test_initial_camera_uses_all_floating_base_poses() -> None:
    half_turn_about_z = np.array([0.0, 0.0, 0.0, 1.0])
    view = compute_initial_camera_view(
        (np.array([0.0, -2.0, 1.0]), np.array([0.0, 2.0, 1.0])),
        (np.array([1.0, 0.0, 0.0, 0.0]), half_turn_about_z),
    )

    np.testing.assert_allclose(view.look_at, [0.0, 0.0, 1.0])
    # Opposing headings fall back to the first base, and the actor spread widens the view.
    np.testing.assert_allclose(view.position, [6.0, 0.0, 3.1])


def test_initial_camera_rotates_with_mean_floating_base_heading() -> None:
    quarter_turn_about_z = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    view = compute_initial_camera_view(
        (np.array([1.0, 2.0, 0.8]),),
        (quarter_turn_about_z,),
    )

    np.testing.assert_allclose(view.look_at, [1.0, 2.0, 0.8])
    np.testing.assert_allclose(view.position, [1.0, 5.0, 1.85], atol=1e-12)


def test_camera_follow_controller_preserves_view_offset_and_can_be_disabled() -> None:
    class FakeCheckbox:
        def __init__(self, initial_value: bool) -> None:
            self.value = initial_value
            self.callback = None

        def on_update(self, callback):
            self.callback = callback
            return callback

    checkbox = FakeCheckbox(False)
    camera = SimpleNamespace(
        position=np.array([4.0, -3.0, 2.0]),
        look_at=np.array([1.0, 0.0, 1.0]),
    )
    client = SimpleNamespace(camera=camera)
    server = SimpleNamespace(
        gui=SimpleNamespace(add_checkbox=lambda *_args, **_kwargs: checkbox),
        on_client_connect=lambda callback: callback,
        get_clients=lambda: {0: client},
        atomic=nullcontext,
    )
    controller = CameraFollowController(server)

    controller.update_target(np.array([2.0, 2.0, 1.0]))
    np.testing.assert_array_equal(camera.position, [4.0, -3.0, 2.0])

    checkbox.value = True
    checkbox.callback(None)
    np.testing.assert_array_equal(camera.look_at, [2.0, 2.0, 1.0])
    np.testing.assert_array_equal(camera.position, [5.0, -1.0, 2.0])

    controller.update_target(np.array([3.0, 2.0, 1.0]))
    np.testing.assert_array_equal(camera.look_at, [3.0, 2.0, 1.0])
    np.testing.assert_array_equal(camera.position, [6.0, -1.0, 2.0])

    checkbox.value = False
    controller.update_target(np.array([9.0, 9.0, 9.0]))
    np.testing.assert_array_equal(camera.position, [6.0, -1.0, 2.0])


def test_xsens_options_are_not_part_of_global_viser_config() -> None:
    base = ViserConfig()
    xsens = XsensViserConfig(actor_modes=("xsens", "g1_xsens"), xsens_hdf5="motion.hdf5")

    assert not hasattr(base, "actor_modes")
    assert not hasattr(base, "xsens_hdf5")
    assert not hasattr(base, "show_tennis_racket")
    assert xsens.actor_modes == ("xsens", "g1_xsens")
    assert xsens.xsens_hdf5 == "motion.hdf5"
    assert xsens.actor_spacing_m == 2.0
    assert xsens.show_tennis_racket is True
    assert xsens.g1_xsens_root_motion.mode == "scale_by_leg_length"
    assert isinstance(get_default_xsens_viser_config(), XsensViserConfig)


def test_tennis_control_updates_all_rackets_without_affecting_other_bodies() -> None:
    class FakeCheckbox:
        def __init__(self, initial_value: bool) -> None:
            self.value = initial_value
            self.callback = None

        def on_update(self, callback):
            self.callback = callback
            return callback

    class FakeGui:
        def __init__(self) -> None:
            self.folders = []
            self.checkbox = None

        def add_folder(self, label, *, order=None):
            self.folders.append((label, order))
            return nullcontext()

        def add_checkbox(self, label, *, initial_value):
            assert label == "Show tennis racket"
            self.checkbox = FakeCheckbox(initial_value)
            return self.checkbox

    pelvis_a = SimpleNamespace(visible=True)
    pelvis_b = SimpleNamespace(visible=True)
    racket_a = SimpleNamespace(visible=True)
    racket_b = SimpleNamespace(visible=True)
    actors = (
        SimpleNamespace(body_frames={"Pelvis": pelvis_a, "TennisRacket": racket_a}),
        SimpleNamespace(body_frames={"Pelvis": pelvis_b, "TennisRacket": racket_b}),
    )
    gui = FakeGui()

    checkbox = add_tennis_racket_control(
        SimpleNamespace(gui=gui),
        actors,
        initial_visible=False,
    )

    assert checkbox is gui.checkbox
    assert gui.folders == [("Tennis", 50.0)]
    assert racket_a.visible is False
    assert racket_b.visible is False
    assert pelvis_a.visible is True
    assert pelvis_b.visible is True

    checkbox.value = True
    checkbox.callback(None)
    assert racket_a.visible is True
    assert racket_b.visible is True
    assert pelvis_a.visible is True
    assert pelvis_b.visible is True


def test_tennis_control_is_absent_when_no_actor_has_a_racket() -> None:
    gui = SimpleNamespace(
        add_folder=lambda *_args, **_kwargs: pytest.fail("unexpected tennis folder"),
        add_checkbox=lambda *_args, **_kwargs: pytest.fail("unexpected tennis checkbox"),
    )
    actor = SimpleNamespace(body_frames={"Pelvis": SimpleNamespace(visible=True)})

    checkbox = add_tennis_racket_control(
        SimpleNamespace(gui=gui),
        (actor,),
        initial_visible=True,
    )

    assert checkbox is None


def test_record_path_defaults_to_xsens_hdf5_sibling() -> None:
    config = XsensViserConfig(
        actor_modes=("xsens", "g1_xsens"),
        xsens_hdf5="/tmp/session/stream_log.hdf5",
    )

    assert Path(resolve_record_output_path(config)) == Path("/tmp/session/stream_log.mp4").resolve()


def test_explicit_record_path_overrides_derived_path() -> None:
    config = XsensViserConfig(
        actor_modes=("xsens",),
        xsens_hdf5="motion.hdf5",
        record_path="videos/custom.gif",
    )

    assert resolve_record_output_path(config) == "videos/custom.gif"


def test_actor_modes_are_composable_unique_and_canonically_ordered() -> None:
    assert resolve_actor_modes(("g1_xsens", "xsens", "g1_xsens")) == ("xsens", "g1_xsens")


def test_all_actor_alias_expands_to_every_actor() -> None:
    assert resolve_actor_modes(("all",)) == ("robot", "xsens", "g1_xsens")
    assert resolve_actor_modes(("xsens", "all")) == ("robot", "xsens", "g1_xsens")


def test_actor_modes_reject_empty_and_unknown_selections() -> None:
    with pytest.raises(ValueError, match="at least one actor"):
        resolve_actor_modes(())
    with pytest.raises(ValueError, match="Unknown actor modes"):
        resolve_actor_modes(("not_an_actor",))


def test_actor_offsets_equally_space_all_actors_along_a_centered_line() -> None:
    offsets = resolve_actor_offsets(("all",), 2.0)

    assert tuple(offsets) == ("xsens", "g1_xsens", "robot")
    np.testing.assert_array_equal(offsets["xsens"], [0.0, -2.0, 0.0])
    np.testing.assert_array_equal(offsets["g1_xsens"], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(offsets["robot"], [0.0, 2.0, 0.0])


def test_actor_offsets_center_subsets_and_single_actors() -> None:
    paired = resolve_actor_offsets(("xsens", "robot"), 2.0)
    single = resolve_actor_offsets(("g1_xsens",), 2.0)

    np.testing.assert_array_equal(paired["xsens"], [0.0, -1.0, 0.0])
    np.testing.assert_array_equal(paired["robot"], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(single["g1_xsens"], [0.0, 0.0, 0.0])


def test_actor_spacing_requires_a_finite_non_negative_value() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        resolve_actor_offsets(("all",), -1.0)
    with pytest.raises(ValueError, match="finite non-negative"):
        resolve_actor_offsets(("all",), np.nan)


def test_qpos_offset_translates_robot_and_tracked_object() -> None:
    qpos = np.zeros((2, 7 + 2 + 7))
    qpos[:, 0:3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    qpos[:, -7:-4] = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]

    shifted = offset_qpos_positions(qpos, (0.0, 2.0, 0.0), has_object_input=True)

    np.testing.assert_array_equal(qpos[:, 0:3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(shifted[:, 0:3], [[1.0, 4.0, 3.0], [4.0, 7.0, 6.0]])
    np.testing.assert_array_equal(shifted[:, -7:-4], [[7.0, 10.0, 9.0], [10.0, 13.0, 12.0]])
    np.testing.assert_array_equal(shifted[:, 3:-7], qpos[:, 3:-7])


def test_robot_ground_bounds_include_base_and_tracked_object() -> None:
    robot_dof = 2
    qpos = np.zeros((2, 7 + robot_dof + 7))
    qpos[:, 0:2] = [[10.0, -2.0], [14.0, 6.0]]
    qpos[:, -7:-5] = [[8.0, -5.0], [20.0, 9.0]]

    bounds = compute_ground_plane_bounds(
        qpos=qpos,
        robot_dof=robot_dof,
        contains_object_in_qpos=True,
        padding_m=1.0,
    )

    np.testing.assert_allclose(bounds.center_xy, [14.0, 2.0])
    assert bounds.width == 14.0
    assert bounds.height == 16.0


def test_xsens_and_g1_xsens_ground_bounds_use_all_segment_positions() -> None:
    positions = np.array(
        [
            [[-2.0, 1.0, 0.8], [1.0, 3.0, 1.7]],
            [[4.0, 9.0, 0.8], [2.0, 5.0, 1.7]],
        ]
    )

    bounds = compute_ground_plane_bounds(xsens_positions_m=positions, padding_m=1.0)

    np.testing.assert_allclose(bounds.center_xy, [1.0, 5.0])
    assert bounds.width == 8.0
    assert bounds.height == 10.0


def test_combined_ground_bounds_cover_both_data_sources() -> None:
    qpos = np.zeros((2, 7))
    qpos[:, 0:2] = [[-10.0, 2.0], [-8.0, 4.0]]
    positions = np.array([[[3.0, -6.0, 1.0]], [[7.0, -2.0, 1.0]]])

    bounds = compute_ground_plane_bounds(
        qpos=qpos,
        xsens_positions_m=positions,
        padding_m=0.5,
    )

    np.testing.assert_allclose(bounds.center_xy, [-1.5, -1.0])
    assert bounds.width == 18.0
    assert bounds.height == 11.0


def test_ground_bounds_cover_composed_xsens_actor_offsets() -> None:
    positions = np.array([[[0.0, 2.0, 1.0]], [[1.0, 4.0, 1.0]]])
    combined_positions = np.concatenate(
        [positions, positions + np.array([1.5, -0.5, 0.0])[None, None, :]],
        axis=1,
    )

    bounds = compute_ground_plane_bounds(xsens_positions_m=combined_positions, padding_m=0.5)

    np.testing.assert_allclose(bounds.center_xy, [1.25, 2.75])
    assert bounds.width == 3.5
    assert bounds.height == 3.5


def test_explicit_grid_extents_act_only_as_minimums() -> None:
    qpos = np.zeros((1, 7))

    bounds = compute_ground_plane_bounds(
        qpos=qpos,
        padding_m=1.0,
        minimum_width_m=8.0,
        minimum_height_m=6.0,
    )

    np.testing.assert_array_equal(bounds.center_xy, [0.0, 0.0])
    assert bounds.width == 8.0
    assert bounds.height == 6.0


def test_ground_bounds_require_finite_motion_positions() -> None:
    with pytest.raises(ValueError, match="finite horizontal positions"):
        compute_ground_plane_bounds(qpos=np.full((1, 7), np.nan))


def test_frame_times_use_fps_when_timestamps_are_unavailable() -> None:
    np.testing.assert_allclose(
        resolve_frame_times(4, initial_fps=2.0),
        [0.0, 0.5, 1.0, 1.5],
    )


def test_frame_times_preserve_irregular_timestamp_correspondence() -> None:
    np.testing.assert_allclose(
        resolve_frame_times(
            3,
            initial_fps=30.0,
            frame_times_s=np.array([100.0, 100.01, 100.035]),
        ),
        [0.0, 0.01, 0.035],
    )
