"""URDF colours that the IsaacSim spawn writes onto converted visual meshes (pure XML)."""

from __future__ import annotations

from pathlib import Path

import pytest

from holosoma.config_values.robot import g1_29dof, yam
from holosoma.simulator.isaacsim.urdf_visual_colors import (
    match_visual_colors_to_groups,
    urdf_visual_color_groups,
)
from holosoma.utils.path import resolve_asset_path

pytestmark = pytest.mark.no_sim


def test_yam_links_map_to_the_two_palette_colours() -> None:
    colors = urdf_visual_color_groups(resolve_asset_path(yam.asset.urdf_file, yam.asset.asset_root))
    assert set(colors) == set(yam.body_names)
    for name, per_visual in colors.items():
        expected = (0.9, 0.9, 0.9) if name in ("link2", "link3") else (0.25, 0.25, 0.25)
        assert per_visual and all(color == expected for color in per_visual)


def test_g1_visual_colours_are_read_for_every_coloured_link() -> None:
    colors = urdf_visual_color_groups(resolve_asset_path(g1_29dof.asset.urdf_file, g1_29dof.asset.asset_root))
    # Literal oracle checked against data/robots/g1/g1_29dof.urdf, including
    # transparent helper visuals whose RGB colour is black.
    expected = {
        "pelvis": (0.2, 0.2, 0.2),
        "pelvis_contour_link": (0.7, 0.7, 0.7),
        "left_hip_pitch_link": (0.2, 0.2, 0.2),
        "left_hip_roll_link": (0.7, 0.7, 0.7),
        "left_hip_yaw_link": (0.7, 0.7, 0.7),
        "left_knee_link": (0.7, 0.7, 0.7),
        "left_ankle_pitch_link": (0.7, 0.7, 0.7),
        "left_ankle_roll_link": (0.2, 0.2, 0.2),
        "left_foot_contact_point": (0.0, 0.0, 0.0),
        "right_hip_pitch_link": (0.2, 0.2, 0.2),
        "right_hip_roll_link": (0.7, 0.7, 0.7),
        "right_hip_yaw_link": (0.7, 0.7, 0.7),
        "right_knee_link": (0.7, 0.7, 0.7),
        "right_ankle_pitch_link": (0.7, 0.7, 0.7),
        "right_ankle_roll_link": (0.2, 0.2, 0.2),
        "right_foot_contact_point": (0.0, 0.0, 0.0),
        "waist_yaw_link": (0.7, 0.7, 0.7),
        "waist_roll_link": (0.7, 0.7, 0.7),
        "torso_link": (0.7, 0.7, 0.7),
        "logo_link": (0.2, 0.2, 0.2),
        "head_link": (0.2, 0.2, 0.2),
        "imu_in_torso": (0.0, 0.0, 0.0),
        "imu_in_pelvis": (0.0, 0.0, 0.0),
        "mid360_link": (0.0, 0.0, 0.0),
        "left_shoulder_pitch_link": (0.7, 0.7, 0.7),
        "left_shoulder_roll_link": (0.7, 0.7, 0.7),
        "left_shoulder_yaw_link": (0.7, 0.7, 0.7),
        "left_elbow_link": (0.7, 0.7, 0.7),
        "left_wrist_roll_link": (0.7, 0.7, 0.7),
        "left_wrist_pitch_link": (0.7, 0.7, 0.7),
        "left_wrist_yaw_link": (0.7, 0.7, 0.7),
        "left_rubber_hand": (0.7, 0.7, 0.7),
        "right_shoulder_pitch_link": (0.7, 0.7, 0.7),
        "right_shoulder_roll_link": (0.7, 0.7, 0.7),
        "right_shoulder_yaw_link": (0.7, 0.7, 0.7),
        "right_elbow_link": (0.7, 0.7, 0.7),
        "right_wrist_roll_link": (0.7, 0.7, 0.7),
        "right_wrist_pitch_link": (0.7, 0.7, 0.7),
        "right_wrist_yaw_link": (0.7, 0.7, 0.7),
        "right_rubber_hand": (0.7, 0.7, 0.7),
        "LL_FOOT": (0.0, 0.0, 0.0),
        "LR_FOOT": (0.0, 0.0, 0.0),
    }
    assert set(colors) == set(expected)
    for name, per_visual in colors.items():
        assert per_visual and all(color == expected[name] for color in per_visual)


def test_named_material_references_and_uncoloured_links(tmp_path: Path) -> None:
    urdf = tmp_path / "r.urdf"
    urdf.write_text(
        """<robot name="r">
          <material name="red"><color rgba="1 0 0 1"/></material>
          <link name="a"><visual><geometry><box size="1 1 1"/></geometry><material name="red"/></visual></link>
          <link name="b"><visual><geometry><box size="1 1 1"/></geometry>
            <material name="inline"><color rgba="0 0.5 1 1"/></material></visual></link>
          <link name="c"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
          <link name="d"/>
        </robot>"""
    )
    assert urdf_visual_color_groups(str(urdf)) == {
        "a": [(1.0, 0.0, 0.0)],
        "b": [(0.0, 0.5, 1.0)],
        "c": [None],
    }


def test_links_preserve_each_visual_color_in_document_order(tmp_path: Path) -> None:
    # A grey mesh plus a red marker on one link (a FAR-pi alpha-style URDF) must keep both colours.
    urdf = tmp_path / "r.urdf"
    urdf.write_text(
        """<robot name="r">
          <link name="mixed">
            <visual><geometry><box size="1 1 1"/></geometry><material name="g"><color rgba="0.5 0.5 0.5 1"/></material></visual>
            <visual><geometry><sphere radius="0.1"/></geometry><material name="r"><color rgba="1 0 0 1"/></material></visual>
          </link>
          <link name="partial">
            <visual><geometry><box size="1 1 1"/></geometry><material name="g"><color rgba="0.5 0.5 0.5 1"/></material></visual>
            <visual><geometry><sphere radius="0.1"/></geometry></visual>
          </link>
          <link name="agree">
            <visual><geometry><box size="1 1 1"/></geometry><material name="g"><color rgba="0.5 0.5 0.5 1"/></material></visual>
            <visual><geometry><sphere radius="0.1"/></geometry><material name="g2"><color rgba="0.5 0.5 0.5 1"/></material></visual>
          </link>
        </robot>"""
    )
    groups = urdf_visual_color_groups(str(urdf))
    assert groups == {
        "mixed": [(0.5, 0.5, 0.5), (1.0, 0.0, 0.0)],
        "partial": [(0.5, 0.5, 0.5), None],
        "agree": [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    }


def test_visual_colors_map_individually_and_preserve_uncoloured_groups() -> None:
    assert match_visual_colors_to_groups(
        [(0.5, 0.5, 0.5), (1.0, 0.0, 0.0), None], ["body_mesh", "marker", "unpainted"]
    ) == {"body_mesh": (0.5, 0.5, 0.5), "marker": (1.0, 0.0, 0.0)}


def test_uniform_color_can_cover_importer_split_groups_but_mixed_mismatch_is_not_guessed() -> None:
    assert match_visual_colors_to_groups([(0.25, 0.25, 0.25)], ["mesh_a", "mesh_b"]) == {
        "mesh_a": (0.25, 0.25, 0.25),
        "mesh_b": (0.25, 0.25, 0.25),
    }
    assert match_visual_colors_to_groups([(0.5, 0.5, 0.5), (1.0, 0.0, 0.0)], ["merged"]) == {}


def test_prior_inline_material_is_reused_on_the_same_and_another_link(tmp_path: Path) -> None:
    urdf = tmp_path / "inline.urdf"
    urdf.write_text(
        '<robot name="inline"><link name="arm-link">'
        '<visual><geometry><box size="1 1 1"/></geometry>'
        '<material name="red"><color rgba="1 0 0 1"/></material></visual>'
        '<visual><geometry><sphere radius="1"/></geometry><material name="red"/></visual></link>'
        '<link name="_tip"><visual><geometry><sphere radius="1"/></geometry>'
        '<material name="red"/></visual></link></robot>'
    )
    # Parser keys remain the original URDF names; normalization belongs to the writer boundary.
    assert urdf_visual_color_groups(str(urdf)) == {
        "arm-link": [(1, 0, 0), (1, 0, 0)],
        "_tip": [(1, 0, 0)],
    }


@pytest.mark.parametrize("top_level", [False, True], ids=["prior-inline", "top-level"])
def test_explicit_visual_colour_does_not_replace_prior_named_colour(tmp_path: Path, *, top_level: bool) -> None:
    urdf = tmp_path / "precedence.urdf"
    definition = '<material name="paint"><color rgba="1 0 0 1"/></material>'
    urdf.write_text(
        '<robot name="precedence">'
        + (definition if top_level else "")
        + '<link name="link">'
        + ("" if top_level else '<visual><geometry><box size="1 1 1"/></geometry>' + definition + "</visual>")
        + '<visual><geometry><box size="1 1 1"/></geometry>'
        '<material name="paint"><color rgba="0 0 1 1"/></material></visual>'
        '<visual><geometry><box size="1 1 1"/></geometry><material name="paint"/></visual>'
        "</link></robot>"
    )
    expected = [(0, 0, 1), (1, 0, 0)] if top_level else [(1, 0, 0), (0, 0, 1), (1, 0, 0)]
    assert urdf_visual_color_groups(str(urdf)) == {"link": expected}
