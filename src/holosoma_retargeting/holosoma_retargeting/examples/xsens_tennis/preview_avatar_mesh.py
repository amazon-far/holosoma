#!/usr/bin/env python3
"""Preview the subject-proportioned procedural Xsens avatar in a static T-pose."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
import viser  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.xsens.avatar_mesh import (  # noqa: E402
    AvatarMeshPart,
    build_tennis_racket_meshes,
    build_xsens_avatar_meshes,
    load_xsens_avatar_proportions,
    validate_avatar_mesh_parts,
)
from holosoma_retargeting.xsens.kinematic_model import XSENS_RACKET_SOURCE_SEGMENT  # noqa: E402


@dataclass(frozen=True)
class XsensAvatarMeshPreviewConfig:
    """Configuration for the static Xsens avatar mesh preview."""

    hdf5_path: Path = Path("demo_data/xsens_tennis/2026-07-10_16-38-58_streamLog_tennis_S16.hdf5")
    """Xsens HDF5 file providing subject T-pose proportions."""

    tpose_variant: str = "Tpose"
    """Static Xsens pose variant to preview."""

    port: int = 8080
    """Viser server port."""

    sections: int = 14
    """Radial polygon count for generated segment shells."""


def _add_part(server: viser.ViserServer, path: str, part: AvatarMeshPart):
    return server.scene.add_mesh_simple(
        path,
        vertices=part.mesh.vertices,
        faces=part.mesh.faces,
        color=part.color,
        material="toon5",
        flat_shading=False,
        side="double",
    )


def make_preview(config: XsensAvatarMeshPreviewConfig) -> viser.ViserServer:
    proportions = load_xsens_avatar_proportions(config.hdf5_path, variant=config.tpose_variant)
    avatar_parts = build_xsens_avatar_meshes(proportions, sections=config.sections)
    validate_avatar_mesh_parts(avatar_parts)
    racket_parts = build_tennis_racket_meshes()

    server = viser.ViserServer(port=config.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=4.0, height=4.0, position=(0.0, 0.0, 0.0))
    server.scene.add_frame("/xsens", show_axes=False)

    shell_handles = []
    panel_handles = []
    orientation_cue_handles = []
    accent_handles = []
    diagnostic_handles = []
    racket_handles = []
    for segment_idx, segment_name in enumerate(proportions.segment_names):
        if segment_name == XSENS_RACKET_SOURCE_SEGMENT:
            continue
        segment_path = f"/xsens/segments/{segment_name.replace(' ', '_')}"
        server.scene.add_frame(
            segment_path,
            show_axes=False,
            position=proportions.tpose_positions_m[segment_idx],
            wxyz=proportions.tpose_quaternions_wijk[segment_idx],
        )
        diagnostic_handles.append(
            server.scene.add_frame(
                f"{segment_path}/diagnostic_axes",
                show_axes=True,
                axes_length=0.09,
                axes_radius=0.003,
                visible=False,
            )
        )
        for part in avatar_parts.get(segment_name, ()):
            handle = _add_part(server, f"{segment_path}/{part.name}", part)
            if part.category == "accent":
                accent_handles.append(handle)
            elif part.category == "orientation_cue":
                orientation_cue_handles.append(handle)
            elif part.category == "panel":
                panel_handles.append(handle)
            else:
                shell_handles.append(handle)

    # The source HDF5 stream calls this frame RightHandSword for historical
    # compatibility; the scene exposes the actual tracked prop instead.
    prop_idx = proportions.segment_index(XSENS_RACKET_SOURCE_SEGMENT)
    racket_path = "/xsens/props/tennis_racket"
    server.scene.add_frame(
        racket_path,
        show_axes=False,
        position=proportions.tpose_positions_m[prop_idx],
        wxyz=proportions.tpose_quaternions_wijk[prop_idx],
    )
    diagnostic_handles.append(
        server.scene.add_frame(
            f"{racket_path}/diagnostic_axes",
            show_axes=True,
            axes_length=0.12,
            axes_radius=0.003,
            visible=False,
        )
    )
    racket_handles.extend(_add_part(server, f"{racket_path}/{part.name}", part) for part in racket_parts)

    with server.gui.add_folder("Xsens avatar"):
        show_shells = server.gui.add_checkbox("Shells", initial_value=True)
        show_panels = server.gui.add_checkbox("Dark panels", initial_value=True)
        show_orientation_cues = server.gui.add_checkbox("Hand orientation cues", initial_value=True)
        show_accents = server.gui.add_checkbox("Orange accents", initial_value=True)
        show_racket = server.gui.add_checkbox("Tennis racket", initial_value=True)
        show_axes = server.gui.add_checkbox("Segment axes", initial_value=False)

    @show_shells.on_update
    def _(_) -> None:
        for handle in shell_handles:
            handle.visible = bool(show_shells.value)

    @show_panels.on_update
    def _(_) -> None:
        for handle in panel_handles:
            handle.visible = bool(show_panels.value)

    @show_orientation_cues.on_update
    def _(_) -> None:
        for handle in orientation_cue_handles:
            handle.visible = bool(show_orientation_cues.value)

    @show_accents.on_update
    def _(_) -> None:
        for handle in accent_handles:
            handle.visible = bool(show_accents.value)

    @show_racket.on_update
    def _(_) -> None:
        for handle in racket_handles:
            handle.visible = bool(show_racket.value)

    @show_axes.on_update
    def _(_) -> None:
        for handle in diagnostic_handles:
            handle.visible = bool(show_axes.value)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Three-quarter view keeps the racket face, separated legs, and hand
        # silhouette simultaneously readable.
        client.camera.position = (1.8, -2.2, 1.62)
        client.camera.look_at = (0.0, 0.0, 0.92)
        client.camera.up = (0.0, 0.0, 1.0)

    print(f"[preview_avatar_mesh] Subject: {config.hdf5_path}")
    print(f"[preview_avatar_mesh] Generated {sum(len(value) for value in avatar_parts.values())} avatar parts")
    return server


def main(config: XsensAvatarMeshPreviewConfig) -> None:
    make_preview(config)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(XsensAvatarMeshPreviewConfig))
