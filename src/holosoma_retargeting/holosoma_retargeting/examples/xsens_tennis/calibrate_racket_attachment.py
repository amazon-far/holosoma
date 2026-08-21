#!/usr/bin/env python3
"""Interactively inspect and save the G1 palm-to-tennis-racket attachment."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.src.paths import PACKAGE_ROOT  # noqa: E402
from holosoma_retargeting.viser_player import (  # noqa: E402
    add_g1_tennis_racket,
    update_g1_tennis_racket_pose,
)
from holosoma_retargeting.xsens.tennis_racket import (  # noqa: E402
    TennisRacketAttachment,
    attachment_handle_intersects_palm,
    load_tennis_racket_attachment,
    save_tennis_racket_attachment,
)


@dataclass(frozen=True)
class Config:
    robot_urdf: Path = Path("models/g1/g1_29dof.urdf")
    attachment_path: Path | None = None
    save_path: Path = Path("tennis_racket_attachment_override.json")
    port: int = 8080


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else PACKAGE_ROOT / expanded


def _attachment_from_controls(
    base: TennisRacketAttachment,
    xyz: list[object],
    rpy: list[object],
) -> TennisRacketAttachment:
    quaternion_xyzw = Rotation.from_euler("xyz", [float(control.value) for control in rpy], degrees=True).as_quat()
    return replace(
        base,
        position_m=np.asarray([float(control.value) for control in xyz]),
        quaternion_wxyz=quaternion_xyzw[[3, 0, 1, 2]],
        calibration_source="global",
    )


def main(config: Config) -> None:
    attachment = load_tennis_racket_attachment(
        None if config.attachment_path is None else _resolve(config.attachment_path)
    )
    robot_path = _resolve(config.robot_urdf)
    robot = yourdfpy.URDF.load(
        str(robot_path),
        mesh_dir=str(robot_path.parent),
        load_meshes=True,
        build_scene_graph=True,
    )
    server = viser.ViserServer(port=config.port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)
    server.scene.add_frame("/robot", show_axes=False)
    robot_handle = ViserUrdf(server, urdf_or_path=robot, root_node_name="/robot")
    robot_handle.update_cfg(np.zeros(len(robot_handle.get_actuated_joint_limits())))
    racket_frame, _ = add_g1_tennis_racket(server)

    initial_rpy = Rotation.from_quat(attachment.quaternion_wxyz[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    with server.gui.add_folder("Rigid hand → racket attachment"):
        xyz = [
            server.gui.add_number(label, initial_value=float(value), step=0.001)
            for label, value in zip(("X (m)", "Y (m)", "Z (m)"), attachment.position_m, strict=True)
        ]
        rpy = [
            server.gui.add_number(label, initial_value=float(value), step=1.0)
            for label, value in zip(("Roll (deg)", "Pitch (deg)", "Yaw (deg)"), initial_rpy, strict=True)
        ]
        status = server.gui.add_text("Palm validation", initial_value="")
        save_button = server.gui.add_button("Save override")

    current: dict[str, TennisRacketAttachment] = {"value": attachment}

    def refresh() -> None:
        current["value"] = _attachment_from_controls(attachment, xyz, rpy)
        update_g1_tennis_racket_pose(racket_frame, robot, current["value"])
        status.value = (
            "PASS: handle center is inside the palm interior"
            if attachment_handle_intersects_palm(current["value"])
            else "FAIL: move the handle center off the hand surface and into the palm interior"
        )

    for control in (*xyz, *rpy):

        @control.on_update
        def _(_event) -> None:
            refresh()

    @save_button.on_click
    def _(_event) -> None:
        output = _resolve(config.save_path)
        save_tennis_racket_attachment(current["value"], output)
        status.value = f"Saved {output}"

    refresh()
    print(f"Open the Viser URL above. The override will be saved to {_resolve(config.save_path)}.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Config))
