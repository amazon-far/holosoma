"""Configuration support and runtime evidence for the pinned Isaac Lab 3.0 release."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from holosoma.config_types.simulator import IsaacPhysicsBackend

if TYPE_CHECKING:
    from holosoma.config_types.simulator import SimulatorInitConfig


class Verification(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    UNPORTED = "unported"


@dataclass(frozen=True)
class BackendStatus:
    backend: IsaacPhysicsBackend
    usable: bool
    reachable: bool
    summary: str
    detail: str


BACKEND_STATUS: tuple[BackendStatus, ...] = (
    BackendStatus(
        IsaacPhysicsBackend.PHYSX,
        True,
        True,
        "Four-environment state, cloning and contact-support checks pass on GA.",
        "",
    ),
    BackendStatus(
        IsaacPhysicsBackend.NEWTON,
        True,
        True,
        "Four-environment state, cloning and contact-support checks pass on GA.",
        "",
    ),
    BackendStatus(
        IsaacPhysicsBackend.OVPHYSX,
        False,
        False,
        "Holosoma's current asset and sensor pipeline requires Kit.",
        "OvPhysX is a Kit-free runtime and cannot share a process with Kit. "
        "Holosoma currently uses Kit for USD/URDF import and RTX sensors. Select PhysX or Newton.",
    ),
)

HEADLESS = "(none)"
RENDERER_AGNOSTIC: tuple[str, ...] = ()
KNOWN_VISUALIZERS = ("kit", "newton_gl", "newton_rtx", "rerun", "viser")


@dataclass(frozen=True)
class Capability:
    """Recorded results; an unmeasured backend never inherits another backend's verdict."""

    name: str
    physx: Verification
    note: str
    newton: Verification | None = None
    renderers: tuple[str, ...] = RENDERER_AGNOSTIC

    @property
    def ported(self) -> bool:
        return self.physx is not Verification.UNPORTED

    def on(self, backend: IsaacPhysicsBackend) -> Verification:
        if not self.ported:
            return Verification.UNPORTED
        if not _backend_status(backend).reachable:
            return Verification.BLOCKED
        if backend is IsaacPhysicsBackend.PHYSX:
            return self.physx
        if backend is IsaacPhysicsBackend.NEWTON and self.newton is not None:
            return self.newton
        return Verification.UNVERIFIED


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "Rigid-body state read/write",
        Verification.VERIFIED,
        "static_move_assert: four environments, static relocation and dynamic contact support.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Articulation state read/write",
        Verification.VERIFIED,
        "all_root_states_unified_assert: robot/object reads, full and partial writes.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Kinematic playback",
        Verification.VERIFIED,
        "playback_assert: PhysX clip tracking, FK and frozen physics clock pass. "
        "Newton validation is blocked by native Kit startup crashes before the scene is built.",
        newton=Verification.UNVERIFIED,
    ),
    Capability(
        "Scene spawn (USD, multi-body)",
        Verification.VERIFIED,
        "scene_spawn_assert: four-environment multibody and multibody-override scenes; object_spawner_cfg_assert.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Environment cloning",
        Verification.VERIFIED,
        "static_move_assert: four distinct environments populated through native ClonePlan.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Lighting rig",
        Verification.VERIFIED,
        "scene_spawn_assert --probe-lights: UsdLux types, properties and orientations.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Contact sensors",
        Verification.VERIFIED,
        "isaac_sensor_state_assert: finite forces/history and contact response in all four environments.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Height-scanner ray caster",
        Verification.VERIFIED,
        "isaac_sensor_state_assert: finite ray hits at the known ground height in four environments.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Cameras (robot-link / world mount)",
        Verification.VERIFIED,
        "Headless RGB and depth capture; geometric and cadence harnesses cover the shared camera path.",
        newton=Verification.VERIFIED,
        renderers=(HEADLESS,),
    ),
    Capability(
        "Domain randomization (mass, friction, inertia, CoM)",
        Verification.VERIFIED,
        "dr_matrix_assert: read-back of robot and object properties. Newton has one friction coefficient.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "External wrench / push events",
        Verification.VERIFIED,
        "dr_matrix_assert: robot and object pushes with velocity read-back.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Debug draw / viewer keyboard",
        Verification.UNVERIFIED,
        "Shared viewport tracking and Kit keyboard controller; desktop viewers were not launched.",
    ),
    Capability(
        "Cameras (actor mount)",
        Verification.VERIFIED,
        "camera_actor_mount_assert: nonblank frames change when the owning actor moves.",
        newton=Verification.VERIFIED,
        renderers=(HEADLESS,),
    ),
    Capability(
        "LiDAR",
        Verification.VERIFIED,
        "lidar_assert: two environments, static/moving meshes, ray cycling, filters and update cadence.",
        newton=Verification.VERIFIED,
    ),
    Capability(
        "Video recording",
        Verification.VERIFIED,
        "video_recording_assert: nonblank frames and encoded MP4 from environment 1.",
        newton=Verification.VERIFIED,
        renderers=(HEADLESS,),
    ),
)


def _backend_status(backend: IsaacPhysicsBackend) -> BackendStatus:
    return next(status for status in BACKEND_STATUS if status.backend is backend)


def validate(config: SimulatorInitConfig, sensors: Mapping[str, object]) -> None:
    """Validate engine and visualizer settings before importing simulator extensions.

    Existing Holosoma sensor types use the shared Isaac Sim builders. Verification
    metadata above records measured coverage; it does not disable unmeasured features.
    """
    status = _backend_status(config.isaac_physics_backend)
    if not status.usable:
        raise NotImplementedError(
            f"The {status.backend.value!r} physics backend is not supported by holosoma on "
            f"Isaac Sim 6.2. {status.detail} The Isaac Sim 5.1 preset is simulator:isaacsim."
        )
    if config.isaac_physics_backend is not IsaacPhysicsBackend.NEWTON:
        newton_set = {name: value for name, value in vars(config.isaac_newton).items() if value is not None}
        if newton_set:
            raise ValueError(
                f"isaac_newton tuning {newton_set} was supplied, but isaac_physics_backend is "
                f"{config.isaac_physics_backend.value!r}. Select isaac_physics_backend='newton' "
                "or remove the Newton settings."
            )
    unknown = sorted(set(config.isaac_visualizers) - set(KNOWN_VISUALIZERS))
    if unknown:
        raise ValueError(f"Unknown isaac_visualizers {unknown}. Isaac Lab 3.0 accepts {list(KNOWN_VISUALIZERS)}.")
