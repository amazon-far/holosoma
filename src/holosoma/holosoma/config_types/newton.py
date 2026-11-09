"""Newton controls that do not belong to the shared MuJoCo configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveCount = Annotated[int, Field(gt=0, strict=True)]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class NewtonCollisionConfig:
    """Newton collision detection, active when ``collision_source="newton"``."""

    broad_phase: Literal["explicit", "nxn", "sap"] = "explicit"
    """Candidate-pair algorithm; explicit uses the authored collision pairs."""
    reduce_contacts: bool = True
    """Reduce collision manifolds before solving."""
    rigid_contact_max: PositiveCount | None = None
    """Total rigid-contact capacity; None lets Newton estimate it."""
    max_triangle_pairs: PositiveCount = 1_000_000
    """Triangle-pair capacity for mesh collision detection."""
    soft_contact_max: PositiveCount | None = None
    """Soft-contact capacity; None lets Newton estimate it."""
    soft_contact_margin: Nonnegative = 0.01
    """Soft-contact detection margin in metres."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class NewtonPhysicsConfig:
    """Newton contact and mesh controls, authored before physics compilation.

    MuJoCo friction, solver parameters and gravity compensation use ``mujoco``.
    Mass and density use the existing shared PhysicsConfig fields.
    """

    contact_stiffness: Nonnegative | None = None
    """Contact spring stiffness."""
    contact_damping: Nonnegative | None = None
    """Contact spring damping."""
    contact_friction_gain: Nonnegative | None = None
    """Tangential contact stiffness."""
    contact_adhesion: Nonnegative | None = None
    """Contact adhesion force."""
    max_hull_vertices: PositiveCount | None = None
    """Maximum convex-hull vertices."""
    sdf_max_resolution: PositiveCount | None = None
    """Maximum signed-distance-field resolution."""
    sdf_target_voxel_size: Nonnegative | None = None
    """Target signed-distance-field voxel size in metres."""
    sdf_padding: Nonnegative | None = None
    """Signed-distance-field padding in metres."""
    hydroelastic_enabled: bool | None = None
    """Enable hydroelastic collision for this shape."""
    hydroelastic_stiffness: Nonnegative | None = None
    """Hydroelastic contact stiffness."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class IsaacLab3RendererConfig:
    """Camera options, interpreted by the selected renderer.

    None retains the selected SDK renderer's default. Explicit unsupported
    options fail at startup. Kit launch presets remain in ``isaacsim``.
    """

    enable_shadows: bool | None = None
    """Enable shadows on Warp or OVRTX."""
    use_cuda_graph: bool | None = None
    """Capture Warp rendering kernels on CUDA."""
    enable_textures: bool | None = None
    """Enable Warp texture sampling."""
    enable_ambient_lighting: bool | None = None
    """Enable Warp ambient lighting."""
    enable_backface_culling: bool | None = None
    """Cull back faces in Warp."""
    max_distance: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    """Maximum Warp ray distance in metres."""
    depth_clipping_behavior: Literal["max", "zero", "none"] | None = None
    """Warp depth values beyond the maximum ray distance."""
    colorize_semantic_segmentation: bool | None = None
    """Return colors instead of semantic IDs."""
    colorize_instance_segmentation: bool | None = None
    """Return colors instead of instance IDs."""
