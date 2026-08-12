"""Backend-independent rigid-body kinematic model definitions."""

from .model import (
    KinematicTree,
    MeshAttachment,
    PointSetAttachment,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
    ValidationReport,
    compute_joint_positions,
    compute_reference_joint_positions,
    validate_kinematic_tree,
    with_body_attachments,
)
from .morphology import (
    GroundingSurface,
    KinematicMorphologyAdapter,
    KinematicMotion,
    KinematicPose,
    LowestSurfaceGrounding,
    SurfacePoseEvaluator,
    reference_grounding_offset_m,
    reference_root_floor_clearance_m,
)

__all__ = [
    "GroundingSurface",
    "KinematicMorphologyAdapter",
    "KinematicMotion",
    "KinematicPose",
    "KinematicTree",
    "LowestSurfaceGrounding",
    "MeshAttachment",
    "PointSetAttachment",
    "RigidBodyDefinition",
    "SphericalJointDefinition",
    "SurfacePoseEvaluator",
    "Transform",
    "ValidationReport",
    "compute_joint_positions",
    "compute_reference_joint_positions",
    "reference_grounding_offset_m",
    "reference_root_floor_clearance_m",
    "validate_kinematic_tree",
    "with_body_attachments",
]
