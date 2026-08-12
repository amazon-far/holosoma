"""Xsens-specific loading, calibration, model generation, and tracking utilities."""

from .g1_kinematic_reduction import (
    G1Anthropometry,
    G1XsensProportionReport,
    G1XsensReductionConfig,
    build_g1_proportioned_xsens_tree,
    export_g1_proportioned_xsens_usd,
    extract_g1_anthropometry,
    g1_anthropometry_to_xsens_avatar_proportions,
)
from .morphology_adaptation import (
    XsensGroundingMode,
    XsensRootMotionConfig,
    XsensRootMotionMode,
    XsensRootMotionReport,
    adapt_xsens_motion_to_g1,
    apply_xsens_root_motion,
    build_subject_xsens_reference_model,
    build_xsens_morphology_adapter,
    xsens_body_to_source_mapping,
)

__all__ = [
    "G1Anthropometry",
    "G1XsensProportionReport",
    "G1XsensReductionConfig",
    "XsensGroundingMode",
    "XsensRootMotionConfig",
    "XsensRootMotionMode",
    "XsensRootMotionReport",
    "adapt_xsens_motion_to_g1",
    "apply_xsens_root_motion",
    "build_g1_proportioned_xsens_tree",
    "build_subject_xsens_reference_model",
    "build_xsens_morphology_adapter",
    "export_g1_proportioned_xsens_usd",
    "extract_g1_anthropometry",
    "g1_anthropometry_to_xsens_avatar_proportions",
    "xsens_body_to_source_mapping",
]
