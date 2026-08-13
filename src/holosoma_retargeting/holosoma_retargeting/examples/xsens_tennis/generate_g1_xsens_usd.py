"""Generate a source-independent Xsens USD with G1 proportions.

Example:
    python examples/xsens_tennis/generate_g1_xsens_usd.py \
        --output-path demo_results/g1/models/g1_proportioned_xsens.usda

Add ``--preserve-joint-offsets`` to retain the translations between the axes
inside G1 compound shoulder, hip, wrist, ankle, and waist joints.  The default
collapses those axes to idealized Xsens spherical joints while straight adapter
spans retain the shoulder and hip cluster extents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from holosoma_retargeting.xsens.g1_kinematic_reduction import (
    G1XsensReductionConfig,
    export_g1_proportioned_xsens_usd,
)
from holosoma_retargeting.src.paths import DEMO_RESULTS_DIR


@dataclass(frozen=True)
class G1XsensUsdCli:
    output_path: Path = DEMO_RESULTS_DIR / "g1/models/g1_proportioned_xsens.usda"
    """Generated OpenUSD ASCII stage."""

    robot_model_path: Path | None = None
    """Optional G1 MuJoCo XML override; defaults to the packaged 29-DoF model."""

    report_path: Path | None = None
    """Optional JSON report path; defaults to the output path with a .json suffix."""

    preserve_joint_offsets: bool = False
    """Retain translations between G1 compound-joint axes."""

    include_visuals: bool = True
    """Include G1-proportioned procedural Xsens avatar geometry."""


def main(cli: G1XsensUsdCli) -> None:
    report = export_g1_proportioned_xsens_usd(
        cli.output_path,
        robot_model_path=cli.robot_model_path,
        report_path=cli.report_path,
        config=G1XsensReductionConfig(
            preserve_joint_offsets=cli.preserve_joint_offsets,
            include_visuals=cli.include_visuals,
        ),
    )
    print(
        f"{report.source_path} -> {report.output_path} "
        f"({report.body_count} bodies, {report.joint_count} joints, "
        f"preserve_joint_offsets={report.preserve_joint_offsets}, "
        f"max length error {report.max_length_error_m:.3g} m, "
        f"max anchor residual {report.max_joint_residual_m:.3g} m)"
    )
    print(f"Proportion report: {report.report_path}")


if __name__ == "__main__":
    main(tyro.cli(G1XsensUsdCli))
