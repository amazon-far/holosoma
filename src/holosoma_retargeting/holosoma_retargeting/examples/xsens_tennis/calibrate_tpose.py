"""
Calibrate a clean G1 T-pose from Xsens tennis HDF5 data.

Usage:
    python examples/xsens_tennis/calibrate_tpose.py \
    --data-path demo_data/xsens_tennis \
    --task-name 2026-06-14_tennis_S02_xsens_myo_data_02 \
    --robot g1 \
    --variant Tpose \
    --save-path demo_results/g1/calibration/xsens_tennis/2026-06-14_tennis_S02_xsens_myo_data_02_tpose_calibration.npz
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.xsens.tpose_calibration import (  # noqa: E402
    XsensTposeCalibrationConfig,
    resolve_xsens_tennis_hdf5_path,
    save_xsens_tpose_calibration,
    solve_xsens_tpose_calibration,
)
from holosoma_retargeting.src.paths import DEMO_RESULTS_DIR  # noqa: E402


@dataclass
class XsensTennisTposeCalibrationCli:
    """CLI configuration for Xsens tennis T-pose calibration."""

    data_path: Path = Path("demo_data/xsens_tennis")
    """Directory containing Xsens tennis HDF5 files."""

    task_name: str = "2026-06-14_tennis_S02_xsens_myo_data_01"
    """Task stem or explicit HDF5 filename."""

    robot: str = "g1"
    """Robot type. First implementation supports g1."""

    variant: str = "Tpose"
    """T-pose variant under xsens-segments-tpose."""

    save_path: Path | None = None
    """Output calibration .npz path."""

    robot_urdf_file: str | None = None
    """Optional robot URDF path override."""

    default_human_height: float = 1.78
    """Human height used to scale Xsens T-pose to the robot."""

    max_nfev: int = 400
    """Maximum function evaluations for least-squares IK."""

    verbose: int = 1
    """SciPy least_squares verbosity: 0, 1, or 2."""


def _default_save_path(robot: str, task_name: str) -> Path:
    task_stem = Path(task_name).stem
    return DEMO_RESULTS_DIR / robot / "calibration" / "xsens_tennis" / f"{task_stem}_tpose_calibration.npz"


def main(cfg: XsensTennisTposeCalibrationCli) -> None:
    hdf5_path = resolve_xsens_tennis_hdf5_path(cfg.data_path, cfg.task_name)
    save_path = cfg.save_path or _default_save_path(cfg.robot, cfg.task_name)
    calibration_config = XsensTposeCalibrationConfig(
        robot_type=cfg.robot,
        variant=cfg.variant,
        robot_urdf_file=cfg.robot_urdf_file,
        default_human_height=cfg.default_human_height,
        max_nfev=cfg.max_nfev,
        verbose=cfg.verbose,
    )
    print(f"[xsens_tpose_calibration] Loading: {hdf5_path}")
    result = solve_xsens_tpose_calibration(hdf5_path, config=calibration_config)
    save_xsens_tpose_calibration(result, save_path, fps=calibration_config.fps)

    print(f"[xsens_tpose_calibration] Saved: {save_path}")
    print(f"[xsens_tpose_calibration] solver_success={result.solver_success} cost={result.solver_cost:.4f}")
    print(f"[xsens_tpose_calibration] head_candidate_status={result.head_candidate_status}")
    if result.active_orientation_mapping_names:
        active_pairs = ", ".join(
            f"{xsens}->{link}" for xsens, link in zip(result.active_orientation_mapping_names, result.robot_link_names)
        )
    else:
        active_pairs = "(none)"
    print(f"[xsens_tpose_calibration] active_orientation_mapping={active_pairs}")


if __name__ == "__main__":
    main(tyro.cli(XsensTennisTposeCalibrationCli))
