"""Export one independent calibrated XSens model USD per input HDF5 file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from holosoma_retargeting.xsens.usd_conversion import convert_xsens_hdf5_to_usd


@dataclass(frozen=True)
class XsensUsdExportCli:
    hdf5_path: Path | None = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    include_visuals: bool = True
    include_landmarks: bool = True
    include_tennis_racket: bool = True


def _inputs(config: XsensUsdExportCli) -> tuple[Path, ...]:
    if (config.hdf5_path is None) == (config.input_dir is None):
        raise ValueError("Specify exactly one of --hdf5-path or --input-dir")
    if config.hdf5_path is not None:
        return (config.hdf5_path,)
    assert config.input_dir is not None
    paths = tuple(sorted((*config.input_dir.glob("*.hdf5"), *config.input_dir.glob("*.h5"))))
    if not paths:
        raise FileNotFoundError(f"No .hdf5/.h5 files found in {config.input_dir}")
    return paths


def main(config: XsensUsdExportCli) -> None:
    failures: list[str] = []
    for hdf5_path in _inputs(config):
        output_path = None
        if config.output_dir is not None:
            output_path = config.output_dir / f"{hdf5_path.stem}_xsens_model.usda"
        try:
            report = convert_xsens_hdf5_to_usd(
                hdf5_path,
                output_path,
                include_visuals=config.include_visuals,
                include_landmarks=config.include_landmarks,
                include_tennis_racket=config.include_tennis_racket,
            )
            print(
                f"{report.source_path} -> {report.output_path} "
                f"({report.body_count} bodies, {report.joint_count} joints, "
                f"max anchor residual {report.max_joint_residual_m:.3g} m)"
            )
        except Exception as exc:
            failures.append(f"{hdf5_path}: {exc}")
    if failures:
        raise RuntimeError("Failed XSens USD exports:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main(tyro.cli(XsensUsdExportCli))
